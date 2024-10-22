import argparse
import json
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import mlx.core as mx
from mlx_lm.utils import load as load_mlx_lm
from mlx_lm.tokenizer_utils import load_tokenizer
from torch.utils.data.sampler import Sampler
from mlx_model import load_entropix_model
from mlx_generate import generate_step as generate_step
from mlx_lm.utils import generate_step as generate_step_mlx_lm
from mlx_lm.server import APIHandler, stopping_criteria, ModelProvider, sequence_overlap
from mlx_attention_sampler import SamplerConfig
from typing import List, Union, Literal, Optional, Dict

class EntropixModelProvider(ModelProvider):
    def __init__(self, cli_args: argparse.Namespace):
        self.cli_args = cli_args
        self.normal = cli_args.normal
        self.model_path = Path(cli_args.model_path)
        self.model = None
        self.is_model_loaded = False
        self.load(requested_model="default_model")

    def load(self, requested_model, requested_adapter = None):
        if requested_model != "default_model":
            self.model_path = Path(requested_model)
        if self.is_model_loaded:
            return self.model, self.tokenizer
        model_path = self.model_path
        try:
            logging.info(f"Loading model from {model_path}")
            if self.normal:
                self.model, self.tokenizer = load_mlx_lm(str(model_path))
            else:
                self.model = load_entropix_model(model_path)
                self.tokenizer = load_tokenizer(model_path)
            self.is_model_loaded = True
            print(f"Model loaded from {model_path} successfully!")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            self.is_model_loaded = False
            raise

        return self.model, self.tokenizer

class EntropixAPIHandler(APIHandler):
    def __init__(self, model_provider: ModelProvider, *args, **kwargs):
        super().__init__(model_provider, *args, **kwargs)

    def handle_completion(self, prompt, stop_id_sequences: List[List[int]]):
        """
        Generate completions (non-streaming) for the given prompt
        """
        detokenizer = self.tokenizer.detokenizer
        detokenizer.reset()

        tokens = []
        metrics_list = []
        finish_reason = "length"
        stop_sequence_suffix = None
        prompt = mx.array(prompt)

        logging.debug(f"Starting completion:")
        for (token, metrics), _ in zip(
            generate_step(
                prompt,
                model = self.model,
                sampler_config=self.sampler_config,
                key=self.key,
            ),
            range(self.max_tokens)
        ):
            detokenizer.add_token(token)
            logging.debug(detokenizer.text)
            tokens.append(token)
            metrics["token"] = self.tokenizer.decode([token])
            metrics_list.append(metrics)

            stop_condition = stopping_criteria(
                tokens, stop_id_sequences, self.tokenizer.eos_token_id
            )

            if stop_condition.stop_met:
                finish_reason = "stop"
                if stop_condition.trim_length:
                    stop_sequence_suffix = self.tokenizer.decode(
                        tokens[-stop_condition.trim_length :]
                    )
                break
        detokenizer.finalize()
        text = (
            detokenizer.text
            if stop_sequence_suffix is None
            else detokenizer.text[: -len(stop_sequence_suffix)]
        )
        response = self.generate_metrics_response(
            text=text,
            metrics=metrics_list,
            finish_reason=finish_reason,
            prompt_token_count=len(prompt),
            completion_token_count=len(tokens),
        )

        response_json = json.dumps(response).encode()
        indent = "\t"
        logging.debug(f"Outgoing response: {json.dumps(response, indent = indent)}")

         # Send an additional Content-Length header when it is known
        self.send_header("Content-Length", str(len(response_json)))
        self.end_headers()

        self.wfile.write(response_json)
        self.wfile.flush()

    def handle_stream(self, prompt, stop_id_sequences: List[List[int]]):
        """
        Generate response to prompt and forward it to the client using a Server Sent Events
        (SSE) stream.
        """

        self.end_headers()

        detokenizer = self.tokenizer.detokenizer
        detokenizer.reset()

        tokens = []
        metrics_list = []

        stop_sequence_suffix = None
        prompt = mx.array(prompt)

        logging.debug(f"Starting stream:")

        for (token, metrics), _ in zip(
            generate_step(
                prompt,
                model=self.model,
                sampler_config=self.sampler_config,
                key=self.key,
            ),
            range(self.max_tokens)
        ):
            detokenizer.add_token(token)
            logging.debug(detokenizer.text)
            tokens.append(token)
            metrics["token"] = self.tokenizer.decode([token])
            metrics_list.append(metrics)

            stop_condition = stopping_criteria(
                tokens, stop_id_sequences, self.tokenizer.eos_token_id
            )

            if stop_condition.stop_met:
                if stop_condition.trim_length:
                    stop_sequence_suffix = self.tokenizer.decode(
                        tokens[-stop_condition.trim_length :]
                    )
                break

            if any(
                (sequence_overlap(tokens, sequence) for sequence in stop_id_sequences)
            ):
                continue

            new_text = detokenizer.last_segment
            response = self.generate_metrics_response(new_text, None, metrics = metrics)

            self.wfile.write(f"data: {json.dumps(response)}\n\n".encode())
            self.wfile.flush()

        detokenizer.finalize()
        last_segment = detokenizer.last_segment
        if last_segment:
            if stop_sequence_suffix is not None:
                last_segment = last_segment[: -len(stop_sequence_suffix)]
            response = self.generate_metrics_response(last_segment, "length", metrics = metrics_list)
            self.wfile.write(f"data: {json.dumps(response)}\n\n".encode())
            self.wfile.flush()

        if self.stream_options is not None and self.stream_options["include_usage"]:
            response = self.completion_usage_response(len(prompt), len(tokens))
            self.wfile.write(f"data: {json.dumps(response)}\n\n".encode())

        self.wfile.write("data: [DONE]\n\n".encode())
        self.wfile.flush()

    def do_POST(self):
        """
        Respond to a POST request from a client.
        """
        endpoints = {
            "/v1/completions": self.handle_text_completions,
            "/v1/chat/completions": self.handle_chat_completions,
            "/chat/completions": self.handle_chat_completions,
        }

        if self.path not in endpoints:
            self._set_completion_headers(404)
            self.end_headers()
            self.wfile.write(b"Not Found")
            return

        # Fetch and parse request body
        content_length = int(self.headers["Content-Length"])
        raw_body = self.rfile.read(content_length)
        self.body = json.loads(raw_body.decode())
        indent = "\t"  # Backslashes can't be inside of f-strings
        logging.debug(f"Incoming Request Body: {json.dumps(self.body, indent=indent)}")
        assert isinstance(
            self.body, dict
        ), f"Request should be dict, but got {type(self.body)}"

        # Extract request parameters from the body
        self.stream = self.body.get("stream", False)
        self.seed = self.body.get("seed", None)
        sampler_config_params = self.body.get("sampler_config", {})
        self.sampler_config = SamplerConfig(**sampler_config_params)
        self.stream_options = self.body.get("stream_options", None)
        self.requested_model = self.body.get("model", "default_model")
        self.adapter = self.body.get("adapters", None)
        self.max_tokens = self.body.get("max_completion_tokens", None)
        if self.max_tokens is None:
            self.max_tokens = self.body.get("max_tokens", 512)
        self.temperature = self.body.get("temperature", 1.0)
        self.top_p = self.body.get("top_p", 1.0)
        self.repetition_penalty = self.body.get("repetition_penalty", 1.0)
        self.repetition_context_size = self.body.get("repetition_context_size", 20)
        self.logit_bias = self.body.get("logit_bias", None)
        self.logprobs = self.body.get("logprobs", -1)
        self.validate_model_parameters()

        if self.seed is not None:
            self.key = mx.random.key(seed=self.seed)
        else:
            self.key = None

        # Load the model if needed
        try:
            self.model, self.tokenizer = self.model_provider.load(
                self.requested_model, self.adapter
            )
        except:
            self._set_completion_headers(404)
            self.end_headers()
            self.wfile.write(b"Not Found")
            return

        # Get stop id sequences, if provided
        stop_words = self.body.get("stop")
        stop_words = stop_words or []
        stop_words = [stop_words] if isinstance(stop_words, str) else stop_words
        stop_id_sequences = [
            self.tokenizer.encode(stop_word, add_special_tokens=False)
            for stop_word in stop_words
        ]

        # Send header type
        (
            self._set_stream_headers(200)
            if self.stream
            else self._set_completion_headers(200)
        )

        # Call endpoint specific method
        prompt = endpoints[self.path]()

        # Call method based on response type
        method = self.handle_stream if self.stream else self.handle_completion
        method(prompt, stop_id_sequences)

    def generate_metrics_response(
       self,
       text: str,
       finish_reason: Union[Literal["length", "stop"], None],
       metrics: Optional[Union[Dict[str, float], List[Dict[str, float]]]] = None,
       prompt_token_count: Optional[int] = None,
       completion_token_count: Optional[int] = None,
   ) -> dict:
       """
       Generate a single response packet including the generated text and metrics.

       Args:
           text (str): Text generated by model
           finish_reason (Union[Literal["length", "stop"], None]): The reason the
             response is being sent: "length", "stop" or `None`.
           metrics (Optional[Dict[str, float]]): A dictionary of calculated metrics.
           prompt_token_count (Optional[int]): The number of tokens in the prompt,
             used to populate the "usage" field (not used when stream).
           completion_token_count (Optional[int]): The number of tokens in the
             response, used to populate the "usage" field (not used when stream).

       Returns:
           dict: A dictionary containing the response, including text and metrics.
       """
       response = {
           "id": self.request_id,
           "object": self.object_type,
           "model": self.requested_model,
           "created": self.created,
           "choices": [
               {
                   "index": 0,
                   "finish_reason": finish_reason,
               }
           ],
       }

       if metrics:
           response["metrics"] = metrics

       if not self.stream:
           if isinstance(prompt_token_count, int) and isinstance(completion_token_count, int):
               response["usage"] = {
                   "prompt_tokens": prompt_token_count,
                   "completion_tokens": completion_token_count,
                   "total_tokens": prompt_token_count + completion_token_count,
               }

       choice = response["choices"][0]

       if self.object_type.startswith("chat.completion"):
           key_name = "delta" if self.stream else "message"
           choice[key_name] = {"role": "assistant", "content": text}
       elif self.object_type == "text_completion":
           choice.update(text=text)
       else:
           raise ValueError(f"Unsupported response type: {self.object_type}")

       return response


def run_server(host, port, model_provider, normal = False):
    server_address = (host, port)
    if normal:
        handler_class = lambda *args, **kwargs: APIHandler(model_provider, *args, **kwargs)
    else:
        def create_handler(*args, **kwargs):
            return EntropixAPIHandler(model_provider, *args, **kwargs)
        handler_class = create_handler
    httpd = HTTPServer(server_address, handler_class)
    print(f"Starting server on {host}:{port}")
    httpd.serve_forever()

def main():
    parser = argparse.ArgumentParser(description="Run an MLX model server")
    parser.add_argument("--model_path", type=str, default = "weights/Llama-3.2-1B-Instruct", help="Path to the model weights")
    parser.add_argument("--normal", action="store_true", help="Use normal model for generation")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    args = parser.parse_args()

    if args.normal:
        args.model = args.model_path
        args.trust_remote_code = False
        args.chat_template = ""
        args.max_tokens = 2048
        args.adapter_path = None
        args.use_default_chat_template = True
        model_provider = ModelProvider(cli_args = args)
    else:
        model_provider = EntropixModelProvider(cli_args = args)
    run_server(args.host, args.port, model_provider, normal = args.normal)

if __name__ == "__main__":
    main()
