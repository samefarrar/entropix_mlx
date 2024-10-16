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

logging.basicConfig(level=logging.INFO)

class EntropixModelProvider(ModelProvider):
    def __init__(self, cli_args: argparse.Namespace):
        self.cli_args = cli_args
        self.normal = cli_args.normal
        self.model_path = Path(cli_args.model_path)
        self.model = None
        self.tokenizer = None  # Initialize tokenizer to avoid reference before assignment
        self.is_model_loaded = False
        self.load(requested_model="default_model")

    def load(self, requested_model: str, requested_adapter: Optional[str] = None):
        if requested_model != "default_model":
            self.model_path = Path(requested_model)
        
        logging.info(f"Loading model from {self.model_path}...")
        
        if self.is_model_loaded:
            logging.info("Model already loaded. Returning...")
            return self.model, self.tokenizer
        
        try:
            if self.normal:
                self.model, self.tokenizer = load_mlx_lm(str(self.model_path))
            else:
                self.model = load_entropix_model(self.model_path)
                self.tokenizer = load_tokenizer(self.model_path)
            self.is_model_loaded = True
            logging.info(f"Model loaded from {self.model_path} successfully!")
        except (FileNotFoundError, ValueError) as e:
            logging.error(f"Error loading model: {e}")
            self.is_model_loaded = False
            raise

        return self.model, self.tokenizer

class EntropixAPIHandler(APIHandler):
    def __init__(self, model_provider: ModelProvider, *args, **kwargs):
        super().__init__(model_provider, *args, **kwargs)

    def handle_completion(self, prompt: str, stop_id_sequences: List[List[int]]):
        """
        Generate completions (non-streaming) for the given prompt.
        """
        detokenizer = self.tokenizer.detokenizer
        detokenizer.reset()

        tokens = []
        metrics_list = []
        finish_reason = "length"
        stop_sequence_suffix = None
        prompt = mx.array(prompt)

        logging.debug("Starting completion:")
        
        for (token, metrics), _ in zip(generate_step(prompt, model=self.model, sampler_config=self.sampler_config), range(self.max_tokens)):
            detokenizer.add_token(token)
            logging.debug(detokenizer.text)
            tokens.append(token)
            metrics["token"] = self.tokenizer.decode([token])
            metrics_list.append(metrics)

            stop_condition = stopping_criteria(tokens, stop_id_sequences, self.tokenizer.eos_token_id)

            if stop_condition.stop_met:
                finish_reason = "stop"
                if stop_condition.trim_length:
                    stop_sequence_suffix = self.tokenizer.decode(tokens[-stop_condition.trim_length:])
                break

        detokenizer.finalize()
        
        text = detokenizer.text if stop_sequence_suffix is None else detokenizer.text[:-len(stop_sequence_suffix)]
        
        response = self.generate_metrics_response(
            text=text,
            metrics=metrics_list,
            finish_reason=finish_reason,
            prompt_token_count=len(prompt),
            completion_token_count=len(tokens),
        )

        logging.debug(f"Outgoing response: {json.dumps(response)}")
        
        response_json = json.dumps(response).encode()
        
        # Send an additional Content-Length header when it is known
        self.send_header("Content-Length", str(len(response_json)))
        self.end_headers()
        
        self.wfile.write(response_json)
        self.wfile.flush()

    def handle_stream(self, prompt: str, stop_id_sequences: List[List[int]]):
        """
        Generate response to prompt and forward it to the client using a Server Sent Events (SSE) stream.
        """
        
        detokenizer = self.tokenizer.detokenizer
        detokenizer.reset()

        tokens = []
        metrics_list = []
        
        stop_sequence_suffix = None
        prompt = mx.array(prompt)

        logging.debug("Starting stream:")

        for (token, metrics), _ in zip(generate_step(prompt, model=self.model, sampler_config=self.sampler_config), range(self.max_tokens)):
            detokenizer.add_token(token)
            logging.debug(detokenizer.text)
            tokens.append(token)
            metrics["token"] = self.tokenizer.decode([token])
            metrics_list.append(metrics)

            stop_condition = stopping_criteria(tokens, stop_id_sequences, self.tokenizer.eos_token_id)

            if stop_condition.stop_met:
                if stop_condition.trim_length:
                    stop_sequence_suffix = self.tokenizer.decode(tokens[-stop_condition.trim_length:])
                break

            if any(sequence_overlap(tokens, sequence) for sequence in stop_id_sequences):
                continue

            new_text = detokenizer.last_segment
            response = self.generate_metrics_response(new_text, None, metrics=metrics)

            self.wfile.write(f"data: {json.dumps(response)}\n\n".encode())
            self.wfile.flush()

        detokenizer.finalize()
        
        last_segment = detokenizer.last_segment
        
        if last_segment:
            if stop_sequence_suffix is not None:
                last_segment = last_segment[:-len(stop_sequence_suffix)]
                
            response = self.generate_metrics_response(last_segment, "length", metrics=metrics_list)
            
            self.wfile.write(f"data: {json.dumps(response)}\n\n".encode())
            self.wfile.flush()

        if getattr(self.stream_options, "include_usage", False):
            response = self.completion_usage_response(len(prompt), len(tokens))
            self.wfile.write(f"data: {json.dumps(response)}\n\n".encode())

        # Indicate the end of the stream
        self.wfile.write("data: [DONE]\n\n".encode())
        self.wfile.flush()

    def do_POST(self):
        """Handle POST requests."""
        
        # Initialize sampler configuration on each POST request.
        self.sampler_config = SamplerConfig()
        
        return super().do_POST()

    def generate_metrics_response(
       self,
       text: str,
       finish_reason: Union[Literal["length", "stop"], None],
       metrics: Optional[Union[Dict[str, float], List[Dict[str, float]]] ]=None,
       prompt_token_count: Optional[int] = None,
       completion_token_count: Optional[int] = None,
   ) -> dict:
       """
       Generate a single response packet including the generated text and metrics.
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

       if not getattr(self.stream_options, 'stream', False):
           if isinstance(prompt_token_count, int) and isinstance(completion_token_count, int):
               response["usage"] = {
                   "prompt_tokens": prompt_token_count,
                   "completion_tokens": completion_token_count,
                   "total_tokens": prompt_token_count + completion_token_count,
               }

       choice = response["choices"][0]

       if self.object_type.startswith("chat.completion"):
           key_name = "delta" if getattr(self.stream_options,'stream', False) else "message"
           choice[key_name] = {"role": "assistant", "content": text}
       elif self.object_type == "text_completion":
           choice.update(text=text)
       else:
           raise ValueError(f"Unsupported response type: {self.object_type}")

       return response


def run_server(host:str , port:int , model_provider: ModelProvider , normal=False):
    """Run the HTTP server."""
    
    server_address = (host, port)
    
    handler_class = EntropixAPIHandler if not normal else APIHandler
    
    httpd = HTTPServer(server_address , handler_class(model_provider))
    
    logging.info(f"Starting server on {host}:{port}")
    
    httpd.serve_forever()

def main():
    """Main entry point for the server."""
    
    parser = argparse.ArgumentParser(description="Run an MLX model server")
    parser.add_argument("--model_path", type=str , default="weights/Llama-3.2-1B-Instruct", help="Path to the model weights")
    parser.add_argument("--normal", action="store_true", help="Use normal model for generation")
    parser.add_argument("--host", type=str , default="localhost", help="Server host")
    parser.add_argument("--port", type=int , default=8000 , help="Server port")
    
    args = parser.parse_args()

    model_provider_class = ModelProvider if args.normal else EntropixModelProvider
    
    model_provider_instance= model_provider_class(cli_args=args)
    
    run_server(args.host , args.port , model_provider_instance , normal=args.normal)

if __name__ == "__main__":
    main()
