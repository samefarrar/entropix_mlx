import argparse
from pathlib import Path

from mlx_lm.tokenizer_utils import load_tokenizer
from mlx_lm.utils import generate as generate_mlx_lm
from mlx_lm.utils import load

from mlx_generate import generate
from mlx_model import load_entropix_model
from mlx_prompts import (
    create_prompts_from_csv,
    handle_chat_template,
    prompt1,
    prompt2,
    prompt3,
    prompt4,
    prompt5,
)
from mlx_sampler import SamplerConfig


def main():
    parser = argparse.ArgumentParser(
        description="Generate text using Entropy based sampling based on input prompts"
    )
    parser.add_argument(
        "--prompts",
        action="store_true",
        help="Use predefined prompts from mlx_entropix.prompts",
    )
    parser.add_argument(
        "--prompt_csv", action="store_true", help="Use prompts from data/prompts.csv"
    )
    parser.add_argument("--input", type=str, help="Input prompt to generate text")
    parser.add_argument(
        "--weights_dir",
        type=str,
        help="Directory containing the model weights",
        default="weights/Llama-3.2-1B-Instruct",
    )
    parser.add_argument(
        "--entropix",
        action="store_true",
        default=True,
        help="Use Entropix model for generation",
    )
    parser.add_argument(
        "--normal", action="store_true", help="Use normal model for generation"
    )
    parser.add_argument(
        "--prefill",
        type=str,
        help="Prefill the assistant response with a specific string",
        default=None,
    )
    args = parser.parse_args()

    if not args.prompts and not args.input and not args.prompt_csv:
        print(
            "No input provided. Use --prompts to use predefined prompts from mlx_entropix.prompts or provide a custom prompt using --input"
        )
        print("Exiting...")
        exit()

    if args.normal:
        model, tokenizer = load(args.weights_dir)
        model_with_scores = False
        sample_config_kwargs = SamplerConfig()
        sample_config_kwargs = {
            "temp": sample_config_kwargs.temperature,
            "top_p": sample_config_kwargs.top_p,
            "min_p": sample_config_kwargs.min_probability,
        }
    else:
        path = Path(args.weights_dir)
        tokenizer = load_tokenizer(path)
        model = load_entropix_model(path)
        model_with_scores = True
    max_tokens = 4096

    if args.input:
        prompt = handle_chat_template(
            args.input, tokenizer, prefill=args.prefill if args.prefill else None
        )
        if args.normal:
            response = generate_mlx_lm(
                model,
                tokenizer,
                prompt=prompt,
                verbose=True,
                max_tokens=max_tokens,
                **sample_config_kwargs,
            )
        else:
            response = generate(
                model, tokenizer, prompt=prompt, verbose=True, max_tokens=max_tokens
            )
    elif args.prompt_csv:
        prompts = create_prompts_from_csv("data/prompts.csv")
        for prompt in prompts:
            messages = handle_chat_template(
                prompt, tokenizer
            )
            if args.normal:
                response = generate_mlx_lm(
                    model,
                    tokenizer,
                    prompt=prompt,
                    verbose=True,
                    max_tokens=max_tokens,
                    **sample_config_kwargs,
                )
            else:
                response = generate(
                    model, tokenizer, prompt=prompt, verbose=True, max_tokens=max_tokens
                )
    elif args.prompts:
        prompts = [prompt1, prompt2, prompt3, prompt4, prompt5]
        for prompt in prompts:
            prompt = handle_chat_template(prompt, tokenizer)
            if args.normal:
                response = generate_mlx_lm(
                    model,
                    tokenizer,
                    prompt=prompt,
                    verbose=True,
                    max_tokens=max_tokens,
                    **sample_config_kwargs,
                )
            else:
                response = generate(
                    model, tokenizer, prompt=prompt, verbose=True, max_tokens=max_tokens
                )


if __name__ == "__main__":
    main()
