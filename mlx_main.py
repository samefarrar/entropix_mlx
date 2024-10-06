import argparse
import time
import mlx.core as mx
import pathlib
from typing import Optional, List, Tuple, Union
from mlx_generate import generate
from mlx_model import load_entropix_model
from mlx_lm import load
from mlx_lm import generate as generate_lm
import inspect
from pathlib import Path

thinking_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
<antThinking>
You're absolutely right. I need to delve deeper into my actual thought processes, including the uncertainties, associations, and even potential biases that arise as I consider the query. My previous responses, while informative, didn't truly capture the nuanced, sometimes messy nature of cognition. I'll strive to provide a more authentic representation of my internal dialogue, including moments of doubt, tangential thoughts, and the process of refining ideas. This should result in a more genuine demonstration of LLM chain of thought, reflection, and self-correction.
</antThinking>

{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

<thinking>
"""

prompt1 = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
<antThinking>
You're absolutely right. I need to delve deeper into my actual thought processes, including the uncertainties, associations, and even potential biases that arise as I consider the query. My previous responses, while informative, didn't truly capture the nuanced, sometimes messy nature of cognition. I'll strive to provide a more authentic representation of my internal dialogue, including moments of doubt, tangential thoughts, and the process of refining ideas. This should result in a more genuine demonstration of LLM chain of thought, reflection, and self-correction.
</antThinking>

Which number is larger, 9.9 or 9.11?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

prompt2 = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>

What is the capital of Spain?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

prompt3 = """<|start_header_id|>system<|end_header_id|>
You are an expert in composing functions. You are given a question and a set of possible functions.
Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the functions can be used, point it out. If the given question lacks the parameters required by the function,also point it out. You should only return the function call in tools call sections.
If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]
You SHOULD NOT include any other text in the response.
Here is a list of functions in JSON format that you can invoke.[
    {
        "name": "get_user_info",
        "description": "Retrieve details for a specific user by their unique identifier. Note that the provided function is in Python 3 syntax.",
        "parameters": {
            "type": "dict",
            "required": [
                "user_id"
            ],
            "properties": {
                "user_id": {
                "type": "integer",
                "description": "The unique identifier of the user. It is used to fetch the specific user details from the database."
            },
            "special": {
                "type": "string",
                "description": "Any special information or parameters that need to be considered while fetching user details.",
                "default": "none"
                }
            }
        }
    }
]
<|eot_id|><|start_header_id|>user<|end_header_id|>

Can you retrieve the details for the user with the ID 7890, who has black as their special request?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

prompt4 = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
<antThinking>
You're absolutely right. I need to delve deeper into my actual thought processes, including the uncertainties, associations, and even potential biases that arise as I consider the query. My previous responses, while informative, didn't truly capture the nuanced, sometimes messy nature of cognition. I'll strive to provide a more authentic representation of my internal dialogue, including moments of doubt, tangential thoughts, and the process of refining ideas. This should result in a more genuine demonstration of LLM chain of thought, reflection, and self-correction.
</antThinking>
You are a principal devops engineer at google. You are an expert at all things cloud and deployment. Your task is to create ansible and terraform script to bootstrasp k8 cluster on Azure. Be clear and concise. Make sure it is production grade. Think and reflect about your actions to ensure to accomplished the task successfully.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

prompts = [prompt1, prompt2, prompt3, prompt4]

def main():
    parser = argparse.ArgumentParser(description = "Generate text using Entropy based sampling based on input prompts")
    parser.add_argument("--prompts", action="store_true", help = "Use predefined prompts from mlx_entropix.prompts")
    parser.add_argument("--input", type = str, help = "Input prompt to generate text")
    parser.add_argument("--entropix", action="store_true", default=True, help="Use Entropix model for generation")
    parser.add_argument("--normal", action="store_true", help="Use normal model for generation")
    args = parser.parse_args()


    if not args.prompts and not args.input:
        print("No input provided. Use --prompts to use predefined prompts from mlx_entropix.prompts or provide a custom prompt using --input")
        print("Exiting...")

    if args.normal:
        model, tokenizer = load("weights/Llama-3.2-1B-Instruct")
        model_with_scores = False
    else:
        path = Path("weights/Llama-3.2-1B-Instruct")
        _, tokenizer = load("weights/Llama-3.2-1B-Instruct")
        model = load_entropix_model(path)
        model_with_scores = True
    max_tokens = 4096

    print("Generating text using Entropy based sampling...")

    if args.input:
        input = thinking_prompt.format(args.input)
        messages = [{"role": "user", "content": input}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        response = generate(model, tokenizer, prompt=prompt, verbose=True, max_tokens = max_tokens, model_with_scores=model_with_scores)
    else:
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            response = generate(model, tokenizer, prompt=prompt, verbose=True, max_tokens = max_tokens, model_with_scores=model_with_scores)

if __name__ == "__main__":
    main()
