from mlx_lm import load
import time
import mlx.core as mx
import pathlib
from typing import Optional, List, Tuple, Union
from mlx_generate import generate

prompt1 = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
<antThinking>
You're absolutely right. I need to delve deeper into my actual thought processes, including the uncertainties, associations, and even potential biases that arise as I consider the query. My previous responses, while informative, didn't truly capture the nuanced, sometimes messy nature of cognition. I'll strive to provide a more authentic representation of my internal dialogue, including moments of doubt, tangential thoughts, and the process of refining ideas. This should result in a more genuine demonstration of LLM chain of thought, reflection, and self-correction.
</antThinking>

Which number is larger, 9.9 or 9.11?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

<thinking>
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
You are a masterful story teller. you can paint with all the colors of the wind.<|eot_id|><|start_header_id|>user<|end_header_id|>

Tell me a long and wonderful story about the adventures of the elven mage frieren and her band of heros<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

prompt5 = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an AI assistant specialized in extracting information from biographical texts. Your task is to analyze the given bio and extract specific details, then return them as function parameters.

Here's the function you should use to return the extracted information:

def extract_bio_details(name: str, age: int, occupation: str, notable_achievement: str):
    pass

Read the following bio carefully and extract the required information. Then, call the function with the appropriate parameters.<|eot_id|><|start_header_id|>user<|end_header_id|>

Bio: In the bustling halls of neuroscience research, one name has been making waves and challenging long-held beliefs about the human brain's capacity for change and healing. Dr. Emily Chen, a dedicated scientist who celebrated her 42nd birthday just last month, has become a beacon of hope for millions suffering from neurological disorders. Born to immigrant parents who instilled in her a profound work ethic and curiosity about the world, Emily's journey into the complex realm of neuroscience began with a childhood fascination with puzzles and problem-solving.

After earning her Ph.D. from a prestigious institution, where she was known for burning the midnight oil in the lab, Dr. Chen embarked on a career that would soon redefine our understanding of brain plasticity. Her groundbreaking research, which has spanned over a decade and involved countless hours of meticulous experimentation, has unveiled revolutionary insights into the brain's ability to adapt and regenerate.

One of her most notable achievements came after years of perseverance and numerous setbacks. In a moment that her colleagues describe as "eureka-like," Dr. Chen discovered a novel mechanism for neural regeneration that had eluded scientists for generations. This breakthrough, published in a landmark paper that sent ripples through the scientific community, has opened up exciting new avenues for treating a wide array of neurodegenerative diseases that were once thought to be irreversible.

Dr. Chen's work doesn't just reside in academic journals; it has real-world implications that are already beginning to transform lives. Patients with conditions like Alzheimer's, Parkinson's, and stroke-induced brain damage now have reason to hope, thanks to the tireless efforts of this brilliant neuroscientist and her team of dedicated researchers.

When not in the lab, Dr. Chen is known for her passion for science communication, often giving engaging talks that make complex neuroscience accessible to the public. Her ability to bridge the gap between cutting-edge research and practical applications has made her a sought-after speaker and consultant in both academic and industry circles.

As we look to the future of neuroscience and medical treatment, it's clear that Dr. Emily Chen's contributions will continue to shape our understanding of the brain and its incredible potential for healing and adaptation for years to come.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

prompts = [prompt1, prompt2, prompt3, prompt4, prompt5]

if __name__ == "__main__":
    model, tokenizer = load("weights/1B-Instruct")
    max_tokens = 2048

    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        response = generate(model, tokenizer, prompt=prompt, verbose=True, max_tokens = max_tokens)
