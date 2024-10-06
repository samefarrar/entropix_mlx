from mlx_lm import load
import argparse
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
The following is a multiple choice question in a Medical exam. Please provide the correct answer, along with a final explanation after
thinking through the question. Provide your final answer, e.g. "A", "B", "C", or "D", within <answer> tags.

For example:
<|eot_id|><|start_header_id|>user<|end_header_id|>
A patient presents with a history of recurrent episodes of abdominal pain, bloating, and diarrhea. The symptoms are relieved by defecation. Which of the following is the most likely diagnosis?
A. Irritable bowel syndrome
B. Crohn's disease
C. Ulcerative colitis
D. Celiac disease

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
<thinking> Let's think through this step-by-step:

1. The patient has recurrent episodes, which suggests a chronic condition.
2. The main symptoms are:
   - Abdominal pain
   - Bloating
   - Diarrhea
3. Importantly, the symptoms are relieved by defecation.

Now, let's consider each option:

A. Irritable Bowel Syndrome (IBS):
   - IBS is characterized by recurrent abdominal pain, bloating, and changes in bowel habits.
   - Symptoms are often relieved by defecation.
   - It's a functional disorder without structural abnormalities.

B. Crohn's Disease:
   - This is an inflammatory bowel disease.
   - It can cause abdominal pain and diarrhea, but symptoms are not typically relieved by defecation.
   - Often involves more severe symptoms and can affect any part of the digestive tract.

C. Ulcerative Colitis:
   - Another inflammatory bowel disease.
   - Causes abdominal pain and diarrhea, but symptoms are not typically relieved by defecation.
   - Specifically affects the colon and rectum.

D. Celiac Disease:
   - An autoimmune disorder triggered by gluten.
   - Can cause abdominal pain and diarrhea, but symptoms are not typically relieved by defecation.
   - Usually accompanied by other symptoms like fatigue and weight loss.

Given the description, especially the relief of symptoms after defecation, the most likely diagnosis is Irritable Bowel Syndrome (IBS). IBS is a functional gastrointestinal disorder that matches the described symptoms perfectly, including the relief after bowel movements.
</thinking>
<answer>A</answer>

<|eot_id|><|start_header_id|>user<|end_header_id|>
Which of the following is a correct description of a language learning disability?
A It shows difficulties in reading and writing.
B It does not show problems with higher-level language abilities.
C It occurs due to visual or motor disabilities.
D It often accompanies autism spectrum disorder

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
<thinking>
"""

prompts = [prompt1, prompt2, prompt3, prompt4, prompt5]

def main():
    parser = argparse.ArgumentParser(description = "Generate text using Entropy based sampling based on input prompts")
    parser.add_argument("--prompt", action="store_true", help = "Use predefined prompts from mlx_entropix.prompts")
    parser.add_argument("--input", type = str, help = "Input prompt to generate text")
    args = parser.parse_args()


    if args.input:
        prompts_to_use = [args.input]
    elif args.prompt:
        prompts_to_use = prompts
    else:
        print("No input provided. Use --prompts to use predefined prompts from mlx_entropix.prompts or provide a custom prompt using --input")
        print("Exiting...")

    model, tokenizer = load("weights/1B-Instruct")
    max_tokens = 8192

    for prompt in prompts_to_use:
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        response = generate(model, tokenizer, prompt=prompt, verbose=True, max_tokens = max_tokens)

if __name__ == "__main__":
    main()
