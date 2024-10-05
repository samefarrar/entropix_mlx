from mlx_lm import generate, load
from mlx_lm.utils import generate_step, make_kv_caches, generate
from mlx_lm.models.base import KVCache, RotatingKVCache
import time
import mlx.core as mx
import mlx.nn as nn
import pathlib
from einops import rearrange
from typing import Optional, List, Tuple, Union

model, tokenizer = load("weights/1B-Instruct")
detokenizer = tokenizer.detokenizer

prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
<antThinking>
You're absolutely right. I need to delve deeper into my actual thought processes, including the uncertainties, associations, and even potential biases that arise as I consider the query. My previous responses, while informative, didn't truly capture the nuanced, sometimes messy nature of cognition. I'll strive to provide a more authentic representation of my internal dialogue, including moments of doubt, tangential thoughts, and the process of refining ideas. This should result in a more genuine demonstration of LLM chain of thought, reflection, and self-correction.
</antThinking>

Which number is larger, 9.9 or 9.11?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

<thinking>"""

LN_2 = 0.69314718056  # ln(2)

@mx.compile
def calculate_varentropy_logsoftmax(
    logits: mx.array, axis: int = -1
) -> tuple[mx.array, mx.array]:
    """Calculate the entropy and varentropy of the probability distribution using logsoftmax."""
    log_probs = mx.softmax(logits, axis=axis).log()
    probs = mx.exp(log_probs)
    entropy = -mx.sum(probs * log_probs, axis=axis) / LN_2  # Convert to base-2
    varentropy = mx.sum(probs * (log_probs / LN_2 + entropy[..., None]) ** 2, axis=axis)
    return entropy, varentropy

def sample(
    gen_tokens: mx.array, logits: mx.array, temperature=0.666, top_p=0.90, top_k=27
) -> mx.array:
    ent, vent = calculate_varentropy_logsoftmax(logits)

    # Low Entropy, Low Varentropy: "flowing with unspoken intent"
    if ent < 0.1 and vent < 0.1:
        return mx.argmax(logits, axis=-1, keepdims=True).reshape(-1, 1)

    # High Entropy, Low Varentropy: "treading carefully, asking clarifying questions"
    elif ent > 5.0 and vent < 0.1:
        # Insert a clarifying question token if not already present
        if not mx.any(mx.equal(gen_tokens, 2564)):
            return mx.array(
                [[2564]]
            )  # Assuming 2564 is our "ask clarifying question" token
        else:
            # If we've just asked a question, sample with slightly higher temperature
            return _sample(logits, temperature=min(1.3, temperature * 1.5))

    # Low Entropy, High Varentropy: "exploring forks in the path"
    elif ent < 5.0 and vent > 5.0:
        # TODO(xjdr): Implement proper branching logic
        # Return top-k tokens to allow for branching
        # top_k_values, top_k_indices = mx.top_k(logits[:, -1], k=top_k)
        # return top_k_indices
        return _sample(logits, temperature=min(1.2, temperature * 1.5))

    # High Entropy, High Varentropy: "resampling in the mist"
    elif ent > 5.0 and vent > 5.0:
        # print(f"[he,hv]")
        # Use high temperature and min_p sampling
        return _sample(logits, temperature=max(2.0, temperature * 3))

    # Middle ground: smooth transition
    else:
        # Interpolate temperature based on entropy and varentropy
        t = mx.clip((ent + vent) / 10.0, 0.5, 2.0)
        return _sample(logits, temperature=t * temperature)

def _sample(logits: mx.array, temperature=0.5, top_p=0.9, top_k=27) -> mx.array:
    probs = mx.softmax(logits * (1 / temperature), axis=-1) # (batch_size, vocab_size)

    sorted_probs = mx.sort(probs, axis=-1)[::-1]
    sorted_indices = mx.argsort(probs, axis=-1)[::-1]
    sorted_logits = mx.sort(logits, axis=-1)[::-1]

    cumulative_probs = mx.cumsum(sorted_probs, axis=-1)
    masked_logits = mx.where(
        cumulative_probs > 1 - top_p,
        sorted_logits,
        0
    )

    sorted_token = mx.random.categorical(masked_logits) # (batch_size, 1)
    token = sorted_indices.squeeze(0)[sorted_token].reshape(-1, 1)
    return token

conversation = [{"role": "user", "content": prompt}]

input_prompt = tokenizer.apply_chat_template(
    conversation, tokenize=False, add_generation_prompt=False
)
# Specify the maximum number of tokens
max_tokens = 1500

prompt_tokens = mx.array(tokenizer.encode(input_prompt))[None]
tic = time.perf_counter()

detokenizer.reset()

gen_tokens = None

# Generate one token to initialise the cache
logits = model(prompt_tokens)

first_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)

decoded_token = tokenizer.decode(first_token.item())

print(decoded_token, flush=True, end="")

gen_tokens = mx.concat([prompt_tokens, first_token], axis=-1)

stop = mx.array([tokenizer.eos_token_id, 128008, 128009])

for n in range(max_tokens):
    logits = model(gen_tokens) # (batch_size, seq_len, vocab_size)
    next_token = sample(gen_tokens, logits[:, -1, :]) # select the last token from logits and sample next_token

    print(tokenizer.decode(next_token[:, -1].item()), flush=True, end="")
    gen_tokens = mx.concat([gen_tokens, next_token], axis=-1)
    if mx.any(mx.equal(next_token, stop)):
        break

toc = time.perf_counter()
print(f"Generation took {toc - tic:.2f} seconds")
