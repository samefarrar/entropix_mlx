from mlx_lm.utils import make_kv_caches
import mlx.core as mx
import mlx.nn as nn
from typing import Optional, List, Tuple, Union

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
        return mx.argmax(logits, axis=-1)

    # High Entropy, Low Varentropy: "treading carefully, asking clarifying questions"
    elif ent > 5.0 and vent < 0.1:
        #print(f"[he,lv]", flush = True, end = "")
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
        #print(f"[me,hv]", flush = True, end = "")
        # TODO(xjdr): Implement proper branching logic
        # Return top-k tokens to allow for branching
        # top_k_values, top_k_indices = mx.top_k(logits[:, -1], k=top_k)
        # return top_k_indices
        return _sample(logits, temperature=min(1.2, temperature * 1.5))

    # High Entropy, High Varentropy: "resampling in the mist"
    elif ent > 5.0 and vent > 5.0:
        #print(f"[he,hv]", flush = True, end = "")
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
    token = sorted_indices.squeeze(0)[sorted_token]
    return token
