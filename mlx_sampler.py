from mlx_lm.utils import make_kv_caches
import mlx.core as mx
import mlx.nn as nn
from typing import Optional, List, Tuple, Union, Dict

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

@mx.compile
def calculate_metrics(logits: mx.array, attention_scores: mx.array) -> Dict[str, mx.array]:
    entropy, varentropy = calculate_varentropy_logsoftmax(logits)

    attention_probs = mx.softmax(attention_scores, axis=-1)
    attention_entropy = -mx.sum(attention_probs * mx.log2(mx.clip(attention_probs, 1e-10, 1.0)), axis = -1)
    attention_varentropy = mx.var(attention_entropy, axis = -1)

    mean_attention = mx.mean(attention_probs, axis = -1)
    agreement = mx.mean(mx.abs(attention_probs - mean_attention[:, :, :, None]), axis = (1, 2))

    interaction_strength = mx.mean(mx.abs(attention_scores), axis = (1, 2, 3))

    return {
        "logits_entropy": mx.mean(entropy),
        "logits_varentropy": mx.mean(varentropy),
        "attention_entropy": mx.mean(attention_entropy),
        "attention_varentropy": mx.mean(attention_varentropy),
        "agreement": mx.mean(agreement),
        "interaction_strength": interaction_strength
    }

def sample(
    gen_tokens: mx.array, logits: mx.array, scores: mx.array, temperature=0.666, top_p=0.90, top_k=27, min_p: float = 0.0
) -> mx.array:
    metrics = calculate_metrics(logits, scores)
    ent, vent = metrics["logits_entropy"], metrics["logits_varentropy"]
    attention_entropy, attention_varentropy = metrics["attention_entropy"], metrics["attention_varentropy"]
    agreement, interaction_strength = metrics["agreement"], metrics["interaction_strength"]

    # Low Entropy, Low Varentropy: "flowing with unspoken intent"
    if ent < 0.1 and vent < 0.1:
        return mx.argmax(logits[:, -1], axis=-1)

    # High Entropy, Low Varentropy: "treading carefully, asking clarifying questions"
    elif ent > 3.0 and vent < 0.1:
        #print("ε", flush = True, end = "")
        # Insert a clarifying question token if not already present
        if not mx.any(mx.equal(gen_tokens[:, -1], 2564).any()):
            return mx.array(
                [[2564]]
            )  # Assuming 2564 is our "ask clarifying question" token
        else:
            # If we've just asked a question, sample with slightly higher temperature
            temp_adj = 1.3 + 0.2 * attention_entropy # Increase temperature
            return _sample(logits, temperature=min(1.5, temperature * temp_adj), top_p = top_p, top_k = top_k, min_p = 0.0)

    # Low Entropy, High Varentropy: "exploring forks in the path"
    elif ent < 5.0 and vent > 5.0:
        #print("Ψ", flush = True, end = "")
        # TODO(xjdr): Implement proper branching logic
        # Return top-k tokens to allow for branching
        # top_k_values, top_k_indices = mx.top_k(logits[:, -1], k=top_k)
        # return top_k_indices
        temp_adj = 1.2 + 0.3 * interaction_strength
        top_k_adj = max(5, int(top_k * (1 + 0.5 * (1 - agreement))))
        return _sample(logits, temperature=min(1,5, temperature * temp_adj), top_p = top_p, top_k = top_k_adj, min_p = min_p)

    # High Entropy, High Varentropy: "resampling in the mist"
    elif ent > 5.0 and vent > 5.0:
        #print("!", flush = True, end = "")
        # Use high temperature and min_p sampling
        temp_adj = 2.0 + 0.5 * attention_varentropy
        top_p_adj = max(0.5, top_p - 0.2 * attention_entropy)
        return _sample(logits, temperature=max(2.0, temperature * temp_adj), top_p = top_p_adj, top_k = top_k, min_p = min_p)

    # Middle ground: smooth transition
    else:
        # Adjust temperature and top_k based on attention metrics
        #print("~", flush = True, end = "")
        t = mx.clip((ent + vent) / 10.0, 0.5, 2.0)
        temp_adj = t + 0.2 * attention_entropy + 0.1 * attention_varentropy
        top_k_adj = max(5, int(top_k * (1 + 0.3 * interaction_strength - 0.2 * agreement)))
        return _sample(logits, temperature=temp_adj * temperature, top_p=top_p, top_k=top_k_adj, min_p=min_p)
        # Interpolate temperature based on entropy and varentropy
        # return adaptive_sample(
        #     logits, metrics, gen_tokens, n_samples=2, base_temperature=temperature, base_top_p=top_p, base_top_k=top_k, base_min_p=min_p
        # )

def adaptive_sample(
    logits: mx.array,
    metrics: Dict[str, mx.array],
    gen_tokens: mx.array,
    n_samples: int = 5,
    base_temperature: float = 0.666,
    base_top_p: float = 0.90,
    base_top_k: int = 27,
    base_min_p: float = 0.0,
):
    logits_uncertainty = metrics["logits_entropy"] + metrics["logits_varentropy"]
    attention_uncertainty = metrics["attention_entropy"] + metrics["attention_varentropy"]
    temperature = base_temperature * (1 + 0.3 * logits_uncertainty + 0.2 * attention_uncertainty - 0.2 * metrics["agreement"])
    top_p = mx.clip(base_top_p * (1 + 0.1 * metrics["attention_varentropy"]), 0.1, 1.0)
    top_k = int(
        mx.clip(
            mx.round(base_top_k * (1 + 0.3 * metrics["interaction_strength"].item() - 0.2 * metrics['agreement'].item())),
            a_min = 1,
            a_max = 100
        )
    )
    min_p = mx.clip(base_min_p * (1 - 0.5 * logits_uncertainty), 0.01, 0.5)

    samples = []
    for _ in range(n_samples):
        sample = _sample(logits, temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p)
        samples.append(sample)

    def score_sample(sample):
        one_hot = mx.zeros((sample, logits.shape[-1]))
        one_hot[mx.arange(sample.size), logits.shape[-1]] = 1
        log_prob = mx.sum(mx.softmax(logits[:, -1], axis=-1) * one_hot)

        confidence_score = (
            (1 - metrics["logits_entropy"]) * 0.1 +
            (1 - metrics["attention_entropy"]) * 0.2 +
            (1 - metrics["logits_varentropy"]) * 0.3 +
            (1 - metrics["attention_varentropy"]) * 0.4 +
            metrics["agreement"] * 0.5 +
            metrics["interaction_strength"] * 0.6
        )
        return log_prob + confidence_score

    sample_scores = [score_sample(sample) for sample in samples]
    best_sample_idx = mx.argmax(mx.array(sample_scores)).item()
    return samples[best_sample_idx]

def _sample(logits: mx.array, temperature=0.666, top_p=0.9, top_k:int = 27, min_p: float = 0.0, min_tokens_to_keep: int = 2) -> mx.array:
    logit = logits[:, -1] / temperature  # (batch_size, vocab_size)

    # Calculate probabilities by softmaxing the temparature-scaled logits
    probs = mx.softmax(logit, axis=-1)

    # Sort probabilities in descending order
    # This should then look like
    sorted_indices = mx.argsort(-probs, axis=-1).squeeze(0) # e.g. [3, 1280, 1, 0, 2, ...]
    sorted_probs = probs[..., sorted_indices] # e.g. [0.9, 0.05, 0.02, 0.01, 0.01, ...]

    # Apply min_p sampling
    if min_p > 0:
        top_prob = sorted_probs[..., 0] # Highest probability e.g. [0.9]
        scaled_min_p = min_p * top_prob # e.g. 0.9 * 0.1 = 0.09
        min_p_mask = sorted_probs > scaled_min_p # e.g. [True, False, False, False, False, ...]
        min_p_mask[..., :min_tokens_to_keep] = True # Keep at least min_tokens_to_keep tokens, e.g. [True, True, True, False, False, ...]
        sorted_probs = mx.where(min_p_mask, sorted_probs, 0.0) # e.g. [0.9, 0.0, 0.0, 0.0, 0.0, ...]

    # Apply top_p (nucleus) sampling
    cumulative_probs = mx.cumsum(sorted_probs, axis=-1, inclusive = False) # e.g. [0.9, 0.95, 0.97, 0.98, 0.99, ...]
    # or, if min_p is applied, [0.9, 0.0, 0.0, 0.0, 0.0, ...]
    top_p_mask = cumulative_probs <= top_p # e.g. [True, True, True, True, True, ...]
    # or, if min_p is applied, [True, False, False, False, False, ...]
    top_p_mask[..., :min_tokens_to_keep] = True #
    sorted_probs = mx.where(top_p_mask, sorted_probs, 0.0)

    # Optionally apply top_k sampling
    sorted_probs[..., top_k:] = 0.0

    # Sample token
    sorted_token = mx.random.categorical(mx.log(sorted_probs))
    token = sorted_indices[sorted_token]

    return token
