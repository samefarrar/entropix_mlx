from mlx_lm.utils import make_kv_caches
import mlx.core as mx
import mlx.nn as nn
from typing import Optional, List, Tuple, Union, Dict
from mlx_attention_sampler import SamplerConfig
import numpy as np

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
    attention_varentropy = mx.var(attention_entropy, axis = 1)

    mean_attention = mx.mean(attention_probs, axis = 1)
    agreement = mx.mean(mx.abs(attention_probs - mean_attention[:, None, :]), axis = (1, 2))

    interaction_strength = mx.mean(mx.abs(attention_scores), axis = (1, 2, 3))
    return {
        "logits_entropy": mx.mean(entropy),
        "logits_varentropy": mx.mean(varentropy),
        "attention_entropy": mx.mean(attention_entropy),
        "attention_varentropy": mx.mean(attention_varentropy),
        "agreement": mx.mean(agreement),
        "interaction_strength": interaction_strength
    }

def _sample(logits: mx.array, temperature=0.666, top_p=0.9, top_k: int = 27, min_p: float = 0.0, min_tokens_to_keep: int = 2) -> mx.array:
    batch_size = logits.shape[0]
    logit = logits[:, -1] / temperature  # (batch_size, vocab_size)

    # Calculate probabilities by softmaxing the temparature-scaled logits
    probs = mx.softmax(logit, axis=-1)

    # Sort probabilities in descending order
    # This should then look like
    sorted_indices = mx.argsort(-probs, axis=-1) # e.g. (bsz x [3, 1280, 1, 0, 2, ...])
    sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1) # e.g. (bsz x [0.9, 0.05, 0.02, 0.01, 0.01, ...])

    # Apply min_p sampling
    if min_p > 0:
        top_prob = sorted_probs[..., 0] # Highest probability e.g. (bsz x[0.9])
        scaled_min_p = min_p * top_prob # e.g. 0.9 * 0.1 = 0.09, (bsz x[0.09])
        min_p_mask = sorted_probs > scaled_min_p[..., None] # e.g. (bsz * [True, False, False, False, False, ...])
        min_p_mask[..., :min_tokens_to_keep] = True # Keep at least min_tokens_to_keep tokens, e.g. (bsz * [True, True, True, False, False, ...])
        sorted_probs = mx.where(min_p_mask, sorted_probs, 0.0) # e.g. (bsz * [0.9, 0.0, 0.0, 0.0, 0.0, ...])

    # Apply top_p (nucleus) sampling
    cumulative_probs = mx.cumsum(sorted_probs, axis=-1, inclusive = False) # e.g. (bsz * [0.9, 0.95, 0.97, 0.98, 0.99, ...]
    # or, if min_p is applied, (bsz * [0.9, 0.0, 0.0, 0.0, 0.0, ...]
    top_p_mask = cumulative_probs <= top_p # e.g. (bsz * [True, True, True, True, True, ...]
    # or, if min_p is applied, (bsz * [True, False, False, False, False, ...]
    top_p_mask[..., :min_tokens_to_keep] = True # Keep at least min_tokens_to_keep tokens, e.g. (bsz * [True, True, True, False, False, ...])
    sorted_probs = mx.where(top_p_mask, sorted_probs, 0.0) # e.g. (bsz * [0.9, 0.05, 0.02, 0.01, 0.01, ...])

    # Optionally apply top_k sampling
    sorted_probs[..., top_k:] = 0.0 # e.g. (bsz * [0.9, 0.05, 0.0, 0.0, 0.0, ...])

    # Sample token
    sorted_token = mx.random.categorical(mx.log(sorted_probs))[..., None] # e.g. (bsz * [1390, 3, 2791, 1381, 12476, ...])
    token = mx.take_along_axis(sorted_indices, sorted_token, axis=-1) # e.g. [3,] in shape (batch_size,)
    return token

@mx.compile
def score_sample(
    sample: mx.array,
    logits: mx.array,
    logits_entropy: float,
    attention_entropy: float,
    logits_varentropy: float,
    attention_varentropy: float,
    agreement: float,
    interaction_strength: float,
    ADA_SCORE_LOGITS_ENT: float,
    ADA_SCORE_ATT_ENT: float,
    ADA_SCORE_LOGITS_VAR: float,
    ADA_SCORE_ATT_VAR: float,
    ADA_SCORE_AGREEMENT: float,
    ADA_SCORE_INTERACTION: float,
    HIGH_LOGITS_ENTROPY_THRESHOLD: float,
    HIGH_LOGITS_VARENTROPY_THRESHOLD: float,
    HIGH_ATTENTION_ENTROPY_THRESHOLD: float,
    HIGH_ATTENTION_VARENTROPY_THRESHOLD: float,
    HIGH_AGREEMENT_THRESHOLD: float,
    HIGH_INTERACTION_STRENGTH_THRESHOLD: float,
) -> mx.array:
    batch_size, seq_length = sample.shape
    vocab_size = logits.shape[-1]

    # Create one-hot encoding
    one_hot = mx.zeros((batch_size, seq_length, vocab_size))
    one_hot[mx.arange(batch_size)[:, None], mx.arange(seq_length)[None, :], sample] = 1

    # Calculate log probability
    log_probs = mx.sum(mx.softmax(logits[:, -1], axis=-1).log()[:, None, :] * one_hot, axis=(1, 2))

    # Calculate confidence score
    confidence_scores = (
        (1 - logits_entropy / HIGH_LOGITS_ENTROPY_THRESHOLD) * ADA_SCORE_LOGITS_ENT +
        (1 - attention_entropy / HIGH_ATTENTION_ENTROPY_THRESHOLD) * ADA_SCORE_ATT_ENT +
        (1 - logits_varentropy / HIGH_LOGITS_VARENTROPY_THRESHOLD) * ADA_SCORE_LOGITS_VAR +
        (1 - attention_varentropy / HIGH_ATTENTION_VARENTROPY_THRESHOLD) * ADA_SCORE_ATT_VAR +
        agreement / HIGH_AGREEMENT_THRESHOLD * ADA_SCORE_AGREEMENT +
        interaction_strength / HIGH_INTERACTION_STRENGTH_THRESHOLD * ADA_SCORE_INTERACTION
    )

    return log_probs + confidence_scores

def sample(
    gen_tokens: mx.array, logits: mx.array, scores: mx.array, cfg: SamplerConfig, clarifying_question_token: int = 2564
) -> Tuple[mx.array, Dict[str, float]]:
    metrics = calculate_metrics(logits, scores)
    ent, vent = metrics["logits_entropy"], metrics["logits_varentropy"]
    attention_entropy, attention_varentropy = metrics["attention_entropy"], metrics["attention_varentropy"]
    agreement, interaction_strength = metrics["agreement"], metrics["interaction_strength"]

    # Low Entropy, Low Varentropy: "flowing with unspoken intent"
    if (ent < cfg.low_logits_entropy_threshold and
        vent < cfg.low_logits_varentropy_threshold and
        attention_entropy < cfg.low_attention_entropy_threshold and
        attention_varentropy < cfg.low_attention_varentropy_threshold
        and agreement > cfg.medium_agreement_threshold and
        interaction_strength < cfg.low_interaction_strength_threshold):
        return mx.argmax(logits[:, -1], axis=-1, keepdims=True), metrics

    # High Entropy, Low Varentropy: "treading carefully, asking clarifying questions"
    elif (ent > cfg.medium_logits_entropy_threshold and
        vent < cfg.medium_logits_varentropy_threshold and
        attention_entropy < cfg.high_attention_entropy_threshold and
        attention_varentropy < cfg.low_attention_varentropy_threshold and
        agreement < cfg.medium_agreement_threshold and
        interaction_strength < cfg.low_interaction_strength_threshold):
        #print("ε", flush = True, end = "")
        # Insert a clarifying question token if not already present
        if not mx.any(mx.equal(gen_tokens[:, -1], clarifying_question_token).any()):
            return mx.array(
                [[clarifying_question_token]]
            ), metrics  # Assuming 2564 is our "ask clarifying question" token
        else:
            # If we've just asked a question, sample with slightly higher temperature
            temp_adj = cfg.high_entropy_attention_offset + cfg.high_entropy_varentropy_attention_coefficient * attention_entropy # Increase temperature
            return _sample(logits, temperature=min(1.5, cfg.temperature * temp_adj), top_p = cfg.top_p, top_k = cfg.top_k, min_p = cfg.min_probability), metrics

    # Low Entropy, High Varentropy: "exploring forks in the path"
    elif (ent < cfg.high_logits_entropy_threshold and
        vent > cfg.high_logits_varentropy_threshold and
        attention_entropy < cfg.low_attention_entropy_threshold and
        attention_varentropy > cfg.medium_attention_varentropy_threshold and
        agreement < cfg.low_agreement_threshold and
        interaction_strength > cfg.low_interaction_strength_threshold):
        #print("Ψ", flush = True, end = "")
        # TODO(xjdr): Implement proper branching logic
        # Return top-k tokens to allow for branching
        # top_k_values, top_k_indices = mx.top_k(logits[:, -1], k=top_k)
        # return top_k_indices
        temp_adj = cfg.low_entropy_interaction_strength_offset + cfg.low_entropy_interaction_strength_coefficient * interaction_strength
        top_k_adj = max(cfg.top_k, int(cfg.top_k * (1 + 0.5 * (1 - agreement))))
        return _sample(logits, temperature=min(1.5, cfg.temperature * temp_adj), top_p = cfg.top_p, top_k = top_k_adj, min_p = cfg.min_probability), metrics

    # High Entropy, High Varentropy: "resampling in the mist"
    elif (ent > cfg.medium_logits_entropy_threshold and
        vent > cfg.high_logits_varentropy_threshold and
        attention_entropy > cfg.high_attention_entropy_threshold and
        attention_varentropy > cfg.high_attention_varentropy_threshold and
        agreement > cfg.high_agreement_threshold and
        interaction_strength > cfg.high_interaction_strength_threshold):
        #print("!", flush = True, end = "")
        # Use high temperature and min_p sampling
        temp_adj = cfg.high_entropy_varentropy_attention_offset + cfg.high_entropy_varentropy_attention_coefficient * attention_varentropy
        top_p_adj = max(0.5, cfg.top_p - cfg.high_entropy_attention_coefficient * attention_entropy)
        return _sample(logits, temperature=max(2.0, cfg.temperature * temp_adj), top_p = top_p_adj, top_k = cfg.top_k, min_p = cfg.min_probability), metrics

    # Middle ground: smooth transition
    else:
        # Interpolate temperature based on entropy and varentropy
        logits_uncertainty = metrics["logits_entropy"] + metrics["logits_varentropy"]
        attention_uncertainty = metrics["attention_entropy"] + metrics["attention_varentropy"]

        temperature = cfg.temperature * (1 + cfg.adaptive_temperature_logits_coefficient * ent + cfg.adaptive_temperature_attention_coefficient * attention_entropy - cfg.adaptive_temperature_agreement_coefficient * agreement)
        top_p = mx.clip(cfg.top_p * (cfg.adaptive_top_p_coefficient * attention_varentropy), 0.1, 1.0)
        top_k = int(
            mx.clip(
                mx.round(cfg.top_k * (0.6 + cfg.adaptive_top_k_interaction_coefficient * interaction_strength.item() - cfg.adaptive_top_k_agreement_coefficient * agreement.item())),
                a_min = 1,
                a_max = 100
            )
        )
        min_p = mx.clip(cfg.min_probability * (1 - cfg.adaptive_min_p_coefficient * logits_uncertainty), 0.01, 0.4)

        # Sample from the logits
        perturbed_logits = mx.repeat(logits, cfg.n_adaptive_samples, axis = 0)
        gumbel_noise = mx.random.gumbel(perturbed_logits.shape) * cfg.ada_noise_scale
        perturbed_logits = perturbed_logits + gumbel_noise
        samples = _sample(perturbed_logits, temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p)

        sample_scores = score_sample(
            samples,
            perturbed_logits,
            ent,
            attention_entropy,
            vent,
            attention_varentropy,
            agreement,
            interaction_strength,
            ADA_SCORE_LOGITS_ENT = cfg.adaptive_score_logits_entropy_coefficient,
            ADA_SCORE_ATT_ENT = cfg.adaptive_score_attention_entropy_coefficient,
            ADA_SCORE_LOGITS_VAR = cfg.adaptive_score_logits_varentropy_coefficient,
            ADA_SCORE_ATT_VAR = cfg.adaptive_score_attention_varentropy_coefficient,
            ADA_SCORE_AGREEMENT = cfg.adaptive_score_agreement_coefficient,
            ADA_SCORE_INTERACTION = cfg.adaptive_score_interaction_strength_coefficient,
            HIGH_LOGITS_ENTROPY_THRESHOLD = cfg.high_logits_entropy_threshold,
            HIGH_ATTENTION_ENTROPY_THRESHOLD = cfg.high_attention_entropy_threshold,
            HIGH_LOGITS_VARENTROPY_THRESHOLD = cfg.high_logits_varentropy_threshold,
            HIGH_ATTENTION_VARENTROPY_THRESHOLD = cfg.high_attention_varentropy_threshold,
            HIGH_AGREEMENT_THRESHOLD = cfg.high_agreement_threshold,
            HIGH_INTERACTION_STRENGTH_THRESHOLD = cfg.high_interaction_strength_threshold,
        )

        best_sample_idx = mx.argmax(sample_scores)
        return samples[best_sample_idx][None], metrics
