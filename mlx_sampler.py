import mlx.core as mx
import mlx.nn as nn

from mlx_attention_sampler import SamplerConfig
from typing import Union

LN_2 = 0.69314718056  # ln(2)


@mx.compile
def calculate_varentropy_logsoftmax(
    logits: mx.array, axis: int = -1
) -> tuple[mx.array, mx.array]:
    """Calculate the entropy and varentropy of the probability distribution using logsoftmax."""
    log_probs = nn.log_softmax(logits, axis = axis)
    probs = mx.exp(log_probs)
    entropy = -mx.sum(probs * log_probs, axis=axis) / LN_2  # Convert to base-2
    varentropy = mx.sum(probs * (log_probs / LN_2 + entropy[..., None]) ** 2, axis=axis)
    return entropy, varentropy


@mx.compile
def calculate_metrics(
    logits: mx.array, attention_scores: mx.array
) -> dict[str, mx.array]:
    entropy, varentropy = calculate_varentropy_logsoftmax(logits)

    attention_probs = mx.softmax(attention_scores, axis=-1)
    attention_entropy = -mx.sum(
        attention_probs * mx.log2(mx.clip(attention_probs, 1e-10, 1.0)), axis=-1
    )
    attention_varentropy = mx.var(attention_entropy, axis=1)

    mean_attention = mx.mean(attention_probs, axis=1)
    agreement = mx.mean(
        mx.abs(attention_probs - mean_attention[:, None, :]), axis=(1, 2)
    )

    interaction_strength = mx.mean(mx.abs(attention_scores), axis=(1, 2, 3))
    return {
        "logits_entropy": mx.mean(entropy),
        "logits_varentropy": mx.mean(varentropy),
        "attention_entropy": mx.mean(attention_entropy),
        "attention_varentropy": mx.mean(attention_varentropy),
        "agreement": mx.mean(agreement),
        "interaction_strength": interaction_strength,
    }


def _sample(
    logits: mx.array,
    temperature=0.666,
    top_p=0.9,
    top_k: int = 33,
    min_p: float = 0.0,
    key: Union[mx.array, None] = None,
) -> mx.array:
    batch_size = logits.shape[0]
    #print(f"t({temperature.item()}), top_p({top_p.item()}), top_k({top_k}), min_p({min_p.item()})")
    logit = logits[:, -1] / temperature  # (batch_size, vocab_size)

    # Calculate probabilities by softmaxing the temparature-scaled logits
    probs = mx.softmax(logit, axis=-1)

    sorted_indices = mx.argsort(-probs, axis=-1)  # e.g. (bsz x [3, 1280, 1, 0, 2, ...])
    sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)  # e.g. (bsz x [0.9, 0.05, 0.02, 0.01, 0.01, ...])

    # Apply min_p sampling
    if min_p > 0:
        top_prob = sorted_probs[..., 0]  # Highest probability e.g. (bsz x[0.9])
        scaled_min_p = min_p * top_prob  # e.g. 0.9 * 0.1 = 0.09, (bsz x[0.09])
        min_p_mask = (sorted_probs > scaled_min_p[..., None])  # e.g. (bsz * [True, False, False, False, False, ...])
        sorted_probs = mx.where(min_p_mask, sorted_probs, 0.0)  # e.g. (bsz * [0.9, 0.0, 0.0, 0.0, 0.0, ...])

    # Apply top_p (nucleus) sampling
    cumulative_probs = mx.cumsum(sorted_probs, axis=-1)  # e.g. (bsz * [0.9, 0.95, 0.97, 0.98, 0.99, ...]
    # or, if min_p is applied, (bsz * [0.9, 0.0, 0.0, 0.0, 0.0, ...]
    top_p_mask = (
        cumulative_probs <= top_p
    )  # e.g. (bsz * [True, True, True, True, True, ...]
    # or, if min_p is applied, (bsz * [True, False, False, False, False, ...]
    sorted_probs = mx.where(
        top_p_mask, sorted_probs, 0.0
    )  # e.g. (bsz * [0.9, 0.05, 0.02, 0.01, 0.01, ...])

    # Optionally apply top_k sampling
    sorted_probs[..., top_k:] = 0.0  # e.g. (bsz * [0.9, 0.05, 0.0, 0.0, 0.0, ...])

    sorted_probs = sorted_probs / mx.sum(sorted_probs, axis = -1)

    # Sample token
    sorted_token = mx.random.categorical(mx.log(sorted_probs / (1 - sorted_probs)), key=key)[
        ..., None
    ]  # e.g. (bsz * [1390, 3, 2791, 1381, 12476, ...])
    token = mx.take_along_axis(
        sorted_indices, sorted_token, axis=-1
    )  # e.g. [3,] in shape (batch_size,)
    return token


@mx.compile
def score_sample(
    sample: mx.array,
    logits: mx.array,
) -> mx.array:
    batch_size, seq_length = sample.shape
    vocab_size = logits.shape[-1]

    # Create one-hot encoding
    one_hot = mx.zeros((batch_size, seq_length, vocab_size))
    one_hot[mx.arange(batch_size)[:, None], mx.arange(seq_length)[None, :], sample] = 1

    # Calculate log probability
    log_probs = mx.sum(
        mx.softmax(logits[:, -1], axis=-1).log()[:, None, :] * one_hot, axis=(1, 2)
    )

    return log_probs


def sample(
    gen_tokens: mx.array,
    logits: mx.array,
    scores: mx.array,
    cfg: SamplerConfig,
    clarifying_question_token: int = 2564,
    key: Union[mx.array, None] = None,
) -> tuple[mx.array, dict[str, float]]:
    metrics = calculate_metrics(logits, scores)
    ent, vent = metrics["logits_entropy"], metrics["logits_varentropy"]
    attention_entropy, attention_varentropy = (
        metrics["attention_entropy"],
        metrics["attention_varentropy"],
    )
    agreement, interaction_strength = (
        metrics["agreement"],
        metrics["interaction_strength"],
    )

    # Low Entropy, Low Varentropy: "flowing with unspoken intent"
    if (
        ent < cfg.low_logits_entropy_threshold
        and vent < cfg.low_logits_varentropy_threshold
    ):
        #print("🌊", flush = True, end = "")
        return mx.argmax(logits[:, -1], axis=-1, keepdims=True), metrics

    # High Entropy, Low Varentropy: "treading carefully, asking clarifying questions"
    elif (
        ent > cfg.high_logits_entropy_threshold
        and vent < cfg.low_logits_varentropy_threshold
    ):
        #print("ε", flush = True, end = "")
        # Insert a clarifying question token if not already present
        if not mx.any(mx.equal(gen_tokens[:, -1], clarifying_question_token).any()):
            return mx.array(
                [[clarifying_question_token]]
            ), metrics  # Assuming 2564 is our "ask clarifying question" token
        else:
            # If we've just asked a question, sample with slightly higher temperature
            temp_adj = (
                cfg.high_entropy_attention_offset
                + cfg.high_entropy_varentropy_attention_coefficient * attention_entropy
            )  # Increase temperature
            return _sample(
                logits,
                temperature=min(1.5, cfg.temperature * temp_adj),
                top_p=cfg.top_p,
                top_k=cfg.top_k,
                min_p=cfg.min_probability,
                key=key,
            ), metrics

    # Low Entropy, High Varentropy: "exploring forks in the path"
    elif (
        ent < cfg.low_logits_entropy_threshold
        and vent > cfg.high_logits_varentropy_threshold
        and attention_entropy > cfg.low_attention_entropy_threshold
        and attention_varentropy < cfg.medium_attention_varentropy_threshold
    ):
        #print("Ψ", flush = True, end = "")
        # TODO(xjdr): Implement proper branching logic
        # Return top-k tokens to allow for branching
        # top_k_values, top_k_indices = mx.top_k(logits[:, -1], k=top_k)
        # return top_k_indices
        temp_adj = (
            cfg.low_entropy_interaction_strength_offset
            + cfg.low_entropy_interaction_strength_coefficient * interaction_strength
        )
        top_k_adj = max(cfg.top_k, int(cfg.top_k * (1 + 0.5 * (1 - agreement))))
        return _sample(
            logits,
            temperature=min(1.5, cfg.temperature * temp_adj),
            top_p=cfg.top_p,
            top_k=top_k_adj,
            min_p=cfg.min_probability,
            key=key,
        ), metrics

    # High Entropy, High Varentropy: "resampling in the mist"
    elif (
        ent > cfg.high_logits_entropy_threshold
        and vent > cfg.high_logits_varentropy_threshold
        and attention_entropy > cfg.high_attention_entropy_threshold
    ):
        #print("!", flush = True, end = "")
        # Use high temperature and min_p sampling
        temp_adj = (
            cfg.high_entropy_varentropy_attention_offset
            + cfg.high_entropy_varentropy_attention_coefficient * attention_varentropy
        )
        top_p_adj = max(
            0.5, cfg.top_p - cfg.high_entropy_attention_coefficient * attention_entropy
        )
        return _sample(
            logits,
            temperature=max(2.0, cfg.temperature * temp_adj),
            top_p=top_p_adj,
            top_k=cfg.top_k,
            min_p=cfg.min_probability,
            key=key,
        ), metrics

    # Middle ground: smooth transition
    else:
        # Interpolate temperature based on entropy and varentropy
        logits_uncertainty = metrics["logits_entropy"] + metrics["logits_varentropy"]
        attention_uncertainty = (
            metrics["attention_entropy"] + metrics["attention_varentropy"]
        )

        temperature = mx.clip(cfg.temperature * (
            1
            + cfg.adaptive_temperature_logits_coefficient * ent
            + cfg.adaptive_temperature_attention_coefficient * attention_entropy
            - cfg.adaptive_temperature_agreement_coefficient * agreement
        ),
        0.0,
        2.0)
        top_p = mx.clip(
            cfg.top_p * (1 - cfg.adaptive_top_p_coefficient * attention_varentropy),
            0.6,
            1.0,
        )
        top_k = int(
            mx.clip(
                mx.round(
                    cfg.top_k
                    * (
                        0.6
                        + cfg.adaptive_top_k_interaction_coefficient
                        * interaction_strength.item()
                        - cfg.adaptive_top_k_agreement_coefficient * agreement.item()
                    )
                ),
                a_min=10,
                a_max=60,
            )
        )
        min_p = mx.clip(
            cfg.min_probability
            * (1 - cfg.adaptive_min_p_coefficient * logits_uncertainty),
            0.01,
            0.4,
        )
        #print(f"({temperature.item():.2f},{top_p.item():.2f},{top_k},{min_p.item():.2f})", flush = True, end = "")
        return _sample(
            logits,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            key=key,
        ), metrics
