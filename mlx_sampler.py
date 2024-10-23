import mlx.core as mx
import mlx.nn as nn

from mlx_attention_sampler import SamplerConfig
from typing import Union, Tuple

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


def adaptive_sample(
    logits: mx.array,
    *,
    key: Union[mx.array, None] = None,
    epsilon: float = 0.01,
    cfg: SamplerConfig
) -> mx.array:
    batch_size = logits.shape[0]

    logit = logits[:, -1] / cfg.temperature
    probs = mx.softmax(logit, axis=-1)

    sorted_indices = mx.argsort(-probs, axis=-1)  # e.g. (bsz x [3, 1280, 1, 0, 2, ...])
    sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)  # e.g. (bsz x [0.9, 0.05, 0.02, 0.01, 0.01, ...])

    mask = mx.zeros_like(sorted_probs)

    counter = 1
    cumulative_entropy = mx.zeros((batch_size, ))
    cumulative_varentropy = mx.zeros((batch_size, ))

    current_entropy = -mx.sum(sorted_probs[:, :counter] * mx.log2(mx.clip(sorted_probs[:, :counter], 1e-10, 1.0)))
    current_varentropy = mx.sum(sorted_probs[:, :counter] * (mx.log2(mx.clip(sorted_probs[:, :counter], 1e-10, 1.0)) + current_entropy[..., None]) ** 2)

    cumulative_entropy[: counter] = current_entropy
    cumulative_varentropy[: counter] = current_varentropy

    entropy_reduction = current_entropy
    while (entropy_reduction >= epsilon) and (counter < sorted_probs.shape[-1]):
        previous_entropy = current_entropy
        previous_varentropy = current_varentropy

        counter += 1
        current_probs = sorted_probs[:, :counter]

        # Update entropy and varentropy with current token
        current_entropy = -mx.sum(current_probs * mx.log2(mx.clip(current_probs, 1e-10, 1.0)))
        current_varentropy = mx.sum(current_probs * (mx.log2(mx.clip(current_probs, 1e-10, 1.0)) + current_entropy[..., None]) ** 2)

        cumulative_entropy[: counter] = current_entropy
        cumulative_varentropy[: counter] = current_varentropy

        entropy_reduction = current_entropy - previous_entropy
        varentropy_reduction = current_varentropy - previous_varentropy

        mask = mx.concatenate([mx.ones((batch_size, counter)), mx.zeros((batch_size, sorted_probs.shape[-1] - counter))], axis = -1)

    final_mask = mask
    candidate_probs = sorted_probs * final_mask
    candidate_probs = candidate_probs / mx.sum(candidate_probs, axis=-1, keepdims=True)

    # Sample token
    sorted_token = mx.random.categorical(mx.log(candidate_probs), key=key)[
        ..., None
    ]  # e.g. (bsz * [1390, 3, 2791, 1381, 12476, ...])
    token = mx.take_along_axis(
        sorted_indices, sorted_token, axis=-1
    )  # e.g. [[3]] in shape (batch_size, 1)
    return token

def new_sample(
    logits: mx.array,
    attention_scores: mx.array,
    key: Union[mx.array, None] = None,
    cfg: SamplerConfig = SamplerConfig(),
) -> Tuple[mx.array, dict[str, mx.array]]:
    metrics = calculate_metrics(logits, attention_scores)

    # Calculate adaptive sampling parameters based on all metrics
    adaptive_score = (
        metrics["logits_entropy"] * cfg.adaptive_score_logits_entropy_coefficient +
        metrics["attention_entropy"] * cfg.adaptive_score_attention_entropy_coefficient +
        metrics["logits_varentropy"] * cfg.adaptive_score_logits_varentropy_coefficient +
        metrics["attention_varentropy"] * cfg.adaptive_score_attention_varentropy_coefficient +
        metrics["agreement"] * cfg.adaptive_score_agreement_coefficient +
        metrics["interaction_strength"] * cfg.adaptive_score_interaction_strength_coefficient
    )

    # Use adaptive score to adjust epsilon
    dynamic_epsilon = cfg.epsilon * (1 + adaptive_score * 0.1)

    token = adaptive_sample(
        logits,
        key=key,
        epsilon=dynamic_epsilon,
        cfg=cfg
    )

    return token, metrics

# Old Sampler with top_p and temperature

# def sample(
#     logits: mx.array,
#     attention_scores: mx.array,
#     cfg: SamplerConfig = SamplerConfig(),
#     key: Union[mx.array, None] = None,
# ) -> Tuple[mx.array, dict[str, mx.array]]:
#     batch_size = logits.shape[0]
#     logit = logits[:, -1] / cfg.temperature  # (batch_size, vocab_size)
#     probs = mx.softmax(logit, axis=-1)

#     sorted_indices = mx.argsort(-probs, axis=-1)  # e.g. (bsz x [3, 1280, 1, 0, 2, ...])
#     sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)  # e.g. (bsz x [0.9, 0.05, 0.02, 0.01, 0.01, ...])
#     cumulative_probs = mx.cumsum(sorted_probs, axis=-1)
#     mask = cumulative_probs < cfg.top_p

#     sorted_probs = mx.where(mask, sorted_probs, 0.0)
#     sorted_probs = sorted_probs / mx.sum(sorted_probs, axis=-1, keepdims=True)
#     sorted_token = mx.random.categorical(mx.log(sorted_probs / (1 - sorted_probs)), key=key)[
#         ..., None]

#     token = mx.take_along_axis(
#         sorted_indices, sorted_token, axis=-1
#     )
#     return token, calculate_metrics(logits, attention_scores)
