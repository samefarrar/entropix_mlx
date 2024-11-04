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

    return {
        "logits_entropy": mx.mean(entropy),
        "logits_varentropy": mx.mean(varentropy),
        "attention_entropy": mx.mean(attention_entropy),
        "attention_varentropy": mx.mean(attention_varentropy),
    }

def one_hot(
    indices: mx.array,
    num_classes: int,
    axis = -1
):
    one_hot_matrix = mx.zeros((indices.shape[0], num_classes))
    if indices.size == 0:
        return one_hot_matrix
    one_hot_matrix[mx.arange(indices.shape[0]), indices] = 1
    return one_hot_matrix

def score_sample(
    logits: mx.array,
    sample: mx.array,
    attention_entropy: mx.array,
    attention_varentropy: mx.array,
    logits_entropy: mx.array,
    logits_varentropy: mx.array,
    cfg: SamplerConfig = SamplerConfig(),
):
    log_prob = mx.sum(
        nn.log_softmax(logits, axis=-1) * one_hot(sample, logits.shape[-1]),
    )

    confidence_score = (
        (1 - logits_entropy / cfg.high_logits_entropy_threshold)
        * cfg.adaptive_score_logits_entropy_coefficient
        + (1 - attention_entropy / cfg.high_attention_entropy_threshold)
        * cfg.adaptive_score_attention_entropy_coefficient
        + (1 - logits_varentropy / cfg.high_logits_varentropy_threshold)
        * cfg.adaptive_score_logits_varentropy_coefficient
        + (1 - attention_varentropy / cfg.high_attention_varentropy_threshold)
        * cfg.adaptive_score_attention_varentropy_coefficient
        )

    return log_prob * confidence_score

def adaptive_sample(
    logits: mx.array,
    logits_entropy: mx.array,
    logits_varentropy: mx.array,
    attention_entropy: mx.array,
    attention_varentropy: mx.array,
    key: mx.array,
    cfg: SamplerConfig = SamplerConfig(),
) -> mx.array:
    batch_size = logits.shape[0]

    log_probs = nn.log_softmax(logits, axis=-1)
    cross_entropy = -mx.sum(mx.exp(log_probs) * log_probs, axis=-1)

    target_ce = cfg.target_ce_alpha * cross_entropy + cfg.target_ce_beta
    temperature = cfg.temperature * mx.sqrt(target_ce / (cross_entropy + 1e-6))
    temperature = mx.clip(temperature, cfg.min_temperature, cfg.max_temperature)

    min_p = mx.clip(
        cfg.min_probability * (1 - cfg.adaptive_min_p_coefficient * logits_varentropy),
        a_min = 0.01,
        a_max = 0.5
    )

    keys = mx.random.split(key, num = cfg.n_adaptive_samples)

    samples = []
    for sample_key in keys:
        sample = _sample(
            logits = logits,
            temperature=temperature,
            top_p = cfg.top_p,
            top_k = cfg.top_k,
            min_p = min_p,
            key = sample_key
        )
        samples.append(sample)

    sample_scores = mx.array([score_sample(logits,
            sample,
            logits_entropy=logits_entropy,
            logits_varentropy=logits_varentropy,
            attention_entropy=attention_entropy,
            attention_varentropy=attention_varentropy) for sample in samples])
    best_sample = mx.array(samples)[mx.argmax(sample_scores)]
    return best_sample

def sample(
    logits: mx.array,
    gen_tokens: mx.array,
    attention_scores: mx.array,
    key: mx.array,
    cfg: SamplerConfig = SamplerConfig(),
    clarifying_question_token: int = 2564,
) -> Tuple[mx.array, dict[str, mx.array]]:
    metrics = calculate_metrics(logits, attention_scores)
    logits_entropy, logits_varentropy, attention_entropy, attention_varentropy  = (
        metrics["logits_entropy"], metrics["logits_varentropy"], metrics["attention_entropy"], metrics["attention_varentropy"]
    )

    # Low Entropy, Low Varentropy: "flowing with unspoken intent"
    if (
        logits_entropy < cfg.low_logits_entropy_threshold and logits_varentropy < cfg.low_logits_varentropy_threshold
    ):
        return mx.argmax(logits[:, -1], axis = -1, keepdims = True), metrics
    # High Entropy, Low Varentropy: "treading carefully, asking clarifying questions"
    elif (
        logits_entropy > cfg.high_logits_entropy_threshold and
        logits_varentropy < cfg.low_logits_varentropy_threshold
    ):
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
                temperature=min(2.0, cfg.temperature * temp_adj),
                top_p=cfg.top_p,
                top_k=cfg.top_k,
                min_p=cfg.min_probability,
                key = key,
            ), metrics
    # Low Entropy, High Varentropy: "exploring forks in the path"
    elif (
        logits_entropy < cfg.low_logits_entropy_threshold
        and logits_varentropy > cfg.high_logits_varentropy_threshold
        and attention_entropy > cfg.low_attention_entropy_threshold
        and attention_varentropy < cfg.medium_attention_varentropy_threshold
    ):
        return _sample(
            logits,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            min_p=cfg.min_probability,
            key = key
        ), metrics
    # High Entropy, High Varentropy: "resampling in the mist"
    elif (
        logits_entropy > cfg.high_logits_entropy_threshold and
        logits_varentropy > cfg.high_logits_varentropy_threshold
        and attention_entropy > cfg.high_attention_entropy_threshold):
        temp_adj = (
            cfg.high_entropy_varentropy_attention_offset +
            cfg.high_entropy_varentropy_attention_coefficient * attention_varentropy
        )
        top_p_adj = (
            cfg.high_entropy_varentropy_attention_offset + cfg.high_entropy_varentropy_attention_coefficient * attention_entropy)
        return _sample(
            logits,
            temperature=min(2.3, cfg.temperature * temp_adj),
            top_p = mx.clip(top_p_adj, a_min = 0.6, a_max = 1.0),
            top_k = cfg.top_k,
            min_p = cfg.min_probability,
            key = key
        ), metrics
    else:
        token = adaptive_sample(
            logits,
            logits_entropy = logits_entropy,
            logits_varentropy = logits_varentropy,
            attention_entropy = attention_entropy,
            attention_varentropy = attention_varentropy,
            key=key,
            cfg=cfg
        )
        return token, metrics

# Old Sampler with top_p and temperature
def _sample(
    logits: mx.array,
    temperature: float | mx.array,
    top_p: float | mx.array,
    top_k: int | mx.array,
    min_p: float | mx.array,
    key: Union[mx.array, None] = None,
) -> mx.array:
    batch_size = logits.shape[0]
    logit = logits[:, -1] / temperature  # (batch_size, vocab_size)

    # Calculate probabilities by softmaxing the temparature-scaled logits
    probs = mx.softmax(logit, axis=-1)

    # Sort probabilities in descending order
    # This should then look like
    sorted_indices = mx.argsort(-probs, axis=-1) # e.g. (bsz x [3, 1280, 1, 0, 2, ...])
    sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1) # e.g. (bsz x [0.9, 0.05, 0.02, 0.01, 0.01, ...])

    # Apply min_p sampling
    top_prob = sorted_probs[..., 0] # Highest probability e.g. (bsz x[0.9])
    scaled_min_p = min_p * top_prob # e.g. 0.9 * 0.1 = 0.09, (bsz x[0.09])
    min_p_mask = sorted_probs > scaled_min_p[..., None] # e.g. (bsz * [True, False, False, False, False, ...])
    sorted_probs = mx.where(min_p_mask, sorted_probs, 0.0) # e.g. (bsz * [0.9, 0.0, 0.0, 0.0, 0.0, ...])

    # Apply top_p (nucleus) sampling
    cumulative_probs = mx.cumsum(sorted_probs, axis=-1, inclusive = False) # e.g. (bsz * [0.9, 0.95, 0.97, 0.98, 0.99, ...]
    # or, if min_p is applied, (bsz * [0.9, 0.0, 0.0, 0.0, 0.0, ...]
    top_p_mask = cumulative_probs <= top_p # e.g. (bsz * [True, True, True, True, True, ...]
    # or, if min_p is applied, (bsz * [True, False, False, False, False, ...]
    sorted_probs = mx.where(top_p_mask, sorted_probs, 0.0) # e.g. (bsz * [0.9, 0.05, 0.02, 0.01, 0.01, ...])

    # Optionally apply top_k sampling
    sorted_probs[..., top_k:] = 0.0 # e.g. (bsz * [0.9, 0.05, 0.0, 0.0, 0.0, ...])

    sorted_probs = sorted_probs / mx.sum(sorted_probs, axis=-1, keepdims=True) # Normalize probabilities, e.g. (bsz * [0.9, 0.05, 0.0, 0.0, 0.0, ...])

    # Sample token
    sorted_token = mx.random.categorical(mx.log(sorted_probs), key=key)[..., None] # e.g. (bsz * [1390, 3, 2791, 1381, 12476, ...])
    token = mx.take_along_axis(sorted_indices, sorted_token, axis=-1) # e.g. [3,] in shape (batch_size,)
    return token

def nucleus_sample(
    logits: mx.array,
    temperature=0.666,
    top_p=0.9,
    top_k=27,
    key: Union[mx.array, None] = None,
) -> mx.array:
    batch_size = logits.shape[0]
    logit = logits[:, -1] / temperature  # (batch_size, vocab_size)
    probs = mx.softmax(logit, axis=-1)

    sorted_indices = mx.argsort(-probs, axis=-1)  # e.g. (bsz x [3, 1280, 1, 0, 2, ...])
    sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)  # e.g. (bsz x [0.9, 0.05, 0.02, 0.01, 0.01, ...])
    cumulative_probs = mx.cumsum(sorted_probs, axis=-1)
    mask = cumulative_probs < top_p

    sorted_probs = mx.where(mask, sorted_probs, 0.0)
    sorted_probs[..., top_k:] = 0.0
    sorted_probs = sorted_probs / mx.sum(sorted_probs, axis=-1, keepdims=True)
    sorted_token = mx.random.categorical(mx.log(sorted_probs), key=key)[
        ..., None]

    token = mx.take_along_axis(
        sorted_indices, sorted_token, axis=-1
    )
    return token
