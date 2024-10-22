from pydantic import BaseModel


class SamplerConfig(BaseModel):
    """
    Encapsulation of all available sampler hyperparameters.
    This should be a good starting point for baselining experiments.
    """

    temperature: float = 0.666
    top_p: float = 1.0
    top_k: int = 33
    min_probability: float = 0.03  # Turn this down to 0.01 to reduce the shoggoth

    # Logits entropy thresholds
    low_logits_entropy_threshold: float = 0.004
    medium_logits_entropy_threshold: float = 2.7
    high_logits_entropy_threshold: float = 3.7

    # Logits varentropy thresholds
    low_logits_varentropy_threshold: float = 0.0045
    medium_logits_varentropy_threshold: float = 3.0
    high_logits_varentropy_threshold: float = 6.0

    # Attention Entropy Thresholds
    low_attention_entropy_threshold: float = 1.6
    medium_attention_entropy_threshold: float = 3.0
    high_attention_entropy_threshold: float = 5.0

    # Attention Varentropy Thresholds
    low_attention_varentropy_threshold: float = 0.25
    medium_attention_varentropy_threshold: float = 0.8
    high_attention_varentropy_threshold: float = 1.5

    # Agreement thresholds
    low_agreement_threshold: float = 2.2e-3
    medium_agreement_threshold: float = 3e-3
    high_agreement_threshold: float = 3.8e-3

    # Interaction strength thresholds
    low_interaction_strength_threshold: float = 6.06
    medium_interaction_strength_threshold: float = 6.4
    high_interaction_strength_threshold: float = 7.0

    # TODO this is a bit of a nasty mess, but also makes all the hyperparameters visible
    high_entropy_attention_offset: float = 1.3
    high_entropy_attention_coefficient: float = 0.05

    low_entropy_interaction_strength_offset: float = 1.5
    low_entropy_interaction_strength_coefficient: float = 0.10

    high_entropy_varentropy_attention_offset: float = 1.2
    high_entropy_varentropy_attention_coefficient: float = 0.7

    # TODO not convinced this should
    n_adaptive_samples: int = 5

    adaptive_temperature_logits_coefficient: float = 0.5
    adaptive_temperature_attention_coefficient: float = 0.5
    adaptive_temperature_agreement_coefficient: float = 800
    adaptive_top_p_coefficient: float = 0.6
    adaptive_top_k_coefficient: float = 0.1
    adaptive_top_k_interaction_coefficient: float = 0.1
    adaptive_top_k_agreement_coefficient: float = 200
    adaptive_min_p_coefficient: float = 0.05
    adaptive_score_logits_entropy_coefficient: float = 0.1
    adaptive_score_attention_entropy_coefficient: float = 0.2 / 0.29
    adaptive_score_logits_varentropy_coefficient: float = 0.3
    adaptive_score_attention_varentropy_coefficient: float = 0.4 * 100
    adaptive_score_agreement_coefficient: float = 0.5
    adaptive_score_interaction_strength_coefficient: float = 0.6
