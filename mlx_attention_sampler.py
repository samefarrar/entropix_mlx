from pydantic import BaseModel
class SamplerConfig(BaseModel):
    """
    Encapsulation of all available sampler hyperparameters.

    This should be a good starting point for baselining experiments.
    """

    temp: float = 0.666
    top_p: float = 0.90
    top_k: int = 27
    min_p: float = 0.03  # Turn this down to 0.01 to reduce the shoggoth

    low_ent_thresh: float = 0.1
    low_vent_thresh: float = 0.1
    med_ent_thresh: float = 3.0
    high_ent_thresh: float = 5.0
    high_vent_thresh: float = 5.0

    # TODO this is a bit of a nasty mess, but also makes all the hyperparameters visible
    helv_attn_ent_offset: float = 1.3
    helv_attn_ent_coef: float = 0.2

    lehv_interaction_strength_offset: float = 1.2
    lehv_interaction_strength_coef: float = 0.3

    hehv_attn_ent_coef: float = 0.2
    hehv_attn_vent_offset: float = 2.0
    hehv_attn_vent_coef: float = 0.5

    # TODO not convinced this should
    n_adaptive_samples: int = 5

    # Adaptive sampling parameters
    ada_temp_logits: float = 0.3
    ada_temp_attn: float = 0.2
    ada_temp_agree: float = 0.2
    ada_top_p: float = 0.1
    ada_top_k_int: float = 0.3
    ada_top_k_agree: float = 0.2
    ada_min_p: float = 0.5
    ada_score_logits_ent: float = 0.1
    ada_score_attn_ent: float = 0.2
    ada_score_logits_vent: float = 0.3
    ada_score_attn_vent: float = 0.4
    ada_score_agree: float = 0.5
    ada_score_int: float = 0.6
