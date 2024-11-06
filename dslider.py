import mlx.core as mx
import mlx.nn as nn
from einops.array_api import rearrange
from scipy.stats import dirichlet
import scipy.special as sp

from typing import NamedTuple, Tuple
from dslider_config import EPS, MAX_TEMP, MIN_TEMP, DSConfig
from dslider_utils import fit_dirichlet, temp_tune

def lgamma(alpha: mx.array):
    lgamma = sp.loggamma(alpha)
    return mx.array(lgamma)

def digamma(alpha: mx.array):
    digamma = sp.digamma(alpha)
    return mx.array(digamma)

def polygamma(n: int, alpha: mx.array):
    polygamma = sp.polygamma(n, alpha)
    return mx.array(polygamma)

def kl_divergence(logp: mx.array, logq: mx.array) -> mx.array:
    """
    Computes the KL divergence between two sets of log probabilities.
    """
    p = mx.exp(logp)
    kl_elements = p * (logp - logq)
    kl_elements = mx.where(p > 0, kl_elements, 0)
    return mx.sum(kl_elements, axis = -1)

def ent_varent(logp: mx.array):
    """
    Compute entropy and variance of entropy from log-probabilities
    """
    p = mx.exp(logp)
    ent = -mx.sum(p * logp, axis = -1)
    diff = logp + rearrange(ent, "v -> v 1")
    varent = mx.sum(p * diff **2, axis = -1)
    return ent, varent

def dirichlet_expectation(alpha: mx.array) -> mx.array:
    """
    Computes the Dirichlet expectation.
    """
    return alpha / mx.sum(alpha, axis = -1, keepdims = True)

def sample_dirichlet(alpha: mx.array, key: mx.array) -> mx.array:
    """
    Samples from a Dirichlet distribution.
    """
    return mx.array(dirichlet(alpha, int(key[0])))

class DSState(NamedTuple):
    """State maintained by the Adaptive Dirichlet Sampler"""

    emwa_dir: mx.array
    emwa_logp_dir_supp: mx.array
    emwa_temp: mx.array
    emwa_ent_scaffold: mx.array
    emwa_ent_naked: mx.array
    emwa_varent_scaffold: mx.array
    emwa_varent_naked: mx.array
    token_cross_ent_scaffold: mx.array
    token_cross_ent_naked: mx.array
    token_cross_var_scaffold: mx.array
    token_cross_var_naked: mx.array
    emwa_dir_ent: mx.array
    emwa_topk_ent_naked: mx.array

def dirichlet_expected_entropy(alpha: mx.array) -> mx.array:
    """
    Compute the expected entropy of a Dirichlet distribution.
    """
    alpha_sum = mx.sum(alpha, axis = -1, keepdims = True)
    K = alpha.shape[-1]

    # ln B(alpha) term
    log_beta = mx.sum(lgamma(alpha), axis = -1) - lgamma(rearrange(alpha_sum, "v -> v 1"))

    # (alpha_0 - K) * ψ(alpha_0) term
    digamma_sum = digamma(alpha_sum)
    second_term = (rearrange(alpha_sum, "1 v -> v") - K) * rearrange(digamma_sum, "1 v -> v")

    # -sum((alpha_j - 1) * ψ(alpha_j)) term
    digamma_alpha = digamma(alpha)
    third_term = -mx.sum((alpha - 1) * digamma_alpha, axis = -1)

    return log_beta + second_term + third_term

def dirichlet_log_likelihood_from_logprob(logprobs: mx.array, alpha: mx.array) -> mx.array:
    """
    Compute the log likelihood of a set of log probabilities under a Dirichlet distribution.
    """
    return (
        mx.sum((alpha - 1) * logprobs, axis = -1)
        - lgamma(mx.sum(alpha, axis = -1))
        + mx.sum(lgamma(alpha), axis = -1)
    )

def dirichlet_expected_varentropy(alpha: mx.array) -> mx.array:
    """
    Compute the expected variance of entropy of a Dirichlet distribution.
    """
    alpha_sum = mx.sum(alpha, axis = -1, keepdims = True)

    # E[X] = α / α_0
    digamma_alpha = digamma(alpha)
    trigamma_alpha = polygamma(1, alpha)

    squared_plus_deriv = digamma_alpha ** 2 + trigamma_alpha

    return mx.sum(expected_x * squared_plus_deriv, axis = -1)

def initialize_state(bsz: int, vsz: int, config: DSConfig) -> DSState:
    """Initialize the DSState with specified dtype."""
    state = DSState(
        emwa_dir=mx.ones((bsz, config.dirichlet_support.size)),
        emwa_logp_dir_supp=mx.zeros((bsz, config.dirichlet_support.size)),
        emwa_temp=mx.ones((bsz,)),
        emwa_ent_scaffold=mx.zeros((bsz,)),
        emwa_ent_naked=mx.zeros((bsz,)),
        emwa_varent_scaffold=mx.zeros((bsz,)),
        emwa_varent_naked=mx.zeros((bsz,)),
        token_cross_ent_scaffold=mx.zeros((bsz,)),
        token_cross_ent_naked=mx.zeros((bsz,)),
        token_cross_var_scaffold=mx.zeros((bsz,)),
        token_cross_var_naked=mx.zeros((bsz,)),
        emwa_dir_ent=mx.zeros((bsz,)),
        emwa_topk_ent_naked=mx.zeros((bsz,)),
    )
    return state
