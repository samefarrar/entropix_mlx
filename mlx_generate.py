from mlx_lm.utils import apply_repetition_penalty
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.models.cache import KVCache, RotatingKVCache, make_prompt_cache
import mlx.core as mx
import mlx.nn as nn
from transformers import PreTrainedTokenizer
from typing import Union, Optional, Callable, Generator, List, Tuple, Dict
from mlx_lm.sample_utils import top_p_sampling, min_p_sampling, categorical_sampling
import time
from mlx_sampler import sample
from mlx_attention_sampler import SamplerConfig
import numpy as np

LN_2 = 0.69314718056  # ln(2)
max_float32 = np.finfo(np.float32).max
DEFAULT_MASK_VALUE = -0.7 * mx.array(max_float32, dtype=mx.float16)

def generate_step(
    prompt: mx.array,
    model: nn.Module,
    prefill_step_size: int = 4092,
    max_kv_size: Optional[int] = None,
    cache_history: Optional[List[Tuple[mx.array, mx.array]]] = None,
    sampler_config: SamplerConfig = SamplerConfig(),
    key: Union[mx.array, None] = None,
) -> Generator[Tuple[mx.array, Dict[str, float]], None, None]:
    """
    A generator producing token ids based on the given prompt from the model.

    Args:
        prompt (mx.array): The input prompt as a tensor of token ids.
        model (nn.Module): The language model to use for generation.
        temp (float): The temperature for sampling. If 0, argmax is used.
            Default: 0.666.
        top_p (float): The cumulative probability threshold for nucleus sampling.
            Higher values consider more less likely words. Default: 0.9.
        top_k (int): The number of highest probability vocabulary tokens to keep for
            top-k filtering. Default: 0.
        prefill_step_size (int): The number of tokens to process in each prefill step.
            Default: 512.
        max_kv_size (Optional[int]): The maximum size of the key-value cache.
            If None, no limit is applied. Default: None.
        cache_history (Optional[List[Tuple[mx.array, mx.array]]]): The history of
            key-value pairs for each layer to initialize the cache. Default: None.
        model_with_scores (Optional[bool]): Whether the model returns scores in to the
            output. Default: True.

    Yields:
        int: The next generated token id.

    Note:
        This function uses entropy sampling to allow for "thinking" time when the model is less certain.
    """
    y = prompt
    tokens = None

    # Create the KV cache for generation
    cache = make_prompt_cache(model, max_kv_size)

    if cache_history is not None:
        if len(cache_history) != len(cache):
            raise ValueError("Wrong number of layers in the cache history")

        # Set the history in the cache objects and evaluate them to prepare for
        # generation.
        for c, h in zip(cache, cache_history):
            c.update_and_fetch(h[0], h[1])
        mx.eval([c.state for c in cache])

    def _step(y):
        logits, scores, attention_stats = model(y, cache=cache)

        # In the original xjdr repo, scores are calculated on un-masked logits.
        # This means that in order for the scores to be comparable with xjdr thresholds,
        # we need to calculate scores on un-masked logits.
        # pad_length = model.max_seq_len - scores.shape[-1]
        # pad_width = [
        #     (0, 0),  # No padding on batch_size axis
        #     (0, 0),  # No padding on num_heads axis
        #     (0, 0),  # No padding on query_length axis
        #     (0, pad_length)  # Pad 0 before and pad_length after the key_length axis
        # ]
        # padded_scores = mx.pad(scores, pad_width=pad_width)
        y, metrics = sample(y, logits, scores, cfg = sampler_config, key=key) # Convert returned (bsz, 1) to (bsz, )
        metrics = {k: v.item() for k, v in metrics.items()}
        metrics["cur_pos"] = scores.shape[-1]
        return y, metrics

    while y.size > prefill_step_size:
        model(y[:prefill_step_size][None], cache=cache)
        mx.eval([c.state for c in cache])
        y = y[prefill_step_size:]

    y, metrics = _step(y[None])

    mx.async_eval(y)
    while True:
        next_y, next_metrics = _step(y)
        mx.async_eval(next_y)
        yield (y.item(), metrics)
        y, metrics = next_y, next_metrics

def generate(
    model: nn.Module,
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompt: str,
    max_tokens: int = 100,
    verbose: bool = False,
    formatter: Optional[Callable] = None,
    seed: Optional[int] = None,
    **kwargs,
) -> Union[str, Generator[str, None, None]]:
    """
    Generate a complete response from the model.

    Args:
       model (nn.Module): The language model.
       tokenizer (PreTrainedTokenizer): The tokenizer.
       prompt (str): The string prompt.
       max_tokens (int): The maximum number of tokens. Default: ``100``.
       verbose (bool): If ``True``, print tokens and timing information.
           Default: ``False``.
       formatter (Optional[Callable]): A function which takes a token and a
           probability and displays it.
       kwargs: The remaining options get passed to :func:`generate_step`.
          See :func:`generate_step` for more details.
    """
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    if verbose:
        print("=" * 10)
        print("Prompt:", prompt)

    prompt_tokens = mx.array(tokenizer.encode(prompt))
    detokenizer = tokenizer.detokenizer

    tic = time.perf_counter()
    detokenizer.reset()

    sampler_config = SamplerConfig()
    if seed is not None:
        key = mx.random.seed(seed)
    else:
        key = None

    for (token, metrics), n in zip(
        generate_step(prompt_tokens, model, sampler_config = sampler_config, key=key, **kwargs),
        range(max_tokens),
    ):
        if n == 0:
            prompt_time = time.perf_counter() - tic
            tic = time.perf_counter()
        if token == tokenizer.eos_token_id:
            break
        detokenizer.add_token(token)

        if verbose:
            if formatter:
                # We have to finalize so that the prob corresponds to the last segment
                detokenizer.finalize()
                #formatter(detokenizer.last_segment, mx.exp(logprobs[token]).item())
            else:
                print(detokenizer.last_segment, end="", flush=True)

    token_count = n + 1
    detokenizer.finalize()

    if verbose:
        gen_time = time.perf_counter() - tic
        print(detokenizer.last_segment, flush=True)
        print("=" * 10)
        if token_count == 0:
            print("No tokens generated for this prompt")
            return
        prompt_tps = prompt_tokens.size / prompt_time
        gen_tps = (token_count - 1) / gen_time
        print(f"Prompt: {prompt_tokens.size} tokens, {prompt_tps:.3f} tokens-per-sec")
        print(f"Generation: {token_count} tokens, {gen_tps:.3f} tokens-per-sec")
        peak_mem = mx.metal.get_peak_memory() / 2**30
        print(f"Peak memory: {peak_mem:.3f} GB")

    return detokenizer.text
