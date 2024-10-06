from mlx_lm import load
import mlx.nn as nn
import mlx.core as mx
from mlx_lm.models.base import KVCache, RotatingKVCache
from mlx_lm.models.llama import Attention, Model, LlamaModel, TransformerBlock, ModelArgs, create_attention_mask
from typing import Tuple, Optional, Callable
from types import MethodType
from mlx_stats import AttnStats
import inspect
from einops import einsum

class EntropyAttention(Attention):
    def __init__(self, original_attention: Attention, model_args: ModelArgs):
        super().__init__(model_args)
        self.__dict__.update(original_attention.__dict__)
        self.n_rep = self.n_heads // self.n_kv_heads

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> Tuple[mx.array, mx.array]:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3) # (B, n_heads, L, head_dim)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3) # (B, n_kv_heads, L, head_dim)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3) # (B, n_kv_heads, L, head_dim)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        shaped_keys = mx.repeat(keys, repeats=self.n_rep, axis = 1).transpose(0, 1, 3, 2) # (B, n_heads, L, head_dim)
        pre_scores = mx.matmul(queries, shaped_keys) / mx.sqrt(self.n_heads) # (B, n_heads, L, L)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output), pre_scores

class EntropyTransformerBlock(TransformerBlock):
    def __init__(self, original_block: TransformerBlock):
        super().__init__(original_block.args)
        self.__dict__.update(original_block.__dict__)
        self.self_attn = EntropyAttention(self.self_attn, original_block.args)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> Tuple[mx.array, mx.array]:
        r, scores = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out, scores

class EntropyLlamaModel(LlamaModel):
    def __init__(self, original_model: LlamaModel, model_args: ModelArgs):
        super().__init__(original_model.args)
        self.n_heads = model_args.num_attention_heads
        self.__dict__.update(original_model.__dict__)

        for i, layer in enumerate(self.layers):
            self.layers[i] = EntropyTransformerBlock(layer)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        h = self.embed_tokens(inputs)

        mask = create_attention_mask(h, cache)
        attention_stats = AttnStats.new(h.shape[0], n_layers = len(self.layers), n_heads = self.n_heads)

        if cache is None:
            cache = [None] * len(self.layers)

        for i, (layer, c) in enumerate(zip(self.layers, cache)):
            h, scores = layer(h, mask, cache=c)
            attention_stats = attention_stats.update(scores[:, :, -1, :], i)

        return self.norm(h), scores, attention_stats

class EntropixModel(Model):
    def __init__(self, original_model: Model):
        super().__init__(original_model.args)
        self.args = original_model.args
        self.__dict__.update(original_model.__dict__)
        self.model = EntropyLlamaModel(original_model.model, original_model.args)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        out, scores, attention_stats = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out, scores, attention_stats

def load_entropix_model(path: str):
    model, tokenizer = load(path)
    entropix_manipulated_model = EntropixModel(model)
    return entropix_manipulated_model, tokenizer
