from typing import NamedTuple
import mlx.core as mx

class AttnStats(NamedTuple):
    entropy: mx.array  # (batch_size, n_layers, num_heads)
    varentropy: mx.array  # (batch_size, n_layers, num_heads)
    n_layers: int
    n_heads: int

    @classmethod
    def new(cls, batch_size: int, n_layers: int, n_heads: int) -> 'AttnStats':
        return cls(
            entropy=mx.zeros((batch_size, n_layers, n_heads), dtype=mx.float32),
            varentropy=mx.zeros((batch_size, n_layers, n_heads), dtype=mx.float32),
            n_layers=n_layers,
            n_heads=n_heads
        )

    @property
    def avg_entropy(self):
        return self.entropy.sum(axis=-1, keepdims=False)

    @property
    def std_error(self):
        return mx.sqrt(mx.mean(self.varentropy)) / (self.n_heads * self.n_layers)

    def update(self, scores: mx.array, layer_idx: int):
        # scores shape: (batch_size, n_heads, seqlen, n_words)
        probs = mx.softmax(scores, axis=-1)
        new_entropy = -mx.sum(mx.where(probs > 0, probs * mx.log(probs), 0), axis=-1)
        new_varentropy = mx.sum(probs * (mx.log(probs) + new_entropy[..., None])**2, axis=-1)

        self.entropy[:, layer_idx, :] = new_entropy
        self.varentropy[:, layer_idx, :] = new_varentropy

        return AttnStats(
                    entropy=self.entropy,
                    varentropy=self.varentropy,
                    n_layers=self.n_layers,
                    n_heads=self.n_heads
                )
