from __future__ import annotations
import numpy as np
from engine import Tensor

class Linear:
    def __init__(self, in_features, out_features, rng, activation=None):
        # He initialization for ReLU hidden layers; Xavier for linear/tanh layers.
        scale = np.sqrt(2.0 / in_features) if activation == 'relu' else np.sqrt(2.0 / (in_features + out_features))
        self.weight = Tensor(rng.normal(0, scale, (in_features, out_features)), True, name='weight')
        self.bias = Tensor(np.zeros((out_features,)), True, name='bias')
        self.mask = np.ones_like(self.weight.data, dtype=bool)
        self.activation = activation

    def __call__(self, x: Tensor):
        # During training this is a masked dense operation. Sparse-aware inference
        # is implemented separately and skips inactive connections entirely.
        effective_weight = self.weight * self.mask.astype(np.float64)
        out = x @ effective_weight + self.bias
        if self.activation == 'relu': return out.relu()
        if self.activation == 'tanh': return out.tanh()
        return out

    def parameters(self): return [self.weight, self.bias]

    def sparse_forward(self, x: np.ndarray) -> np.ndarray:
        """Sparse-aware inference: only active weight entries are multiplied."""
        result = np.broadcast_to(self.bias.data, (x.shape[0], self.bias.data.size)).copy()
        for j in range(self.weight.shape[1]):
            active = np.flatnonzero(self.mask[:, j])
            if active.size:
                result[:, j] += x[:, active] @ self.weight.data[active, j]
        if self.activation == 'relu': result = np.maximum(result, 0)
        elif self.activation == 'tanh': result = np.tanh(result)
        return result

class MLP:
    def __init__(self, widths, seed=0):
        rng = np.random.default_rng(seed)
        self.layers = []
        for i, (a, b) in enumerate(zip(widths[:-1], widths[1:])):
            act = 'relu' if i < len(widths) - 2 else None
            self.layers.append(Linear(a, b, rng, activation=act))
    def __call__(self, x):
        for layer in self.layers: x = layer(x)
        return x
    def parameters(self): return [p for layer in self.layers for p in layer.parameters()]
    def weights(self): return [layer.weight for layer in self.layers]
    def masks(self): return [layer.mask for layer in self.layers]
    def predict_sparse(self, x):
        for layer in self.layers: x = layer.sparse_forward(x)
        return x
    def active_weight_count(self): return int(sum(m.sum() for m in self.masks()))
    def total_weight_count(self): return int(sum(m.size for m in self.masks()))
