"""Minimal reverse-mode autodiff engine implemented with NumPy only."""
from __future__ import annotations
import numpy as np


def _unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Sum broadcast axes so grad has exactly `shape`."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    def __init__(self, data, requires_grad: bool = False, _parents=(), _backward=None, name=None):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = bool(requires_grad)
        self.grad = np.zeros_like(self.data) if self.requires_grad else None
        self._parents = tuple(p for p in _parents if p.requires_grad)
        self._backward = _backward or (lambda grad, grads: None)
        self.name = name

    @property
    def shape(self):
        return self.data.shape

    def zero_grad(self):
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    def backward(self, grad=None):
        if not self.requires_grad:
            raise RuntimeError("Cannot call backward on a tensor that does not require gradients")
        if grad is None:
            if self.data.size != 1:
                raise ValueError("grad must be provided for non-scalar outputs")
            grad = np.ones_like(self.data)
        grad = np.asarray(grad, dtype=np.float64)
        if grad.shape != self.shape:
            raise ValueError(f"upstream gradient shape {grad.shape} != output shape {self.shape}")
        topo, seen = [], set()
        def visit(node):
            if id(node) not in seen:
                seen.add(id(node))
                for parent in node._parents:
                    visit(parent)
                topo.append(node)
        visit(self)
        # Each backward call computes fresh local gradients; leaf gradients accumulate.
        grads = {id(self): grad.copy()}
        for node in reversed(topo):
            g = grads.get(id(node))
            if g is None:
                continue
            if node.requires_grad:
                node.grad += g
            node._backward_with(g, grads)

    def _backward_with(self, grad, grads):
        # Operation closures add contributions to this call's local gradient map.
        self._backward(grad, grads)

    @staticmethod
    def _accumulate(grads, node, value):
        if node.requires_grad:
            key = id(node)
            grads[key] = grads.get(key, 0) + value

    def __add__(self, other):
        other = as_tensor(other)
        out = Tensor(self.data + other.data, self.requires_grad or other.requires_grad, (self, other))
        def back(g, grads):
            self._accumulate(grads, self, _unbroadcast(g, self.shape))
            self._accumulate(grads, other, _unbroadcast(g, other.shape))
        out._backward = back
        return out
    __radd__ = __add__

    def __neg__(self):
        out = Tensor(-self.data, self.requires_grad, (self,))
        out._backward = lambda g, grads: self._accumulate(grads, self, -g)
        return out

    def __sub__(self, other): return self + (-as_tensor(other))
    def __rsub__(self, other): return as_tensor(other) - self

    def __mul__(self, other):
        other = as_tensor(other)
        out = Tensor(self.data * other.data, self.requires_grad or other.requires_grad, (self, other))
        def back(g, grads):
            self._accumulate(grads, self, _unbroadcast(g * other.data, self.shape))
            self._accumulate(grads, other, _unbroadcast(g * self.data, other.shape))
        out._backward = back
        return out
    __rmul__ = __mul__

    def __truediv__(self, other):
        other = as_tensor(other)
        if np.any(other.data == 0): raise ZeroDivisionError("Tensor division by zero")
        out = Tensor(self.data / other.data, self.requires_grad or other.requires_grad, (self, other))
        def back(g, grads):
            self._accumulate(grads, self, _unbroadcast(g / other.data, self.shape))
            self._accumulate(grads, other, _unbroadcast(-g * self.data / (other.data ** 2), other.shape))
        out._backward = back
        return out

    def __matmul__(self, other):
        other = as_tensor(other)
        out = Tensor(self.data @ other.data, self.requires_grad or other.requires_grad, (self, other))
        def back(g, grads):
            if self.requires_grad: self._accumulate(grads, self, g @ np.swapaxes(other.data, -1, -2))
            if other.requires_grad: self._accumulate(grads, other, np.swapaxes(self.data, -1, -2) @ g)
        out._backward = back
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), self.requires_grad, (self,))
        def back(g, grads):
            expanded = g
            if axis is not None and not keepdims:
                axes = (axis,) if isinstance(axis, int) else tuple(axis)
                for ax in sorted((a % self.data.ndim for a in axes)):
                    expanded = np.expand_dims(expanded, ax)
            self._accumulate(grads, self, np.broadcast_to(expanded, self.shape))
        out._backward = back
        return out

    def mean(self, axis=None, keepdims=False):
        if axis is None: count = self.data.size
        else:
            axes = (axis,) if isinstance(axis, int) else tuple(axis)
            count = int(np.prod([self.shape[a] for a in axes]))
        return self.sum(axis=axis, keepdims=keepdims) / count

    def relu(self):
        out = Tensor(np.maximum(self.data, 0), self.requires_grad, (self,))
        mask = self.data > 0
        out._backward = lambda g, grads: self._accumulate(grads, self, g * mask)
        return out

    def tanh(self):
        value = np.tanh(self.data)
        out = Tensor(value, self.requires_grad, (self,))
        out._backward = lambda g, grads: self._accumulate(grads, self, g * (1 - value * value))
        return out

    def sigmoid(self):
        # Stable sigmoid without overflow for large negative inputs.
        value = np.empty_like(self.data)
        pos = self.data >= 0
        value[pos] = 1.0 / (1.0 + np.exp(-self.data[pos]))
        ex = np.exp(self.data[~pos])
        value[~pos] = ex / (1.0 + ex)
        out = Tensor(value, self.requires_grad, (self,))
        out._backward = lambda g, grads: self._accumulate(grads, self, g * value * (1 - value))
        return out

    def __repr__(self): return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"


def as_tensor(x): return x if isinstance(x, Tensor) else Tensor(x)


def softmax_cross_entropy(logits: Tensor, labels: np.ndarray) -> Tensor:
    """Mean multiclass cross entropy; labels are integer class IDs."""
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if logits.data.ndim != 2 or logits.data.shape[0] != len(y):
        raise ValueError("logits must have shape (batch, classes) and match labels")
    shifted = logits.data - logits.data.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    n = len(y)
    loss_value = -np.log(np.maximum(probs[np.arange(n), y], np.finfo(np.float64).tiny)).mean()
    out = Tensor(loss_value, logits.requires_grad, (logits,))
    grad_logits = probs.copy()
    grad_logits[np.arange(n), y] -= 1.0
    grad_logits /= n
    out._backward = lambda g, grads: Tensor._accumulate(grads, logits, grad_logits * g)
    return out
