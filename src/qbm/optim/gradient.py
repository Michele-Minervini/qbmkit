"""First-order optimizers: plain gradient descent and Adam."""

from __future__ import annotations

import numpy as np

from .base import Optimizer


class GradientDescent(Optimizer):
    """``theta <- theta - lr * grad`` (optional gradient clipping)."""

    def __init__(self, lr: float = 0.1, clip: float | None = None):
        self.lr = lr
        self.clip = clip

    def step(self, theta, grad, state=None):
        grad = np.asarray(grad, dtype=float)
        if self.clip is not None:
            grad = np.clip(grad, -self.clip, self.clip)
        return theta - self.lr * grad


class Adam(Optimizer):
    """Adam optimizer (Kingma & Ba)."""

    def __init__(self, lr=0.05, beta1=0.9, beta2=0.999, eps=1e-8, clip=None):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.clip = clip
        self.reset()

    def reset(self):
        self._m = None
        self._v = None
        self._t = 0

    def step(self, theta, grad, state=None):
        grad = np.asarray(grad, dtype=float)
        if self.clip is not None:
            grad = np.clip(grad, -self.clip, self.clip)
        if self._m is None:
            self._m = np.zeros_like(grad)
            self._v = np.zeros_like(grad)
        self._t += 1
        self._m = self.beta1 * self._m + (1 - self.beta1) * grad
        self._v = self.beta2 * self._v + (1 - self.beta2) * grad ** 2
        mhat = self._m / (1 - self.beta1 ** self._t)
        vhat = self._v / (1 - self.beta2 ** self._t)
        return theta - self.lr * mhat / (np.sqrt(vhat) + self.eps)
