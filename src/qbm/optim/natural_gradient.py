"""Quantum natural gradient: ``theta <- theta - lr (g + reg I)^{-1} grad``.

``g`` is a quantum-Fisher-information matrix (default Kubo-Mori, the free-energy
Hessian for Gibbs states).  The ridge ``reg`` keeps the (often ill-conditioned)
metric invertible.  Paper prefactors (e.g. 4*lr for Fisher-Bures, 2*lr for
Kubo-Mori) are absorbed into ``lr``.
"""

from __future__ import annotations

import numpy as np

from ..metrics.base import get_metric
from .base import Optimizer


class NaturalGradient(Optimizer):
    """Metric-preconditioned gradient step."""

    def __init__(self, metric="kubo_mori", lr=0.05, reg=1e-4):
        self.metric = get_metric(metric)
        self.lr = lr
        self.reg = reg

    def step(self, theta, grad, state=None):
        if state is None:
            raise ValueError("NaturalGradient.step requires the current ThermalState")
        grad = np.asarray(grad, dtype=float)
        g = self.metric.matrix(state)
        if not np.all(np.isfinite(g)):
            raise FloatingPointError(
                f"the {self.metric.kind} metric contains non-finite entries, so the "
                "natural-gradient step is undefined. This usually means the model "
                "hit a degenerate spectrum on a backend whose derivatives are not "
                "degeneracy-safe. Try backend='dense', a different initialisation, "
                "or a plain optimizer (Adam / GradientDescent)."
            )
        g = g + self.reg * np.eye(g.shape[0])
        # lstsq is robust to a (near-)singular metric; the ridge usually prevents it
        try:
            nat = np.linalg.solve(g, grad)
        except np.linalg.LinAlgError:
            nat = np.linalg.lstsq(g, grad, rcond=None)[0]
        return theta - self.lr * nat
