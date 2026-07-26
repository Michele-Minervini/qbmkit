"""Loss base class.

A loss is a pure object holding its target; it computes ``value(state)`` and
``grad(state)`` using only the backend-agnostic ThermalState protocol, so it
works on any backend.
"""

from __future__ import annotations

import numpy as np


class Loss:
    """Abstract objective ``L(rho(theta))``."""

    def value(self, state) -> float:
        raise NotImplementedError

    def grad(self, state) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, state):
        return self.value(state), self.grad(state)
