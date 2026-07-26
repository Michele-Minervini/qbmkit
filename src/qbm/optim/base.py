"""Optimizer base class.

``step(theta, grad, state=None)`` returns the updated parameter vector.  The
optional ``state`` lets geometry-aware optimizers (natural gradient, Newton)
query the QFI metric without changing the calling convention.
"""

from __future__ import annotations

import numpy as np


class Optimizer:
    """Abstract parameter-update rule."""

    def step(self, theta: np.ndarray, grad: np.ndarray, state=None) -> np.ndarray:
        raise NotImplementedError

    def reset(self) -> None:
        """Clear any internal state (e.g. Adam moments)."""
