"""Energy / observable-expectation loss ``L = Tr(O rho)``.

Used for ground-state-energy estimation (``O`` = target Hamiltonian) and any
variational observable-minimisation task.  Gradient: ``d_j L = Tr(O d_j rho)``.
"""

from __future__ import annotations

import numpy as np

from .base import Loss


class Energy(Loss):
    """``L(theta) = Tr(O rho(theta))`` for a fixed Hermitian observable ``O``."""

    def __init__(self, observable: np.ndarray):
        self.O = np.asarray(observable, dtype=complex)

    def value(self, state) -> float:
        return state.expect(self.O)

    def grad(self, state) -> np.ndarray:
        return state.observable_gradient(self.O)
