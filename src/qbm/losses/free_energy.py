"""Free-energy objective ``F = <H> - T S(rho)``.

Minimising ``F`` over the QBM parameters variationally prepares the Gibbs state of
``H`` at temperature ``T`` (the global minimum is the exact free energy
``-T log Tr exp(-H/T)``).  This is the entry point for free-energy minimisation
and the entropy-regularised view of SDP / Gibbs-state preparation.

Gradient (using the Gibbs identity ``d_j S = Tr[G(theta) d_j rho]``)::

    d_j F = Tr[(H - T G(theta)) d_j rho]
"""

from __future__ import annotations

import numpy as np

from .base import Loss


class FreeEnergy(Loss):
    """``L(theta) = Tr(H rho) - T * S(rho)`` for a fixed Hamiltonian ``H``."""

    def __init__(self, hamiltonian: np.ndarray, temperature: float = 1.0):
        self.H = np.asarray(hamiltonian, dtype=complex)
        self.T = float(temperature)

    def value(self, state) -> float:
        return state.expect(self.H) - self.T * state.entropy()

    def grad(self, state) -> np.ndarray:
        G = state.ham.matrix(state.theta)
        return state.observable_gradient(self.H - self.T * G)
