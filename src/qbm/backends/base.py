"""Backend and ThermalState protocols (the swap seam).

Everything above the backend layer (models, losses, metrics, optimizers, tasks)
is written against these two protocols only, expressed in the language of QBM
math.  A new backend (tensor network, circuit, hardware) implements the same
methods and unlocks the whole library with no change to user code.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ThermalState(Protocol):
    """A handle to ``rho(theta) = exp(-G(theta)) / Z`` answering QBM questions."""

    def expect(self, op: np.ndarray) -> float:
        """Expectation value ``<O> = Tr(rho O)``."""

    def generator_expectations(self) -> np.ndarray:
        """Vector of model-term expectations ``[<G_j>]``."""

    def observable_gradient(self, op: np.ndarray) -> np.ndarray:
        """Gradient ``[Tr(O d_j rho)]`` of an observable expectation w.r.t. theta."""

    def belief_prop(self, op: np.ndarray) -> np.ndarray:
        """Apply the belief-propagation channel ``Phi(op)``."""

    def metric(self, kind: str = "kubo_mori") -> np.ndarray:
        """Quantum-Fisher-information matrix for the chosen monotone metric."""

    def diagonal_gradient(self) -> np.ndarray:
        """``d_j`` of the computational-basis probability vector, shape ``(J, dim)``."""

    def state_derivatives(self) -> np.ndarray:
        """Full state derivatives ``[d_j rho]`` in the computational basis.

        Shape ``(J, dim, dim)``.
        """

    def entropy(self) -> float:
        """von Neumann entropy ``S(rho)``."""

    def log_partition(self) -> float:
        """``log Z(theta)``."""

    def density_matrix(self) -> np.ndarray:
        """Dense ``rho`` (small systems only)."""

    def probabilities(self) -> np.ndarray:
        """Computational-basis probability vector ``diag(rho)``."""

    def sample(self, n: int, rng=None) -> np.ndarray:
        """Draw ``n`` computational-basis bitstring indices from ``diag(rho)``."""


@runtime_checkable
class Backend(Protocol):
    """Turns a ``ParamHamiltonian`` + parameters into a ``ThermalState``."""

    def thermal_state(self, ham, theta: np.ndarray) -> ThermalState: ...
