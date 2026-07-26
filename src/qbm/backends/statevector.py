"""Statevector backend via thermofield-double purification (with optional shot noise).

Represents the Gibbs state by its TFD purification on a doubled register and computes
expectations through the pure state ``<psi| O ⊗ I |psi>``.  With ``shots`` set, the
measurement-type quantities (``expect``, ``generator_expectations``, ``sample``)
return finite-sample estimates, modelling hardware readout — so sample-complexity /
shot-budget studies (cf. arXiv:2511.11802) work through the same training loop.

In simulation it shares the eigendecomposition with the dense engine, so the exact
analytic quantities (gradients, metrics, entropy) are reused; on hardware those
would be replaced by their estimator circuits (Hadamard test / QBGE), a future path.
"""

from __future__ import annotations

import numpy as np

from .. import purification
from .dense import DenseThermalState


class StatevectorThermalState(DenseThermalState):
    """A dense thermal state that also carries (and measures through) its TFD purification."""

    def __init__(self, ham, theta, shots=None, rng=None):
        super().__init__(ham, theta)
        self.shots = shots
        self.rng = np.random.default_rng() if rng is None else rng
        self.psi = purification.tfd_statevector(self.V, self.p)

    def tfd_state(self) -> np.ndarray:
        """The thermofield-double statevector (length ``dim**2``)."""
        return self.psi

    def entanglement_entropy(self) -> float:
        return purification.entanglement_entropy(self.psi, self.dim)

    # -- measurement-type quantities (shot-aware) -------------------------
    def _shot_expect(self, op: np.ndarray) -> float:
        o, R = np.linalg.eigh(np.asarray(op, dtype=complex))
        rho = self.density_matrix()
        probs = np.real(np.einsum("ik,ij,jk->k", np.conj(R), rho, R))
        probs = np.clip(probs, 0.0, None)
        probs = probs / probs.sum()
        draws = self.rng.choice(len(o), size=self.shots, p=probs)
        return float(np.mean(o[draws]))

    def expect(self, op) -> float:
        if self.shots is None:
            return purification.expectation(self.psi, np.asarray(op, complex), self.dim)
        return self._shot_expect(op)

    def generator_expectations(self) -> np.ndarray:
        if self.shots is None:
            return super().generator_expectations()
        return np.array([self._shot_expect(g) for g in self.ham.generators])

    def sample(self, n: int, rng=None) -> np.ndarray:
        return super().sample(n, rng if rng is not None else self.rng)


class StatevectorBackend:
    """Builds :class:`StatevectorThermalState` objects (TFD purification, optional shots)."""

    name = "statevector"

    def __init__(self, shots=None, seed=None):
        self.shots = shots
        self._rng = np.random.default_rng(seed)

    def thermal_state(self, ham, theta) -> StatevectorThermalState:
        return StatevectorThermalState(ham, theta, shots=self.shots, rng=self._rng)

    def __repr__(self) -> str:
        return f"StatevectorBackend(shots={self.shots})"
