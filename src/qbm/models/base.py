"""Base model: a parameter vector that produces a thermal state via a backend."""

from __future__ import annotations

import numpy as np

from ..backends import get_backend


class Model:
    """A QBM model: generators ``G_j``, free parameters ``theta``, and a backend.

    Subclasses typically just supply a default generator set; everything else
    (state construction, expectations, sampling, metrics) is inherited.
    """

    def __init__(self, ham, theta=None, backend=None):
        self.ham = ham
        self.backend = get_backend(backend)
        if theta is None:
            theta = np.zeros(ham.n_params)
        self.theta = np.asarray(theta, dtype=float).copy()
        if self.theta.shape != (ham.n_params,):
            raise ValueError(f"theta must have shape ({ham.n_params},)")

    @property
    def n_params(self) -> int:
        return self.ham.n_params

    @property
    def n_qubits(self) -> int:
        return self.ham.n_qubits

    def state(self):
        """Build the current thermal state ``rho(theta)``."""
        return self.backend.thermal_state(self.ham, self.theta)

    # -- convenience read-outs --------------------------------------------
    def density_matrix(self) -> np.ndarray:
        return self.state().density_matrix()

    def probabilities(self) -> np.ndarray:
        return self.state().probabilities()

    def sample(self, n: int, rng=None) -> np.ndarray:
        return self.state().sample(n, rng)

    def energy(self, observable: np.ndarray) -> float:
        return self.state().expect(observable)

    def kl(self, q: np.ndarray) -> float:
        """KL divergence ``D(q || p_model)`` of the computational-basis marginal."""
        q = np.asarray(q, dtype=float)
        p = self.probabilities()
        mask = q > 0
        return float(np.sum(q[mask] * (np.log(q[mask]) - np.log(np.clip(p[mask], 1e-300, None)))))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(n_qubits={self.n_qubits}, n_params={self.n_params})"
