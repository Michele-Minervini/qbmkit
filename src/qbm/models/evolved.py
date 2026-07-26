"""Evolved Quantum Boltzmann Machine (EQBM), arXiv:2501.03367.

The model state is a Gibbs state of ``G(theta)`` conjugated by real-time evolution
under a second Hamiltonian ``H(phi)``:

    omega(theta, phi) = e^{-iH(phi)} rho(theta) e^{+iH(phi)},   rho(theta) = e^{-G(theta)}/Z .

The standard QBM is the special case ``phi = 0``.  Because ``omega`` is just ``rho``
conjugated by a unitary ``U = e^{-iH(phi)}``, every quantity reduces to objects we
already have:

* ``omega = U rho U^dag``;  its eigenvalues are those of ``rho`` (the populations
  ``p_k``) and its eigenvectors are ``U V``.
* theta-derivatives: ``d_theta omega = U (d_theta rho) U^dag`` (U is theta-independent).
* phi-derivatives: ``d_phi omega = (d_phi U) rho U^dag + U rho (d_phi U)^dag`` with
  ``d_phi U`` from the Daleckii-Krein kernel (see :func:`qbm.linalg.unitary_derivative_kernel`).

Exposing the full set of state derivatives ``{d_i omega}`` and ``omega``'s spectrum
lets the existing ``Energy`` / ``MarginalRelativeEntropy`` losses, the three QFI
metrics, and ``NaturalGradient`` all operate on the EQBM unchanged.
"""

from __future__ import annotations

import numpy as np

from ..backends import get_backend
from ..linalg import unitary_derivative_kernel
from ..metrics.monotone import mc_weight
from ..operators import ParamHamiltonian


class EvolvedState:
    """Exact dense state of an EQBM, exposing the generic state/derivative interface."""

    def __init__(self, G_ham, H_ham, theta, phi, backend):
        self.inner = backend.thermal_state(G_ham, theta)  # rho(theta)
        self.G_ham = G_ham
        self.H_ham = H_ham
        self.theta = np.asarray(theta, float)
        self.phi = np.asarray(phi, float)
        self.n_theta = G_ham.n_params
        self.n_phi = H_ham.n_params
        self.dim = G_ham.dim
        self.n_qubits = G_ham.n_qubits

        eps, W = np.linalg.eigh(H_ham.matrix(self.phi))
        self.eps, self.W = eps, W
        self.U = (W * np.exp(-1j * eps)) @ W.conj().T
        self.p = self.inner.p
        self.rho = self.inner.density_matrix()
        self.omega = self.U @ self.rho @ self.U.conj().T
        self._eigvecs = self.U @ self.inner.V  # eigenvectors of omega (eigvals = p)
        self._Dfull = None

    # -- generic state interface (shared by every loss/metric) ------------
    def density_matrix(self) -> np.ndarray:
        return self.omega

    def expect(self, op) -> float:
        return float(np.real(np.sum(np.asarray(op, complex) * self.omega.T)))

    def _dU(self, Hk: np.ndarray) -> np.ndarray:
        Hk_e = self.W.conj().T @ Hk @ self.W
        Lam = unitary_derivative_kernel(self.eps)
        return self.W @ (Hk_e * Lam) @ self.W.conj().T

    def state_derivatives(self) -> np.ndarray:
        if self._Dfull is None:
            U, Ud = self.U, self.U.conj().T
            drho = self.inner.state_derivatives()  # (n_theta, dim, dim)
            out = np.empty((self.n_theta + self.n_phi, self.dim, self.dim), dtype=complex)
            for j in range(self.n_theta):
                out[j] = U @ drho[j] @ Ud
            for k in range(self.n_phi):
                dUk = self._dU(self.H_ham.generators[k])
                out[self.n_theta + k] = dUk @ self.rho @ Ud + U @ self.rho @ dUk.conj().T
            self._Dfull = out
        return self._Dfull

    def observable_gradient(self, op) -> np.ndarray:
        D = self.state_derivatives()
        return np.real(np.einsum("kl,jlk->j", np.asarray(op, complex), D))

    def metric(self, kind: str = "kubo_mori") -> np.ndarray:
        D = self.state_derivatives()
        Q = self._eigvecs
        Qd = Q.conj().T
        M = np.empty_like(D)
        for i in range(D.shape[0]):
            M[i] = Qd @ D[i] @ Q
        W = mc_weight(self.p, kind)
        g = np.real(np.einsum("ikl,kl,jkl->ij", M, W, np.conj(M)))
        return 0.5 * (g + g.T)

    def probabilities(self) -> np.ndarray:
        return np.real(np.diag(self.omega))

    def sample(self, n: int, rng=None) -> np.ndarray:
        rng = np.random.default_rng() if rng is None else rng
        p = np.clip(self.probabilities(), 0.0, None)
        return rng.choice(self.dim, size=n, p=p / p.sum())

    def log_partition(self) -> float:
        return self.inner.log_partition()


class EvolvedQBM:
    """An Evolved QBM with imaginary-time generators ``G_j`` and real-time generators ``H_k``.

    The flat parameter vector is ``theta = [gibbs params] ++ [evolution params]``.
    Use ``Energy`` for ground-state energy and ``MarginalRelativeEntropy`` (with
    ``n_visible = n_qubits``) for state learning / generative modelling; both yield
    the full ``(theta, phi)`` gradient.
    """

    def __init__(self, gibbs_generators, evolution_generators, theta=None, phi=None, backend=None):
        self.G = ParamHamiltonian(gibbs_generators)
        self.H = ParamHamiltonian(evolution_generators)
        if self.G.dim != self.H.dim:
            raise ValueError("gibbs and evolution generators must act on the same space")
        self.n_theta = self.G.n_params
        self.n_phi = self.H.n_params
        self.dim = self.G.dim
        self.n_qubits = self.G.n_qubits
        self.backend = get_backend(backend)
        th = np.zeros(self.n_theta) if theta is None else np.asarray(theta, float)
        ph = np.zeros(self.n_phi) if phi is None else np.asarray(phi, float)
        self.theta = np.concatenate([th, ph])

    @property
    def n_params(self) -> int:
        return self.n_theta + self.n_phi

    @property
    def gibbs_theta(self) -> np.ndarray:
        return self.theta[: self.n_theta]

    @property
    def evolution_phi(self) -> np.ndarray:
        return self.theta[self.n_theta :]

    def state(self) -> EvolvedState:
        return EvolvedState(self.G, self.H, self.gibbs_theta, self.evolution_phi, self.backend)

    def density_matrix(self) -> np.ndarray:
        return self.state().density_matrix()

    def probabilities(self) -> np.ndarray:
        return self.state().probabilities()

    def sample(self, n: int, rng=None) -> np.ndarray:
        return self.state().sample(n, rng)

    def energy(self, observable) -> float:
        return self.state().expect(observable)

    def kl(self, q: np.ndarray) -> float:
        q = np.asarray(q, dtype=float)
        p = self.probabilities()
        mask = q > 0
        return float(np.sum(q[mask] * (np.log(q[mask]) - np.log(np.clip(p[mask], 1e-300, None)))))

    def __repr__(self) -> str:
        return f"EvolvedQBM(n_qubits={self.n_qubits}, n_theta={self.n_theta}, n_phi={self.n_phi})"
