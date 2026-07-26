"""JAX backend: autodiff-powered Gibbs-state engine (optional, ``pip install qbmkit[jax]``).

The Gibbs state is built in JAX (``jax.numpy.linalg.eigh`` with a ground-energy
shift), and **gradients and quantum-Fisher-information metrics are obtained by
automatic differentiation** of the state construction (``jax.grad`` / ``jax.jacrev``)
rather than from the hand-derived analytic kernels.  This is the strongest possible
cross-check of the dense engine: a JAX-backend result that matches the dense result
independently validates the belief-propagation gradient and the Kubo-Mori/Fisher-
Bures/Wigner-Yanase metric formulas.  It also runs on GPU/TPU.

Note: ``eigh`` autodiff is fragile at exact eigenvalue degeneracies; generic
parameters are fine.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)  # match the dense engine's float64 precision

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from ..metrics.monotone import mc_weight  # noqa: E402


def gibbs_density_matrix(theta, generator_stack):
    """``rho(theta) = e^{-G(theta)}/Z`` in JAX, overflow-safe via a ground-energy shift."""
    G = jnp.tensordot(theta.astype(generator_stack.dtype), generator_stack, axes=1)
    w, V = jnp.linalg.eigh(G)
    shifted = jnp.exp(-(w - w[0]))
    p = shifted / jnp.sum(shifted)
    return (V * p) @ V.conj().T


class JaxThermalState:
    """Thermal state whose gradients/metrics come from JAX autodiff."""

    def __init__(self, ham, theta):
        self.ham = ham
        self.theta = np.asarray(theta, dtype=float)
        self.dim = ham.dim
        self.n_qubits = ham.n_qubits
        self._Gs = jnp.asarray(np.stack([np.asarray(g, dtype=complex) for g in ham.generators]))
        self._theta = jnp.asarray(self.theta)
        self._rho = None
        self._D = None

    def _rho_fn(self, theta):
        return gibbs_density_matrix(theta, self._Gs)

    def density_matrix(self) -> np.ndarray:
        if self._rho is None:
            self._rho = np.asarray(self._rho_fn(self._theta))
        return self._rho

    def expect(self, op) -> float:
        Oj = jnp.asarray(np.asarray(op, dtype=complex))
        return float(jnp.real(jnp.trace(Oj @ self._rho_fn(self._theta))))

    def generator_expectations(self) -> np.ndarray:
        rho = self._rho_fn(self._theta)
        return np.asarray(jnp.real(jnp.einsum("jab,ba->j", self._Gs, rho)))

    def observable_gradient(self, op) -> np.ndarray:
        Oj = jnp.asarray(np.asarray(op, dtype=complex))

        def f(theta):
            return jnp.real(jnp.trace(Oj @ self._rho_fn(theta)))

        return np.asarray(jax.grad(f)(self._theta))

    def state_derivatives(self) -> np.ndarray:
        # real/imag split keeps jacrev on a real-valued output (robust for complex rho)
        if self._D is None:
            def rho_ri(theta):
                r = self._rho_fn(theta)
                return jnp.stack([jnp.real(r), jnp.imag(r)])

            jac = jax.jacrev(rho_ri)(self._theta)        # (2, dim, dim, J)
            D = np.asarray(jac[0]) + 1j * np.asarray(jac[1])
            self._D = np.moveaxis(D, -1, 0)              # (J, dim, dim)
        return self._D

    def diagonal_gradient(self) -> np.ndarray:
        return np.real(np.einsum("jaa->ja", self.state_derivatives()))

    def metric(self, kind: str = "kubo_mori") -> np.ndarray:
        rho = self.density_matrix()
        lam, Q = np.linalg.eigh(rho)
        lam = np.clip(lam, 1e-300, None)
        D = self.state_derivatives()
        M = np.einsum("ab,jbc,cd->jad", Q.conj().T, D, Q)
        W = mc_weight(lam, kind)
        g = np.real(np.einsum("ikl,kl,jkl->ij", M, W, np.conj(M)))
        return 0.5 * (g + g.T)

    def entropy(self) -> float:
        lam = np.clip(np.linalg.eigvalsh(self.density_matrix()), 1e-300, None)
        lam = lam[lam > 1e-300]
        return float(-np.sum(lam * np.log(lam)))

    def log_partition(self) -> float:
        w = np.linalg.eigvalsh(self.ham.matrix(self.theta))
        w0 = w[0]
        return float(-w0 + np.log(np.sum(np.exp(-(w - w0)))))

    def probabilities(self) -> np.ndarray:
        return np.real(np.diag(self.density_matrix()))

    def sample(self, n: int, rng=None) -> np.ndarray:
        rng = np.random.default_rng() if rng is None else rng
        p = np.clip(self.probabilities(), 0.0, None)
        return rng.choice(self.dim, size=n, p=p / p.sum())


class JaxBackend:
    """Builds :class:`JaxThermalState` objects (autodiff gradients/metrics, GPU-capable)."""

    name = "jax"

    def thermal_state(self, ham, theta) -> JaxThermalState:
        return JaxThermalState(ham, theta)

    def __repr__(self) -> str:
        return "JaxBackend()"
