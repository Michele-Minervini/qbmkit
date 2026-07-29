"""Dense NumPy/SciPy backend (the v1 default and exact validation oracle).

The Gibbs state is built by **eigendecomposition** of ``G(theta)`` (not ``expm``):
overflow-safe via a ground-energy shift, and one factorization yields rho, log Z,
the belief-propagation channel, all gradients and all metrics.
"""

from __future__ import annotations

import numpy as np

from ..channels import bp_multiplier, state_derivative_kernel
from ..metrics.monotone import mc_weight


class DenseThermalState:
    """Exact dense thermal state with cached eigendata of ``G(theta)``."""

    def __init__(self, ham, theta):
        self.ham = ham
        self.theta = np.asarray(theta, dtype=float)
        G = ham.matrix(self.theta)
        w, V = np.linalg.eigh(G)
        w0 = float(w[0])  # eigh returns ascending eigenvalues
        shifted = np.exp(-(w - w0))
        ztilde = float(shifted.sum())
        self.w = w
        self.V = V
        self.p = shifted / ztilde
        self.logZ = -w0 + np.log(ztilde)
        self.dim = ham.dim
        self.n_qubits = ham.n_qubits
        self._geig = None  # generators in the eigenbasis, list of (dim, dim)
        self._deig = None  # eigenbasis d_j rho, array (J, dim, dim)
        self._genexp = None

    # -- eigenbasis helpers ------------------------------------------------
    def _to_eig(self, op: np.ndarray) -> np.ndarray:
        return self.V.conj().T @ op @ self.V

    def _geig_list(self):
        if self._geig is None:
            self._geig = [self._to_eig(g) for g in self.ham.generators]
        return self._geig

    def generator_expectations(self) -> np.ndarray:
        # <G_j> = Tr(rho G_j) directly: O(dim^2) per generator, no eigenbasis
        # transform.  This is the only primitive the relative-entropy / NLL
        # gradient needs, so generative and state-learning training stays fast
        # even with O(n^2) generators.
        if self._genexp is None:
            rho = self.density_matrix()
            self._genexp = np.array(
                [float(np.real(np.sum(rho * g.T))) for g in self.ham.generators]
            )
        return self._genexp

    def _deig_list(self) -> np.ndarray:
        if self._deig is None:
            K = state_derivative_kernel(self.w, self.p)
            geig = self._geig_list()
            ge = self.generator_expectations()
            D = np.empty((len(geig), self.dim, self.dim), dtype=complex)
            diag = np.diag_indices(self.dim)
            for j, g in enumerate(geig):
                Dj = g * K
                Dj[diag] += self.p * ge[j]
                D[j] = Dj
            self._deig = D
        return self._deig

    # -- ThermalState protocol --------------------------------------------
    def expect(self, op: np.ndarray) -> float:
        oeig = self._to_eig(np.asarray(op, dtype=complex))
        return float(np.real(np.sum(self.p * np.diag(oeig))))

    def observable_gradient(self, op: np.ndarray) -> np.ndarray:
        oeig = self._to_eig(np.asarray(op, dtype=complex))
        D = self._deig_list()
        # Tr(O d_j rho) = sum_{k,l} O_eig[k,l] (d_j rho)_eig[l,k]
        grad = np.einsum("kl,jlk->j", oeig, D)
        return np.real(grad)

    def belief_prop(self, op: np.ndarray) -> np.ndarray:
        oeig = self._to_eig(np.asarray(op, dtype=complex))
        out = oeig * bp_multiplier(self.w)
        return self.V @ out @ self.V.conj().T

    def metric(self, kind: str = "kubo_mori") -> np.ndarray:
        D = self._deig_list()
        W = mc_weight(self.p, kind)
        g = np.einsum("ikl,kl,jkl->ij", D, W, np.conj(D))
        g = np.real(g)
        return 0.5 * (g + g.T)  # symmetrise away round-off

    def diagonal_gradient(self) -> np.ndarray:
        """``d_j`` of the computational-basis probability vector, shape ``(J, dim)``.

        ``d_j p(a) = (d_j rho)_{aa}`` in the computational basis.  Used by
        marginal-likelihood losses for visible/hidden models (the hidden trace is
        just a reshape-and-sum over this).

        Only the diagonal of ``V D V^dag`` is needed, so we contract it directly
        (``diag(V A V^dag)_a = sum_l (V A)_{al} conj(V)_{al}``) instead of forming the
        full transformed matrix -- one batched matmul rather than two per generator.
        """
        D = self._deig_list()
        M = np.einsum("ab,jbc->jac", self.V, D, optimize=True)
        return np.real(np.einsum("jac,ac->ja", M, self.V.conj(), optimize=True))

    def state_derivatives(self) -> np.ndarray:
        """Full state derivatives ``[d_j rho]`` in the computational basis, shape ``(J, dim, dim)``.

        Needed by losses that differentiate functions of a *reduced* state
        (e.g. relative entropy to a quantum target with hidden units).
        """
        D = self._deig_list()
        out = np.empty_like(D)
        for j in range(D.shape[0]):
            out[j] = self.V @ D[j] @ self.V.conj().T
        return out

    def entropy(self) -> float:
        """von Neumann entropy ``S(rho) = -sum_k p_k ln p_k``."""
        p = self.p[self.p > 1e-300]
        return float(-np.sum(p * np.log(p)))

    def log_partition(self) -> float:
        return float(self.logZ)

    def density_matrix(self) -> np.ndarray:
        return (self.V * self.p) @ self.V.conj().T

    def probabilities(self) -> np.ndarray:
        # diag(rho) in the computational basis = sum_k |V[i,k]|^2 p_k
        return (np.abs(self.V) ** 2) @ self.p

    def sample(self, n: int, rng=None) -> np.ndarray:
        rng = np.random.default_rng() if rng is None else rng
        probs = self.probabilities()
        probs = np.clip(probs, 0.0, None)
        probs = probs / probs.sum()
        return rng.choice(self.dim, size=n, p=probs)


class DenseBackend:
    """Builds :class:`DenseThermalState` objects."""

    name = "dense"

    def thermal_state(self, ham, theta) -> DenseThermalState:
        return DenseThermalState(ham, theta)

    def __repr__(self) -> str:
        return "DenseBackend()"
