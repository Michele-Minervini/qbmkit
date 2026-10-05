"""Quantum relative entropy ``D(sigma || rho(theta))``.

For a fixed target ``sigma`` (a quantum state, or a classical distribution ``q``
standing for ``diag(q)``) the gradient is the classic positive/negative phase
difference::

    d_j D(sigma || rho) = <G_j>_sigma - <G_j>_rho

which is exact for fully-visible QBMs even with non-commuting generators, and is
cheap (only generator expectations).  Note: for non-commuting generators
``D(diag(q) || rho)`` is *not* the measured-distribution likelihood (that is
:class:`qbm.losses.NLL`); it is the relative-entropy objective, equal to the NLL
only in the commuting/diagonal case and an upper bound otherwise.

**Scaling.**  The target side ``<G_j>_sigma`` is evaluated from the Pauli *labels* of
the generators, never from their dense matrices: a Pauli string has one nonzero per
row, so ``Tr(sigma P)`` is a sum over ``2^n`` entries of ``sigma``, and for a classical
target ``q`` every ``Z``-string moment is one entry of the Walsh-Hadamard transform of
``q``.  A backend that prepares ``rho(theta)`` without dense matrices (tensor network,
Pauli propagation) therefore trains on this loss without anything of size ``4^n`` ever
being built.
"""

from __future__ import annotations

import numpy as np

from ..linalg import log_divided_differences, partial_trace_hidden
from ..pauli_prop import _popcount, _popcount_array, _walsh_hadamard, label_to_xz
from .base import Loss


def _entropy_term(sigma: np.ndarray) -> float:
    """``Tr(sigma ln sigma)`` (the theta-independent part of the relative entropy)."""
    vals = np.linalg.eigvalsh(sigma)
    vals = np.clip(np.real(vals), 0.0, None)
    nz = vals[vals > 1e-15]
    return float(np.sum(nz * np.log(nz)))


class RelativeEntropy(Loss):
    """``L(theta) = D(sigma || rho(theta))`` for a fixed target.

    Parameters
    ----------
    sigma : ndarray
        The target.  A 2-D array is a density matrix.  A 1-D array is a classical
        distribution ``q`` over the ``2^n`` computational-basis states (qubit 0 the most
        significant bit) and stands for ``diag(q)`` -- which is never built, so classical
        data costs ``2^n`` numbers rather than ``4^n``.
    """

    def __init__(self, sigma: np.ndarray):
        sigma = np.asarray(sigma)
        if sigma.ndim == 1:  # classical target q, standing for diag(q)
            if sigma.size & (sigma.size - 1) or sigma.size < 2:
                raise ValueError(f"a probability vector needs length 2^n; got {sigma.size}")
            self.sigma = None
            self.q = np.real(sigma).astype(float)
            # <Z-string>_q for every Z-string at once: moments[z] = sum_v q_v (-1)^{|v & z|}
            self._moments = _walsh_hadamard(self.q)
            support = self.q[self.q > 0]
            self._const = float(np.sum(support * np.log(support)))
        elif sigma.ndim == 2 and sigma.shape[0] == sigma.shape[1]:
            self.sigma = sigma.astype(complex)
            self._index = np.arange(sigma.shape[0])
            self._const = _entropy_term(self.sigma)
        else:
            raise ValueError(
                "the target must be a density matrix (2-D, square) or a probability "
                f"vector (1-D); got an array of shape {sigma.shape}"
            )
        self._sigma_gen_exp = None  # cache of [Tr(sigma G_j)] keyed on ham id
        self._sigma_gen_key = None

    # -- target expectations ------------------------------------------------
    def _expect_pauli(self, label: str) -> float:
        """``Tr(sigma P)`` for a Pauli string, from its label alone.

        ``P = i^{|x & z|} X^x Z^z`` sends ``|c>`` to ``(-1)^{|c & z|} |c xor x>`` (up to
        that phase), so it has a single nonzero per column and the trace is a sum over
        the ``2^n`` entries ``sigma[c, c xor x]``.
        """
        x, z = label_to_xz(label)
        if self.sigma is None:  # X/Y strings are off-diagonal: no weight on diag(q)
            return 0.0 if x else float(self._moments[z])
        c = self._index
        signs = 1 - 2 * (_popcount_array(c & z) & 1)
        phase = 1j ** (_popcount(x & z) % 4)
        return float(np.real(phase * np.sum(self.sigma[c, c ^ x] * signs)))

    def _expect_dense(self, op: np.ndarray) -> float:
        """``Tr(sigma O)`` for a dense operator (matrix generators, the offset)."""
        if self.sigma is None:
            return float(np.real(np.diag(op)) @ self.q)
        return float(np.real(np.sum(self.sigma * op.T)))

    def _target_generator_expectations(self, state) -> np.ndarray:
        ham = state.ham
        key = id(ham)
        if self._sigma_gen_key != key:
            labels = getattr(ham, "pauli_labels", None)
            if labels is not None:  # the scalable route: dense generators stay unbuilt
                vals = [self._expect_pauli(lbl) for lbl in labels]
            else:
                vals = [self._expect_dense(g) for g in ham.generators]
            self._sigma_gen_exp = np.array(vals)
            self._sigma_gen_key = key
        return self._sigma_gen_exp

    def value(self, state) -> float:
        # D = Tr(sigma ln sigma) - Tr(sigma ln rho);  ln rho = -G(theta) - ln Z
        #   = const + Tr(sigma G(theta)) + ln Z,  Tr(sigma G) = sum_j theta_j <G_j>_sigma
        # ln Z comes first: a backend that cannot provide it says so before any work
        log_z = state.log_partition()
        tr_sigma_G = float(self._target_generator_expectations(state) @ state.theta)
        offset = getattr(state.ham, "offset", None)
        if offset is not None:
            tr_sigma_G += self._expect_dense(offset)
        return self._const + tr_sigma_G + log_z

    def grad(self, state) -> np.ndarray:
        return self._target_generator_expectations(state) - state.generator_expectations()


class MarginalRelativeEntropy(Loss):
    """Relative entropy ``D(rho || sigma_v(theta))`` to a quantum target with hidden units.

    ``sigma_v = Tr_h rho_vh`` is the visible reduced state of the QBM and ``rho`` is
    a target density matrix on the visible qubits.  Unlike the fully-visible case,
    ``ln sigma_v`` does not simplify, so the gradient

        d_j D = -Tr[rho * d_j ln sigma_v]

    is computed exactly via the Frechet derivative of ``log`` (the dense-exact
    counterpart of the modular-flow lift of arXiv:2512.19819).  For a model with no
    hidden units this reduces to :class:`RelativeEntropy`.
    """

    def __init__(self, rho: np.ndarray, n_visible: int):
        self.rho = np.asarray(rho, dtype=complex)
        self.n_visible = n_visible
        self._const = _entropy_term(self.rho)  # Tr(rho ln rho)

    def _reduced(self, state):
        nh = state.n_qubits - self.n_visible
        sv = partial_trace_hidden(state.density_matrix(), self.n_visible, nh)
        mu, U = np.linalg.eigh(sv)
        mu = np.clip(np.real(mu), 1e-300, None)
        return sv, mu, U, nh

    def value(self, state) -> float:
        _, mu, U, _ = self._reduced(state)
        ln_sv = (U * np.log(mu)) @ U.conj().T
        return float(self._const - np.real(np.trace(self.rho @ ln_sv)))

    def grad(self, state) -> np.ndarray:
        _, mu, U, nh = self._reduced(state)
        L = log_divided_differences(mu)  # (dv, dv) Frechet kernel for log
        rho_eig = U.conj().T @ self.rho @ U
        D = state.state_derivatives()  # (J, dim, dim) d_j rho_vh
        grad = np.empty(D.shape[0])
        for j in range(D.shape[0]):
            d_sv = partial_trace_hidden(D[j], self.n_visible, nh)
            d_ln = (U.conj().T @ d_sv @ U) * L  # d_j ln sigma_v in its eigenbasis
            grad[j] = -np.real(np.sum(rho_eig * d_ln.T))  # -Tr(rho * d_j ln sigma_v)
        return grad
