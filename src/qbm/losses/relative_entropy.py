"""Quantum relative entropy ``D(sigma || rho(theta))``.

For a fixed target ``sigma`` (a quantum state, or ``diag(q)`` for a classical
distribution) the gradient is the classic positive/negative phase difference::

    d_j D(sigma || rho) = <G_j>_sigma - <G_j>_rho

which is exact for fully-visible QBMs even with non-commuting generators, and is
cheap (only generator expectations).  Note: for non-commuting generators
``D(diag(q) || rho)`` is *not* the measured-distribution likelihood (that is
:class:`qbm.losses.NLL`); it is the relative-entropy objective, equal to the NLL
only in the commuting/diagonal case and an upper bound otherwise.
"""

from __future__ import annotations

import numpy as np

from ..linalg import log_divided_differences, partial_trace_hidden
from .base import Loss


def _entropy_term(sigma: np.ndarray) -> float:
    """``Tr(sigma ln sigma)`` (the theta-independent part of the relative entropy)."""
    vals = np.linalg.eigvalsh(sigma)
    vals = np.clip(np.real(vals), 0.0, None)
    nz = vals[vals > 1e-15]
    return float(np.sum(nz * np.log(nz)))


class RelativeEntropy(Loss):
    """``L(theta) = D(sigma || rho(theta))`` for a fixed target density matrix."""

    def __init__(self, sigma: np.ndarray):
        self.sigma = np.asarray(sigma, dtype=complex)
        self._const = _entropy_term(self.sigma)
        self._sigma_gen_exp = None  # cache of [Tr(sigma G_j)] keyed on ham id
        self._sigma_gen_key = None

    def _target_generator_expectations(self, state) -> np.ndarray:
        key = id(state.ham)
        if self._sigma_gen_key != key:
            self._sigma_gen_exp = np.array(
                [float(np.real(np.trace(self.sigma @ g))) for g in state.ham.generators]
            )
            self._sigma_gen_key = key
        return self._sigma_gen_exp

    def value(self, state) -> float:
        # D = Tr(sigma ln sigma) - Tr(sigma ln rho);  ln rho = -G(theta) - ln Z
        #   = const + Tr(sigma G(theta)) + ln Z
        G = state.ham.matrix(state.theta)
        tr_sigma_G = float(np.real(np.trace(self.sigma @ G)))
        return self._const + tr_sigma_G + state.log_partition()

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
        L = log_divided_differences(mu)            # (dv, dv) Frechet kernel for log
        rho_eig = U.conj().T @ self.rho @ U
        D = state.state_derivatives()              # (J, dim, dim) d_j rho_vh
        grad = np.empty(D.shape[0])
        for j in range(D.shape[0]):
            d_sv = partial_trace_hidden(D[j], self.n_visible, nh)
            d_ln = (U.conj().T @ d_sv @ U) * L     # d_j ln sigma_v in its eigenbasis
            grad[j] = -np.real(np.sum(rho_eig * d_ln.T))  # -Tr(rho * d_j ln sigma_v)
        return grad
