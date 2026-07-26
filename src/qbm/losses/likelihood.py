"""Marginal negative log-likelihood for visible/hidden QBMs.

Trains the visible marginal ``p(v) = Tr_h rho_vh`` to match a classical target
``q``.  The gradient is exact via ``d_j p(v) = Tr[(|v><v| (x) I) d_j rho_vh]``,
assembled from the backend's :meth:`diagonal_gradient` primitive.  For a model
with no hidden units this reduces to the ordinary NLL.
"""

from __future__ import annotations

import numpy as np

from .base import Loss


class MarginalNLL(Loss):
    """``L = -sum_v q(v) log p(v)`` over the visible marginal of a QBM."""

    def __init__(self, q: np.ndarray, n_visible: int):
        self.q = np.asarray(q, dtype=float)
        self.n_visible = n_visible

    def _visible_marginal(self, state) -> np.ndarray:
        nh = state.n_qubits - self.n_visible
        p = state.probabilities().reshape(1 << self.n_visible, 1 << nh)
        return p.sum(axis=1)

    def value(self, state) -> float:
        pv = self._visible_marginal(state)
        mask = self.q > 0
        return float(-np.sum(self.q[mask] * np.log(np.clip(pv[mask], 1e-300, None))))

    def grad(self, state) -> np.ndarray:
        nh = state.n_qubits - self.n_visible
        dg = state.diagonal_gradient()  # (J, dim)
        dpv = dg.reshape(dg.shape[0], 1 << self.n_visible, 1 << nh).sum(axis=2)  # (J, 2^nv)
        pv = self._visible_marginal(state)
        coef = np.where(self.q > 0, self.q / np.clip(pv, 1e-300, None), 0.0)
        return -(dpv @ coef)


class SqRBMNLL(Loss):
    """Negative log-likelihood for a :class:`~qbm.SemiQuantumRBM` (closed form).

    Consumes the model's :class:`SqRBMState` (visible marginal + log-derivative
    table); no Gibbs-state diagonalisation, so training scales with the number of
    visible configurations only, not with the hidden count.
    """

    def __init__(self, q: np.ndarray):
        self.q = np.asarray(q, dtype=float)

    def value(self, state) -> float:
        pv = state.pv
        mask = self.q > 0
        return float(-np.sum(self.q[mask] * np.log(np.clip(pv[mask], 1e-300, None))))

    def grad(self, state) -> np.ndarray:
        # d NLL / d theta = sum_v (p_v - q_v) d log p~(v)/d theta
        return state.dlog @ (state.pv - self.q)


def NLL(q: np.ndarray, n_visible: int | None = None) -> MarginalNLL:
    """True negative log-likelihood of a classical distribution ``q``.

    Matches the *measured* (computational-basis) distribution ``p(v) = <v|rho_v|v>``
    to ``q``.  For a fully-visible model omit ``n_visible`` (inferred from ``q``).
    For non-commuting models this differs from ``RelativeEntropy(diag(q))`` (see
    that class); use ``RelativeEntropy`` when you want the cheaper relative-entropy
    objective instead.
    """
    q = np.asarray(q, dtype=float)
    if n_visible is None:
        n_visible = int(round(np.log2(len(q))))
    return MarginalNLL(q, n_visible=n_visible)
