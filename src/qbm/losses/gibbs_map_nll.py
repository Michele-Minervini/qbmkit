"""Negative log-likelihood of a visible/hidden QBM via the exact Gibbs map.

:class:`~qbm.losses.MarginalNLL` is exact but needs ``d_j rho`` (the backend's
``diagonal_gradient``), which only the dense and JAX engines provide -- so hidden-unit
models were capped at the dense ceiling.  This loss computes the *same* gradient from

    d_j L  =  sum_v q(v) <G_j>_{sigma_v}  -  <G_j>_rho ,

where the positive phase comes from the exact conditional hidden Gibbs states
(:class:`~qbm.gibbs_map.GibbsMap`, cost independent of the number of visible units) and
the negative phase is ``generator_expectations()`` -- which **every** backend supplies.
So hidden-unit training runs on the tensor-network, circuit and Pauli-propagation
backends too, and it is exact for non-commuting hidden operators (unlike contrastive
divergence, see :mod:`qbm.sampling`).
"""

from __future__ import annotations

import numpy as np

from ..gibbs_map import GibbsMap, _as_visible_distribution
from .base import Loss


class GibbsMapNLL(Loss):
    """``L = -sum_v q(v) log p(v)`` for a visible/hidden QBM, via exact hidden marginalisation.

    Parameters
    ----------
    data : array
        Probability vector over ``2^n_visible`` outcomes, or a 1-D integer array of
        visible basis-state samples (only distinct configurations are visited).
    n_visible : int
        Number of leading qubits treated as visible.

    Notes
    -----
    Requires the visible register to be diagonal (``I``/``Z`` generators), i.e. the
    RBM / semi-quantum-RBM structure.  The **gradient** works on any backend; the
    **value** additionally needs ``log Z``, obtained by enumerating the visible register,
    so it is available only for modest ``n_visible`` (a ``NotImplementedError`` is raised
    otherwise, which :func:`qbm.fit` records as ``nan`` while training continues on the
    gradient).
    """

    def __init__(self, data, n_visible: int):
        self.n_visible = n_visible
        self._data = data
        self._map = None
        self._configs = None
        self._weights = None
        self._pos = None
        self._pos_key = None

    def _prepare(self, state) -> GibbsMap:
        if self._map is None or self._map.ham is not state.ham:
            self._map = GibbsMap(state.ham, self.n_visible)
            self._configs, self._weights = _as_visible_distribution(self._data, self.n_visible)
            self._pos_key = None
        return self._map

    def _positive_phase(self, state) -> np.ndarray:
        """``sum_v q(v) <G_j>_{sigma_v}``, cached per parameter vector."""
        gmap = self._prepare(state)
        key = state.theta.tobytes()
        if self._pos_key != key:
            _, exps = gmap.conditional(state.theta, self._configs)
            self._pos = self._weights @ exps
            self._pos_key = key
        return self._pos

    def value(self, state) -> float:
        gmap = self._prepare(state)
        log_Zv = gmap.log_unnormalised_marginal(state.theta, self._configs)
        return float(-(self._weights @ log_Zv) + gmap.log_partition(state.theta))

    def grad(self, state) -> np.ndarray:
        return self._positive_phase(state) - np.asarray(state.generator_expectations())

    def marginal(self, state) -> np.ndarray:
        """Exact model marginal ``p(v)`` (handy as a training monitor)."""
        return self._prepare(state).marginal(state.theta)


__all__ = ["GibbsMapNLL"]
