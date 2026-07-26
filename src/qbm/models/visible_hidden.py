"""Visible + hidden QBM (restricted / semi-quantum Boltzmann machine).

The joint Gibbs state ``rho_vh = e^{-G(theta)}/Z`` lives on visible + hidden
qubits; the model distribution is the visible marginal ``p(v) = Tr_h rho_vh``
read in the computational basis.  Visible qubits are the leading tensor factors,
hidden qubits the trailing ones.

The marginal-likelihood gradient is exact (no Golden-Thompson bound needed): we
differentiate the joint Gibbs state and project, ``d_j p(v) = Tr[(|v><v| (x) I) d_j rho_vh]``,
which reuses the same eigenbasis state-derivative kernel as the fully-visible model.
"""

from __future__ import annotations

import numpy as np

from ..operators import ParamHamiltonian, rbm_generators
from .base import Model


class VisibleHiddenQBM(Model):
    """A restricted / semi-quantum QBM with ``n_visible`` visible and ``n_hidden`` hidden units."""

    def __init__(
        self,
        n_visible,
        n_hidden,
        hidden_paulis=("Z", "X"),
        generators=None,
        ham=None,
        theta=None,
        backend=None,
    ):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        if ham is None:
            if generators is None:
                generators = rbm_generators(n_visible, n_hidden, hidden_paulis=hidden_paulis)
            ham = ParamHamiltonian(generators, n_qubits=n_visible + n_hidden)
        super().__init__(ham, theta=theta, backend=backend)

    def visible_probabilities(self) -> np.ndarray:
        """Marginal distribution ``p(v)`` over the ``2^n_visible`` visible outcomes."""
        p = self.state().probabilities()
        return p.reshape(1 << self.n_visible, 1 << self.n_hidden).sum(axis=1)

    # the visible marginal is what users see/sample/compare against data
    def probabilities(self) -> np.ndarray:
        return self.visible_probabilities()

    def sample(self, n: int, rng=None) -> np.ndarray:
        rng = np.random.default_rng() if rng is None else rng
        pv = self.visible_probabilities()
        pv = np.clip(pv, 0.0, None)
        pv = pv / pv.sum()
        return rng.choice(1 << self.n_visible, size=n, p=pv)
