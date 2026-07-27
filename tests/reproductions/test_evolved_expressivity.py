"""Reproduction -- arXiv:2501.03367, "Evolved Quantum Boltzmann Machines".

The EQBM wraps a Gibbs state in real-time evolution, omega = e^{-iH(phi)} rho e^{+iH(phi)},
making it strictly more expressive than the plain QBM.  We reproduce this: an
evolved target that is *not* a Gibbs state of the chosen generators cannot be
represented by a plain QBM (its relative entropy floors well above zero), while the
EQBM drives it to zero.
"""

import numpy as np

import qbm
from qbm.losses import MarginalRelativeEntropy

_G = qbm.local_pauli_generators(3)
_HGEN = ["XII", "IXI", "IIX"]


def _evolved_target(seed=3):
    src = qbm.EvolvedQBM(_G, _HGEN)
    rng = np.random.default_rng(seed)
    src.theta = np.concatenate(
        [rng.normal(scale=0.5, size=src.n_theta), rng.normal(scale=0.6, size=src.n_phi)]
    )
    return src.density_matrix()


def test_plain_qbm_cannot_represent_evolved_target():
    target = _evolved_target()
    rng = np.random.default_rng(3)
    model = qbm.FullyVisibleQBM(generators=_G)
    model.theta = rng.normal(scale=0.1, size=model.n_params)
    hist = qbm.fit(
        model, MarginalRelativeEntropy(target, n_visible=3), qbm.optim.Adam(lr=0.1), steps=400
    )
    assert hist.final_loss > 0.05  # calibrated ~0.22: hits a floor


def test_evolved_qbm_represents_evolved_target():
    target = _evolved_target()
    rng = np.random.default_rng(4)
    model = qbm.EvolvedQBM(_G, _HGEN)
    model.theta = np.concatenate(
        [rng.normal(scale=0.1, size=model.n_theta), rng.normal(scale=0.1, size=model.n_phi)]
    )
    hist = qbm.fit(
        model, MarginalRelativeEntropy(target, n_visible=3), qbm.optim.Adam(lr=0.1), steps=400
    )
    assert hist.final_loss < 1e-3  # calibrated ~1e-15: reaches the target
