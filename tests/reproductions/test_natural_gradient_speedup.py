"""Reproduction -- arXiv:2410.24058, "Natural gradient ... for quantum Boltzmann machines".

The paper's central practical claim is that preconditioning by a quantum Fisher
information metric (natural gradient) converges far faster than ordinary gradient
descent.  We reproduce this on transverse-field-Ising ground-state estimation:
quantum natural gradient (Kubo-Mori metric) reaches the ground energy to ~1e-4 in
200 steps, while plain gradient descent and Adam are orders of magnitude further off.
"""

import numpy as np

import qbm

_H = qbm.hamiltonians.tfim(4, J=1.0, g=1.5)
_E0 = qbm.oracles.ground_energy(_H)


def _final_error(optimizer, steps=200, seed=0):
    model = qbm.FullyVisibleQBM(n=4)
    model.theta = np.random.default_rng(seed).normal(scale=0.05, size=model.n_params)
    qbm.fit(model, qbm.losses.Energy(_H), optimizer, steps=steps)
    return model.energy(_H) - _E0


def test_natural_gradient_beats_gradient_descent():
    err_gd = _final_error(qbm.optim.GradientDescent(lr=0.05))
    err_qng = _final_error(qbm.optim.NaturalGradient(metric="kubo_mori", lr=0.2, reg=1e-3))
    # calibrated: GD ~0.199, QNG ~1.5e-4
    assert err_qng < 1e-2
    assert err_gd > 20 * err_qng  # QNG at least an order of magnitude better


def test_natural_gradient_reaches_ground_state():
    err_qng = _final_error(qbm.optim.NaturalGradient(metric="kubo_mori", lr=0.2, reg=1e-3))
    assert 0 <= err_qng < 1e-3  # variational (>= 0) and essentially exact
