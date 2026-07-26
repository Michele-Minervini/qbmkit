"""Analytic gradients vs central finite differences (the core correctness check)."""

import numpy as np
import pytest

import qbm
from qbm.losses import NLL, Energy, RelativeEntropy
from qbm.operators import ParamHamiltonian, local_pauli_generators


def _finite_diff(loss, ham, theta, eps=1e-6):
    backend = qbm.DenseBackend()
    g = np.zeros_like(theta)
    for j in range(len(theta)):
        tp = theta.copy()
        tp[j] += eps
        tm = theta.copy()
        tm[j] -= eps
        vp = loss.value(backend.thermal_state(ham, tp))
        vm = loss.value(backend.thermal_state(ham, tm))
        g[j] = (vp - vm) / (2 * eps)
    return g


def _analytic(loss, ham, theta):
    state = qbm.DenseBackend().thermal_state(ham, theta)
    return loss.grad(state)


@pytest.fixture
def setup():
    rng = np.random.default_rng(7)
    n = 3
    ham = ParamHamiltonian(local_pauli_generators(n))
    theta = rng.normal(scale=0.4, size=ham.n_params)
    return rng, n, ham, theta


def test_energy_gradient(setup):
    rng, n, ham, theta = setup
    O = qbm.hamiltonians.tfim(n, J=1.0, g=1.4)
    loss = Energy(O)
    assert np.allclose(_analytic(loss, ham, theta), _finite_diff(loss, ham, theta), atol=1e-6)


def test_relative_entropy_gradient(setup):
    rng, n, ham, theta = setup
    sigma = qbm.oracles.gibbs(qbm.hamiltonians.heisenberg(n), beta=0.8)
    loss = RelativeEntropy(sigma)
    assert np.allclose(_analytic(loss, ham, theta), _finite_diff(loss, ham, theta), atol=1e-6)


def test_nll_gradient(setup):
    rng, n, ham, theta = setup
    q = rng.random(2**n)
    q /= q.sum()
    loss = NLL(q)
    assert np.allclose(_analytic(loss, ham, theta), _finite_diff(loss, ham, theta), atol=1e-6)


def test_relative_entropy_nonnegative_and_zero_at_self():
    # D(sigma||sigma) = 0 and D >= 0
    rng = np.random.default_rng(11)
    ham = ParamHamiltonian(local_pauli_generators(3))
    theta = rng.normal(scale=0.5, size=ham.n_params)
    state = qbm.DenseBackend().thermal_state(ham, theta)
    sigma = state.density_matrix()
    loss = RelativeEntropy(sigma)
    assert abs(loss.value(state)) < 1e-9
    other = qbm.DenseBackend().thermal_state(ham, np.zeros(ham.n_params))
    assert loss.value(other) > -1e-9
