"""Milestone v0.2/v0.3: visible+hidden marginal likelihood, free energy, diagnostics."""

import numpy as np

import qbm
from qbm.losses import NLL, FreeEnergy, MarginalNLL
from qbm.operators import ParamHamiltonian, local_pauli_generators


def _fd_grad(loss, ham, theta, eps=1e-6):
    backend = qbm.DenseBackend()
    g = np.zeros_like(theta)
    for j in range(len(theta)):
        tp = theta.copy(); tp[j] += eps
        tm = theta.copy(); tm[j] -= eps
        g[j] = (loss.value(backend.thermal_state(ham, tp))
                - loss.value(backend.thermal_state(ham, tm))) / (2 * eps)
    return g


def test_marginal_nll_gradient_finite_diff():
    rng = np.random.default_rng(0)
    model = qbm.VisibleHiddenQBM(n_visible=2, n_hidden=2)
    theta = rng.normal(scale=0.4, size=model.n_params)
    model.theta = theta
    q = rng.random(4); q /= q.sum()
    loss = MarginalNLL(q, n_visible=2)
    analytic = loss.grad(model.state())
    assert np.allclose(analytic, _fd_grad(loss, model.ham, theta), atol=1e-6)


def test_marginal_nll_equals_relative_entropy_for_commuting_model():
    # For a diagonal (commuting) model rho is diagonal, so the measured-distribution
    # NLL and the relative-entropy objective D(diag(q)||rho) have the same gradient.
    # (They differ for non-commuting models -- that is the point of having both.)
    from qbm.losses import RelativeEntropy

    rng = np.random.default_rng(1)
    ham = ParamHamiltonian(["ZII", "IZI", "IIZ", "ZZI", "IZZ"])  # diagonal
    theta = rng.normal(scale=0.4, size=ham.n_params)
    q = rng.random(8); q /= q.sum()
    state = qbm.DenseBackend().thermal_state(ham, theta)
    g_marg = MarginalNLL(q, n_visible=3).grad(state)
    g_relent = RelativeEntropy(np.diag(q.astype(complex))).grad(state)
    assert np.allclose(g_marg, g_relent, atol=1e-8)


def test_free_energy_gradient_finite_diff():
    rng = np.random.default_rng(2)
    n = 3
    ham = ParamHamiltonian(local_pauli_generators(n))
    theta = rng.normal(scale=0.4, size=ham.n_params)
    H = qbm.hamiltonians.heisenberg(n)
    loss = FreeEnergy(H, temperature=0.8)
    state = qbm.DenseBackend().thermal_state(ham, theta)
    assert np.allclose(loss.grad(state), _fd_grad(loss, ham, theta), atol=1e-6)


def test_free_energy_reaches_exact_minimum():
    # H is in the model's generator span -> variational F reaches the exact value.
    n = 3
    T = 1.0
    H = qbm.hamiltonians.tfim(n, J=1.0, g=1.0)
    model = qbm.FullyVisibleQBM(n=n)
    hist = qbm.fit(
        model,
        FreeEnergy(H, temperature=T),
        qbm.optim.NaturalGradient(metric="kubo_mori", lr=0.3, reg=1e-4),
        steps=400,
    )
    f_exact = qbm.oracles.free_energy(H, beta=1.0 / T)
    assert abs(hist.final_loss - f_exact) < 1e-2


def test_hidden_units_improve_generative_fit():
    # A QBM with hidden units fits a classical target via marginal NLL.
    # Random init breaks the symmetry saddle at theta=0 (flip-symmetric targets).
    q = qbm.datasets.bars_and_stripes(grid=2)  # 4 visible qubits
    model = qbm.VisibleHiddenQBM(n_visible=4, n_hidden=2)
    model.theta = np.random.default_rng(0).normal(scale=0.1, size=model.n_params)
    loss = MarginalNLL(q, n_visible=4)
    hist = qbm.fit(model, loss, qbm.optim.Adam(lr=0.1), steps=400)
    assert hist.final_loss < hist.loss[0]
    assert model.kl(q) < 1.0


def test_barren_plateau_scan_runs():
    def make_model_for_n(n):
        return lambda: qbm.FullyVisibleQBM(n=n)

    def make_loss_for_n(n):
        return qbm.losses.Energy(qbm.hamiltonians.tfim(n, g=1.0))

    var = qbm.diagnostics.barren_plateau_scan(
        make_model_for_n, make_loss_for_n, sizes=[2, 3], n_samples=20, scale=1.0
    )
    assert set(var) == {2, 3}
    assert all(v > 0 for v in var.values())
