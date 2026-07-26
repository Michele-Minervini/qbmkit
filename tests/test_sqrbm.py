"""v0.3: semi-quantum RBM closed form and quantum-target hidden-unit relative entropy."""

import numpy as np

import qbm
from qbm.linalg import partial_trace_hidden
from qbm.losses import MarginalRelativeEntropy, RelativeEntropy, SqRBMNLL
from qbm.operators import ParamHamiltonian, local_pauli_generators


# ---------------------------------------------------------------------------
# sqRBM closed form
# ---------------------------------------------------------------------------
def test_sqrbm_closed_form_matches_exact_gibbs():
    # The closed-form visible marginal must equal the exact dense Gibbs marginal.
    rng = np.random.default_rng(0)
    model = qbm.SemiQuantumRBM(n_visible=3, n_hidden=2, hidden_paulis=("X", "Z"))
    model.theta = rng.normal(scale=0.5, size=model.n_params)

    pv_closed = model.visible_probabilities()

    ham = model.to_hamiltonian()
    state = qbm.DenseBackend().thermal_state(ham, model.theta)
    rho = state.density_matrix()
    sv = partial_trace_hidden(rho, model.n_visible, model.n_hidden)
    pv_exact = np.real(np.diag(sv))

    assert np.allclose(pv_closed, pv_exact, atol=1e-10)


def test_sqrbm_nll_gradient_finite_diff():
    rng = np.random.default_rng(1)
    model = qbm.SemiQuantumRBM(n_visible=3, n_hidden=2)
    theta = rng.normal(scale=0.4, size=model.n_params)
    model.theta = theta.copy()
    q = rng.random(8)
    q /= q.sum()
    loss = SqRBMNLL(q)

    analytic = loss.grad(model.state())

    eps = 1e-6
    fd = np.zeros_like(theta)
    for j in range(len(theta)):
        for sign in (+1, -1):
            t = theta.copy()
            t[j] += sign * eps
            m = qbm.SemiQuantumRBM(n_visible=3, n_hidden=2, theta=t)
            fd[j] += sign * loss.value(m.state())
        fd[j] /= 2 * eps
    assert np.allclose(analytic, fd, atol=1e-6)


def test_sqrbm_trains_distribution():
    q = qbm.datasets.bars_and_stripes(grid=2)  # 4 visible qubits
    model = qbm.SemiQuantumRBM(n_visible=4, n_hidden=4, hidden_paulis=("X", "Z"))
    model.theta = np.random.default_rng(0).normal(scale=0.1, size=model.n_params)
    hist = qbm.fit(model, SqRBMNLL(q), qbm.optim.Adam(lr=0.1), steps=400)
    assert hist.final_loss < hist.loss[0]
    assert model.kl(q) < 0.3


# ---------------------------------------------------------------------------
# Quantum-target relative entropy with hidden units
# ---------------------------------------------------------------------------
def _fd_grad(loss, ham, theta, eps=1e-6):
    backend = qbm.DenseBackend()
    g = np.zeros_like(theta)
    for j in range(len(theta)):
        tp = theta.copy()
        tp[j] += eps
        tm = theta.copy()
        tm[j] -= eps
        g[j] = (
            loss.value(backend.thermal_state(ham, tp)) - loss.value(backend.thermal_state(ham, tm))
        ) / (2 * eps)
    return g


def test_marginal_relative_entropy_gradient_finite_diff():
    rng = np.random.default_rng(2)
    model = qbm.VisibleHiddenQBM(n_visible=2, n_hidden=2)
    theta = rng.normal(scale=0.4, size=model.n_params)
    model.theta = theta
    # a random 2-qubit target state on the visible qubits
    target = qbm.oracles.gibbs(qbm.hamiltonians.heisenberg(2), beta=0.7)
    loss = MarginalRelativeEntropy(target, n_visible=2)
    analytic = loss.grad(model.state())
    assert np.allclose(analytic, _fd_grad(loss, model.ham, theta), atol=1e-6)


def test_marginal_relative_entropy_reduces_to_relative_entropy_fully_visible():
    # With no hidden units, MarginalRelativeEntropy must match RelativeEntropy.
    rng = np.random.default_rng(3)
    n = 3
    ham = ParamHamiltonian(local_pauli_generators(n))
    theta = rng.normal(scale=0.4, size=ham.n_params)
    sigma = qbm.oracles.gibbs(qbm.hamiltonians.tfim(n, g=1.1), beta=0.9)
    state = qbm.DenseBackend().thermal_state(ham, theta)
    g_marg = MarginalRelativeEntropy(sigma, n_visible=n).grad(state)
    g_full = RelativeEntropy(sigma).grad(state)
    assert np.allclose(g_marg, g_full, atol=1e-7)


def test_quantum_state_learning_with_hidden_units():
    # Learn a visible-qubit quantum target with extra hidden units. Use a target in
    # the model's visible-marginal class (realizable) so the optimum is D = 0.
    n_vis, n_hid = 2, 2
    src = qbm.VisibleHiddenQBM(n_visible=n_vis, n_hidden=n_hid)
    src.theta = np.random.default_rng(5).normal(scale=0.5, size=src.n_params)
    target = partial_trace_hidden(src.state().density_matrix(), n_vis, n_hid)

    model = qbm.VisibleHiddenQBM(n_visible=n_vis, n_hidden=n_hid)
    model.theta = np.random.default_rng(0).normal(scale=0.1, size=model.n_params)
    loss = MarginalRelativeEntropy(target, n_visible=n_vis)
    hist = qbm.fit(model, loss, qbm.optim.Adam(lr=0.1), steps=400)
    assert hist.final_loss < hist.loss[0]
    assert hist.final_loss < 1e-6
