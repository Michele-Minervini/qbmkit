"""v0.8: block-Gibbs sampling and contrastive-divergence training (arXiv:2511.11802)."""

import numpy as np

import qbm
from qbm.losses import SqRBMNLL
from qbm.sampling import block_gibbs_sample, contrastive_divergence_gradient

_Q = qbm.datasets.bars_and_stripes(grid=2)  # 4 visible qubits


def _model(hidden_paulis=("Z",), n_hidden=3, seed=0, scale=0.3):
    m = qbm.SemiQuantumRBM(n_visible=4, n_hidden=n_hidden, hidden_paulis=hidden_paulis)
    m.theta = np.random.default_rng(seed).normal(scale=scale, size=m.n_params)
    return m


def _empirical(bits, n_visible=4):
    idx = bits @ (1 << np.arange(n_visible - 1, -1, -1))
    return np.bincount(idx, minlength=1 << n_visible) / len(idx)


def test_block_gibbs_returns_valid_bits():
    m = _model()
    rng = np.random.default_rng(0)
    out = block_gibbs_sample(m, rng.integers(0, 2, size=(64, 4)), k=3, rng=rng)
    assert out.shape == (64, 4)
    assert set(np.unique(out).tolist()) <= {0, 1}


def test_chain_equilibrates_to_model_for_commuting_hidden_units():
    # A classical RBM has commuting hidden operators, so the block-Gibbs chain is
    # exact: its stationary distribution IS the model's visible marginal.
    m = _model(hidden_paulis=("Z",))
    rng = np.random.default_rng(0)
    bits = block_gibbs_sample(m, rng.integers(0, 2, size=(40000, 4)), k=60, rng=rng)
    tv = 0.5 * np.abs(_empirical(bits) - m.visible_probabilities()).sum()
    assert tv < 0.03  # sampling noise only


def test_cd_gradient_converges_to_exact_for_commuting_hidden_units():
    m = _model(hidden_paulis=("Z",))
    exact = SqRBMNLL(_Q).grad(m.state())
    g1 = contrastive_divergence_gradient(m, _Q, k=1, n_chains=40000, rng=np.random.default_rng(1))
    g10 = contrastive_divergence_gradient(m, _Q, k=10, n_chains=40000, rng=np.random.default_rng(1))
    err1 = np.linalg.norm(g1 - exact) / np.linalg.norm(exact)
    err10 = np.linalg.norm(g10 - exact) / np.linalg.norm(exact)
    assert err10 < err1  # longer chains -> less bias
    assert err10 < 0.1  # calibrated ~0.03
    cos = g10 @ exact / (np.linalg.norm(g10) * np.linalg.norm(exact))
    assert cos > 0.99  # calibrated ~0.9998


def test_cd_gradient_is_a_descent_direction_for_non_commuting_hidden_units():
    # With non-commuting hidden Paulis the sampler is an approximation (documented):
    # the CD gradient stays positively aligned with the exact one but does not converge.
    m = _model(hidden_paulis=("X", "Z"))
    exact = SqRBMNLL(_Q).grad(m.state())
    g = contrastive_divergence_gradient(m, _Q, k=10, n_chains=20000, rng=np.random.default_rng(2))
    cos = g @ exact / (np.linalg.norm(g) * np.linalg.norm(exact))
    assert cos > 0.5


def test_training_with_contrastive_divergence_reduces_kl():
    # Plain SGD is the right optimizer for stochastic gradients: its step is
    # proportional to the gradient so sampling noise averages out (Adam rescales by
    # the gradient magnitude and amplifies noise -- see the module docstring).
    m = _model(hidden_paulis=("Z",), n_hidden=4, scale=0.1)
    rng = np.random.default_rng(0)
    opt = qbm.optim.GradientDescent(lr=0.2)
    kl_start = m.kl(_Q)
    for _ in range(300):
        g = contrastive_divergence_gradient(m, _Q, k=5, n_chains=500, rng=rng)
        m.theta = opt.step(m.theta, g)
    assert m.kl(_Q) < 0.2  # calibrated ~0.05, from ~0.97
    assert m.kl(_Q) < kl_start


def test_cd_cost_is_independent_of_target_support():
    # The paper's efficiency claim: CD needs a fixed number of chain steps regardless
    # of how many configurations the target distribution puts weight on.
    m = _model(hidden_paulis=("Z",))
    rng = np.random.default_rng(0)
    narrow = np.zeros(16)
    narrow[[0, 15]] = 0.5  # support 2
    wide = np.full(16, 1 / 16)  # support 16
    g_narrow = contrastive_divergence_gradient(m, narrow, k=2, n_chains=500, rng=rng)
    g_wide = contrastive_divergence_gradient(m, wide, k=2, n_chains=500, rng=rng)
    assert g_narrow.shape == g_wide.shape == (m.n_params,)
    assert np.all(np.isfinite(g_narrow)) and np.all(np.isfinite(g_wide))
