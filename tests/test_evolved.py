"""v0.4: Evolved QBM (omega = e^{-iH(phi)} rho(theta) e^{+iH(phi)})."""

import numpy as np

import qbm
from qbm.losses import Energy, MarginalRelativeEntropy
from qbm.operators import ParamHamiltonian, local_pauli_generators


def _model(n=2, seed=0, phi_scale=0.5):
    rng = np.random.default_rng(seed)
    G = local_pauli_generators(n)
    Hgen = [["I"] * n for _ in range(n)]
    for i in range(n):
        Hgen[i][i] = "Y"
    Hgen = ["".join(s) for s in Hgen]  # Y field on each qubit (does not commute with G)
    m = qbm.EvolvedQBM(G, Hgen)
    m.theta = np.concatenate(
        [
            rng.normal(scale=0.4, size=m.n_theta),
            rng.normal(scale=phi_scale, size=m.n_phi),
        ]
    )
    return m


def _fd(model, loss, eps=1e-6):
    base = model.theta.copy()
    g = np.zeros_like(base)
    for j in range(len(base)):
        model.theta = base.copy()
        model.theta[j] += eps
        vp = loss.value(model.state())
        model.theta = base.copy()
        model.theta[j] -= eps
        vm = loss.value(model.state())
        g[j] = (vp - vm) / (2 * eps)
    model.theta = base
    return g


def test_reduces_to_qbm_at_phi_zero():
    rng = np.random.default_rng(1)
    n = 3
    G = local_pauli_generators(n)
    theta = rng.normal(scale=0.5, size=len(G))
    m = qbm.EvolvedQBM(G, ["XII", "IXI", "IIX"], theta=theta, phi=np.zeros(3))
    rho = qbm.DenseBackend().thermal_state(ParamHamiltonian(G), theta).density_matrix()
    assert np.allclose(m.density_matrix(), rho, atol=1e-12)


def test_phi_changes_the_state():
    m = _model(n=2, phi_scale=0.7)
    omega = m.density_matrix()
    rho = m.state().rho
    assert not np.allclose(omega, rho, atol=1e-6)  # real-time evolution does something


def test_energy_gradient_finite_diff():
    m = _model(n=3, seed=2)
    O = qbm.hamiltonians.tfim(3, J=1.0, g=1.2)
    loss = Energy(O)
    analytic = loss.grad(m.state())
    assert analytic.shape == (m.n_params,)
    assert np.allclose(analytic, _fd(m, loss), atol=1e-6)


def test_relative_entropy_gradient_finite_diff():
    m = _model(n=2, seed=3)
    target = qbm.oracles.gibbs(qbm.hamiltonians.heisenberg(2), beta=0.6)
    loss = MarginalRelativeEntropy(target, n_visible=2)
    analytic = loss.grad(m.state())
    assert np.allclose(analytic, _fd(m, loss), atol=1e-6)


def test_metric_properties_and_reduction():
    # symmetric, PSD, Loewner orderings; and the theta-block at phi=0 equals the
    # plain-QBM metric.
    m = _model(n=2, seed=4)
    st = m.state()
    for kind in ("kubo_mori", "fisher_bures", "wigner_yanase"):
        g = st.metric(kind)
        assert np.allclose(g, g.T, atol=1e-9)
        assert np.min(np.linalg.eigvalsh(g)) > -1e-8
    fb, wy, km = st.metric("fisher_bures"), st.metric("wigner_yanase"), st.metric("kubo_mori")
    assert np.min(np.linalg.eigvalsh(wy - fb)) > -1e-8
    assert np.min(np.linalg.eigvalsh(2 * fb - wy)) > -1e-8
    assert np.min(np.linalg.eigvalsh(km - fb)) > -1e-8

    # phi = 0 reduction of the theta-theta block
    rng = np.random.default_rng(7)
    G = local_pauli_generators(2)
    theta = rng.normal(scale=0.4, size=len(G))
    m0 = qbm.EvolvedQBM(G, ["YI", "IY"], theta=theta, phi=np.zeros(2))
    inner = qbm.DenseBackend().thermal_state(ParamHamiltonian(G), theta)
    g_full = m0.state().metric("kubo_mori")
    assert np.allclose(g_full[: len(G), : len(G)], inner.metric("kubo_mori"), atol=1e-8)


def test_ground_state_energy_with_evolution():
    n = 3
    H = qbm.hamiltonians.tfim(n, J=1.0, g=1.2)
    e0 = qbm.oracles.ground_energy(H)
    G = local_pauli_generators(n)
    m = qbm.EvolvedQBM(G, ["XII", "IXI", "IIX"])
    m.theta = np.concatenate(
        [
            np.random.default_rng(0).normal(scale=0.05, size=m.n_theta),
            np.zeros(m.n_phi),
        ]
    )
    hist = qbm.fit(
        m, Energy(H), qbm.optim.NaturalGradient(metric="kubo_mori", lr=0.2, reg=1e-3), steps=400
    )
    assert m.energy(H) < hist.loss[0]
    assert (m.energy(H) - e0) < 0.15
