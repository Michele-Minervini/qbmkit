"""End-to-end task tests: the v0.1 thin slice across three QBM use cases."""

import numpy as np

import qbm
from qbm.losses import Energy, RelativeEntropy
from qbm.operators import ParamHamiltonian, local_pauli_generators


def test_classical_bm_reproduces_boltzmann_distribution():
    # A fully-visible QBM with only diagonal (Z, ZZ) generators is a classical BM:
    # its marginal must be the Boltzmann distribution exp(-E_v)/Z, E_v = <v|G|v>.
    gens = ["ZII", "IZI", "IIZ", "ZZI", "IZZ"]
    ham = ParamHamiltonian(gens)
    rng = np.random.default_rng(0)
    theta = rng.normal(scale=0.5, size=ham.n_params)
    G = ham.matrix(theta)
    E = np.real(np.diag(G))
    boltz = np.exp(-(E - E.min()))
    boltz /= boltz.sum()
    p = qbm.DenseBackend().thermal_state(ham, theta).probabilities()
    assert np.allclose(p, boltz, atol=1e-10)


def test_state_learning_recovers_realizable_target():
    # Target is a Gibbs state in the model's generator span -> loss must reach ~0.
    n = 3
    ham = ParamHamiltonian(local_pauli_generators(n))
    rng = np.random.default_rng(1)
    theta_true = rng.normal(scale=0.4, size=ham.n_params)
    sigma = qbm.DenseBackend().thermal_state(ham, theta_true).density_matrix()

    model = qbm.FullyVisibleQBM(n=n)
    hist = qbm.fit(
        model,
        RelativeEntropy(sigma),
        qbm.optim.NaturalGradient(metric="kubo_mori", lr=0.4, reg=1e-6),
        steps=300,
    )
    assert hist.final_loss < 1e-3
    assert hist.loss[0] > hist.final_loss


def test_ground_state_energy_estimation():
    n = 3
    H = qbm.hamiltonians.tfim(n, J=1.0, g=1.2)
    e0 = qbm.oracles.ground_energy(H)
    model = qbm.FullyVisibleQBM(n=n)
    hist = qbm.fit(
        model,
        Energy(H),
        qbm.optim.NaturalGradient(metric="kubo_mori", lr=0.2, reg=1e-3),
        steps=500,
    )
    e_est = model.energy(H)
    assert e_est < hist.loss[0]  # improved over the start
    assert (e_est - e0) < 0.1  # close to the true ground energy


def test_generative_learns_realizable_distribution():
    # A target that the (classical/diagonal) model can represent exactly: the
    # convex NLL must drive KL to ~0.
    n = 4
    ham = ParamHamiltonian(local_pauli_generators(n, fields=("Z",), couplings=("ZZ",)))
    rng = np.random.default_rng(0)
    theta_true = rng.normal(scale=0.5, size=ham.n_params)
    q = qbm.DenseBackend().thermal_state(ham, theta_true).probabilities()

    model = qbm.FullyVisibleQBM(n=n, fields=("Z",), couplings=("ZZ",))
    qbm.fit(
        model,
        qbm.losses.NLL(q),
        qbm.optim.NaturalGradient(metric="kubo_mori", lr=0.5, reg=1e-8),
        steps=300,
    )
    assert model.kl(q) < 1e-6


def test_generative_bars_and_stripes():
    # BAS needs richer-than-nearest-neighbour correlations; the all-to-all
    # default of qbm.learn captures it well.
    q = qbm.datasets.bars_and_stripes(grid=2)  # n = 4
    kl_init = qbm.FullyVisibleQBM(n=4, connectivity="all").kl(q)
    model = qbm.learn(q, steps=600, lr=0.1)
    assert model.kl(q) < 0.05
    assert model.kl(q) < kl_init


def test_learn_facade_returns_trained_model():
    q = qbm.datasets.parity(n=3)
    model = qbm.learn(q, steps=200)
    assert isinstance(model, qbm.FullyVisibleQBM)
    assert hasattr(model, "history")
    assert len(model.history) > 0
