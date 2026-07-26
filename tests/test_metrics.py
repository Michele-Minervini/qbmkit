"""Quantum-Fisher-information metric properties (Loewner orderings, limits)."""

import numpy as np

import qbm
from qbm.operators import ParamHamiltonian, local_pauli_generators


def _state(generators, scale=0.5, seed=0, n=3):
    rng = np.random.default_rng(seed)
    ham = ParamHamiltonian(generators)
    theta = rng.normal(scale=scale, size=ham.n_params)
    return qbm.DenseBackend().thermal_state(ham, theta)


def _psd(M, tol=1e-9):
    return np.min(np.linalg.eigvalsh(M)) >= -tol


def test_metrics_symmetric_and_psd():
    state = _state(local_pauli_generators(3), seed=1)
    for kind in ("kubo_mori", "fisher_bures", "wigner_yanase"):
        g = state.metric(kind)
        assert np.allclose(g, g.T, atol=1e-9)
        assert _psd(g)


def test_loewner_orderings():
    # g_FB <= g_WY <= 2 g_FB   and   g_KM >= g_FB
    state = _state(local_pauli_generators(3), seed=2, scale=0.6)
    fb = state.metric("fisher_bures")
    wy = state.metric("wigner_yanase")
    km = state.metric("kubo_mori")
    assert _psd(wy - fb)
    assert _psd(2 * fb - wy)
    assert _psd(km - fb)


def test_classical_limit_metrics_coincide():
    # Diagonal (commuting) generators -> all monotone metrics equal the classical
    # Fisher information.
    diag_gens = ["ZII", "IZI", "IIZ", "ZZI", "IZZ"]
    state = _state(diag_gens, seed=3)
    km = state.metric("kubo_mori")
    fb = state.metric("fisher_bures")
    wy = state.metric("wigner_yanase")
    assert np.allclose(km, fb, atol=1e-8)
    assert np.allclose(km, wy, atol=1e-8)


def test_kubo_mori_is_free_energy_hessian():
    # The Kubo-Mori metric equals the Hessian of log Z(theta) (= covariance of
    # generators in the canonical-correlation sense). Check against a finite-diff
    # Hessian of log Z.
    rng = np.random.default_rng(5)
    ham = ParamHamiltonian(local_pauli_generators(3))
    theta = rng.normal(scale=0.4, size=ham.n_params)
    backend = qbm.DenseBackend()

    def logZ(t):
        return backend.thermal_state(ham, t).log_partition()

    eps = 1e-4
    J = ham.n_params
    H = np.zeros((J, J))
    for i in range(J):
        for j in range(J):
            tpp = theta.copy(); tpp[i] += eps; tpp[j] += eps
            tpm = theta.copy(); tpm[i] += eps; tpm[j] -= eps
            tmp = theta.copy(); tmp[i] -= eps; tmp[j] += eps
            tmm = theta.copy(); tmm[i] -= eps; tmm[j] -= eps
            H[i, j] = (logZ(tpp) - logZ(tpm) - logZ(tmp) + logZ(tmm)) / (4 * eps ** 2)
    km = backend.thermal_state(ham, theta).metric("kubo_mori")
    assert np.allclose(km, H, atol=1e-4)
