"""Dense thermal-state correctness against direct/expm references."""

import numpy as np
import scipy.linalg as sla

import qbm
from qbm.backends.dense import DenseThermalState
from qbm.operators import ParamHamiltonian, local_pauli_generators


def _random_model(n=3, scale=0.4, seed=0):
    rng = np.random.default_rng(seed)
    ham = ParamHamiltonian(local_pauli_generators(n))
    theta = rng.normal(scale=scale, size=ham.n_params)
    return ham, theta


def test_gibbs_matches_expm():
    ham, theta = _random_model()
    state = DenseThermalState(ham, theta)
    G = ham.matrix(theta)
    rho_expm = sla.expm(-G)
    rho_expm /= np.trace(rho_expm)
    assert np.allclose(state.density_matrix(), rho_expm, atol=1e-10)


def test_density_matrix_is_valid_state():
    ham, theta = _random_model(seed=1)
    rho = DenseThermalState(ham, theta).density_matrix()
    assert np.allclose(rho, rho.conj().T, atol=1e-12)          # Hermitian
    assert np.isclose(np.trace(rho).real, 1.0, atol=1e-12)     # unit trace
    assert np.min(np.linalg.eigvalsh(rho)) > -1e-12            # PSD


def test_probabilities_sum_to_one():
    ham, theta = _random_model(seed=2)
    state = DenseThermalState(ham, theta)
    p = state.probabilities()
    assert np.all(p >= -1e-12)
    assert np.isclose(p.sum(), 1.0, atol=1e-12)
    # probabilities equal the diagonal of rho in the computational basis
    assert np.allclose(p, np.real(np.diag(state.density_matrix())), atol=1e-12)


def test_expect_matches_trace():
    ham, theta = _random_model(seed=3)
    state = DenseThermalState(ham, theta)
    rho = state.density_matrix()
    O = qbm.hamiltonians.tfim(ham.n_qubits, J=0.7, g=1.3)
    assert np.isclose(state.expect(O), np.real(np.trace(rho @ O)), atol=1e-10)


def test_overflow_safe():
    # A Hamiltonian with a strongly negative eigenvalue: naive expm(-G) overflows,
    # eigendecomposition stays finite and normalised.
    ham = ParamHamiltonian(local_pauli_generators(4))
    theta = np.full(ham.n_params, 3.0)
    state = DenseThermalState(ham, theta)
    assert np.isfinite(state.log_partition())
    assert np.isclose(state.probabilities().sum(), 1.0, atol=1e-10)
