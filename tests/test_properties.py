"""Property-based tests: invariants that must hold for *any* Hamiltonian and parameters.

Hypothesis generates adversarial (size, seed, scale) combinations; these check the
fundamental guarantees rather than hand-picked cases.
"""

import numpy as np
import pytest

pytest.importorskip("hypothesis")

from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

import qbm  # noqa: E402
from qbm import purification  # noqa: E402
from qbm.operators import ParamHamiltonian, local_pauli_generators  # noqa: E402

_SETTINGS = settings(max_examples=60, deadline=None)


@st.composite
def gibbs_states(draw):
    n = draw(st.integers(min_value=2, max_value=3))
    seed = draw(st.integers(min_value=0, max_value=10_000))
    scale = draw(st.floats(min_value=0.05, max_value=1.5))
    ham = ParamHamiltonian(local_pauli_generators(n))
    theta = np.random.default_rng(seed).normal(scale=scale, size=ham.n_params)
    return ham, theta


@_SETTINGS
@given(gibbs_states())
def test_density_matrix_is_a_valid_state(ht):
    ham, theta = ht
    rho = qbm.DenseBackend().thermal_state(ham, theta).density_matrix()
    assert np.allclose(rho, rho.conj().T, atol=1e-9)
    assert np.isclose(np.trace(rho).real, 1.0, atol=1e-9)
    assert np.min(np.linalg.eigvalsh(rho)) > -1e-9


@_SETTINGS
@given(gibbs_states())
def test_probabilities_are_a_distribution(ht):
    ham, theta = ht
    p = qbm.DenseBackend().thermal_state(ham, theta).probabilities()
    assert np.all(p > -1e-12)
    assert np.isclose(p.sum(), 1.0, atol=1e-9)


@_SETTINGS
@given(gibbs_states())
def test_metrics_are_symmetric_psd_and_ordered(ht):
    ham, theta = ht
    state = qbm.DenseBackend().thermal_state(ham, theta)
    fb = state.metric("fisher_bures")
    wy = state.metric("wigner_yanase")
    km = state.metric("kubo_mori")
    for g in (fb, wy, km):
        assert np.all(np.isfinite(g))
        assert np.allclose(g, g.T, atol=1e-8)
        assert np.min(np.linalg.eigvalsh(g)) > -1e-7
    # Loewner orderings: FB <= WY <= 2 FB and KM >= FB
    assert np.min(np.linalg.eigvalsh(wy - fb)) > -1e-7
    assert np.min(np.linalg.eigvalsh(2 * fb - wy)) > -1e-7
    assert np.min(np.linalg.eigvalsh(km - fb)) > -1e-7


@_SETTINGS
@given(gibbs_states())
def test_gradient_is_finite_and_matches_finite_difference(ht):
    ham, theta = ht
    O = qbm.hamiltonians.tfim(ham.n_qubits, g=1.0)
    backend = qbm.DenseBackend()
    analytic = backend.thermal_state(ham, theta).observable_gradient(O)
    assert np.all(np.isfinite(analytic))
    eps = 1e-6
    fd = np.zeros_like(theta)
    for j in range(len(theta)):
        tp = theta.copy()
        tp[j] += eps
        tm = theta.copy()
        tm[j] -= eps
        fd[j] = (
            backend.thermal_state(ham, tp).expect(O) - backend.thermal_state(ham, tm).expect(O)
        ) / (2 * eps)
    assert np.allclose(analytic, fd, atol=1e-5)


@_SETTINGS
@given(gibbs_states())
def test_tfd_purification_reduces_to_rho(ht):
    ham, theta = ht
    sv = qbm.StatevectorBackend().thermal_state(ham, theta)
    rho = purification.reduced_system_state(sv.tfd_state(), sv.dim)
    assert np.allclose(rho, sv.density_matrix(), atol=1e-9)
    assert np.isclose(sv.entanglement_entropy(), sv.entropy(), atol=1e-9)


@_SETTINGS
@given(gibbs_states())
def test_statevector_agrees_with_dense(ht):
    ham, theta = ht
    O = qbm.hamiltonians.tfim(ham.n_qubits, g=1.1)
    dense = qbm.DenseBackend().thermal_state(ham, theta)
    sv = qbm.StatevectorBackend().thermal_state(ham, theta)
    assert np.isclose(dense.expect(O), sv.expect(O), atol=1e-9)
    assert np.allclose(dense.observable_gradient(O), sv.observable_gradient(O), atol=1e-9)
    assert np.allclose(dense.metric("kubo_mori"), sv.metric("kubo_mori"), atol=1e-9)
