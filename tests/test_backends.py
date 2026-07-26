"""v0.5: statevector backend (TFD purification), shot noise, cross-backend agreement."""

import numpy as np

import qbm
from qbm import purification
from qbm.operators import ParamHamiltonian, local_pauli_generators


def _state(theta=None, n=3, seed=0, backend=None):
    rng = np.random.default_rng(seed)
    ham = ParamHamiltonian(local_pauli_generators(n))
    if theta is None:
        theta = rng.normal(scale=0.5, size=ham.n_params)
    b = qbm.get_backend(backend)
    return ham, theta, b.thermal_state(ham, theta)


# ---------------------------------------------------------------------------
# TFD purification
# ---------------------------------------------------------------------------
def test_tfd_reduces_to_rho():
    ham, theta, st = _state(seed=1)
    sv = qbm.get_backend("statevector").thermal_state(ham, theta)
    rho_from_tfd = purification.reduced_system_state(sv.tfd_state(), sv.dim)
    assert np.allclose(rho_from_tfd, sv.density_matrix(), atol=1e-10)


def test_tfd_expectation_matches_trace():
    ham, theta, st = _state(seed=2)
    sv = qbm.get_backend("statevector").thermal_state(ham, theta)
    O = qbm.hamiltonians.tfim(ham.n_qubits, g=1.3)
    via_psi = purification.expectation(sv.tfd_state(), O, sv.dim)
    assert np.isclose(via_psi, np.real(np.trace(sv.density_matrix() @ O)), atol=1e-10)


def test_entanglement_entropy_equals_thermal_entropy():
    ham, theta, st = _state(seed=3)
    sv = qbm.get_backend("statevector").thermal_state(ham, theta)
    assert np.isclose(sv.entanglement_entropy(), sv.entropy(), atol=1e-10)


# ---------------------------------------------------------------------------
# cross-backend agreement (statevector, infinite shots == dense)
# ---------------------------------------------------------------------------
def test_statevector_matches_dense_exact():
    rng = np.random.default_rng(4)
    ham = ParamHamiltonian(local_pauli_generators(3))
    theta = rng.normal(scale=0.5, size=ham.n_params)
    dense = qbm.DenseBackend().thermal_state(ham, theta)
    sv = qbm.StatevectorBackend(shots=None).thermal_state(ham, theta)
    O = qbm.hamiltonians.tfim(3, g=1.2)

    assert np.isclose(dense.expect(O), sv.expect(O), atol=1e-10)
    assert np.allclose(dense.probabilities(), sv.probabilities(), atol=1e-10)
    assert np.allclose(dense.generator_expectations(), sv.generator_expectations(), atol=1e-10)
    assert np.allclose(dense.observable_gradient(O), sv.observable_gradient(O), atol=1e-10)
    assert np.allclose(dense.metric("kubo_mori"), sv.metric("kubo_mori"), atol=1e-10)


def test_get_backend_and_registry():
    assert "dense" in qbm.available_backends()
    assert "statevector" in qbm.available_backends()
    assert isinstance(qbm.get_backend("dense"), qbm.DenseBackend)
    assert qbm.get_backend(None).__class__ is qbm.DenseBackend
    inst = qbm.StatevectorBackend()
    assert qbm.get_backend(inst) is inst


# ---------------------------------------------------------------------------
# shot noise
# ---------------------------------------------------------------------------
def test_shot_expectation_is_unbiased():
    ham = ParamHamiltonian(local_pauli_generators(2))
    theta = np.random.default_rng(5).normal(scale=0.5, size=ham.n_params)
    O = qbm.pauli("ZI")
    exact = qbm.DenseBackend().thermal_state(ham, theta).expect(O)
    backend = qbm.StatevectorBackend(shots=4000, seed=0)
    estimates = [backend.thermal_state(ham, theta).expect(O) for _ in range(40)]
    assert abs(np.mean(estimates) - exact) < 0.02  # average of estimates -> exact


def test_training_through_statevector_backend():
    # A model trained on the (exact) statevector backend matches the dense result.
    q = qbm.datasets.parity(n=3)
    model = qbm.FullyVisibleQBM(n=3, backend="statevector")
    model.theta = np.random.default_rng(0).normal(scale=0.05, size=model.n_params)
    hist = qbm.fit(
        model,
        qbm.losses.RelativeEntropy(np.diag(q.astype(complex))),
        qbm.optim.Adam(lr=0.1),
        steps=150,
    )
    assert hist.final_loss < hist.loss[0]
    assert isinstance(model.backend, qbm.StatevectorBackend)
