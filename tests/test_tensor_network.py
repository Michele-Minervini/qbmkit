"""v0.8: tensor-network backend (purified MPS). Skipped without the ``tn`` extra."""

import numpy as np
import pytest

pytest.importorskip("quimb")

import qbm  # noqa: E402
from qbm.operators import ParamHamiltonian  # noqa: E402


def _chain(n):
    """1- and 2-body Pauli generators on an n-qubit chain."""
    return (
        [f"{'I' * i}Z{'I' * (n - i - 1)}" for i in range(n)]
        + [f"{'I' * i}X{'I' * (n - i - 1)}" for i in range(n)]
        + [f"{'I' * i}ZZ{'I' * (n - i - 2)}" for i in range(n - 1)]
    )


def _pair(n=4, seed=0, scale=0.4):
    gens = _chain(n)
    theta = np.random.default_rng(seed).normal(scale=scale, size=len(gens))
    return ParamHamiltonian(gens), theta


def test_registered_and_constructible():
    assert "tensor_network" in qbm.available_backends()
    for alias in ("tensor_network", "tn"):
        assert qbm.get_backend(alias).name == "tensor_network"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_generator_expectations_match_dense(seed):
    # THE key cross-backend check: the purified MPS reproduces the exact thermal state.
    ham, theta = _pair(seed=seed)
    dense = qbm.DenseBackend().thermal_state(ham, theta)
    tn = qbm.get_backend("tn").thermal_state(ham, theta)
    assert np.allclose(dense.generator_expectations(), tn.generator_expectations(), atol=1e-4)


def test_density_matrix_and_probabilities_match_dense():
    ham, theta = _pair(seed=3)
    dense = qbm.DenseBackend().thermal_state(ham, theta)
    tn = qbm.get_backend("tn").thermal_state(ham, theta)
    rho = tn.density_matrix()
    assert np.allclose(rho, rho.conj().T, atol=1e-8)
    assert np.isclose(np.trace(rho).real, 1.0, atol=1e-8)
    assert np.allclose(rho, dense.density_matrix(), atol=1e-4)
    assert np.allclose(tn.probabilities(), dense.probabilities(), atol=1e-4)


def test_expect_accepts_label_and_dense_operator():
    ham, theta = _pair(seed=4)
    dense = qbm.DenseBackend().thermal_state(ham, theta)
    tn = qbm.get_backend("tn").thermal_state(ham, theta)
    label = "Z" + "I" * (ham.n_qubits - 1)
    assert np.isclose(tn.expect(label), dense.expect(qbm.pauli(label)), atol=1e-4)
    O = qbm.hamiltonians.tfim(ham.n_qubits, g=1.0)
    assert np.isclose(tn.expect(O), dense.expect(O), atol=1e-4)


def test_accuracy_improves_with_trotter_steps():
    ham, theta = _pair(seed=5)
    exact = qbm.DenseBackend().thermal_state(ham, theta).generator_expectations()
    errs = []
    for steps in (10, 100):
        tn = qbm.get_backend("tn", trotter_steps=steps).thermal_state(ham, theta)
        errs.append(np.max(np.abs(tn.generator_expectations() - exact)))
    assert errs[1] < errs[0]  # second-order Trotter: error falls as dt^2


def test_sampling_returns_valid_basis_states():
    ham, theta = _pair(seed=6)
    tn = qbm.get_backend("tn").thermal_state(ham, theta)
    s = tn.sample(64, rng=np.random.default_rng(0))
    assert s.shape == (64,)
    assert s.min() >= 0 and s.max() < ham.dim


def test_unsupported_quantities_raise_clearly():
    # The backend must refuse rather than silently return something wrong.
    ham, theta = _pair(seed=7)
    tn = qbm.get_backend("tn").thermal_state(ham, theta)
    for call in (
        lambda: tn.metric("kubo_mori"),
        lambda: tn.observable_gradient(qbm.hamiltonians.tfim(ham.n_qubits)),
        lambda: tn.state_derivatives(),
        lambda: tn.log_partition(),
        lambda: tn.entropy(),
    ):
        with pytest.raises(NotImplementedError):
            call()


def test_rejects_more_than_two_body_generators():
    ham = ParamHamiltonian(["ZZZ"])
    with pytest.raises(ValueError, match="1- and 2-body"):
        qbm.get_backend("tn").thermal_state(ham, np.array([0.3]))


def test_scales_past_the_dense_ceiling():
    # 16 qubits: a dense density matrix would be 2^16 x 2^16 (~34 GB); the purified
    # MPS handles it in seconds at small bond dimension.
    n = 16
    gens = _chain(n)
    ham = ParamHamiltonian(gens)
    theta = np.random.default_rng(0).normal(scale=0.3, size=len(gens))
    tn = qbm.get_backend("tn", trotter_steps=20).thermal_state(ham, theta)
    ge = tn.generator_expectations()
    assert ge.shape == (len(gens),)
    assert np.all(np.isfinite(ge))
    assert np.all(np.abs(ge) <= 1.0 + 1e-9)  # Pauli expectations live in [-1, 1]


def test_lazy_generators_do_not_materialise_dense_matrices():
    # Regression: building a 20-qubit ParamHamiltonian must not allocate 2^20 x 2^20
    # matrices (that would be ~17 TB and caps every backend at ~13 qubits).
    ham = ParamHamiltonian(_chain(20))
    assert ham.n_params == 59
    assert ham.dim == 2**20
    assert ham._mats is None  # nothing materialised
    assert len(ham.labels) == 59
