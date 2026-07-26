"""v0.5: JAX backend -- autodiff gradients/metrics cross-validated against the dense engine.

Skipped automatically if JAX is not installed.
"""

import numpy as np
import pytest

pytest.importorskip("jax")

import qbm
from qbm.operators import ParamHamiltonian, local_pauli_generators


def _ham_theta(n=3, seed=0):
    rng = np.random.default_rng(seed)
    ham = ParamHamiltonian(local_pauli_generators(n))
    return ham, rng.normal(scale=0.5, size=ham.n_params)


def test_jax_in_available_backends():
    assert "jax" in qbm.available_backends()
    assert qbm.get_backend("jax").name == "jax"


def test_jax_matches_dense_states_and_expectations():
    ham, theta = _ham_theta(seed=1)
    dense = qbm.DenseBackend().thermal_state(ham, theta)
    jx = qbm.get_backend("jax").thermal_state(ham, theta)
    O = qbm.hamiltonians.tfim(ham.n_qubits, g=1.3)
    assert np.allclose(dense.density_matrix(), jx.density_matrix(), atol=1e-9)
    assert np.isclose(dense.expect(O), jx.expect(O), atol=1e-9)
    assert np.allclose(dense.generator_expectations(), jx.generator_expectations(), atol=1e-9)
    assert np.isclose(dense.entropy(), jx.entropy(), atol=1e-9)
    assert np.isclose(dense.log_partition(), jx.log_partition(), atol=1e-9)


def test_autodiff_observable_gradient_matches_analytic():
    # jax.grad of Tr[O rho] must equal the hand-derived belief-propagation gradient.
    ham, theta = _ham_theta(seed=2)
    O = qbm.hamiltonians.heisenberg(ham.n_qubits)
    g_analytic = qbm.DenseBackend().thermal_state(ham, theta).observable_gradient(O)
    g_autodiff = qbm.get_backend("jax").thermal_state(ham, theta).observable_gradient(O)
    assert np.allclose(g_analytic, g_autodiff, atol=1e-7)


def test_autodiff_metrics_match_analytic():
    # jacrev-based QFI must equal the analytic Kubo-Mori / Fisher-Bures / Wigner-Yanase.
    ham, theta = _ham_theta(seed=3)
    dense = qbm.DenseBackend().thermal_state(ham, theta)
    jx = qbm.get_backend("jax").thermal_state(ham, theta)
    for kind in ("kubo_mori", "fisher_bures", "wigner_yanase"):
        assert np.allclose(dense.metric(kind), jx.metric(kind), atol=1e-7)


def test_autodiff_state_and_diagonal_derivatives():
    ham, theta = _ham_theta(seed=4)
    dense = qbm.DenseBackend().thermal_state(ham, theta)
    jx = qbm.get_backend("jax").thermal_state(ham, theta)
    assert np.allclose(dense.state_derivatives(), jx.state_derivatives(), atol=1e-7)
    assert np.allclose(dense.diagonal_gradient(), jx.diagonal_gradient(), atol=1e-7)


def test_value_and_grad_energy_matches_analytic():
    import jax.numpy as jnp

    from qbm import autodiff

    ham, theta = _ham_theta(seed=5)
    O = qbm.hamiltonians.tfim(ham.n_qubits, g=1.1)
    Oj = jnp.asarray(O.astype(complex))
    val, g = autodiff.value_and_grad(
        ham.generators, theta, lambda rho: jnp.real(jnp.trace(Oj @ rho))
    )
    dense = qbm.DenseBackend().thermal_state(ham, theta)
    assert np.isclose(val, dense.expect(O), atol=1e-9)
    assert np.allclose(g, dense.observable_gradient(O), atol=1e-7)


def test_training_on_jax_backend():
    n = 3
    H = qbm.hamiltonians.tfim(n, J=1.0, g=1.2)
    e0 = qbm.oracles.ground_energy(H)
    model = qbm.FullyVisibleQBM(n=n, backend="jax")
    model.theta = np.random.default_rng(0).normal(scale=0.05, size=model.n_params)
    hist = qbm.fit(
        model,
        qbm.losses.Energy(H),
        qbm.optim.NaturalGradient(metric="kubo_mori", lr=0.2, reg=1e-3),
        steps=300,
    )
    assert model.energy(H) < hist.loss[0]
    assert (model.energy(H) - e0) < 0.15
