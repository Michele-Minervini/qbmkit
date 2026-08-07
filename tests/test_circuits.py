"""Circuit backend: IR + simulator (Tier 1), Hadamard-test estimators (Tier 2),
and vendor adapters / QASM export (Tier 3)."""

import numpy as np
import pytest

import qbm
from qbm import purification
from qbm.backends.circuit import CircuitBackend
from qbm.circuits import Circuit, builder, densities, simulator
from qbm.circuits.adapters import available_adapters, to_qasm3
from qbm.circuits.estimators import energy_gradient, information_matrix
from qbm.metrics import AlphaZ
from qbm.operators import ParamHamiltonian, local_pauli_generators


def _model(n=3, seed=0, scale=0.4):
    ham = ParamHamiltonian(local_pauli_generators(n))
    theta = np.random.default_rng(seed).normal(scale=scale, size=ham.n_params)
    return ham, theta


def _require_adapter(engine):
    """Skip unless the SDK is actually *usable*, not merely present.

    ``pytest.importorskip`` only catches ``ImportError``, so an SDK that installs but
    breaks on import (e.g. a PennyLane / autoray version mismatch) errors the suite.
    The library's own capability probe treats any failure as unavailable -- reusing it
    here keeps the tests honest about the environment instead of red because of it.
    """
    if engine not in available_adapters():
        pytest.skip(f"{engine} is installed but not usable in this environment")


# ---------------------------------------------------------------------------
# Tier 1 -- IR, simulator, and the circuit backend
# ---------------------------------------------------------------------------
def test_simulator_basic_gates():
    c = Circuit(2)
    c.h(0)
    c.cx(0, 1)
    psi = simulator.run(c)
    expected = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    assert np.allclose(psi, expected, atol=1e-12)


def test_qubit_ordering_is_qubit0_most_significant():
    c = Circuit(2)
    c.x(0)  # |10> -> index 2
    assert np.argmax(np.abs(simulator.run(c))) == 2
    c2 = Circuit(2)
    c2.x(1)  # |01> -> index 1
    assert np.argmax(np.abs(simulator.run(c2))) == 1


def test_circuit_prepares_the_thermal_state_exactly():
    ham, theta = _model()
    psi = simulator.run(builder.gibbs_preparation(ham, theta))
    rho_circ = purification.reduced_system_state(psi, ham.dim)
    rho_dense = qbm.DenseBackend().thermal_state(ham, theta).density_matrix()
    assert np.allclose(rho_circ, rho_dense, atol=1e-10)


@pytest.mark.parametrize("imaginary", [False, True])
def test_hadamard_test_is_exact(imaginary):
    ham, theta = _model()
    rho = qbm.DenseBackend().thermal_state(ham, theta).density_matrix()
    U = builder.evolution_unitary(ham, theta, 0.7)
    circ = builder.hadamard_test(
        builder.gibbs_preparation(ham, theta, ancilla_offset=1),
        U,
        range(1, ham.n_qubits + 1),
        imaginary=imaginary,
    )
    est = simulator.measure_z(simulator.run(circ), 0, circ.n_qubits)
    ref = np.imag(np.trace(U @ rho)) if imaginary else np.real(np.trace(U @ rho))
    assert np.isclose(est, ref, atol=1e-10)


def test_circuit_backend_matches_dense_on_measurable_quantities():
    ham, theta = _model()
    dense = qbm.DenseBackend().thermal_state(ham, theta)
    circ = CircuitBackend(seed=0).thermal_state(ham, theta)
    O = qbm.hamiltonians.tfim(ham.n_qubits, g=1.2)
    assert np.isclose(circ.expect(O), dense.expect(O), atol=1e-10)
    assert np.allclose(circ.generator_expectations(), dense.generator_expectations(), atol=1e-10)
    assert np.allclose(circ.probabilities(), dense.probabilities(), atol=1e-10)


def test_circuit_backend_is_registered():
    assert "circuit" in qbm.available_backends()
    assert qbm.get_backend("circuit").name == "circuit"


def test_shot_noise_is_unbiased():
    ham, theta = _model(n=2)
    exact = qbm.DenseBackend().thermal_state(ham, theta).expect(qbm.pauli("ZI"))
    backend = CircuitBackend(shots=4000, seed=0)
    est = [backend.thermal_state(ham, theta).expect(qbm.pauli("ZI")) for _ in range(30)]
    assert abs(np.mean(est) - exact) < 0.03


def test_spectrum_dependent_quantities_are_refused():
    ham, theta = _model()
    st = CircuitBackend().thermal_state(ham, theta)
    for call in (st.entropy, st.log_partition, st.state_derivatives):
        with pytest.raises(NotImplementedError, match="spectrum"):
            call()
    # the diagonal gradient is a different refusal: it needs d_j rho, not the spectrum,
    # and there is now an exact alternative for hidden-unit models
    with pytest.raises(NotImplementedError, match="GibbsMapNLL"):
        st.diagonal_gradient()


def test_unknown_gibbs_strategy_is_explicit():
    ham, theta = _model()
    with pytest.raises(ValueError, match="unknown Gibbs preparation strategy"):
        CircuitBackend(gibbs_prep="qite").thermal_state(ham, theta)


def test_resource_estimate_reports_costs():
    ham, theta = _model()
    r = CircuitBackend(shots=1000).thermal_state(ham, theta).resource_estimate()
    assert r["n_qubits"] == 2 * ham.n_qubits + 1
    assert r["circuits_per_metric"] > r["circuits_per_gradient"]
    assert r["shots_per_metric"] > 0


# ---------------------------------------------------------------------------
# Tier 2 -- smearing densities and Hadamard-test estimators
# ---------------------------------------------------------------------------
def test_densities_are_normalised():
    assert np.isclose(2 * densities.tent_sampler().mass, 1.0, atol=1e-3)
    for a, z in [(0.5, 1.0), (0.3, 0.8), (0.7, 2.0)]:
        assert np.isclose(2 * densities.alpha_z_sampler(a, z).mass, 1.0, atol=1e-3)


def test_tent_sampler_reproduces_its_characteristic_function():
    # the Fourier transform of p(t) is the closed form the dense backend uses
    t = densities.tent_sampler().sample(200000, np.random.default_rng(0))
    for d in (0.5, 1.5, 3.0):
        assert abs(np.mean(np.cos(d * t)) - densities.tent_characteristic(d)) < 0.01


def test_alpha_z_density_rejects_alpha_outside_zero_one():
    with pytest.raises(ValueError, match="alpha in"):
        densities.alpha_z_density(1.0, 2.0, 1.0)


def test_energy_gradient_estimator_converges_to_analytic():
    ham, theta = _model()
    dense = qbm.DenseBackend().thermal_state(ham, theta)
    O = qbm.hamiltonians.tfim(ham.n_qubits, g=1.2)
    exact = dense.observable_gradient(O)
    st = CircuitBackend(seed=1).thermal_state(ham, theta)
    est = energy_gradient(st, O, n_time_samples=20000)
    assert np.max(np.abs(est - exact)) / np.max(np.abs(exact)) < 0.02


@pytest.mark.parametrize(
    "metric, reference",
    [
        ("kubo_mori", "kubo_mori"),
        (AlphaZ(0.5, 1.0), "wigner_yanase"),
        (AlphaZ(0.5, 0.5), "fisher_bures"),
    ],
)
def test_information_matrix_estimator_matches_dense(metric, reference):
    # The paper's channel formula (Eq. 1.1) is the hardware route; it must agree with
    # the spectral formula the dense backend uses.
    ham, theta = _model()
    ref = qbm.DenseBackend().thermal_state(ham, theta).metric(reference)
    st = CircuitBackend(seed=2).thermal_state(ham, theta)
    est = information_matrix(st, metric, n_time_samples=30000)
    assert np.max(np.abs(est - ref)) / np.max(np.abs(ref)) < 0.02


def test_information_matrix_estimator_rejects_metrics_without_a_channel_form():
    ham, theta = _model()
    st = CircuitBackend().thermal_state(ham, theta)
    with pytest.raises(NotImplementedError, match="channel formula"):
        information_matrix(st, "wigner_yanase")


# ---------------------------------------------------------------------------
# Tier 3 -- adapters and export
# ---------------------------------------------------------------------------
def test_qasm3_export_of_a_gate_level_circuit():
    c = Circuit(2)
    c.h(0)
    c.cx(0, 1)
    c.rz(0.7, 1)
    src = to_qasm3(c)
    assert "OPENQASM 3.0;" in src
    assert "h q[0];" in src and "cx q[0], q[1];" in src and "rz(0.7" in src


def test_qasm3_refuses_opaque_unitaries():
    ham, theta = _model(n=2)
    with pytest.raises(ValueError, match="opaque"):
        to_qasm3(builder.gibbs_preparation(ham, theta))


def test_qasm3_always_available():
    assert "qasm3" in available_adapters()


@pytest.mark.parametrize(
    "gate_seq",
    [
        lambda c: (c.h(0), c.cx(0, 1), c.rz(0.7, 2), c.ry(0.3, 1), c.cz(1, 2), c.s(0)),
        lambda c: (c.x(1), c.y(2), c.z(0), c.rx(0.4, 2), c.sdg(1)),
    ],
)
def test_qiskit_adapter_reproduces_our_simulator(gate_seq):
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import Statevector

    from qbm.circuits.adapters import to_qiskit

    c = Circuit(3)
    gate_seq(c)
    ours = simulator.run(c)
    theirs = np.asarray(Statevector(to_qiskit(c)).data)
    assert np.allclose(ours, theirs, atol=1e-10)


def test_qiskit_adapter_handles_the_qbm_preparation_circuit():
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import Statevector

    from qbm.circuits.adapters import to_qiskit

    ham, theta = _model(n=2)
    prep = builder.gibbs_preparation(ham, theta)
    theirs = np.asarray(Statevector(to_qiskit(prep)).data)
    assert np.allclose(simulator.run(prep), theirs, atol=1e-10)


def test_trotter_decomposition_matches_exact_evolution():
    # the hardware-faithful form of e^{-iGt}: more steps -> closer to exact
    ham, theta = _model(n=2, scale=0.3)
    t = 0.5
    exact = builder.evolution_unitary(ham, theta, t)
    errs = []
    for steps in (2, 32):
        circ = Circuit(ham.n_qubits)
        for label, angle in builder.trotter_evolution(ham, theta, t, steps=steps):
            builder.pauli_rotation(circ, label, angle)
        U = np.column_stack(
            [simulator.run(circ, initial_state=np.eye(ham.dim)[:, k]) for k in range(ham.dim)]
        )
        errs.append(np.max(np.abs(U - exact)))
    assert errs[1] < errs[0]
    assert errs[1] < 0.05


# ---------------------------------------------------------------------------
# Emitter / executor swapping (the "no SDK in the core" guarantee)
# ---------------------------------------------------------------------------
def test_core_has_no_sdk_imports_outside_adapters():
    """The algorithms must not import a vendor SDK; only the adapter leaves may."""
    import pathlib
    import re

    root = pathlib.Path(qbm.__file__).parent
    pattern = re.compile(r"(?:^|[^a-z_])(?:import|from)\s+(qiskit|pennylane)\b")
    offenders = [
        str(f.relative_to(root))
        for f in root.rglob("*.py")
        if "adapters" not in f.parts and pattern.search(f.read_text())
    ]
    assert offenders == [], f"SDK imported outside adapters: {offenders}"


def test_importing_qbm_does_not_import_any_sdk():
    """Optional backends are lazy factories, so `import qbm` stays dependency-free."""
    import subprocess
    import sys

    code = (
        "import sys, qbm; "
        "print([m for m in ('qiskit','pennylane','jax','quimb') if m in sys.modules])"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip() == "[]", out.stdout


def test_builtin_executor_is_the_default():
    from qbm.circuits.adapters import executor

    c = Circuit(2)
    c.h(0)
    c.cx(0, 1)
    assert np.allclose(executor("builtin")(c), simulator.run(c), atol=1e-12)


def test_unknown_executor_raises():
    from qbm.circuits.adapters import executor

    with pytest.raises(KeyError, match="unknown executor"):
        executor("not_an_sdk")


@pytest.mark.parametrize("engine", ["qiskit", "pennylane"])
def test_sdk_executors_agree_with_the_builtin_one(engine):
    _require_adapter(engine)
    from qbm.circuits.adapters import executor

    c = Circuit(3)
    c.h(0)
    c.cx(0, 1)
    c.ry(0.4, 2)
    c.cz(1, 2)
    c.rz(0.7, 0)
    assert np.allclose(executor(engine)(c), simulator.run(c), atol=1e-9)


@pytest.mark.parametrize("engine", ["qiskit", "pennylane"])
def test_whole_qbm_runs_through_an_sdk_executor(engine):
    _require_adapter(engine)
    from qbm.circuits.adapters import executor

    ham, theta = _model(n=2)
    dense = qbm.DenseBackend().thermal_state(ham, theta)
    st = CircuitBackend(seed=0, executor=executor(engine)).thermal_state(ham, theta)
    O = qbm.hamiltonians.tfim(2, g=1.2)
    assert np.isclose(st.expect(O), dense.expect(O), atol=1e-9)
    assert np.allclose(st.generator_expectations(), dense.generator_expectations(), atol=1e-9)
