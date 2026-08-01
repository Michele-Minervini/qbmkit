"""VarQITE: variational Gibbs preparation by imaginary-time evolution.

Structured as independent-route checks: the tilt-partner algebra against dense
matrices, the derivative states against finite differences, ``A``/``C`` against
finite differences of the geometry and the energy, the Hadamard-test route against
the exact simulator route, and the prepared state against the exact Gibbs state.
"""

import numpy as np
import pytest

import qbm
from qbm.circuits import simulator
from qbm.circuits import varqite as vq
from qbm.circuits.adapters import to_qasm3
from qbm.operators import ParamHamiltonian, local_pauli_generators, pauli

TFIM_LABELS = ["ZZ", "XI", "IX"]
TFIM_COEFFS = [-1.0, -0.8, -0.8]
ISING_LABELS = ["ZI", "IZ", "ZZ"]
ISING_COEFFS = [0.7, -0.4, 0.9]


def _dense(labels, coeffs):
    return sum(c * pauli(l) for c, l in zip(coeffs, labels))


def _fd_derivatives(ansatz, lam, eps=1e-6):
    cols = []
    for k in range(ansatz.n_params):
        lp = lam.copy()
        lp[k] += eps
        lm = lam.copy()
        lm[k] -= eps
        cols.append((ansatz.state(lp) - ansatz.state(lm)) / (2 * eps))
    return np.stack(cols, axis=1)


# ---------------------------------------------------------------------------
# the algebra: Pauli action and tilt partners
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("label", ["Z", "Y", "X", "XY", "ZZ", "IXYZ", "YZXI"])
def test_pauli_action_matches_the_dense_matrix(label):
    rng = np.random.default_rng(0)
    dim = 1 << len(label)
    v = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    assert np.allclose(vq.apply_pauli(v, vq.pauli_action(label)), pauli(label) @ v, atol=1e-14)


@pytest.mark.parametrize("label", ["Z", "X", "Y", "ZZ", "XX", "XZ", "YY", "ZIX", "ZZZ", "XYZ"])
def test_tilt_partner_generates_imaginary_time_evolution(label):
    """``K |TFD(0)> = -i (P (x) I) |TFD(0)>`` -- the defining property (up to sign)."""
    n = len(label)
    ansatz = vq.PauliRotationAnsatz([vq.tilt_partner(label)], n_system=n)
    tfd0 = ansatz.initial_state()
    lhs = pauli(vq.tilt_partner(label)) @ tfd0
    rhs = -1j * np.kron(pauli(label), np.eye(1 << n)) @ tfd0
    assert np.allclose(lhs, rhs, atol=1e-12) or np.allclose(lhs, -rhs, atol=1e-12)


def test_identity_has_no_tilt_partner():
    with pytest.raises(ValueError, match="identity"):
        vq.tilt_partner("III")


def test_single_qubit_gibbs_state_is_reached_by_one_rotation():
    """``exp(-i lambda Y(x)X / 2)`` on a Bell pair sweeps out every single-qubit Gibbs state."""
    H = 0.9 * pauli("Z")
    res = vq.varqite(H, vq.tfd_ansatz(labels=["Z"], depth=1), tau=0.5, steps=400)
    assert res.ansatz.n_params == 2  # one tilt + one rotation
    assert res.trace_distance() < 1e-4


# ---------------------------------------------------------------------------
# the ansatz
# ---------------------------------------------------------------------------
def test_ansatz_starts_exactly_on_the_infinite_temperature_tfd():
    ansatz = vq.tfd_ansatz(labels=TFIM_LABELS, depth=2)
    rho = qbm.purification.reduced_system_state(ansatz.state(np.zeros(ansatz.n_params)), 4)
    assert np.allclose(rho, np.eye(4) / 4, atol=1e-14)


def test_ansatz_statevector_matches_its_emitted_circuit():
    """The fast statevector path and the gate-level IR circuit must agree exactly."""
    ansatz = vq.tfd_ansatz(labels=TFIM_LABELS, depth=2)
    lam = np.random.default_rng(1).normal(scale=0.7, size=ansatz.n_params)
    assert np.allclose(ansatz.state(lam), simulator.run(ansatz.circuit(lam)), atol=1e-12)


def test_derivative_states_match_finite_differences():
    ansatz = vq.tfd_ansatz(labels=TFIM_LABELS, depth=2)
    lam = np.random.default_rng(2).normal(scale=0.6, size=ansatz.n_params)
    _, D = ansatz.derivative_states(lam)
    assert np.allclose(D, _fd_derivatives(ansatz, lam), atol=1e-8)


def test_ansatz_rejects_wrong_width_paulis():
    with pytest.raises(ValueError, match="length 4"):
        vq.PauliRotationAnsatz(["ZZZ"], n_system=2)


# ---------------------------------------------------------------------------
# McLachlan's variational principle
# ---------------------------------------------------------------------------
def test_C_is_minus_half_the_energy_gradient():
    """The defining identity that makes VarQITE natural-gradient flow on the energy."""
    ansatz = vq.tfd_ansatz(labels=TFIM_LABELS, depth=2)
    lam = np.random.default_rng(3).normal(scale=0.6, size=ansatz.n_params)
    H = qbm.hamiltonians.tfim(2, J=1.0, g=0.8)

    def energy(l):
        psi = ansatz.state(l)
        return float(np.real(np.vdot(psi, (H @ psi.reshape(4, 4)).reshape(-1))))

    eps = 1e-6
    grad = np.array(
        [
            (energy(lam + eps * e) - energy(lam - eps * e)) / (2 * eps)
            for e in np.eye(ansatz.n_params)
        ]
    )
    info = vq.mclachlan_system(ansatz, lam, H)
    assert np.allclose(info["C"], -0.5 * grad, atol=1e-8)


def test_A_is_the_fubini_study_metric_and_is_psd():
    ansatz = vq.tfd_ansatz(labels=TFIM_LABELS, depth=2)
    lam = np.random.default_rng(4).normal(scale=0.6, size=ansatz.n_params)
    H = qbm.hamiltonians.tfim(2, J=1.0, g=0.8)
    info = vq.mclachlan_system(ansatz, lam, H)

    D = _fd_derivatives(ansatz, lam)
    psi = ansatz.state(lam)
    ov = psi.conj() @ D
    fs = np.real(D.conj().T @ D) - np.real(np.outer(ov.conj(), ov))
    assert np.allclose(info["A"], fs, atol=1e-8)
    assert np.min(np.linalg.eigvalsh(info["A"])) > -1e-10


def test_unknown_gauge_is_refused():
    ansatz = vq.tfd_ansatz(labels=TFIM_LABELS, depth=1)
    with pytest.raises(ValueError, match="gauge"):
        vq.mclachlan_system(ansatz, np.zeros(ansatz.n_params), np.eye(4), gauge="nope")


@pytest.mark.parametrize("gauge", ["qgt", "gram"])
def test_hadamard_test_route_reproduces_the_exact_route(gauge):
    """The hardware route: A and C from measurements only, no state vector algebra."""
    ansatz = vq.tfd_ansatz(labels=TFIM_LABELS, depth=1)
    lam = np.random.default_rng(5).normal(scale=0.5, size=ansatz.n_params)
    H = _dense(TFIM_LABELS, TFIM_COEFFS)
    exact = vq.mclachlan_system(ansatz, lam, H, gauge=gauge)
    A, C = vq.measured_mclachlan_system(ansatz, lam, TFIM_LABELS, TFIM_COEFFS, gauge=gauge)
    assert np.allclose(A, exact["A"], atol=1e-12)
    assert np.allclose(C, exact["C"], atol=1e-12)


def test_mclachlan_circuits_are_gate_level():
    """Estimator circuits must export -- no opaque unitaries anywhere in the VarQITE path."""
    ansatz = vq.tfd_ansatz(labels=TFIM_LABELS, depth=1)
    circ = vq.mclachlan_circuit(ansatz, np.zeros(ansatz.n_params), 0, j=2)
    assert "unitary" not in circ.gate_counts()
    assert to_qasm3(circ).startswith("OPENQASM 3.0;")


# ---------------------------------------------------------------------------
# the driver: does it actually prepare the Gibbs state?
# ---------------------------------------------------------------------------
def test_exact_for_a_commuting_hamiltonian_and_converges_in_the_step_size():
    """With tilt partners of every term, a classical Ising TFD is exactly representable.

    The only remaining error is the first-order time discretisation, so refining the
    step size must drive it to zero -- and the (measurable) residual must already say so.
    """
    H = _dense(ISING_LABELS, ISING_COEFFS)
    ansatz = vq.tfd_ansatz(labels=ISING_LABELS, depth=1)
    errs = []
    for steps in (50, 200, 800):
        res = vq.varqite(H, ansatz, tau=0.5, steps=steps)
        assert res.residual < 1e-5  # the ansatz represents the flow exactly
        errs.append(res.infidelity())
    assert errs[-1] < 1e-6
    assert errs[0] > 4 * errs[1] > 16 * errs[2]  # first order in dt


def test_error_decreases_with_ansatz_depth_on_a_non_commuting_hamiltonian():
    H = _dense(TFIM_LABELS, TFIM_COEFFS)
    out = [
        vq.varqite(H, vq.tfd_ansatz(labels=TFIM_LABELS, depth=d), tau=0.5, steps=100)
        for d in (1, 2, 3)
    ]
    infid = [r.infidelity() for r in out]
    resid = [r.residual for r in out]
    assert infid[0] > infid[1] > infid[2]
    assert resid[0] > resid[1] > resid[2]
    assert infid[-1] < 1e-4


def test_residual_flags_an_ansatz_that_cannot_move():
    """System-only rotations leave the maximally mixed state fixed -- residual must be 1.

    This is the diagnostic doing its job: a plausible-looking ansatz with no tilt
    directions is completely stuck, and you can tell from measurements alone.
    """
    H = _dense(TFIM_LABELS, TFIM_COEFFS)
    stuck = vq.PauliRotationAnsatz([lbl + "II" for lbl in TFIM_LABELS], n_system=2)
    res = vq.varqite(H, stuck, tau=0.5, steps=20)
    assert res.residual > 0.99
    assert res.trace_distance() > 0.1
    assert np.allclose(res.density_matrix(), np.eye(4) / 4, atol=1e-10)


def test_beta_zero_is_the_maximally_mixed_state():
    res = vq.prepare_gibbs(qbm.hamiltonians.tfim(2, g=0.8), beta=0.0, depth=1, steps=1)
    assert np.allclose(res.density_matrix(), np.eye(4) / 4, atol=1e-12)


@pytest.mark.parametrize("beta", [0.25, 1.0, 3.0])
def test_prepare_gibbs_matches_the_exact_gibbs_state_across_temperatures(beta):
    H = qbm.hamiltonians.tfim(2, J=1.0, g=0.8)
    res = vq.prepare_gibbs(H, beta=beta, depth=4, steps=150)
    assert res.trace_distance() < 5e-3
    assert np.allclose(res.density_matrix(), vq.exact_gibbs(H, beta), atol=1e-2)


def test_prepare_gibbs_accepts_a_param_hamiltonian_and_matches_the_dense_backend():
    ham = ParamHamiltonian(local_pauli_generators(3))
    theta = np.random.default_rng(0).normal(scale=0.4, size=ham.n_params)
    res = vq.prepare_gibbs(ham, theta, beta=1.0, depth=3, steps=80)
    rho_dense = qbm.DenseBackend().thermal_state(ham, theta).density_matrix()
    assert res.trace_distance(rho_dense) < 5e-3
    assert "VarQITE" in res.report()


def test_param_hamiltonian_requires_theta():
    ham = ParamHamiltonian(local_pauli_generators(2))
    with pytest.raises(ValueError, match="theta"):
        vq.prepare_gibbs(ham)


def test_non_pauli_labels_give_a_clear_error_not_a_key_error():
    with pytest.raises(ValueError, match="Pauli strings over I/X/Y/Z"):
        vq.tfd_ansatz(labels=["G0", "G1"])
    with pytest.raises(ValueError, match="Pauli string"):
        vq.tilt_partner("G0")


def test_dense_generators_fall_back_to_a_generic_ansatz_with_a_warning():
    """A ParamHamiltonian built from matrices has no Pauli labels to adapt to."""
    ham = ParamHamiltonian([pauli("ZI"), pauli("IX")])
    with pytest.warns(RuntimeWarning, match="dense matrices"):
        res = vq.prepare_gibbs(ham, np.array([0.4, -0.3]), depth=2, steps=60)
    assert res.trace_distance() < 1e-3


def test_a_fixed_offset_also_triggers_the_generic_ansatz():
    """The offset's Pauli content is unknown, so a labels-only ansatz could miss it."""
    ham = ParamHamiltonian(["ZI", "IZ"], offset=0.5 * pauli("YY"))
    with pytest.warns(RuntimeWarning, match="offset"):
        res = vq.prepare_gibbs(ham, np.array([0.6, -0.2]), depth=3, steps=80)
    assert res.trace_distance() < 5e-3


def test_an_explicit_ansatz_suppresses_the_fallback_and_its_warning():
    import warnings

    ham = ParamHamiltonian(["ZI", "IZ"], offset=0.5 * pauli("YY"))
    ansatz = vq.tfd_ansatz(n=2, depth=3)  # the generic set, chosen deliberately
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = vq.prepare_gibbs(ham, np.array([0.6, -0.2]), ansatz=ansatz, steps=80)
    assert res.ansatz is ansatz
    assert res.trace_distance() < 5e-3


def test_generic_ansatz_covers_xx_and_yy_couplings():
    """The fallback must handle Heisenberg, not just Ising-like models."""
    H = qbm.hamiltonians.heisenberg(2, J=0.6)
    res = vq.prepare_gibbs(H, beta=1.0, depth=3, steps=100)
    assert res.residual < 1e-4
    assert res.trace_distance() < 5e-3


def test_rk4_is_at_least_as_accurate_as_euler_at_equal_cost():
    H = _dense(TFIM_LABELS, TFIM_COEFFS)
    ansatz = vq.tfd_ansatz(labels=TFIM_LABELS, depth=4)
    euler = vq.varqite(H, ansatz, tau=0.5, steps=200, method="euler")
    rk4 = vq.varqite(H, ansatz, tau=0.5, steps=50, method="rk4")
    assert rk4.infidelity() <= euler.infidelity()


def test_bad_arguments_are_refused():
    ansatz = vq.tfd_ansatz(labels=TFIM_LABELS, depth=1)
    with pytest.raises(ValueError, match="steps"):
        vq.varqite(np.eye(4), ansatz, tau=0.5, steps=0)
    with pytest.raises(ValueError, match="hamiltonian is"):
        vq.varqite(np.eye(8), ansatz, tau=0.5)
    with pytest.raises(ValueError, match="method"):
        vq.varqite(np.eye(4), ansatz, tau=0.5, steps=1, method="verlet")
    with pytest.raises(ValueError, match="labels"):
        vq.tfd_ansatz()


def test_history_records_a_monotone_energy_descent():
    """Imaginary time is energy-decreasing: <H> must fall along the trajectory."""
    H = _dense(TFIM_LABELS, TFIM_COEFFS)
    res = vq.varqite(H, vq.tfd_ansatz(labels=TFIM_LABELS, depth=3), tau=0.5, steps=50)
    e = res.history["energy"]
    assert np.all(np.diff(e) < 1e-12)
    assert e[0] == pytest.approx(np.trace(H).real / 4, abs=1e-10)  # maximally mixed


def test_long_imaginary_time_converges_to_the_ground_state():
    """tau -> infinity is ground-state preparation; a QBM at large beta must find it."""
    H = _dense(TFIM_LABELS, TFIM_COEFFS)
    res = vq.varqite(H, vq.tfd_ansatz(labels=TFIM_LABELS, depth=4), tau=6.0, steps=300)
    assert res.energy == pytest.approx(np.linalg.eigvalsh(H)[0], abs=1e-3)


# ---------------------------------------------------------------------------
# integration with the circuit backend
# ---------------------------------------------------------------------------
def test_circuit_backend_with_varqite_preparation_matches_dense():
    from qbm.backends.circuit import CircuitBackend

    ham = ParamHamiltonian(local_pauli_generators(2))
    theta = np.random.default_rng(0).normal(scale=0.4, size=ham.n_params)
    dense = qbm.DenseBackend().thermal_state(ham, theta)
    st = CircuitBackend(
        seed=0, gibbs_prep="varqite", varqite_options={"depth": 4, "steps": 120}
    ).thermal_state(ham, theta)

    O = qbm.hamiltonians.tfim(2, g=1.2)
    assert st.expect(O) == pytest.approx(dense.expect(O), abs=2e-3)
    assert np.allclose(st.generator_expectations(), dense.generator_expectations(), atol=2e-3)
    assert st.varqite_result().residual < 1e-3


def test_varqite_preparation_circuit_is_gate_level_unlike_exact_synthesis():
    """The point of VarQITE: a circuit a device can run, not an opaque state-prep unitary."""
    from qbm.backends.circuit import CircuitBackend

    ham = ParamHamiltonian(local_pauli_generators(2))
    theta = np.zeros(ham.n_params)
    var = CircuitBackend(gibbs_prep="varqite", varqite_options={"steps": 5}).thermal_state(
        ham, theta
    )
    assert "unitary" not in var.preparation_circuit().gate_counts()
    assert to_qasm3(var.preparation_circuit()).startswith("OPENQASM 3.0;")

    exact = CircuitBackend().thermal_state(ham, theta)
    assert "unitary" in exact.preparation_circuit().gate_counts()
    with pytest.raises(ValueError, match="opaque"):
        to_qasm3(exact.preparation_circuit())


def test_varqite_result_is_only_available_for_that_strategy():
    from qbm.backends.circuit import CircuitBackend

    ham = ParamHamiltonian(local_pauli_generators(2))
    st = CircuitBackend().thermal_state(ham, np.zeros(ham.n_params))
    with pytest.raises(ValueError, match="gibbs_prep='varqite'"):
        st.varqite_result()


def test_resource_estimate_includes_the_preparation_cost():
    from qbm.backends.circuit import CircuitBackend

    ham = ParamHamiltonian(local_pauli_generators(2))
    st = CircuitBackend(
        shots=100, gibbs_prep="varqite", varqite_options={"depth": 1, "steps": 5}
    ).thermal_state(ham, np.zeros(ham.n_params))
    r = st.resource_estimate()
    assert r["circuits_for_preparation"] > 0
    assert r["prep_gates"] > 0
