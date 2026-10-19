"""Pauli propagation: sparse-Pauli imaginary-time thermal-state simulation.

Independent-route checks: the symplectic Pauli algebra against dense matrices, the
imaginary-time state against the exact Gibbs state, the sampler against the exact
diagonal, and end-to-end training against the dense backend.

References: arXiv:2602.04878 (thermal-state Pauli propagation) and *Sampling from Thermal
Quantum States via Pauli Propagation* (the QBM-training application).
"""

import math

import numpy as np
import pytest

import qbm
from qbm import pauli_prop as pp
from qbm.backends.pauli_propagation import PauliPropagationBackend
from qbm.operators import ParamHamiltonian, pauli, pauli_pool

RNG = np.random.default_rng(0)


def _exact_gibbs(labels, coeffs, beta=1.0):
    H = sum(c * pauli(l) for c, l in zip(coeffs, labels))
    w, V = np.linalg.eigh(H)
    p = np.exp(-beta * (w - w[0]))
    return (V * (p / p.sum())) @ V.conj().T


def _trace_distance(a, b):
    return float(0.5 * np.sum(np.abs(np.linalg.eigvalsh(a - b))))


# ---------------------------------------------------------------------------
# the symplectic Pauli algebra
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(6))
def test_pauli_product_phase_matches_dense_matrices(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 5))
    la = "".join(rng.choice(list("IXYZ"), n))
    lb = "".join(rng.choice(list("IXYZ"), n))
    x, z, phase = pp.pauli_product(pp.label_to_xz(la), pp.label_to_xz(lb))
    got = phase * pauli(pp.xz_to_label(x, z, n))
    assert np.allclose(got, pauli(la) @ pauli(lb), atol=1e-12)


@pytest.mark.parametrize("seed", range(6))
def test_commutation_predicate_matches_dense_matrices(seed):
    rng = np.random.default_rng(seed + 100)
    n = int(rng.integers(1, 5))
    la = "".join(rng.choice(list("IXYZ"), n))
    lb = "".join(rng.choice(list("IXYZ"), n))
    A, B = pauli(la), pauli(lb)
    matrices_commute = np.allclose(A @ B, B @ A)
    assert pp.commutes(pp.label_to_xz(la), pp.label_to_xz(lb)) == matrices_commute


def test_label_round_trip():
    for lbl in ["I", "XYZ", "IZXY", "ZZII"]:
        x, z = pp.label_to_xz(lbl)
        assert pp.xz_to_label(x, z, len(lbl)) == lbl


def test_pauli_weight():
    assert pp.pauli_weight(pp.label_to_xz("IXIZ")) == 2
    assert pp.pauli_weight(pp.label_to_xz("IIII")) == 0


# ---------------------------------------------------------------------------
# imaginary-time thermal state
# ---------------------------------------------------------------------------
def test_identity_start_is_the_infinite_temperature_state():
    ps = pp.thermal_state(["ZZ", "XI"], [0.0, 0.0], beta=1.0, trotter_steps=1)
    assert np.allclose(ps.to_matrix(), np.eye(4) / 4, atol=1e-14)


def test_exact_for_commuting_hamiltonians_at_any_trotter_depth():
    """A classical (all-Z) Hamiltonian branches without approximation -> exact at L=1."""
    labels, coeffs = ["ZII", "IZI", "IIZ", "ZZI", "IZZ"], [0.7, -0.4, 0.5, 0.9, -0.3]
    exact = _exact_gibbs(labels, coeffs)
    for L in (1, 4):
        rho = pp.thermal_state(labels, coeffs, trotter_steps=L, coeff_cutoff=0.0).to_matrix()
        assert _trace_distance(rho, exact) < 1e-12


def test_single_qubit_matches_tanh_law():
    rho = pp.thermal_state(["Z"], [0.8], trotter_steps=64, coeff_cutoff=0.0).to_matrix()
    assert np.allclose(rho, (np.eye(2) - np.tanh(0.8) * pauli("Z")) / 2, atol=1e-6)


def test_trotter_error_is_first_order_for_noncommuting_hamiltonians():
    labels, coeffs = ["ZZ", "XI", "IX"], [-1.0, -0.8, -0.8]
    exact = _exact_gibbs(labels, coeffs)
    errs = {
        L: _trace_distance(
            pp.thermal_state(labels, coeffs, trotter_steps=L, coeff_cutoff=0.0).to_matrix(), exact
        )
        for L in (16, 32, 64)
    }
    assert errs[16] > errs[32] > errs[64]
    assert errs[16] / errs[32] == pytest.approx(2.0, abs=0.15)  # halves per doubling


def test_matches_dense_backend_at_high_depth():
    ham = ParamHamiltonian(pauli_pool(3, terms=("Z", "X", "ZZ")))
    theta = RNG.normal(scale=0.4, size=ham.n_params)
    dense = qbm.DenseBackend().thermal_state(ham, theta).density_matrix()
    ps = pp.thermal_state(ham.labels, theta, trotter_steps=256, coeff_cutoff=0.0)
    assert _trace_distance(ps.to_matrix(), dense) < 5e-4


def test_truncation_error_decreases_as_the_cutoff_tightens():
    ham = ParamHamiltonian(pauli_pool(4, terms=("Z", "X", "ZZ")))
    theta = RNG.normal(scale=0.5, size=ham.n_params)
    dense = qbm.DenseBackend().thermal_state(ham, theta).density_matrix()
    errs, terms = [], []
    for cutoff in (1e-1, 1e-2, 1e-4, 1e-8):
        ps = pp.thermal_state(ham.labels, theta, trotter_steps=64, coeff_cutoff=cutoff)
        errs.append(_trace_distance(ps.to_matrix(), dense))
        terms.append(ps.n_terms)
    assert errs[0] > errs[-1]  # tighter cutoff -> smaller error
    assert terms[0] < terms[-1]  # ...bought with more retained terms


def test_non_pauli_generators_are_refused():
    with pytest.raises(ValueError, match="Pauli-string generators"):
        pp.thermal_state(["G0", "G1"], [0.1, 0.2])


# ---------------------------------------------------------------------------
# read-outs and the sampler
# ---------------------------------------------------------------------------
def test_generator_expectations_match_dense():
    ham = ParamHamiltonian(pauli_pool(3, terms=("Z", "X", "ZZ")))
    theta = RNG.normal(scale=0.4, size=ham.n_params)
    dense = qbm.DenseBackend().thermal_state(ham, theta)
    ps = pp.thermal_state(ham.labels, theta, trotter_steps=256, coeff_cutoff=0.0)
    got = np.array([ps.expect_pauli(lbl) for lbl in ham.labels])
    assert np.allclose(got, dense.generator_expectations(), atol=5e-4)


def test_probabilities_sum_to_one_and_match_the_diagonal():
    ham = ParamHamiltonian(pauli_pool(3, terms=("Z", "X", "ZZ")))
    theta = RNG.normal(scale=0.4, size=ham.n_params)
    ps = pp.thermal_state(ham.labels, theta, trotter_steps=128, coeff_cutoff=0.0)
    probs = ps.probabilities()
    assert probs.sum() == pytest.approx(1.0, abs=1e-10)
    assert np.allclose(probs, np.real(np.diag(ps.to_matrix())), atol=1e-12)


def test_sampler_reproduces_the_exact_diagonal_distribution():
    """Algorithm 1: the locally normalised sampler converges to the Born distribution."""
    ham = ParamHamiltonian(pauli_pool(3, terms=("Z", "X", "ZZ")))
    theta = RNG.normal(scale=0.5, size=ham.n_params)
    ps = pp.thermal_state(ham.labels, theta, trotter_steps=128, coeff_cutoff=0.0)
    exact = ps.probabilities()
    samples = ps.sample(60000, rng=np.random.default_rng(0))
    emp = np.bincount(samples, minlength=8) / 60000
    assert 0.5 * np.sum(np.abs(emp - exact)) < 1e-2  # TVD small at 60k shots


def test_exact_pointwise_likelihood_is_consistent_with_the_sampler():
    """log_likelihood over all basis states must give a normalised distribution."""
    ham = ParamHamiltonian(pauli_pool(3, terms=("Z", "X", "ZZ")))
    theta = RNG.normal(scale=0.4, size=ham.n_params)
    ps = pp.thermal_state(ham.labels, theta, trotter_steps=128, coeff_cutoff=0.0)
    p = np.exp(ps.log_likelihood(np.arange(8)))
    assert p.sum() == pytest.approx(1.0, abs=1e-9)
    # with no truncation the sampler's distribution IS the diagonal
    assert np.allclose(p, ps.probabilities(), atol=1e-9)


def test_spectral_negativity_is_zero_without_truncation_and_grows_with_it():
    ham = ParamHamiltonian(pauli_pool(4, terms=("Z", "X", "ZZ")))
    theta = RNG.normal(scale=0.6, size=ham.n_params)
    clean = pp.thermal_state(ham.labels, theta, trotter_steps=64, coeff_cutoff=0.0)
    truncated = pp.thermal_state(ham.labels, theta, trotter_steps=64, coeff_cutoff=1e-2)
    assert clean.spectral_negativity() < 1e-9
    assert truncated.spectral_negativity() >= clean.spectral_negativity()


# ---------------------------------------------------------------------------
# truncation acts after every gate
# ---------------------------------------------------------------------------
def _n_paulis_up_to_weight(n, weight):
    return sum(math.comb(n, k) * 3**k for k in range(weight + 1))


def _chain(n, seed=0, scale=0.3):
    labels = pauli_pool(n, terms=("Z", "X", "ZZ"), connectivity="chain")
    return labels, np.random.default_rng(seed).normal(scale=scale, size=len(labels))


def test_the_gate_never_creates_a_string_above_the_weight_cap():
    start = {pp.label_to_xz("II"): 1.0, pp.label_to_xz("ZI"): 0.5}
    capped, free = dict(start), dict(start)
    pp._apply_ite_gate(capped, pp.label_to_xz("IZ"), 0.3, max_weight=1)
    pp._apply_ite_gate(free, pp.label_to_xz("IZ"), 0.3)
    labels = lambda coeffs: {pp.xz_to_label(x, z, 2) for x, z in coeffs}  # noqa: E731
    assert labels(free) == {"II", "ZI", "IZ", "ZZ"}  # ZI commutes with IZ and spawns ZZ
    assert labels(capped) == {"II", "ZI", "IZ"}  # ... which the cap never lets exist
    for key in capped:  # everything that does exist is treated exactly as without a cap
        assert capped[key] == free[key]


def test_the_working_set_is_bounded_like_the_result():
    # Truncating once per Trotter layer let the sum grow through every gate in between:
    # on this chain it peaked at 15,760 strings in order to return 321.  Per gate, the
    # weight cap holds at every moment, so the peak obeys the same bound as the result.
    n, cap = 8, 3
    labels, theta = _chain(n)
    ps = pp.thermal_state(labels, theta, trotter_steps=8, coeff_cutoff=1e-3, max_weight=cap)
    assert all(pp.pauli_weight(key) <= cap for key in ps.coeffs)
    assert ps.n_terms <= ps.peak_terms <= _n_paulis_up_to_weight(n, cap)  # 1,789
    assert ps.peak_terms < 2 * ps.n_terms  # measured 1.3


@pytest.mark.parametrize("n", [12, 16, 20])
def test_cost_follows_the_retained_terms_beyond_the_dense_ceiling(n):
    # sizes that used not to finish: the 10-qubit version of this state took 18 s and
    # held 184k strings; a dense generator at 20 qubits would be 16 TiB
    labels, theta = _chain(n)
    ps = pp.thermal_state(labels, theta, trotter_steps=8, coeff_cutoff=1e-3, max_weight=3)
    assert ps.peak_terms < 2 * ps.n_terms
    assert ps.n_terms < 150 * n  # grows with the system, not with 4^n (measured ~105 n)
    assert np.isclose(ps.coeffs[(0, 0)] * 2**n, 1.0)  # unit trace
    weights = np.array([pp.pauli_weight(key) for key in ps.coeffs])
    assert weights.max() == 3


def test_accuracy_on_a_classical_chain_past_the_dense_ceiling():
    # 16 spins, where the exact moments are still a sum over 2^16 configurations.  A
    # commuting Hamiltonian has no Trotter error, so one layer isolates the truncation.
    n = 16
    rng = np.random.default_rng(0)
    h, J = rng.normal(scale=0.5, size=n), rng.normal(scale=0.5, size=n - 1)
    v = np.arange(2**n)
    spins = 1 - 2 * ((v[:, None] >> np.arange(n - 1, -1, -1)) & 1)
    energy = spins @ h + (spins[:, :-1] * spins[:, 1:]) @ J
    q = np.exp(-(energy - energy.min()))
    q /= q.sum()
    exact = np.r_[spins.T @ q, (spins[:, :-1] * spins[:, 1:]).T @ q]  # <Z_i>, <Z_i Z_i+1>

    labels = pauli_pool(n, terms=("Z", "ZZ"), connectivity="chain")
    errors = []
    for cutoff in (1e-3, 1e-4):
        ps = pp.thermal_state(labels, np.r_[h, J], trotter_steps=1, coeff_cutoff=cutoff)
        got = np.array([ps.expect_pauli(lbl) for lbl in labels])
        errors.append(np.max(np.abs(got - exact)))
        assert ps.peak_terms < 2 * ps.n_terms
    assert errors[0] < 5e-3 and errors[1] < 5e-4  # measured 1.2e-3 and 1.0e-4
    assert errors[1] < errors[0]


def test_a_cutoff_above_the_entry_size_leaves_the_identity():
    # a gate brings in its string at tanh(|theta_k| / L) relative to the identity; a
    # cutoff above the largest of those lets nothing in, however many layers follow
    labels, theta = _chain(4, seed=3, scale=0.5)
    layers = 16
    entry = np.tanh(np.abs(theta).max() / layers)
    stuck = pp.thermal_state(labels, theta, trotter_steps=layers, coeff_cutoff=1.01 * entry)
    assert stuck.n_terms == 1 and stuck.peak_terms == 2  # one candidate at a time, refused
    moving = pp.thermal_state(labels, theta, trotter_steps=layers, coeff_cutoff=0.5 * entry)
    assert moving.n_terms > 1


def test_the_default_cutoff_is_as_good_as_none():
    labels, theta = _chain(4, seed=1, scale=0.5)
    default = pp.thermal_state(labels, theta, trotter_steps=32)
    untruncated = pp.thermal_state(labels, theta, trotter_steps=32, coeff_cutoff=0.0)
    assert _trace_distance(default.to_matrix(), untruncated.to_matrix()) < 1e-8


def test_peak_terms_is_reported_and_survives_a_copy():
    labels, theta = _chain(5)
    ham = ParamHamiltonian(labels)
    state = PauliPropagationBackend(trotter_steps=8, coeff_cutoff=1e-3).thermal_state(ham, theta)
    estimate = state.resource_estimate()
    assert estimate["peak_pauli_terms"] == state.peak_terms >= estimate["retained_pauli_terms"]
    assert state.pauli_sum().copy().peak_terms == state.peak_terms
    assert pp.PauliSum(3).peak_terms is None  # not built by propagation


# ---------------------------------------------------------------------------
# the distribution a truncated state generates
# ---------------------------------------------------------------------------
def _non_positive_state():
    labels = pauli_pool(5, terms=("Z", "X", "ZZ"))
    theta = np.random.default_rng(1).normal(scale=0.6, size=len(labels))
    ps = pp.thermal_state(labels, theta, trotter_steps=12, coeff_cutoff=3e-2)
    assert ps.probabilities().min() < 0  # truncation has pushed rho out of the positive cone
    return ParamHamiltonian(labels), theta, ps


def test_sampler_distribution_is_the_chain_rule_the_sampler_uses():
    _, _, ps = _non_positive_state()
    p_hat = ps.sampler_probabilities()
    assert np.all(p_hat >= 0) and np.isclose(p_hat.sum(), 1.0, atol=1e-12)
    assert np.allclose(p_hat, np.exp(ps.log_likelihood(np.arange(32))), atol=1e-14)
    draws = ps.sample(30_000, rng=np.random.default_rng(0))  # the per-bit sampler is slow
    empirical = np.bincount(draws, minlength=32) / draws.size
    assert 0.5 * np.abs(empirical - p_hat).sum() < 0.02  # finite-sample floor ~0.01
    assert 0.5 * np.abs(ps.probabilities() - p_hat).sum() > 0.05  # ...and it is not the diagonal


def test_sampler_distribution_is_the_diagonal_when_that_is_a_distribution():
    labels = pauli_pool(4, terms=("Z", "X", "ZZ"))
    theta = np.random.default_rng(2).normal(scale=0.6, size=len(labels))
    ps = pp.thermal_state(labels, theta, trotter_steps=16, coeff_cutoff=0.0)
    assert ps.probabilities().min() > 0
    assert np.allclose(ps.sampler_probabilities(), ps.probabilities(), atol=1e-14)


def test_a_state_reports_the_distribution_it_samples_from():
    # The backend's probabilities() are the sampler's, so everything built on them --
    # model.kl, the KL monitor of qbm.learn, hidden-unit marginals -- sees a distribution.
    # The diagonal of rho stays one call away, on the Pauli sum.
    ham, theta, ps = _non_positive_state()
    state = PauliPropagationBackend(trotter_steps=12, coeff_cutoff=3e-2).thermal_state(ham, theta)
    assert np.allclose(state.probabilities(), ps.sampler_probabilities(), atol=1e-15)
    assert np.allclose(state.pauli_sum().probabilities(), ps.probabilities(), atol=1e-15)
    assert np.allclose(np.exp(state.log_likelihood(np.arange(32))), state.probabilities())


def test_hidden_unit_marginal_is_a_distribution_under_truncation():
    backend = PauliPropagationBackend(trotter_steps=12, coeff_cutoff=3e-2, seed=0)
    model = qbm.VisibleHiddenQBM(n_visible=3, n_hidden=2, backend=backend)
    model.theta = np.random.default_rng(4).normal(scale=0.8, size=model.n_params)
    state = model.state()
    diagonal_marginal = state.pauli_sum().probabilities().reshape(8, 4).sum(axis=1)
    assert diagonal_marginal.min() < 0  # what the visible marginal used to be
    p_visible = model.visible_probabilities()
    assert p_visible.min() >= 0 and np.isclose(p_visible.sum(), 1.0)
    draws = state.sample(20_000, rng=np.random.default_rng(1)) >> 2  # keep the visible bits
    empirical = np.bincount(draws, minlength=8) / draws.size
    assert 0.5 * np.abs(empirical - p_visible).sum() < 0.02


# ---------------------------------------------------------------------------
# the backend and end-to-end training
# ---------------------------------------------------------------------------
def test_backend_is_registered_and_needs_no_sdk():
    assert "pauli_propagation" in qbm.available_backends()
    assert "pauli_propagation" in qbm.available("backend")
    assert qbm.get_backend("pauli_propagation").name == "pauli_propagation"
    assert qbm.get_backend("pauli").name == "pauli_propagation"  # short alias


def test_backend_matches_dense_on_measurable_quantities():
    ham = ParamHamiltonian(pauli_pool(2, terms=("Z", "X", "ZZ")))
    theta = RNG.normal(scale=0.4, size=ham.n_params)
    dense = qbm.DenseBackend().thermal_state(ham, theta)
    st = PauliPropagationBackend(trotter_steps=256, coeff_cutoff=0.0).thermal_state(ham, theta)
    O = qbm.hamiltonians.tfim(2, g=1.2)
    assert st.expect(O) == pytest.approx(dense.expect(O), abs=5e-4)
    assert np.allclose(st.generator_expectations(), dense.generator_expectations(), atol=5e-4)
    assert np.allclose(st.probabilities(), dense.probabilities(), atol=5e-4)


def test_backend_refuses_spectrum_and_channel_quantities():
    ham = ParamHamiltonian(pauli_pool(2, terms=("Z", "X", "ZZ"), connectivity="chain"))
    st = PauliPropagationBackend().thermal_state(ham, np.zeros(ham.n_params))
    for call in (
        lambda: st.metric(),
        st.log_partition,
        st.entropy,
        lambda: st.observable_gradient(np.eye(ham.dim)),
        st.state_derivatives,
        st.diagonal_gradient,
    ):
        with pytest.raises(NotImplementedError):
            call()


def test_offset_hamiltonian_is_refused():
    ham = ParamHamiltonian(["ZI", "IZ"], offset=0.5 * pauli("YY"))
    with pytest.raises(NotImplementedError, match="offset"):
        PauliPropagationBackend().thermal_state(ham, np.zeros(ham.n_params))


def test_generative_training_matches_dense_training():
    """The headline: train a QBM via Pauli propagation with the unchanged qbm.learn API."""
    q = np.array([0.50, 0.20, 0.05, 0.25])
    trained = qbm.learn(
        q, steps=50, lr=0.15, backend=PauliPropagationBackend(trotter_steps=32, coeff_cutoff=1e-8)
    )
    reference = qbm.learn(q, steps=50, lr=0.15)
    assert trained.history.monitor[-1] < 5e-3
    assert np.allclose(trained.probabilities(), q, atol=2e-2)
    assert np.allclose(trained.probabilities(), reference.probabilities(), atol=5e-3)
    # the relative-entropy value needs log Z (refused) -> nan; training rides the gradient
    assert np.all(np.isnan(trained.history.loss))


def test_trained_model_samples_reproduce_the_target():
    q = np.array([0.50, 0.20, 0.05, 0.25])
    model = qbm.learn(
        q,
        steps=60,
        lr=0.15,
        backend=PauliPropagationBackend(trotter_steps=32, coeff_cutoff=1e-8, seed=0),
    )
    samples = model.state().sample(40000)
    emp = np.bincount(samples, minlength=4) / 40000
    assert 0.5 * np.sum(np.abs(emp - q)) < 2e-2


def test_resource_estimate_reports_the_term_count():
    ham = ParamHamiltonian(pauli_pool(3, terms=("Z", "X", "ZZ")))
    theta = RNG.normal(scale=0.4, size=ham.n_params)
    est = PauliPropagationBackend(trotter_steps=16).thermal_state(ham, theta).resource_estimate()
    assert est["retained_pauli_terms"] > 0
    assert est["trotter_steps"] == 16
    assert est["n_qubits"] == 3


def test_truncation_trades_retained_terms_for_training_accuracy():
    """The whole point of the method: looser truncation is cheaper but trains worse.

    A moderate cutoff should reach (nearly) the same KL as a near-untruncated run while
    retaining fewer Pauli strings; a very aggressive cutoff should do measurably worse.
    """
    target = qbm.FullyVisibleQBM(n=4, connectivity="all")
    target.theta = np.random.default_rng(11).normal(scale=0.7, size=target.n_params)
    q = target.probabilities()
    mask = q > 0

    def train(cutoff):
        model = qbm.learn(
            q,
            steps=70,
            connectivity="all",
            optimizer=qbm.optim.GradientDescent(lr=0.1, clip=1.0),
            backend=PauliPropagationBackend(trotter_steps=24, coeff_cutoff=cutoff, seed=0),
        )
        p = np.clip(model.probabilities(), 1e-300, None)
        final_kl = float(np.sum(q[mask] * (np.log(q[mask]) - np.log(p[mask]))))
        return final_kl, model.state().n_terms

    kl_moderate, terms_moderate = train(1e-3)
    kl_tight, terms_tight = train(1e-6)
    kl_aggressive, terms_aggressive = train(3e-1)

    # moderate truncation: essentially full accuracy, at a real term saving
    assert kl_moderate < 5e-2
    assert kl_moderate < 3 * kl_tight
    assert terms_moderate < terms_tight
    # aggressive truncation: cheaper still, but clearly worse training
    assert terms_aggressive < terms_moderate
    assert kl_aggressive > 10 * kl_moderate
