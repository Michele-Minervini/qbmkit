"""The Gibbs map: exact hidden-unit marginalisation, and the removal of the CD bias.

The load-bearing claim is that ``GibbsMapNLL.grad`` equals the exact dense
``MarginalNLL.grad`` for **non-commuting** hidden operators while needing only
``generator_expectations()`` -- so hidden-unit training runs on the scalable backends.
Checked against two independent routes: the dense Frechet-derivative gradient, and the
closed-form semi-quantum RBM.
"""

import numpy as np
import pytest

import qbm
from qbm.gibbs_map import GibbsMap
from qbm.losses import GibbsMapNLL, MarginalNLL
from qbm.operators import ParamHamiltonian, rbm_generators
from qbm.sampling import block_gibbs_sample

NV, NH = 3, 2


def _model(hidden_paulis=("Z", "X"), backend=None, seed=1, nv=NV, nh=NH):
    m = qbm.VisibleHiddenQBM(
        n_visible=nv, n_hidden=nh, hidden_paulis=hidden_paulis, backend=backend
    )
    m.theta = np.random.default_rng(seed).normal(scale=0.6, size=m.n_params)
    return m


def _target(nv=NV, seed=0):
    q = np.random.default_rng(seed).random(1 << nv)
    return q / q.sum()


# ---------------------------------------------------------------------------
# the map itself
# ---------------------------------------------------------------------------
def test_visible_register_is_block_diagonal_so_the_marginal_is_exact():
    m = _model()
    gmap = GibbsMap(m.ham, NV)
    dense_marginal = m.state().probabilities().reshape(1 << NV, 1 << NH).sum(axis=1)
    assert np.allclose(gmap.marginal(m.theta), dense_marginal, atol=1e-12)


def test_marginal_matches_the_independent_closed_form_sqrbm():
    """Cross-check against a completely different implementation (cosh/tanh closed form)."""
    sq = qbm.SemiQuantumRBM(n_visible=NV, n_hidden=NH, hidden_paulis=("X", "Z"))
    sq.theta = np.random.default_rng(3).normal(scale=0.7, size=sq.n_params)
    gmap = GibbsMap(sq.to_hamiltonian(), NV)
    assert np.allclose(gmap.marginal(sq.theta), sq.visible_probabilities(), atol=1e-12)


def test_conditional_hidden_state_is_a_valid_density_matrix():
    m = _model()
    rho_v = GibbsMap(m.ham, NV).conditional_hidden_state(m.theta, 5)
    assert rho_v.shape == (1 << NH, 1 << NH)
    assert np.trace(rho_v).real == pytest.approx(1.0, abs=1e-12)
    assert np.allclose(rho_v, rho_v.conj().T, atol=1e-12)
    assert np.min(np.linalg.eigvalsh(rho_v)) > -1e-12


def test_log_partition_matches_the_dense_backend():
    m = _model()
    assert GibbsMap(m.ham, NV).log_partition(m.theta) == pytest.approx(
        m.state().log_partition(), abs=1e-10
    )


# ---------------------------------------------------------------------------
# the gradient -- the core claim
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("hidden_paulis", [("Z",), ("Z", "X"), ("X", "Y", "Z")])
def test_gradient_equals_the_exact_dense_gradient(hidden_paulis):
    """Exact for commuting *and* non-commuting hidden operators -- unlike CD."""
    m = _model(hidden_paulis=hidden_paulis)
    q = _target()
    exact = MarginalNLL(q, n_visible=NV).grad(m.state())
    got = GibbsMapNLL(q, n_visible=NV).grad(m.state())
    assert np.allclose(got, exact, atol=1e-12)


def test_value_equals_the_exact_dense_value():
    m, q = _model(), _target()
    assert GibbsMapNLL(q, n_visible=NV).value(m.state()) == pytest.approx(
        MarginalNLL(q, n_visible=NV).value(m.state()), abs=1e-10
    )


def test_gradient_accepts_raw_samples_and_only_visits_distinct_configs():
    """The positive phase costs O(distinct configs), not O(2^n_visible)."""
    m = _model()
    q = _target()
    rng = np.random.default_rng(0)
    samples = rng.choice(1 << NV, size=20000, p=q)
    from_samples = GibbsMapNLL(samples, n_visible=NV).grad(m.state())
    empirical = np.bincount(samples, minlength=1 << NV) / len(samples)
    from_dist = GibbsMapNLL(empirical, n_visible=NV).grad(m.state())
    assert np.allclose(from_samples, from_dist, atol=1e-12)


# ---------------------------------------------------------------------------
# what it unlocks: hidden units on the scalable backends
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "backend,tol",
    [(None, 1e-12), ("statevector", 1e-12), ("circuit", 1e-10), ("pauli_propagation", 5e-3)],
)
def test_hidden_unit_gradient_runs_on_every_backend(backend, tol):
    reference = _model()
    q = _target()
    exact = MarginalNLL(q, n_visible=NV).grad(reference.state())

    be = None if backend is None else qbm.get_backend(backend)
    m = _model(backend=be)
    got = GibbsMapNLL(q, n_visible=NV).grad(m.state())
    assert np.allclose(got, exact, atol=tol)


def test_marginal_nll_still_refuses_on_scalable_backends_but_points_at_the_fix():
    """The old route must fail loudly (not with a bare AttributeError) and name the new one."""
    for backend in ("pauli_propagation", "circuit", "tensor_network"):
        pytest.importorskip("quimb") if backend == "tensor_network" else None
        m = _model(backend=qbm.get_backend(backend))
        with pytest.raises(NotImplementedError, match="GibbsMapNLL"):
            MarginalNLL(_target(), n_visible=NV).grad(m.state())


def test_training_a_hidden_unit_model_on_a_scalable_backend():
    """End to end: a non-commuting hidden-unit QBM trained via Pauli propagation."""
    nv, nh = 3, 2
    target = _model(seed=4, nv=nv, nh=nh)
    q = target.state().probabilities().reshape(1 << nv, 1 << nh).sum(axis=1)

    be = qbm.get_backend("pauli_propagation", trotter_steps=48, coeff_cutoff=1e-8)
    m = qbm.VisibleHiddenQBM(n_visible=nv, n_hidden=nh, hidden_paulis=("Z", "X"), backend=be)
    m.theta = np.random.default_rng(0).normal(scale=0.05, size=m.n_params)
    loss = GibbsMapNLL(q, n_visible=nv)
    history = qbm.fit(m, loss, qbm.optim.Adam(lr=0.12), steps=60)

    learned = loss.marginal(m.state())
    kl = float(np.sum(q * (np.log(q) - np.log(np.clip(learned, 1e-300, None)))))
    assert kl < 5e-2
    assert history.loss[-1] < history.loss[0]  # the NLL value is available here too


# ---------------------------------------------------------------------------
# unbiased sampling (what "removes the CD bias" means)
# ---------------------------------------------------------------------------
def test_sampler_is_unbiased_where_contrastive_divergence_is_not():
    """The headline: block-Gibbs CD carries a residual bias for non-commuting hidden units.

    The Gibbs-map chain traces the hidden register out exactly instead of sampling it, so
    it lands at the finite-sample floor while CD does not.
    """
    sq = qbm.SemiQuantumRBM(n_visible=NV, n_hidden=NH, hidden_paulis=("X", "Z"))
    sq.theta = np.random.default_rng(3).normal(scale=0.7, size=sq.n_params)
    exact = sq.visible_probabilities()
    n = 40000

    def tvd(p):
        return 0.5 * float(np.sum(np.abs(p - exact)))

    vk = block_gibbs_sample(
        sq,
        np.random.default_rng(1).integers(0, 2, size=(n, NV)),
        k=40,
        rng=np.random.default_rng(0),
    )
    p_cd = np.bincount(vk @ (1 << np.arange(NV - 1, -1, -1)), minlength=1 << NV) / n

    gmap = GibbsMap(sq.to_hamiltonian(), NV)
    p_map = (
        np.bincount(
            gmap.sample(sq.theta, n, sweeps=12, rng=np.random.default_rng(0)), minlength=1 << NV
        )
        / n
    )

    assert tvd(p_map) < 0.02  # essentially the i.i.d. sampling floor
    assert tvd(p_cd) > 4 * tvd(p_map)  # CD is measurably biased


def test_sampler_reproduces_the_exact_marginal():
    m = _model()
    gmap = GibbsMap(m.ham, NV)
    exact = gmap.marginal(m.theta)
    s = gmap.sample(m.theta, 40000, sweeps=12, rng=np.random.default_rng(0))
    emp = np.bincount(s, minlength=1 << NV) / len(s)
    assert 0.5 * np.sum(np.abs(emp - exact)) < 0.02


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def test_non_diagonal_visible_register_is_refused():
    """X/Y on a visible qubit breaks the block structure -- must fail clearly."""
    ham = ParamHamiltonian(["XII", "IZZ"])
    with pytest.raises(ValueError, match="diagonal"):
        GibbsMap(ham, n_visible=1)


def test_non_pauli_generators_are_refused():
    ham = ParamHamiltonian([qbm.pauli("ZII"), qbm.pauli("IZZ")])
    with pytest.raises(ValueError, match="Pauli-string generators"):
        GibbsMap(ham, n_visible=1)


def test_bad_n_visible_is_refused():
    ham = ParamHamiltonian(rbm_generators(2, 1))
    for bad in (0, 3, 5):
        with pytest.raises(ValueError, match="n_visible"):
            GibbsMap(ham, n_visible=bad)


def test_offset_is_refused():
    ham = ParamHamiltonian(rbm_generators(2, 1), offset=0.3 * qbm.pauli("ZZZ"))
    with pytest.raises(NotImplementedError, match="offset"):
        GibbsMap(ham, n_visible=2)
