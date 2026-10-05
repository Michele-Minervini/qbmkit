"""Training without dense matrices.

``ParamHamiltonian`` keeps Pauli generators as labels until something reads
``.generators``.  That only helps if nothing on the training path reads it: these tests
pin the label route of the relative-entropy loss and of ``qbm.learn``, so a backend that
works from labels (tensor network, Pauli propagation) is not capped by a ``4^n`` object
built elsewhere.
"""

import numpy as np
import pytest
import scipy.linalg as sla

import qbm
from qbm import pauli_prop as pp
from qbm.losses import RelativeEntropy
from qbm.operators import ParamHamiltonian, pauli_pool

TFIM = ("Z", "X", "ZZ")


@pytest.fixture
def no_dense_generators(monkeypatch):
    """Tripwire: reading ``ParamHamiltonian.generators`` fails the test."""

    def tripped(self):
        raise AssertionError("dense generators were materialised")

    monkeypatch.setattr(ParamHamiltonian, "generators", property(tripped))


def _random_state(dim, seed):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    sigma = A @ A.conj().T
    return sigma / np.trace(sigma).real


def _random_distribution(dim, seed, zeros=()):
    q = np.random.default_rng(seed).random(dim)
    q[list(zeros)] = 0.0
    return q / q.sum()


# ---------------------------------------------------------------------------
# the Hamiltonian itself
# ---------------------------------------------------------------------------
def test_construction_allocates_nothing_dense():
    # one dense generator at 20 qubits is 16 TiB; building 59 of them is not an option
    ham = ParamHamiltonian(pauli_pool(20, terms=TFIM, connectivity="chain"))
    assert ham._mats is None
    assert (ham.n_qubits, ham.n_params, ham.dim) == (20, 59, 2**20)
    assert ham.pauli_labels == ham.labels


def test_pauli_labels_reports_whether_the_label_route_applies():
    assert ParamHamiltonian(["ZI", "xx"]).pauli_labels == ["ZI", "XX"]
    # a matrix generator has no label to work from, whatever name it was given
    mixed = ParamHamiltonian(["ZI", qbm.pauli("XX")], labels=["ZI", "XX"])
    assert mixed.pauli_labels is None


# ---------------------------------------------------------------------------
# target expectations from labels
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_quantum_target_expectations_match_the_dense_trace(seed, no_dense_generators):
    n = 4
    labels = pauli_pool(n, locality=3)  # 174 strings, most of them containing Y
    sigma = _random_state(2**n, seed)
    state = qbm.get_backend("pauli_propagation").thermal_state(
        ParamHamiltonian(labels), np.zeros(len(labels))
    )
    got = RelativeEntropy(sigma)._target_generator_expectations(state)
    want = [np.trace(sigma @ qbm.pauli(lbl)).real for lbl in labels]
    assert np.allclose(got, want, atol=1e-14)


def test_classical_target_is_a_vector_not_a_diagonal_matrix():
    n = 4
    q = _random_distribution(2**n, seed=3, zeros=(2, 5, 11))
    ham = ParamHamiltonian(pauli_pool(n))
    theta = np.random.default_rng(0).normal(scale=0.3, size=ham.n_params)
    state = qbm.DenseBackend().thermal_state(ham, theta)
    as_vector, as_matrix = RelativeEntropy(q), RelativeEntropy(np.diag(q.astype(complex)))
    assert as_vector.sigma is None  # nothing 2^n x 2^n is held
    assert np.isclose(as_vector.value(state), as_matrix.value(state), atol=1e-12)
    assert np.allclose(as_vector.grad(state), as_matrix.grad(state), atol=1e-14)
    # only Z-strings have weight on a diagonal target
    moments = as_vector._target_generator_expectations(state)
    for label, m in zip(ham.labels, moments):
        if set(label) - set("IZ"):
            assert m == 0.0


@pytest.mark.parametrize("classical", [False, True])
@pytest.mark.parametrize("with_offset", [False, True])
def test_value_is_the_relative_entropy(classical, with_offset):
    n = 3
    labels = pauli_pool(n)
    offset = 0.3 * qbm.pauli("ZZX") + 0.2 * qbm.pauli("ZIZ") if with_offset else None
    ham = ParamHamiltonian(labels, offset=offset)
    theta = np.random.default_rng(1).normal(scale=0.4, size=ham.n_params)
    state = qbm.DenseBackend().thermal_state(ham, theta)
    if classical:
        q = _random_distribution(2**n, seed=5)
        target, sigma, log_sigma = q, np.diag(q).astype(complex), np.diag(np.log(q))
    else:
        sigma = _random_state(2**n, seed=5)
        target, log_sigma = sigma, sla.logm(sigma)
    want = np.trace(sigma @ (log_sigma - sla.logm(state.density_matrix()))).real
    assert np.isclose(RelativeEntropy(target).value(state), want, atol=1e-10)


def test_matrix_generators_still_work():
    labels = pauli_pool(3, terms=TFIM, connectivity="chain")
    theta = np.random.default_rng(2).normal(scale=0.4, size=len(labels))
    sigma = _random_state(8, seed=6)
    by_label = qbm.DenseBackend().thermal_state(ParamHamiltonian(labels), theta)
    by_matrix = qbm.DenseBackend().thermal_state(
        ParamHamiltonian([qbm.pauli(s) for s in labels]), theta
    )
    loss = RelativeEntropy(sigma)
    assert np.allclose(loss.grad(by_label), RelativeEntropy(sigma).grad(by_matrix), atol=1e-13)
    assert np.isclose(loss.value(by_label), RelativeEntropy(sigma).value(by_matrix), atol=1e-12)


def test_target_shape_is_checked():
    with pytest.raises(ValueError, match="length 2\\^n"):
        RelativeEntropy(np.ones(6) / 6)
    with pytest.raises(ValueError, match="density matrix"):
        RelativeEntropy(np.ones((2, 3)))


# ---------------------------------------------------------------------------
# the whole training path stays on labels
# ---------------------------------------------------------------------------
LABEL_BACKENDS = ["pauli_propagation", "tensor_network"]


def _label_backend(name):
    if name == "tensor_network":
        pytest.importorskip("quimb")
    return qbm.get_backend(name, trotter_steps=8)


@pytest.mark.parametrize("backend", LABEL_BACKENDS)
def test_loss_never_touches_dense_generators(backend, no_dense_generators):
    n = 4
    model = qbm.FullyVisibleQBM(n, connectivity="ring", backend=_label_backend(backend))
    model.theta = np.random.default_rng(0).normal(scale=0.2, size=model.n_params)
    state = model.state()
    for target in (_random_distribution(2**n, seed=1), _random_state(2**n, seed=1)):
        loss = RelativeEntropy(target)
        assert np.all(np.isfinite(loss.grad(state)))
        # the value needs log Z, which these backends do not have -- and they must say
        # so rather than first building G(theta) as a dense matrix
        with pytest.raises(NotImplementedError):
            loss.value(state)


@pytest.mark.parametrize("backend", LABEL_BACKENDS)
def test_the_facade_never_touches_dense_generators(backend, no_dense_generators):
    data = qbm.datasets.bars_and_stripes(grid=2)
    model = qbm.learn(data, steps=4, backend=_label_backend(backend))
    history = model.history
    assert len(history) == 4 and np.all(np.isnan(history.loss))
    assert history.grad_norm[-1] < history.grad_norm[0]
    assert len(history.monitor) == 4  # the KL curve is available at this size


def test_hidden_unit_training_never_touches_dense_generators(no_dense_generators):
    nv, nh = 3, 1
    model = qbm.VisibleHiddenQBM(n_visible=nv, n_hidden=nh, backend="pauli_propagation")
    model.theta = np.random.default_rng(0).normal(scale=0.2, size=model.n_params)
    q = _random_distribution(2**nv, seed=2)
    history = qbm.fit(
        model, qbm.losses.GibbsMapNLL(q, n_visible=nv), qbm.optim.Adam(lr=0.1), steps=3
    )
    assert np.all(np.isfinite(history.grad_norm))


def test_training_past_the_dense_ceiling():
    # 16 qubits: a single dense generator would be 64 GiB, the density matrix as much
    pytest.importorskip("quimb")
    n = 16
    v = np.arange(2**n)
    spins = 1 - 2 * ((v[:, None] >> np.arange(n - 1, -1, -1)) & 1)
    rng = np.random.default_rng(0)
    energy = spins @ rng.normal(scale=0.5, size=n)
    energy += (spins[:, :-1] * spins[:, 1:]) @ rng.normal(scale=0.5, size=n - 1)
    q = np.exp(-(energy - energy.min()))
    q /= q.sum()

    backend = qbm.get_backend("tensor_network", max_bond=4, trotter_steps=4)
    model = qbm.learn(q, steps=3, backend=backend, connectivity="chain")
    assert model.ham._mats is None
    history = model.history
    assert history.grad_norm[-1] < history.grad_norm[0]
    assert history.monitor == []  # no full distribution at this size: the monitor is quiet
    assert np.all(np.isnan(history.loss))  # nor log Z


# ---------------------------------------------------------------------------
# the KL monitor
# ---------------------------------------------------------------------------
class _CountingBackend:
    """Dense backend that counts how many thermal states were built."""

    name = "counting"

    def __init__(self, hide_probabilities=False):
        self.calls = 0
        self.hide_probabilities = hide_probabilities
        self._dense = qbm.DenseBackend()

    def thermal_state(self, ham, theta):
        self.calls += 1
        state = self._dense.thermal_state(ham, theta)
        if self.hide_probabilities:

            def unavailable():
                raise NotImplementedError("no full distribution here")

            state.probabilities = unavailable
        return state


def test_monitor_reuses_the_state_of_the_step():
    data = qbm.datasets.bars_and_stripes(grid=2)
    backend = _CountingBackend()
    model = qbm.learn(data, steps=6, backend=backend)
    assert backend.calls == 6  # one state per step, not a second one for the monitor
    # entry t is the KL at the parameters the step started from
    probe = qbm.FullyVisibleQBM(ham=model.ham, theta=model.history.theta[-1])
    assert np.isclose(model.history.monitor[-1], probe.kl(data), atol=1e-12)


def test_monitor_goes_quiet_when_the_distribution_is_unavailable():
    data = qbm.datasets.bars_and_stripes(grid=2)
    model = qbm.learn(data, steps=6, backend=_CountingBackend(hide_probabilities=True))
    assert model.history.monitor == []
    assert model.history.grad_norm[-1] < model.history.grad_norm[0]  # training went on


def test_tensor_network_distribution_is_refused_beyond_small_sizes():
    pytest.importorskip("quimb")
    ham = ParamHamiltonian(pauli_pool(15, terms=("Z",)))
    state = qbm.get_backend("tensor_network", trotter_steps=1).thermal_state(ham, np.zeros(15))
    with pytest.raises(NotImplementedError, match="full distribution"):
        state.probabilities()


# ---------------------------------------------------------------------------
# Walsh-Hadamard read-out of the Pauli-propagation distribution
# ---------------------------------------------------------------------------
def test_walsh_hadamard_is_the_sign_matrix_transform():
    n = 4
    a = np.random.default_rng(0).normal(size=2**n)
    idx = np.arange(2**n)
    signs = 1 - 2 * (pp._popcount_array(idx[:, None] & idx[None, :]) & 1)
    assert np.allclose(pp._walsh_hadamard(a), signs @ a, atol=1e-12)
    assert np.allclose(pp._walsh_hadamard(pp._walsh_hadamard(a)), 2**n * a, atol=1e-12)


@pytest.mark.parametrize("cutoff", [1e-10, 1e-2])
def test_pauli_propagation_probabilities_are_the_diagonal(cutoff):
    # also under truncation, where the "probabilities" are a quasi-distribution
    ham = ParamHamiltonian(pauli_pool(5, terms=TFIM))
    theta = np.random.default_rng(1).normal(scale=0.4, size=ham.n_params)
    psum = pp.thermal_state(ham.labels, theta, trotter_steps=16, coeff_cutoff=cutoff)
    assert np.allclose(psum.probabilities(), np.real(np.diag(psum.to_matrix())), atol=1e-12)
    assert np.isclose(psum.probabilities().sum(), 1.0, atol=1e-12)
