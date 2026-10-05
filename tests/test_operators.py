"""Generator sets: Pauli words placed on a coupling graph (``qbm.pauli_pool``),
and the bipartite builder (``qbm.rbm_generators``)."""

import itertools

import numpy as np
import pytest

import qbm
from qbm.operators import ParamHamiltonian, _edges, pauli_pool, rbm_generators

TFIM = ("Z", "X", "ZZ")


def _undirected(edges):
    return {frozenset(e) for e in edges}


# ---------------------------------------------------------------------------
# topologies
# ---------------------------------------------------------------------------
def test_named_topologies_give_the_expected_edges():
    assert _edges(4, "chain") == [(0, 1), (1, 2), (2, 3)]
    assert _edges(4, "ring") == [(0, 1), (1, 2), (2, 3), (3, 0)]
    assert _edges(4, "chain", periodic=True) == _edges(4, "ring")
    assert _edges(4, "all") == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    assert _edges(4, "star") == [(0, 1), (0, 2), (0, 3)]
    # 2 x 3 lattice, numbered row by row:   0 1 2
    #                                       3 4 5
    assert _undirected(_edges(6, "grid", shape=(2, 3))) == _undirected(
        [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)]
    )


@pytest.mark.parametrize("n", [3, 4, 5, 8])
def test_edge_counts(n):
    assert len(_edges(n, "chain")) == n - 1
    assert len(_edges(n, "ring")) == n
    assert len(_edges(n, "star")) == n - 1
    assert len(_edges(n, "all")) == n * (n - 1) // 2


@pytest.mark.parametrize("rows, cols", [(2, 2), (2, 3), (3, 3), (3, 4), (1, 5)])
def test_grid_and_torus_edge_counts(rows, cols):
    n = rows * cols
    open_edges = _edges(n, "grid", shape=(rows, cols))
    assert len(open_edges) == rows * (cols - 1) + cols * (rows - 1)
    torus = _edges(n, "grid", shape=(rows, cols), periodic=True)
    # a direction of length L closes with L bonds per line, unless that bond already
    # exists (L = 2) or would join a site to itself (L = 1)
    bonds = lambda length: {1: 0, 2: 1}.get(length, length)  # noqa: E731
    assert len(torus) == rows * bonds(cols) + cols * bonds(rows)
    assert _undirected(open_edges) <= _undirected(torus)


def test_a_one_row_grid_is_a_chain_and_a_one_row_torus_a_ring():
    assert _edges(5, "grid", shape=(1, 5)) == _edges(5, "chain")
    assert _undirected(_edges(5, "grid", shape=(1, 5), periodic=True)) == _undirected(
        _edges(5, "ring")
    )


def test_grid_shape_is_inferred_only_for_a_square():
    assert _edges(9, "grid") == _edges(9, "grid", shape=(3, 3))
    with pytest.raises(ValueError, match="shape"):
        _edges(6, "grid")
    with pytest.raises(ValueError, match="does not tile"):
        _edges(6, "grid", shape=(2, 2))
    with pytest.raises(ValueError, match="pair of integers"):
        _edges(6, "grid", shape=6)


def test_every_grid_site_has_its_lattice_neighbours():
    rows, cols = 3, 4
    und = _undirected(_edges(rows * cols, "grid", shape=(rows, cols)))
    for r, c in itertools.product(range(rows), range(cols)):
        here = r * cols + c
        if c + 1 < cols:
            assert frozenset((here, here + 1)) in und
        if r + 1 < rows:
            assert frozenset((here, here + cols)) in und


def test_explicit_edges_are_validated_and_deduplicated():
    assert _edges(4, [(0, 3), (1, 2)]) == [(0, 3), (1, 2)]
    assert _edges(4, [(0, 1), (1, 0), (0, 1)]) == [(0, 1)]  # undirected
    assert _edges(4, []) == []
    assert _edges(4, ((i, i + 1) for i in range(3))) == _edges(4, "chain")  # any iterable
    assert _edges(4, np.array([[0, 1], [2, 3]])) == [(0, 1), (2, 3)]  # numpy integers
    with pytest.raises(ValueError, match="out of range"):
        _edges(4, [(0, 4)])
    with pytest.raises(ValueError, match="itself"):
        _edges(4, [(2, 2)])
    with pytest.raises(ValueError, match="pair of qubit indices"):
        _edges(4, [(0, 1, 2)])
    with pytest.raises(ValueError, match="pair of qubit indices"):
        _edges(4, [(0.5, 1)])


def test_unknown_or_inconsistent_topology_is_an_error_not_a_silent_default():
    with pytest.raises(ValueError, match="unknown connectivity"):
        _edges(4, "ladder")
    # periodic used to be silently ignored for all-to-all
    for name in ("all", "star"):
        with pytest.raises(ValueError, match="periodic"):
            _edges(4, name, periodic=True)
    with pytest.raises(ValueError, match="explicit list"):
        _edges(4, [(0, 1)], periodic=True)
    with pytest.raises(ValueError, match="shape is only used"):
        _edges(4, "chain", shape=(2, 2))


def test_a_two_site_ring_has_one_bond_not_two():
    # the old periodic chain returned ['ZZ', 'ZZ'] here: a repeated generator, which
    # makes every information metric singular
    assert pauli_pool(2, terms=("ZZ",), connectivity="ring") == ["ZZ"]
    assert pauli_pool(2, terms=("ZZ",), connectivity="chain", periodic=True) == ["ZZ"]
    assert pauli_pool(1, terms=("Z", "ZZ"), connectivity="ring") == ["Z"]
    model = qbm.FullyVisibleQBM(2, connectivity="chain", periodic=True)
    assert model.n_params == 5
    model.theta = np.random.default_rng(0).normal(scale=0.4, size=5)
    assert np.linalg.eigvalsh(model.state().metric("kubo_mori")).min() > 1e-6


# ---------------------------------------------------------------------------
# one builder: words on a graph
# ---------------------------------------------------------------------------
def _all_paulis_up_to(n, k, letters="XYZ"):
    out = set()
    for label in itertools.product("I" + letters, repeat=n):
        if 1 <= sum(c != "I" for c in label) <= k:
            out.add("".join(label))
    return out


@pytest.mark.parametrize("n, k", [(2, 2), (3, 2), (4, 2), (4, 3), (3, 3)])
def test_default_is_the_complete_pool(n, k):
    pool = pauli_pool(n, locality=k)
    assert len(pool) == len(set(pool))
    assert set(pool) == _all_paulis_up_to(n, k)


def test_complete_pool_over_a_restricted_alphabet():
    pool = pauli_pool(3, paulis=("Z", "X"))
    assert set(pool) == _all_paulis_up_to(3, 2, letters="ZX")
    assert set(pauli_pool(3, paulis="ZX")) == set(pool)  # a string is its letters


def test_explicit_terms_recover_the_structured_local_set():
    # exactly what the former local_pauli_generators(3) returned, in the same order
    assert pauli_pool(3, terms=TFIM, connectivity="chain") == [
        "ZII", "IZI", "IIZ", "XII", "IXI", "IIX", "ZZI", "IZZ",
    ]  # fmt: skip
    assert len(pauli_pool(4, terms=TFIM, connectivity="chain")) == 11
    assert len(pauli_pool(8, terms=TFIM, connectivity="chain")) == 23


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_the_two_descriptions_agree(n):
    # every word spelled out == "every word up to locality 2"
    letters = ("X", "Y", "Z")
    words = letters + tuple(a + b for a in letters for b in letters)
    assert pauli_pool(n, terms=words) == pauli_pool(n)
    assert pauli_pool(n, terms=words) == pauli_pool(n, locality=2, paulis=letters)


@pytest.mark.parametrize("topology", ["chain", "ring", "star", "all", "grid"])
def test_pool_restricted_to_a_graph_is_the_matching_subset(topology):
    n = 4
    edges = _undirected(_edges(n, topology))
    pool = pauli_pool(n, connectivity=topology)
    assert len(pool) == len(set(pool)) == 3 * n + 9 * len(edges)
    for label in pool:
        support = frozenset(i for i, c in enumerate(label) if c != "I")
        assert len(support) == 1 or support in edges
    assert set(pool) <= set(pauli_pool(n))


def test_words_are_placed_in_edge_order():
    assert pauli_pool(3, terms=("XY",), connectivity=[(0, 2)]) == ["XIY"]
    assert pauli_pool(3, terms=("XY",), connectivity=[(2, 0)]) == ["YIX"]
    # the wrap-around bond of a ring continues the direction of the chain
    assert pauli_pool(3, terms=("XY",), connectivity="ring") == ["XYI", "IXY", "YIX"]
    assert pauli_pool(3, terms=("XY", "YX"), connectivity="chain") == ["XYI", "IXY", "YXI", "IYX"]


def test_higher_body_words_go_on_connected_sets():
    assert pauli_pool(4, terms=("ZZZ",), connectivity="chain") == ["ZZZI", "IZZZ"]
    assert pauli_pool(4, terms=("ZZZ",), connectivity="ring") == ["ZZZI", "ZZIZ", "ZIZZ", "IZZZ"]
    assert pauli_pool(4, terms=("ZZZ",), connectivity="star") == ["ZZZI", "ZZIZ", "ZIZZ"]
    assert len(pauli_pool(5, terms=("ZZZ",))) == 10  # every triple
    assert pauli_pool(4, terms=("XYZ",), connectivity="chain") == ["XYZI", "IXYZ"]
    assert pauli_pool(2, terms=("ZZZ",)) == []  # no such set of sites
    # 2 x 2 plaquette: all four corners are connected
    assert pauli_pool(4, terms=("ZZZZ",), connectivity="grid") == ["ZZZZ"]


@pytest.mark.parametrize("n", [1, 2, 3, 6])
@pytest.mark.parametrize("topology", ["chain", "ring", "star", "all"])
def test_never_a_repeated_generator(n, topology):
    for kwargs in ({"terms": TFIM}, {}, {"locality": 3}):
        pool = pauli_pool(n, connectivity=topology, **kwargs)
        assert len(pool) == len(set(pool))
        assert all(len(label) == n for label in pool)


def test_term_spelling():
    assert pauli_pool(2, terms="ZZ") == ["ZZ"]  # a bare word, not its letters
    assert pauli_pool(2, terms=("z", "zz")) == ["ZI", "IZ", "ZZ"]
    assert pauli_pool(2, ("Z", "ZZ")) == ["ZI", "IZ", "ZZ"]  # positional


def test_bad_requests_are_rejected():
    with pytest.raises(ValueError, match="not both"):
        pauli_pool(3, terms=("Z",), locality=2)
    with pytest.raises(ValueError, match="not both"):
        pauli_pool(3, terms=("Z",), paulis=("Z",))
    with pytest.raises(ValueError, match="word over X, Y, Z"):
        pauli_pool(3, terms=("ZI",))
    with pytest.raises(ValueError, match="word over X, Y, Z"):
        pauli_pool(3, terms=("Z", ""))
    with pytest.raises(ValueError, match="twice"):
        pauli_pool(3, terms=("Z", "Z"))
    with pytest.raises(TypeError, match="locality=2"):
        pauli_pool(3, 2)  # the old positional locality
    with pytest.raises(ValueError, match="locality must be at least 1"):
        pauli_pool(3, locality=0)
    with pytest.raises(ValueError, match="paulis"):
        pauli_pool(3, paulis=("Z", "Q"))
    with pytest.raises(ValueError, match="positive number of qubits"):
        pauli_pool(0)


# ---------------------------------------------------------------------------
# every topology works end to end
# ---------------------------------------------------------------------------
TOPOLOGIES = [
    {"connectivity": "chain"},
    {"connectivity": "ring"},
    {"connectivity": "chain", "periodic": True},
    {"connectivity": "star"},
    {"connectivity": "all"},
    {"connectivity": "grid"},
    {"connectivity": "grid", "shape": (1, 4)},
    {"connectivity": "grid", "periodic": True},
    {"connectivity": [(0, 2), (1, 3)]},
]


@pytest.mark.parametrize("topology", TOPOLOGIES, ids=str)
def test_relative_entropy_gradient_is_exact_on_every_topology(topology):
    n = 4
    model = qbm.FullyVisibleQBM(n, **topology)
    rng = np.random.default_rng(3)
    model.theta = rng.normal(scale=0.4, size=model.n_params)
    A = rng.normal(size=(16, 16)) + 1j * rng.normal(size=(16, 16))
    sigma = A @ A.conj().T
    sigma /= np.trace(sigma).real
    loss = qbm.losses.RelativeEntropy(sigma)
    grad = loss.grad(model.state())
    h = 1e-5
    for j in range(model.n_params):
        e = np.zeros(model.n_params)
        e[j] = h
        up = loss.value(model.backend.thermal_state(model.ham, model.theta + e))
        dn = loss.value(model.backend.thermal_state(model.ham, model.theta - e))
        assert np.isclose(grad[j], (up - dn) / (2 * h), atol=1e-7)
    # a full-rank metric: no topology produces linearly dependent generators
    assert np.linalg.eigvalsh(model.state().metric("kubo_mori")).min() > 1e-8


@pytest.mark.parametrize("topology", ["ring", "star", "grid", [(0, 1), (1, 2), (2, 3), (0, 2)]])
def test_the_facade_learns_a_target_that_lives_on_its_graph(topology):
    # data drawn from an Ising model on this graph: the model on the same graph can
    # represent it exactly, the model on a different graph cannot
    target = qbm.FullyVisibleQBM(4, terms=("Z", "ZZ"), connectivity=topology)
    target.theta = np.random.default_rng(7).normal(scale=0.8, size=target.n_params)
    data = target.probabilities()

    model = qbm.learn(data, steps=300, connectivity=topology)
    assert model.n_params == 8 + len(_edges(4, topology))
    assert model.history.monitor[-1] < 1e-8
    mismatched = qbm.learn(data, steps=300, connectivity="chain")
    assert mismatched.history.monitor[-1] > 1e-2


def test_matching_the_lattice_of_the_target_matters():
    # a periodic TFIM has the bond (n-1, 0); only the ring model contains it, so only
    # the ring model can hold its Gibbs state
    H = qbm.hamiltonians.tfim(4, J=1.0, g=0.7, periodic=True)
    ring = qbm.free_energy_min(H, steps=300, connectivity="ring")
    chain = qbm.free_energy_min(H, steps=300, connectivity="chain")
    assert abs(ring.error) < 1e-8
    assert chain.error > 1e-2
    assert qbm.ground_state(H, steps=50, connectivity="ring").model.n_params == 12


def test_free_energy_task_accepts_an_edge_list():
    H = qbm.hamiltonians.tfim(3, g=1.0, periodic=True)
    res = qbm.free_energy_min(H, steps=200, connectivity=[(0, 1), (1, 2), (2, 0)])
    assert abs(res.error) < 1e-3


@pytest.mark.parametrize("backend", ["pauli_propagation", "tensor_network"])
@pytest.mark.parametrize("topology", ["ring", "star", "grid"])
def test_label_backends_agree_with_dense_on_new_topologies(backend, topology):
    if backend == "tensor_network":
        pytest.importorskip("quimb")
    ham = ParamHamiltonian(pauli_pool(4, terms=TFIM, connectivity=topology))
    theta = np.random.default_rng(5).normal(scale=0.3, size=ham.n_params)
    exact = qbm.DenseBackend().thermal_state(ham, theta).generator_expectations()
    approx = qbm.get_backend(backend, trotter_steps=200).thermal_state(ham, theta)
    assert np.allclose(approx.generator_expectations(), exact, atol=1e-3)


def test_varqite_ansatz_follows_the_topology():
    from qbm.circuits import varqite as vq

    chain = vq.tfd_ansatz(n=4, depth=1, connectivity="chain")
    star = vq.tfd_ansatz(n=4, depth=1, connectivity="star")
    full = vq.tfd_ansatz(n=4, depth=1)
    assert chain.n_params == star.n_params < full.n_params
    ham = ParamHamiltonian(pauli_pool(3, terms=TFIM, connectivity="ring"))
    theta = np.random.default_rng(0).normal(scale=0.3, size=ham.n_params)
    res = vq.prepare_gibbs(ham, theta, depth=2, steps=60)
    assert res.infidelity() < 1e-3


# ---------------------------------------------------------------------------
# restricted Boltzmann machines: classical, semi-quantum, fully quantum
# ---------------------------------------------------------------------------
def test_default_rbm_is_the_semi_quantum_one():
    # exactly the former rbm_generators(2, 1), in the same order
    assert rbm_generators(2, 1) == ["ZII", "IZI", "IIZ", "IIX", "ZIZ", "ZIX", "IZZ", "IZX"]


@pytest.mark.parametrize("vis", [("Z",), ("Z", "X"), ("X", "Y", "Z")])
@pytest.mark.parametrize("hid", [("Z",), ("Z", "X"), ("X", "Y", "Z")])
def test_rbm_structure(vis, hid):
    nv, nh = 3, 2
    gens = rbm_generators(nv, nh, visible_paulis=vis, hidden_paulis=hid)
    assert len(gens) == len(set(gens))
    assert len(gens) == nv * len(vis) + nh * len(hid) + nv * nh * len(vis) * len(hid)
    for g in gens:
        on_visible = [c for c in g[:nv] if c != "I"]
        on_hidden = [c for c in g[nv:] if c != "I"]
        assert len(on_visible) <= 1 and len(on_hidden) <= 1  # restricted: no intra-layer term
        assert set(on_visible) <= set(vis) and set(on_hidden) <= set(hid)


def test_operator_choice_selects_the_model():
    classical = rbm_generators(2, 2, hidden_paulis=("Z",))
    assert all(set(g) <= set("IZ") for g in classical)
    semi = rbm_generators(2, 2)
    assert all(set(g[:2]) <= set("IZ") for g in semi) and any("X" in g[2:] for g in semi)
    quantum = rbm_generators(2, 2, visible_paulis=("Z", "X"))
    assert any("X" in g[:2] for g in quantum)
    assert set(semi) < set(quantum)  # the semi-quantum machine is a sub-model


def test_rbm_rejects_bad_operators():
    with pytest.raises(ValueError, match="visible_paulis"):
        rbm_generators(2, 1, visible_paulis=())
    with pytest.raises(ValueError, match="hidden_paulis"):
        rbm_generators(2, 1, hidden_paulis=("I",))
    with pytest.raises(ValueError, match="twice"):
        rbm_generators(2, 1, hidden_paulis=("Z", "Z"))


def test_closed_form_sqrbm_uses_the_same_generators():
    sq = qbm.SemiQuantumRBM(n_visible=3, n_hidden=2, hidden_paulis=("X", "Z"))
    assert sq.to_hamiltonian().labels == rbm_generators(3, 2, hidden_paulis=("X", "Z"))


def test_only_a_quantum_visible_register_gives_a_quantum_visible_state():
    from qbm.linalg import partial_trace_hidden

    def off_diagonal_weight(model):
        model.theta = np.random.default_rng(2).normal(scale=0.5, size=model.n_params)
        rho_v = partial_trace_hidden(model.density_matrix(), 2, 1)
        return np.abs(rho_v - np.diag(np.diag(rho_v))).max()

    semi = qbm.VisibleHiddenQBM(n_visible=2, n_hidden=1)
    quantum = qbm.VisibleHiddenQBM(n_visible=2, n_hidden=1, visible_paulis=("Z", "X"))
    assert off_diagonal_weight(semi) < 1e-12  # diagonal whatever the hidden unit does
    assert off_diagonal_weight(quantum) > 1e-2


def test_fully_quantum_rbm_trains_through_the_generic_losses():
    nv, nh = 2, 1
    model = qbm.VisibleHiddenQBM(n_visible=nv, n_hidden=nh, visible_paulis=("Z", "X"))
    rng = np.random.default_rng(4)
    model.theta = rng.normal(scale=0.4, size=model.n_params)
    q = rng.random(1 << nv)
    q /= q.sum()
    loss = qbm.losses.NLL(q, n_visible=nv)
    grad = loss.grad(model.state())
    h = 1e-5
    for j in range(model.n_params):
        e = np.zeros(model.n_params)
        e[j] = h
        up = loss.value(model.backend.thermal_state(model.ham, model.theta + e))
        dn = loss.value(model.backend.thermal_state(model.ham, model.theta - e))
        assert np.isclose(grad[j], (up - dn) / (2 * h), atol=1e-7)
    # the fast hidden-unit path needs a diagonal visible register, and says so
    with pytest.raises(ValueError, match="visible register to be diagonal"):
        qbm.losses.GibbsMapNLL(q, n_visible=nv).grad(model.state())


def test_docstring_examples_are_true():
    import doctest

    import qbm.operators as ops

    result = doctest.testmod(ops)
    assert result.attempted >= 7 and result.failed == 0
