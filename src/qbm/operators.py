"""Operators and parameterized Hamiltonians.

A QBM model Hamiltonian is ``G(theta) = sum_j theta_j G_j`` over fixed Hermitian
generators ``G_j``, usually Pauli strings (qubit 0 is the leftmost tensor factor, e.g.
``pauli("XIZ") == X kron I kron Z``).

Two builders produce every standard generator set, each by a choice of operators:
:func:`pauli_pool` places Pauli words on a coupling graph (one register), and
:func:`rbm_generators` does the same for the bipartite visible/hidden layout.
:class:`ParamHamiltonian` then holds the set, building dense matrices only on demand.
"""

from __future__ import annotations

import functools
import itertools
import math
import numbers
import operator

import numpy as np

_PAULI = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def pauli(label: str) -> np.ndarray:
    """Dense matrix of a Pauli string, e.g. ``"XIZ"`` -> ``X kron I kron Z``."""
    label = label.upper()
    if not label or any(c not in _PAULI for c in label):
        raise ValueError(f"invalid Pauli string: {label!r}")
    return functools.reduce(np.kron, (_PAULI[c] for c in label))


# ---------------------------------------------------------------------------
# generator sets: Pauli words placed on a coupling graph
# ---------------------------------------------------------------------------
def _alphabet(paulis, what: str) -> tuple[str, ...]:
    """Validate a choice of single-qubit Paulis, e.g. ``("Z", "X")`` or ``"XYZ"``."""
    letters = tuple(str(p).upper() for p in paulis)
    if not letters or any(p not in ("X", "Y", "Z") for p in letters):
        raise ValueError(f"{what} must be a non-empty choice of 'X', 'Y', 'Z'; got {paulis!r}")
    if len(set(letters)) != len(letters):
        raise ValueError(f"{what} lists a Pauli twice: {paulis!r}")
    return letters


def _label(n: int, sites, word: str) -> str:
    """Pauli string with ``word[k]`` on qubit ``sites[k]`` and identity elsewhere."""
    out = ["I"] * n
    for site, p in zip(sites, word):
        out[site] = p
    return "".join(out)


def _grid_shape(n: int, shape) -> tuple[int, int]:
    if shape is None:
        side = math.isqrt(n)
        if side * side != n:
            raise ValueError(
                f"connectivity='grid' on {n} qubits needs shape=(rows, cols) with "
                f"rows * cols == {n}; only a square lattice can be inferred. Pass shape "
                "to pauli_pool or FullyVisibleQBM, or give the edges explicitly"
            )
        return side, side
    try:
        rows, cols = (operator.index(s) for s in shape)
    except (TypeError, ValueError):
        raise ValueError(f"shape must be a pair of integers (rows, cols); got {shape!r}") from None
    if rows < 1 or cols < 1 or rows * cols != n:
        raise ValueError(f"shape={shape!r} does not tile {n} qubits (need rows * cols == {n})")
    return rows, cols


def _edges(n: int, connectivity="all", periodic: bool = False, shape=None) -> list[tuple[int, int]]:
    """Resolve a coupling topology to its undirected edges ``(i, j)``, without repeats.

    The order of the pair is kept: a two-letter word ``AB`` puts ``A`` on ``i`` and ``B``
    on ``j``.  Wrap-around edges run from the last site back to the first.
    """
    named = isinstance(connectivity, str)
    if named:
        name = connectivity.lower()
        if shape is not None and name != "grid":
            raise ValueError(f"shape is only used by connectivity='grid', not {connectivity!r}")
        if periodic and name in ("all", "star"):
            raise ValueError(
                f"periodic=True has no meaning for connectivity={connectivity!r}; it closes "
                "a 'chain' into a ring and a 'grid' into a torus"
            )
        if name == "all":
            raw = list(itertools.combinations(range(n), 2))
        elif name in ("chain", "ring"):
            raw = [(i, i + 1) for i in range(n - 1)]
            if periodic or name == "ring":
                raw.append((n - 1, 0))
        elif name == "grid":
            rows, cols = _grid_shape(n, shape)
            raw = []
            for r in range(rows):
                for c in range(cols):
                    here = r * cols + c
                    if c + 1 < cols or periodic:  # neighbour to the right
                        raw.append((here, r * cols + (c + 1) % cols))
                    if r + 1 < rows or periodic:  # neighbour below
                        raw.append((here, ((r + 1) % rows) * cols + c))
        elif name == "star":
            raw = [(0, i) for i in range(1, n)]
        else:
            raise ValueError(
                f"unknown connectivity {connectivity!r}; use 'all', 'chain', 'ring', 'grid', "
                "'star', or an explicit list of edges [(i, j), ...]"
            )
    else:
        if periodic or shape is not None:
            raise ValueError("periodic and shape do not apply to an explicit list of edges")
        raw = []
        for edge in connectivity:
            try:
                i, j = (operator.index(q) for q in edge)
            except (TypeError, ValueError):
                raise ValueError(f"an edge must be a pair of qubit indices; got {edge!r}") from None
            if not (0 <= i < n and 0 <= j < n):
                raise ValueError(f"edge {edge!r} is out of range for {n} qubits")
            if i == j:
                raise ValueError(f"edge {edge!r} couples a qubit to itself")
            raw.append((i, j))

    # undirected and simple: a 2-site ring closes onto its own edge, and a size-1 ring or
    # torus direction onto a single site -- neither is a new coupling
    seen: set[tuple[int, int]] = set()
    edges: list[tuple[int, int]] = []
    for i, j in raw:
        key = (i, j) if i < j else (j, i)
        if i != j and key not in seen:
            seen.add(key)
            edges.append((i, j))
    return edges


def _connected_subsets(n: int, edges, k: int) -> list[tuple[int, ...]]:
    """All sets of ``k`` qubits that are connected in the coupling graph, sorted."""
    adjacent: list[set[int]] = [set() for _ in range(n)]
    for i, j in edges:
        adjacent[i].add(j)
        adjacent[j].add(i)
    level = {frozenset((i,)) for i in range(n)}
    for _ in range(k - 1):  # grow every connected set by one neighbouring qubit
        level = {s | {w} for s in level for v in s for w in adjacent[v] if w not in s}
    return sorted(tuple(sorted(s)) for s in level)


def _words(terms, locality, paulis) -> list[str]:
    """The Pauli words to place: explicit ``terms``, or every word up to ``locality``."""
    if terms is None:
        letters = _alphabet(("X", "Y", "Z") if paulis is None else paulis, "paulis")
        k_max = 2 if locality is None else operator.index(locality)
        if k_max < 1:
            raise ValueError(f"locality must be at least 1; got {locality!r}")
        return [
            "".join(w) for k in range(1, k_max + 1) for w in itertools.product(letters, repeat=k)
        ]
    if locality is not None or paulis is not None:
        raise ValueError(
            "choose the operators one way: explicit `terms`, or `locality`/`paulis` "
            "(every word over those Paulis up to that locality) -- not both"
        )
    if isinstance(terms, numbers.Integral):
        raise TypeError(f"terms must be Pauli words; for a locality write locality={terms}")
    words = [terms.upper()] if isinstance(terms, str) else [str(w).upper() for w in terms]
    for w in words:
        if not w or any(c not in "XYZ" for c in w):
            raise ValueError(f"a term must be a non-empty word over X, Y, Z; got {w!r}")
    if len(set(words)) != len(words):
        raise ValueError(f"terms lists a word twice: {terms!r}")
    return words


def pauli_pool(
    n: int,
    terms=None,
    *,
    locality: int | None = None,
    paulis=None,
    connectivity="all",
    periodic: bool = False,
    shape=None,
) -> list[str]:
    """Generator set of an ``n``-qubit QBM: Pauli words placed on a coupling graph.

    Every generator set in the library is described the same way -- *which operators*,
    and *on which sites*.  A one-letter word (``"Z"``) is a field and goes on every
    qubit; a two-letter word (``"ZZ"``) is a coupling and goes on every edge of
    ``connectivity``; a ``k``-letter word goes on every set of ``k`` qubits that is
    connected in that graph.

    Parameters
    ----------
    n : int
        Number of qubits.
    terms : sequence of str, optional
        The words to place, e.g. ``("Z", "X", "ZZ")`` for transverse-field-Ising-type
        generators.  If omitted, *every* word over ``paulis`` up to ``locality`` is
        used -- the complete pool.
    locality : int, default 2
        Largest number of qubits a generator acts on, when ``terms`` is omitted.
    paulis : sequence of str, default ``("X", "Y", "Z")``
        Single-qubit alphabet, when ``terms`` is omitted.
    connectivity : str or list of (int, int), default ``"all"``
        The coupling graph:

        * ``"all"`` -- every pair of qubits;
        * ``"chain"`` -- nearest neighbours on an open line;
        * ``"ring"`` -- a chain closed on itself (same as ``"chain"`` with ``periodic``);
        * ``"grid"`` -- a 2-D square lattice of ``shape=(rows, cols)``, qubits numbered
          row by row;
        * ``"star"`` -- qubit 0 coupled to all the others;
        * an explicit list of edges ``[(i, j), ...]`` for anything else.

    periodic : bool
        Close a ``"chain"`` into a ring or a ``"grid"`` into a torus.
    shape : (int, int), optional
        Rows and columns of the ``"grid"``; may be omitted when ``n`` is a perfect square.

    Returns
    -------
    list of str
        Pauli strings, word by word in the order given, each word over its placements
        in increasing site order.  No generator appears twice.

    Notes
    -----
    With the defaults this is the **complete** 2-local pool: any 2-local Hamiltonian
    (Heisenberg, TFIM, ...) lies in its span, so its Gibbs state is exactly
    representable -- the natural choice for learning a generic quantum state.  Its size
    grows as ``sum_k C(n, k) |paulis|^k``, and the parameter count drives every cost in
    the library (the metric is ``n_params x n_params``), so a structured choice of
    ``terms`` and ``connectivity`` is the usual one for larger ``n``.

    Edges are undirected, but a word is placed in the order the edge is written: ``"XY"``
    on ``(i, j)`` is ``X_i Y_j``.  Named topologies write every edge from the lower site
    to the higher one, except a wrap-around edge, which runs from the last site back to
    the first.  Ask for both ``"XY"`` and ``"YX"`` to get both orientations.

    Examples
    --------
    >>> pauli_pool(3, terms=("Z", "ZZ"), connectivity="chain")
    ['ZII', 'IZI', 'IIZ', 'ZZI', 'IZZ']
    >>> pauli_pool(3, terms=("ZZ",), connectivity="ring")
    ['ZZI', 'IZZ', 'ZIZ']
    >>> pauli_pool(4, terms=("XX",), connectivity=[(0, 3), (1, 2)])
    ['XIIX', 'IXXI']
    >>> len(pauli_pool(4))                    # all 1- and 2-body Paulis: 12 + 54
    66
    >>> len(pauli_pool(6, terms=("ZZ",), connectivity="grid", shape=(2, 3)))
    7
    """
    n = operator.index(n)
    if n < 1:
        raise ValueError(f"n must be a positive number of qubits; got {n}")
    words = _words(terms, locality, paulis)
    edges = _edges(n, connectivity, periodic, shape)
    placements: dict[int, list] = {1: [(i,) for i in range(n)], 2: edges}
    gens: list[str] = []
    for word in words:
        k = len(word)
        if k not in placements:
            placements[k] = _connected_subsets(n, edges, k)
        gens.extend(_label(n, sites, word) for sites in placements[k])
    return gens


def rbm_generators(
    n_visible: int,
    n_hidden: int,
    visible_paulis=("Z",),
    hidden_paulis=("Z", "X"),
) -> list[str]:
    """Generator set of a restricted Boltzmann machine, classical to fully quantum.

    The bipartite counterpart of :func:`pauli_pool`: fields on every unit and a coupling
    on every visible-hidden pair, with no couplings inside a layer.  Visible qubits come
    first.  The choice of operators selects the model:

    * ``visible_paulis=("Z",)``, ``hidden_paulis=("Z",)`` -- a classical RBM;
    * ``visible_paulis=("Z",)``, non-commuting ``hidden_paulis`` such as ``("Z", "X")``
      -- a semi-quantum RBM (arXiv:2502.17562), the default;
    * non-commuting ``visible_paulis`` as well, e.g. ``("X", "Y", "Z")`` on both
      registers -- a fully quantum RBM.

    Parameters
    ----------
    n_visible, n_hidden : int
        Number of visible and hidden qubits.
    visible_paulis, hidden_paulis : sequence of str
        Single-qubit Paulis on each register.  Every unit gets each of its Paulis as a
        field, and every visible-hidden pair is coupled through each product
        ``P_visible P_hidden``.

    Notes
    -----
    Keeping the visible register diagonal (``visible_paulis=("Z",)``, the default) is
    what makes ``G(theta)`` block diagonal in the visible configuration ``v``.  The
    closed-form semi-quantum marginal (:class:`~qbm.SemiQuantumRBM`), the exact
    Gibbs-map gradient and its sampler (:mod:`qbm.gibbs_map`) all rest on that, and it is
    the right model for classical data.  A non-diagonal visible register gives up those
    fast paths -- only the dense route remains -- but it is what a *quantum* target
    needs: with ``Z`` alone the visible reduced state is diagonal, whatever the hidden
    units do.

    For couplings inside a layer (an unrestricted machine) use :func:`pauli_pool` on
    all ``n_visible + n_hidden`` qubits.

    Examples
    --------
    >>> rbm_generators(1, 1)
    ['ZI', 'IZ', 'IX', 'ZZ', 'ZX']
    >>> rbm_generators(1, 1, visible_paulis=("Z", "X"), hidden_paulis=("Z",))
    ['ZI', 'XI', 'IZ', 'ZZ', 'XZ']
    """
    n_visible, n_hidden = operator.index(n_visible), operator.index(n_hidden)
    if n_visible < 1 or n_hidden < 0:
        raise ValueError("need at least one visible unit and a non-negative number of hidden")
    vis = _alphabet(visible_paulis, "visible_paulis")
    hid = _alphabet(hidden_paulis, "hidden_paulis")
    n = n_visible + n_hidden
    visible, hidden = range(n_visible), range(n_visible, n)
    gens = [_label(n, (i,), p) for i in visible for p in vis]
    gens += [_label(n, (j,), p) for j in hidden for p in hid]
    gens += [
        _label(n, (i, j), pv + ph) for i in visible for j in hidden for pv in vis for ph in hid
    ]
    return gens


class ParamHamiltonian:
    """``G(theta) = offset + sum_j theta_j G_j`` over fixed Hermitian generators.

    Parameters
    ----------
    generators : list of (str | ndarray)
        Pauli-string labels or dense Hermitian matrices.
    labels : list of str, optional
        Names for matrix generators (Pauli labels are kept automatically).
    offset : ndarray, optional
        A fixed (non-trainable) Hermitian term added to ``G``.  Since
        ``dG/dtheta_j = G_j`` is unaffected by it, every gradient and metric in the
        library works unchanged; it exists for models with fixed bias terms and for
        the SDP dual (where the objective matrix sits in the exponent).

    Notes
    -----
    Pauli-string generators are materialised as dense matrices **lazily**, on first
    access to :attr:`generators`.  A dense generator is ``2^n x 2^n``, so eager
    construction would cap every backend at ~13 qubits; backends that work from the
    labels alone (e.g. the tensor-network backend) therefore never pay that cost.
    """

    def __init__(self, generators, labels=None, n_qubits=None, offset=None):
        raw = list(generators)
        if not raw:
            raise ValueError("ParamHamiltonian needs at least one generator")

        lbls: list[str] = []
        for i, g in enumerate(raw):
            if isinstance(g, str):
                lbls.append(g.upper())
            else:
                lbls.append(labels[i] if labels is not None else f"G{i}")

        # infer the dimension without materialising anything
        first = raw[0]
        if isinstance(first, str):
            n_sites = len(first)
            dim = 1 << n_sites
            inferred_qubits = n_sites
        else:
            dim = np.asarray(first).shape[0]
            inferred_qubits = int(round(np.log2(dim)))
        for g in raw:
            size = (1 << len(g)) if isinstance(g, str) else np.asarray(g).shape[0]
            if size != dim:
                raise ValueError("all generators must be square and the same size")

        self._raw = raw
        self._mats: list[np.ndarray] | None = None
        self.labels = lbls
        self.dim = dim
        self.n_qubits = n_qubits if n_qubits is not None else inferred_qubits
        if offset is not None:
            offset = np.asarray(offset, dtype=complex)
            if offset.shape != (dim, dim):
                raise ValueError("offset must match the generator dimension")
        self.offset = offset

    @property
    def generators(self) -> list:
        """Dense generator matrices, built (and cached) on first access."""
        if self._mats is None:
            self._mats = [
                pauli(g) if isinstance(g, str) else np.asarray(g, dtype=complex) for g in self._raw
            ]
        return self._mats

    @property
    def pauli_labels(self) -> list[str] | None:
        """The generators as Pauli strings, or ``None`` if any was given as a matrix.

        This is what lets a backend or a loss work from the labels alone, leaving the
        dense :attr:`generators` unbuilt.
        """
        if all(isinstance(g, str) for g in self._raw):
            return list(self.labels)
        return None

    @property
    def n_params(self) -> int:
        return len(self._raw)

    def matrix(self, theta) -> np.ndarray:
        """Return the dense Hermitian matrix ``offset + sum_j theta_j G_j``."""
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (self.n_params,):
            raise ValueError(f"theta must have shape ({self.n_params},)")
        G = np.zeros((self.dim, self.dim), dtype=complex)
        if self.offset is not None:
            G += self.offset
        for t, g in zip(theta, self.generators):
            if t != 0.0:
                G += t * g
        return G

    def __len__(self) -> int:
        return self.n_params

    def __repr__(self) -> str:
        return f"ParamHamiltonian(n_qubits={self.n_qubits}, n_params={self.n_params})"
