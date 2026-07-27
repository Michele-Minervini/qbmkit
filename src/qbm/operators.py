"""Operators and parameterized Hamiltonians.

A QBM model Hamiltonian is ``G(theta) = sum_j theta_j G_j`` over fixed Hermitian
generators ``G_j``.  In v1 generators are dense matrices built from Pauli strings
(qubit 0 is the leftmost tensor factor, e.g. ``pauli("XIZ") == X kron I kron Z``).
"""

from __future__ import annotations

import functools

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


def _coupling_pairs(n: int, connectivity: str, periodic: bool):
    """Yield index pairs (i, j) for two-body couplings."""
    if connectivity == "all":  # all-to-all (fully connected)
        for i in range(n):
            for j in range(i + 1, n):
                yield i, j
    elif connectivity == "chain":  # nearest-neighbour 1-D chain
        last = n if periodic else n - 1
        for i in range(last):
            yield i, (i + 1) % n
    else:
        raise ValueError(f"unknown connectivity {connectivity!r}; use 'chain' or 'all'")


def local_pauli_generators(
    n: int,
    fields=("Z", "X"),
    couplings=("ZZ",),
    connectivity: str = "chain",
    periodic: bool = False,
) -> list[str]:
    """Standard local generator set for an ``n``-qubit fully-visible QBM.

    Single-site ``fields`` (default ``Z`` and ``X``) on every qubit, plus
    two-body ``couplings`` (default ``ZZ``) over the chosen ``connectivity``:
    ``"chain"`` (nearest neighbour, default) or ``"all"`` (all-to-all).
    All-to-all gives a much better inductive bias for non-1-D targets at the
    cost of ``O(n^2)`` parameters.
    """
    gens: list[str] = []
    for f in fields:
        for i in range(n):
            lbl = ["I"] * n
            lbl[i] = f
            gens.append("".join(lbl))
    for c in couplings:
        a, b = c[0], c[1]
        for i, j in _coupling_pairs(n, connectivity, periodic):
            lbl = ["I"] * n
            lbl[i], lbl[j] = a, b
            gens.append("".join(lbl))
    return gens


def pauli_pool(n: int, locality: int = 2, paulis=("X", "Y", "Z")) -> list[str]:
    """All Pauli strings on ``n`` qubits acting non-trivially on at most ``locality`` sites.

    This is the natural *complete* generator set for learning generic quantum states:
    with ``locality = 2`` any 2-local Hamiltonian (Heisenberg, TFIM, ...) lies in the
    span, so its Gibbs state is exactly representable.  The count grows as
    ``sum_k C(n,k) * |paulis|^k`` -- use a small locality for larger ``n``.
    """
    import itertools

    gens: list[str] = []
    for k in range(1, locality + 1):
        for sites in itertools.combinations(range(n), k):
            for ops in itertools.product(paulis, repeat=k):
                lbl = ["I"] * n
                for s, o in zip(sites, ops):
                    lbl[s] = o
                gens.append("".join(lbl))
    return gens


def rbm_generators(n_visible: int, n_hidden: int, hidden_paulis=("Z", "X")) -> list[str]:
    """Generator set for a (semi-quantum) restricted Boltzmann machine.

    Visible units carry diagonal ``Z`` fields; hidden units carry the Pauli
    fields in ``hidden_paulis`` (default ``Z`` and ``X``); each visible-hidden
    pair is coupled through ``Z_v (hidden Pauli)``.  With ``hidden_paulis=("Z",)``
    this is a classical RBM; with non-commuting hidden Paulis it is the
    semi-quantum RBM of arXiv:2502.17562 / 2511.11802.
    """
    n = n_visible + n_hidden

    def lab(spec: dict) -> str:
        s = ["I"] * n
        for q, o in spec.items():
            s[q] = o
        return "".join(s)

    gens: list[str] = []
    for i in range(n_visible):  # visible Z fields
        gens.append(lab({i: "Z"}))
    for j in range(n_hidden):  # hidden fields
        for P in hidden_paulis:
            gens.append(lab({n_visible + j: P}))
    for i in range(n_visible):  # visible-hidden couplings
        for j in range(n_hidden):
            for P in hidden_paulis:
                gens.append(lab({i: "Z", n_visible + j: P}))
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
