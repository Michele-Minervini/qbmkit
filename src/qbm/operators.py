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
    """``G(theta) = sum_j theta_j G_j`` over fixed Hermitian generators.

    Parameters
    ----------
    generators : list of (str | ndarray)
        Pauli-string labels or dense Hermitian matrices.
    labels : list of str, optional
        Names for matrix generators (Pauli labels are kept automatically).
    """

    def __init__(self, generators, labels=None, n_qubits=None):
        mats: list[np.ndarray] = []
        lbls: list[str] = []
        for i, g in enumerate(generators):
            if isinstance(g, str):
                mats.append(pauli(g))
                lbls.append(g)
            else:
                mats.append(np.asarray(g, dtype=complex))
                lbls.append(labels[i] if labels is not None else f"G{i}")
        if not mats:
            raise ValueError("ParamHamiltonian needs at least one generator")
        dim = mats[0].shape[0]
        for m in mats:
            if m.shape != (dim, dim):
                raise ValueError("all generators must be square and the same size")
        self.generators = mats
        self.labels = lbls
        self.dim = dim
        self.n_qubits = n_qubits if n_qubits is not None else int(round(np.log2(dim)))

    @property
    def n_params(self) -> int:
        return len(self.generators)

    def matrix(self, theta) -> np.ndarray:
        """Return the dense Hermitian matrix ``sum_j theta_j G_j``."""
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (self.n_params,):
            raise ValueError(f"theta must have shape ({self.n_params},)")
        G = np.zeros((self.dim, self.dim), dtype=complex)
        for t, g in zip(theta, self.generators):
            if t != 0.0:
                G += t * g
        return G

    def __len__(self) -> int:
        return self.n_params

    def __repr__(self) -> str:
        return f"ParamHamiltonian(n_qubits={self.n_qubits}, n_params={self.n_params})"
