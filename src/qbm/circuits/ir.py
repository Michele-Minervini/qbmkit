"""A tiny, dependency-free circuit intermediate representation.

The library's quantum algorithms (Gibbs preparation, Hadamard tests, Trotterised
evolution) are expressed **in this IR**, never in a vendor SDK.  Qiskit, PennyLane and
OpenQASM are then *emitters* -- small leaf modules in :mod:`qbm.circuits.adapters`.

Rationale: quantum SDKs churn (Qiskit removed ``opflow``, moved ``qiskit.algorithms``
out of the core, dropped ``execute()``, and revised its primitives twice).  Keeping the
algorithms in our own IR means an SDK break costs one adapter file, not the library.
The IR is deliberately minimal -- just what the QBM estimators need.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# gates that need no matrix
_NAMED_1Q = {"h", "x", "y", "z", "s", "sdg", "t", "tdg"}
_ROT_1Q = {"rx", "ry", "rz", "phase"}


@dataclass(frozen=True)
class Gate:
    """One instruction.

    ``name`` is a named gate (``h``, ``x``, ``rz``, ``cx``, ...), or ``"unitary"`` /
    ``"c-unitary"`` carrying an explicit dense ``matrix`` (the controlled version uses
    ``qubits[0]`` as control and the rest as targets).
    """

    name: str
    qubits: tuple[int, ...]
    params: tuple[float, ...] = ()
    matrix: np.ndarray | None = None
    label: str = ""

    def __repr__(self) -> str:
        p = f"({', '.join(f'{x:.4g}' for x in self.params)})" if self.params else ""
        return f"{self.name}{p} {list(self.qubits)}"


@dataclass
class Circuit:
    """An ordered list of :class:`Gate` on ``n_qubits`` qubits."""

    n_qubits: int
    gates: list[Gate] = field(default_factory=list)
    name: str = "circuit"

    # -- construction helpers ---------------------------------------------
    def _add(self, name, qubits, params=(), matrix=None, label=""):
        for q in qubits:
            if not 0 <= q < self.n_qubits:
                raise ValueError(f"qubit {q} out of range for {self.n_qubits} qubits")
        self.gates.append(Gate(name, tuple(qubits), tuple(params), matrix, label))
        return self

    def h(self, q):
        return self._add("h", (q,))

    def x(self, q):
        return self._add("x", (q,))

    def y(self, q):
        return self._add("y", (q,))

    def z(self, q):
        return self._add("z", (q,))

    def s(self, q):
        return self._add("s", (q,))

    def sdg(self, q):
        return self._add("sdg", (q,))

    def rx(self, theta, q):
        return self._add("rx", (q,), (theta,))

    def ry(self, theta, q):
        return self._add("ry", (q,), (theta,))

    def rz(self, theta, q):
        return self._add("rz", (q,), (theta,))

    def cx(self, control, target):
        return self._add("cx", (control, target))

    def cz(self, control, target):
        return self._add("cz", (control, target))

    def unitary(self, matrix, qubits, label=""):
        """Apply an explicit unitary to ``qubits`` (ordered most-significant first)."""
        return self._add("unitary", tuple(qubits), matrix=np.asarray(matrix, complex), label=label)

    def controlled_unitary(self, matrix, control, targets, label=""):
        """Apply ``matrix`` on ``targets`` conditioned on ``control`` being |1>."""
        return self._add(
            "c-unitary",
            (control, *targets),
            matrix=np.asarray(matrix, complex),
            label=label,
        )

    # -- introspection -----------------------------------------------------
    @property
    def depth(self) -> int:
        """Circuit depth (longest chain of gates sharing a qubit)."""
        last = [0] * self.n_qubits
        for g in self.gates:
            d = max(last[q] for q in g.qubits) + 1
            for q in g.qubits:
                last[q] = d
        return max(last) if last else 0

    def gate_counts(self) -> dict:
        counts: dict[str, int] = {}
        for g in self.gates:
            counts[g.name] = counts.get(g.name, 0) + 1
        return dict(sorted(counts.items()))

    def __len__(self) -> int:
        return len(self.gates)

    def __repr__(self) -> str:
        return (
            f"Circuit({self.name!r}, n_qubits={self.n_qubits}, "
            f"gates={len(self.gates)}, depth={self.depth})"
        )
