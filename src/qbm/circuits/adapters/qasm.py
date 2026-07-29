"""OpenQASM 3 emitter -- the dependency-free, SDK-independent export path.

QASM is the durable interchange format: it outlives SDK refactors and is accepted by
Qiskit, tket, Braket, Cirq and most hardware providers.  Opaque ``unitary`` /
``c-unitary`` instructions (used by exact state preparation and controlled evolution)
have no QASM primitive, so they are emitted only after being decomposed; call with
``decompose=True`` to expand them into elementary rotations first, or export circuits
built from the gate-level (Trotterised) constructions.
"""

from __future__ import annotations

import numpy as np

_SIMPLE = {"h", "x", "y", "z", "s", "sdg", "t", "tdg"}


def _fmt(x: float) -> str:
    """Shortest representation that round-trips exactly (repr semantics)."""
    return repr(float(x))


_ROT = {"rx", "ry", "rz"}


def to_qasm3(circuit, decompose: bool = False) -> str:
    """Emit OpenQASM 3 source for a :class:`~qbm.circuits.ir.Circuit`."""
    if decompose:
        circuit = decompose_unitaries(circuit)
    lines = [
        "OPENQASM 3.0;",
        'include "stdgates.inc";',
        f"qubit[{circuit.n_qubits}] q;",
        f"bit[{circuit.n_qubits}] c;",
    ]
    for g in circuit.gates:
        if g.name in _SIMPLE:
            lines.append(f"{g.name} q[{g.qubits[0]}];")
        elif g.name in _ROT:
            lines.append(f"{g.name}({_fmt(g.params[0])}) q[{g.qubits[0]}];")
        elif g.name == "phase":
            lines.append(f"p({_fmt(g.params[0])}) q[{g.qubits[0]}];")
        elif g.name == "cx":
            lines.append(f"cx q[{g.qubits[0]}], q[{g.qubits[1]}];")
        elif g.name == "cz":
            lines.append(f"cz q[{g.qubits[0]}], q[{g.qubits[1]}];")
        elif g.name in ("unitary", "c-unitary"):
            raise ValueError(
                f"cannot emit an opaque {g.name!r} instruction as QASM; pass "
                "decompose=True, or build the circuit from Trotterised gates "
                "(qbm.circuits.builder.pauli_rotation)."
            )
        else:
            raise ValueError(f"no QASM mapping for gate {g.name!r}")
    lines.append("c = measure q;")
    return "\n".join(lines)


def decompose_unitaries(circuit):
    """Expand opaque unitaries into elementary gates (single-qubit case only).

    A general ``k``-qubit unitary needs full isometry synthesis, which belongs in a
    vendor transpiler rather than here; for those, export via
    :func:`qbm.circuits.adapters.to_qiskit` and let Qiskit transpile.
    """
    from ..ir import Circuit

    out = Circuit(circuit.n_qubits, name=circuit.name + "-decomposed")
    for g in circuit.gates:
        if g.name == "unitary" and len(g.qubits) == 1:
            out.gates.extend(_zyz(g.matrix, g.qubits[0]).gates)
        elif g.name in ("unitary", "c-unitary"):
            raise NotImplementedError(
                f"decomposition of a {len(g.qubits)}-qubit {g.name!r} is not implemented; "
                "use the Qiskit adapter and its transpiler for multi-qubit synthesis."
            )
        else:
            out.gates.append(g)
    return out


def _zyz(u, qubit):
    """ZYZ Euler decomposition of a 2x2 unitary."""
    from ..ir import Circuit

    u = np.asarray(u, dtype=complex)
    det = np.linalg.det(u)
    su = u / np.sqrt(det + 0j)
    theta = 2 * np.arctan2(abs(su[1, 0]), abs(su[0, 0]))
    lam_plus_phi = 2 * np.angle(su[1, 1])
    lam_minus_phi = 2 * np.angle(su[1, 0]) if abs(su[1, 0]) > 1e-12 else 0.0
    phi = (lam_plus_phi + lam_minus_phi) / 2
    lam = (lam_plus_phi - lam_minus_phi) / 2
    c = Circuit(qubit + 1)
    c.rz(lam, qubit)
    c.ry(theta, qubit)
    c.rz(phi, qubit)
    return c
