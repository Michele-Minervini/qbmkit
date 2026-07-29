"""Qiskit adapter (optional): IR -> ``qiskit.QuantumCircuit``, and a hardware runner.

Kept deliberately small.  Only this file needs updating if Qiskit changes its API,
which is the point of routing everything through the internal IR.

Qubit ordering: the IR uses qubit 0 as the *most significant* factor, Qiskit uses
little-endian, so indices are reversed on conversion; the round-trip test in the suite
pins that the resulting statevector matches our simulator.
"""

from __future__ import annotations

import numpy as np


def to_qiskit(circuit):
    """Convert a :class:`~qbm.circuits.ir.Circuit` to ``qiskit.QuantumCircuit``."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate

    n = circuit.n_qubits
    qc = QuantumCircuit(n, name=circuit.name)

    def q(i):  # IR qubit 0 = most significant -> Qiskit little-endian
        return n - 1 - i

    for g in circuit.gates:
        name, qs, p = g.name, [q(i) for i in g.qubits], g.params
        if name in ("h", "x", "y", "z", "s", "sdg", "t", "tdg"):
            getattr(qc, name)(qs[0])
        elif name in ("rx", "ry", "rz"):
            getattr(qc, name)(p[0], qs[0])
        elif name == "phase":
            qc.p(p[0], qs[0])
        elif name == "cx":
            qc.cx(qs[0], qs[1])
        elif name == "cz":
            qc.cz(qs[0], qs[1])
        elif name == "unitary":
            qc.append(UnitaryGate(_reorder(g.matrix, len(g.qubits))), qs)
        elif name == "c-unitary":
            gate = UnitaryGate(_reorder(g.matrix, len(g.qubits) - 1)).control(1)
            qc.append(gate, [qs[0], *qs[1:]])
        else:
            raise ValueError(f"no Qiskit mapping for gate {name!r}")
    return qc


def _reorder(mat, k):
    """Re-index a k-qubit matrix from IR (qubit 0 = MSB) to Qiskit (little-endian)."""
    mat = np.asarray(mat, dtype=complex)
    if k <= 1:
        return mat
    perm = list(range(k))[::-1]
    t = mat.reshape([2] * (2 * k))
    t = np.transpose(t, perm + [k + p for p in perm])
    return t.reshape(2**k, 2**k)


def run_on_backend(circuit, backend, shots=1024, optimization_level=1):
    """Transpile and run on any Qiskit backend (Aer simulator or real hardware).

    Returns the measurement counts.  This is the Tier-3 hardware path: it works, but
    see the module docs on Gibbs-state preparation before expecting research-scale
    results on a device.
    """
    from qiskit import transpile

    qc = to_qiskit(circuit)
    qc.measure_all()
    tqc = transpile(qc, backend, optimization_level=optimization_level)
    job = backend.run(tqc, shots=shots)
    return job.result().get_counts()
