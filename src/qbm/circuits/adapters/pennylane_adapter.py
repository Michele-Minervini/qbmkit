"""PennyLane adapter (optional): IR -> a PennyLane quantum function.

PennyLane's device plugins reach IBM, IonQ, Rigetti, Braket and others, so this is a
convenient hardware route.  Like the Qiskit adapter it is intentionally a thin leaf.
PennyLane wire 0 corresponds to our qubit 0 (both treat wire 0 as most significant in
their default state ordering), so no index reversal is needed.
"""

from __future__ import annotations

import numpy as np


def to_pennylane(circuit):
    """Return a function that applies ``circuit`` to the current PennyLane device."""
    import pennylane as qml

    def apply():
        for g in circuit.gates:
            name, qs, p = g.name, list(g.qubits), g.params
            if name == "h":
                qml.Hadamard(qs[0])
            elif name in ("x", "y", "z"):
                {"x": qml.PauliX, "y": qml.PauliY, "z": qml.PauliZ}[name](qs[0])
            elif name == "s":
                qml.S(qs[0])
            elif name == "sdg":
                qml.adjoint(qml.S)(qs[0])
            elif name == "t":
                qml.T(qs[0])
            elif name == "tdg":
                qml.adjoint(qml.T)(qs[0])
            elif name in ("rx", "ry", "rz"):
                {"rx": qml.RX, "ry": qml.RY, "rz": qml.RZ}[name](p[0], wires=qs[0])
            elif name == "phase":
                qml.PhaseShift(p[0], wires=qs[0])
            elif name == "cx":
                qml.CNOT(wires=qs)
            elif name == "cz":
                qml.CZ(wires=qs)
            elif name == "unitary":
                qml.QubitUnitary(np.asarray(g.matrix, complex), wires=qs)
            elif name == "c-unitary":
                qml.ControlledQubitUnitary(
                    np.asarray(g.matrix, complex), control_wires=[qs[0]], wires=qs[1:]
                )
            else:
                raise ValueError(f"no PennyLane mapping for gate {name!r}")

    return apply
