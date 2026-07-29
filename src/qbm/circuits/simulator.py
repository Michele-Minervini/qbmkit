"""Dependency-free statevector simulator for the circuit IR.

This is the reference executor: it runs :class:`~qbm.circuits.ir.Circuit` objects with
pure NumPy, so the circuit algorithms are testable with no SDK installed.  Vendor
adapters (Qiskit / PennyLane) are alternative executors of the *same* IR, and the test
suite checks they agree with this one.
"""

from __future__ import annotations

import numpy as np

_S2 = 1 / np.sqrt(2)
_GATES = {
    "h": np.array([[_S2, _S2], [_S2, -_S2]], dtype=complex),
    "x": np.array([[0, 1], [1, 0]], dtype=complex),
    "y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "z": np.array([[1, 0], [0, -1]], dtype=complex),
    "s": np.array([[1, 0], [0, 1j]], dtype=complex),
    "sdg": np.array([[1, 0], [0, -1j]], dtype=complex),
    "t": np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex),
    "tdg": np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=complex),
}


def _rot(name, theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    if name == "rx":
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)
    if name == "ry":
        return np.array([[c, -s], [s, c]], dtype=complex)
    if name == "rz":
        return np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=complex)
    if name == "phase":
        return np.array([[1, 0], [0, np.exp(1j * theta)]], dtype=complex)
    raise ValueError(f"unknown rotation {name}")


def _apply(state, mat, qubits, n):
    """Apply ``mat`` (2^k x 2^k) to ``qubits`` of an n-qubit state (qubit 0 = MSB).

    ``reshape([2] * n)`` already makes tensor axis ``j`` the qubit ``j`` index (qubit 0
    most significant), so the axes to move are the qubit indices themselves.
    """
    k = len(qubits)
    state = state.reshape([2] * n)
    axes = list(qubits)
    state = np.moveaxis(state, axes, range(k))
    shape = state.shape
    state = state.reshape(2**k, -1)
    state = mat.reshape(2**k, 2**k) @ state
    state = state.reshape(shape)
    state = np.moveaxis(state, range(k), axes)
    return state.reshape(-1)


def run(circuit, initial_state=None) -> np.ndarray:
    """Execute a circuit and return the final statevector."""
    n = circuit.n_qubits
    if initial_state is None:
        state = np.zeros(2**n, dtype=complex)
        state[0] = 1.0
    else:
        state = np.asarray(initial_state, dtype=complex).reshape(-1).copy()

    for g in circuit.gates:
        if g.name in _GATES:
            state = _apply(state, _GATES[g.name], g.qubits, n)
        elif g.name in ("rx", "ry", "rz", "phase"):
            state = _apply(state, _rot(g.name, g.params[0]), g.qubits, n)
        elif g.name == "cx":
            state = _apply(state, _controlled(_GATES["x"]), g.qubits, n)
        elif g.name == "cz":
            state = _apply(state, _controlled(_GATES["z"]), g.qubits, n)
        elif g.name == "unitary":
            state = _apply(state, g.matrix, g.qubits, n)
        elif g.name == "c-unitary":
            state = _apply(state, _controlled(g.matrix), g.qubits, n)
        else:
            raise ValueError(f"simulator does not know gate {g.name!r}")
    return state


def _controlled(mat):
    """Control an ``m``-qubit unitary on one extra (most significant) qubit."""
    m = mat.shape[0]
    out = np.eye(2 * m, dtype=complex)
    out[m:, m:] = mat
    return out


def measure_z(state, qubit, n, shots=None, rng=None) -> float:
    """Expectation of Z on ``qubit``; with ``shots`` set, a finite-sample estimate."""
    probs = np.abs(state.reshape([2] * n)) ** 2
    p1 = float(np.sum(np.moveaxis(probs, qubit, 0)[1]))
    p1 = min(max(p1, 0.0), 1.0)
    if shots is None:
        return 1.0 - 2.0 * p1
    rng = np.random.default_rng() if rng is None else rng
    ones = rng.binomial(shots, p1)
    return 1.0 - 2.0 * ones / shots


def sample_bitstrings(state, n, shots, rng=None) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    p = np.abs(state) ** 2
    p = np.clip(p, 0, None)
    return rng.choice(len(p), size=shots, p=p / p.sum())
