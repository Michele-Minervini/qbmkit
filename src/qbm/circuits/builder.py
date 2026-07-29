"""Circuit constructions for QBM algorithms, written in the vendor-neutral IR.

Register layout used throughout (qubit 0 is the most significant):

===============  =========================================================
qubit            role
===============  =========================================================
``0``            Hadamard-test ancilla (only in estimator circuits)
``1 .. n``       system qubits carrying ``rho(theta)``
``n+1 .. 2n``    purification ancillas (traced out)
===============  =========================================================

The thermal state is prepared as its **thermofield double**, the same purification the
statevector and tensor-network backends use, so ``Tr_anc |psi><psi| = rho(theta)``.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla

from .ir import Circuit


def state_preparation(amplitudes, n_qubits, offset=0, circuit=None, label="prep") -> Circuit:
    """Prepare an arbitrary normalised statevector on ``n_qubits`` (offset by ``offset``).

    Emitted as a single ``unitary`` instruction whose matrix maps |0...0> to the target
    amplitudes.  A hardware compiler (or the Qiskit/PennyLane adapters) decomposes this
    into elementary gates; keeping it as one instruction keeps the IR small and exact.
    """
    v = np.asarray(amplitudes, dtype=complex).reshape(-1)
    v = v / np.linalg.norm(v)
    dim = 2**n_qubits
    if v.size != dim:
        raise ValueError(f"expected {dim} amplitudes, got {v.size}")
    # any unitary whose first column is v (Householder-style completion)
    M = np.eye(dim, dtype=complex)
    M[:, 0] = v
    Q, R = np.linalg.qr(M)
    Q = Q * (np.sign(np.real(np.diag(R))) + 0j)  # fix QR sign convention
    if not np.allclose(Q[:, 0], v, atol=1e-9):
        Q[:, 0] = v  # QR may flip the first column; enforce it and re-orthogonalise
        Q, _ = np.linalg.qr(Q)
        if not np.allclose(Q[:, 0], v, atol=1e-9):
            Q = _unitary_with_first_column(v)
    circuit = circuit or Circuit(n_qubits + offset, name=label)
    circuit.unitary(Q, tuple(range(offset, offset + n_qubits)), label=label)
    return circuit


def _unitary_with_first_column(v):
    """Householder reflection mapping e_0 -> v (always exact)."""
    dim = v.size
    e0 = np.zeros(dim, dtype=complex)
    e0[0] = 1.0
    phase = np.exp(-1j * np.angle(v[0])) if abs(v[0]) > 1e-14 else 1.0
    w = v * phase - e0
    nw = np.linalg.norm(w)
    if nw < 1e-14:
        return np.eye(dim, dtype=complex) * np.conj(phase)
    w = w / nw
    H = np.eye(dim, dtype=complex) - 2 * np.outer(w, w.conj())
    return H * np.conj(phase)


def thermofield_double(ham, theta) -> np.ndarray:
    """TFD amplitudes of ``rho(theta) = e^{-G(theta)}/Z`` on ``2n`` qubits."""
    G = ham.matrix(np.asarray(theta, dtype=float))
    w, V = np.linalg.eigh(G)
    p = np.exp(-(w - w[0]))
    p = p / p.sum()
    # |TFD> = sum_k sqrt(p_k) |v_k>_sys |k>_anc
    Psi = V * np.sqrt(p)[None, :]
    return Psi.reshape(-1)


def gibbs_preparation(ham, theta, ancilla_offset=0) -> Circuit:
    """Circuit preparing the TFD purification of ``rho(theta)``.

    ``ancilla_offset`` reserves leading qubits (e.g. 1 for a Hadamard-test ancilla).
    """
    n = ham.n_qubits
    amps = thermofield_double(ham, theta)
    circ = Circuit(2 * n + ancilla_offset, name="gibbs-tfd")
    return state_preparation(amps, 2 * n, offset=ancilla_offset, circuit=circ, label="TFD")


def hadamard_test(prep: Circuit, unitary, targets, imaginary=False) -> Circuit:
    """Hadamard test measuring ``Re <U>`` (or ``Im <U>``) in the prepared state.

    Ancilla is qubit 0.  ``P(0) - P(1) = Re <U>`` (with the ``S^dagger`` variant giving
    the imaginary part), so the estimate is ``<Z>`` on the ancilla.
    """
    circ = Circuit(prep.n_qubits, name="hadamard-test")
    circ.gates.extend(prep.gates)
    circ.h(0)
    if imaginary:
        circ.sdg(0)
    circ.controlled_unitary(unitary, 0, tuple(targets), label="c-U")
    circ.h(0)
    return circ


def evolution_unitary(ham, theta, t) -> np.ndarray:
    """Exact ``e^{-i G(theta) t}`` (the simulator path)."""
    G = ham.matrix(np.asarray(theta, dtype=float))
    return sla.expm(-1j * G * t)


def trotter_evolution(ham, theta, t, steps=1) -> list:
    """Second-order Trotter decomposition of ``e^{-iG(theta)t}`` into Pauli rotations.

    Returns a list of ``(pauli_label, angle)`` pairs.  This is the hardware-faithful
    form: each ``exp(-i c P t)`` compiles to a basis change plus a CNOT ladder and one
    ``rz``.  Provided so circuit depth can be counted honestly; the simulator uses the
    exact unitary above.
    """
    terms = [(lbl, c) for lbl, c in zip(ham.labels, np.asarray(theta, float)) if c != 0.0]
    dt = t / steps
    half = [(lbl, c * dt / 2) for lbl, c in terms]
    seq = []
    for _ in range(steps):
        seq.extend(half)
        seq.extend(half[::-1])
    return seq


def pauli_rotation(circ: Circuit, label: str, angle: float, offset: int = 0) -> Circuit:
    """Append ``exp(-i angle P)`` for a Pauli string ``P`` using the CNOT-ladder form."""
    active = [(i, p) for i, p in enumerate(label) if p != "I"]
    if not active:
        return circ
    qs = [offset + i for i, _ in active]
    for (i, p), q in zip(active, qs):  # rotate into the Z basis
        if p == "X":
            circ.h(q)
        elif p == "Y":
            circ.rx(np.pi / 2, q)
    for a, b in zip(qs, qs[1:]):
        circ.cx(a, b)
    circ.rz(2 * angle, qs[-1])
    for a, b in reversed(list(zip(qs, qs[1:]))):
        circ.cx(a, b)
    for (i, p), q in zip(active, qs):
        if p == "X":
            circ.h(q)
        elif p == "Y":
            circ.rx(-np.pi / 2, q)
    return circ
