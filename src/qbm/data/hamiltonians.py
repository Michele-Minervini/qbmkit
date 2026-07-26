"""Standard target / model Hamiltonians, returned as dense Hermitian matrices."""

from __future__ import annotations

import numpy as np

from ..operators import pauli


def _term(n: int, ops: dict) -> np.ndarray:
    """Build a Pauli-string matrix from ``{qubit: 'X'|'Y'|'Z'}``."""
    label = ["I"] * n
    for q, o in ops.items():
        label[q] = o
    return pauli("".join(label))


def tfim(n: int, J: float = 1.0, g: float = 1.0, periodic: bool = False) -> np.ndarray:
    """Transverse-field Ising model ``H = -J sum Z_i Z_{i+1} - g sum X_i``."""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=complex)
    last = n if periodic else n - 1
    for i in range(last):
        H -= J * _term(n, {i: "Z", (i + 1) % n: "Z"})
    for i in range(n):
        H -= g * _term(n, {i: "X"})
    return H


def heisenberg(n: int, J: float = 1.0, periodic: bool = False) -> np.ndarray:
    """Isotropic Heisenberg chain ``H = J sum (X X + Y Y + Z Z)``."""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=complex)
    last = n if periodic else n - 1
    for i in range(last):
        j = (i + 1) % n
        for o in ("X", "Y", "Z"):
            H += J * _term(n, {i: o, j: o})
    return H
