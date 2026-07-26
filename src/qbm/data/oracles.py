"""Exact reference quantities (validation oracles) for small systems."""

from __future__ import annotations

import numpy as np


def gibbs(H: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """Exact Gibbs state ``exp(-beta H) / Z`` via eigendecomposition (overflow-safe)."""
    H = np.asarray(H, dtype=complex)
    w, V = np.linalg.eigh(H)
    shifted = np.exp(-beta * (w - w[0]))
    p = shifted / shifted.sum()
    return (V * p) @ V.conj().T


def ground_energy(H: np.ndarray) -> float:
    """Smallest eigenvalue of ``H``."""
    return float(np.linalg.eigvalsh(np.asarray(H, dtype=complex))[0])


def ground_state(H: np.ndarray):
    """``(E0, psi0)`` for the ground state of ``H``."""
    w, V = np.linalg.eigh(np.asarray(H, dtype=complex))
    return float(w[0]), V[:, 0]


def free_energy(H: np.ndarray, beta: float = 1.0) -> float:
    """Exact free energy ``-(1/beta) log Tr exp(-beta H)``."""
    w = np.linalg.eigvalsh(np.asarray(H, dtype=complex))
    w0 = w[0]
    logZ = -beta * w0 + np.log(np.sum(np.exp(-beta * (w - w0))))
    return float(-logZ / beta)
