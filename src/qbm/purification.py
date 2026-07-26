"""Thermofield-double (TFD) purification of QBM thermal states.

A mixed Gibbs state ``rho = sum_k p_k |v_k><v_k|`` is the reduced state of a pure
state on a doubled Hilbert space (system + ancilla):

    |TFD> = sum_k sqrt(p_k) |v_k>_sys ⊗ |k>_anc,   Tr_anc |TFD><TFD| = rho .

This is the simulation route to circuit / hardware preparation (where the TFD would
be prepared variationally, e.g. VarQITE, instead of from the eigendecomposition),
and the natural object for entanglement-spectrum and barren-plateau studies.

Conventions: the flat statevector has the system register as the most significant
index, ``psi[i * dim + k] = sqrt(p_k) V[i, k]``.
"""

from __future__ import annotations

import numpy as np


def tfd_statevector(eigvecs: np.ndarray, populations: np.ndarray) -> np.ndarray:
    """Build the TFD statevector from the eigenvectors ``V`` and populations ``p`` of rho."""
    Psi = eigvecs * np.sqrt(np.clip(populations, 0.0, None))[None, :]
    return Psi.reshape(-1)


def reduced_system_state(psi: np.ndarray, dim: int) -> np.ndarray:
    """Reduced density matrix on the system register, ``Tr_anc |psi><psi|``."""
    Psi = psi.reshape(dim, dim)
    return Psi @ Psi.conj().T


def expectation(psi: np.ndarray, op: np.ndarray, dim: int) -> float:
    """System-register expectation ``<psi| O ⊗ I |psi>`` (independent of forming rho)."""
    Psi = psi.reshape(dim, dim)
    return float(np.real(np.einsum("ik,ij,jk->", np.conj(Psi), np.asarray(op, complex), Psi)))


def entanglement_entropy(psi: np.ndarray, dim: int) -> float:
    """System-ancilla entanglement entropy of the TFD = thermal entropy ``S(rho)``."""
    Psi = psi.reshape(dim, dim)
    s = np.linalg.svd(Psi, compute_uv=False)
    p = s**2
    p = p[p > 1e-300]
    return float(-np.sum(p * np.log(p)))
