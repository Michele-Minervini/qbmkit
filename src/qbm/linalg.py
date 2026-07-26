"""Small dense linear-algebra helpers shared across the library.

Qubit 0 is the leftmost tensor factor; visible qubits are the leading factors and
hidden qubits the trailing ones (see DESIGN.md, Section 3).
"""

from __future__ import annotations

import numpy as np

_TOL = 1e-12


def partial_trace_hidden(M: np.ndarray, n_visible: int, n_hidden: int) -> np.ndarray:
    """Trace out the trailing ``n_hidden`` qubits of a dense operator ``M``.

    Returns the ``2^n_visible x 2^n_visible`` reduced operator on the visible
    qubits.  ``n_hidden == 0`` returns ``M`` unchanged.
    """
    if n_hidden == 0:
        return M
    dv, dh = 1 << n_visible, 1 << n_hidden
    return np.einsum("ahbh->ab", M.reshape(dv, dh, dv, dh))


def unitary_derivative_kernel(eigvals: np.ndarray) -> np.ndarray:
    """Daleckii-Krein kernel for the derivative of ``U = exp(-i H)``.

    In the eigenbasis ``H = W diag(eps) W^dag``, ``d_x exp(-iH)`` equals
    ``W ((W^dag (d_x H) W) * Lambda) W^dag`` with

        Lambda[a, b] = (e^{-i eps_a} - e^{-i eps_b}) / (eps_a - eps_b),

    and the diagonal/degenerate limit ``Lambda[a, a] = -i e^{-i eps_a}``.  This gives
    the exact derivative of the real-time evolution w.r.t. its parameters (Evolved QBM).
    """
    eps = np.asarray(eigvals, dtype=float)
    fa = np.exp(-1j * eps)
    de = eps[:, None] - eps[None, :]
    small = np.abs(de) < _TOL
    num = fa[:, None] - fa[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        Lam = num / np.where(small, 1.0, de)
    diag = np.broadcast_to((-1j * fa)[:, None], Lam.shape)
    return np.where(small, diag, Lam)


def log_divided_differences(eigvals: np.ndarray) -> np.ndarray:
    """Divided-difference matrix of ``log`` for the Frechet derivative of ``log``.

    ``L[a, b] = (ln m_a - ln m_b) / (m_a - m_b)`` with the diagonal/degenerate
    limit ``1 / m_a``.  Used so that ``d log(sigma) = U ((U^dag dSigma U) * L) U^dag``
    in the eigenbasis ``sigma = U diag(m) U^dag``.
    """
    m = np.asarray(eigvals, dtype=float)
    lm = np.log(m)
    dm = m[:, None] - m[None, :]
    dlm = lm[:, None] - lm[None, :]
    small = np.abs(dm) < _TOL
    with np.errstate(divide="ignore", invalid="ignore"):
        L = dlm / np.where(small, 1.0, dm)
    return np.where(small, 1.0 / m[:, None], L)
