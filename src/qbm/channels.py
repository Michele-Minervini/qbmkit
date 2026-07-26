"""The shared eigenbasis kernels: belief-propagation channel and state derivative.

These are the single "hard primitives" of the whole library.  Working in the
eigenbasis of ``G(theta)`` they are exact and require no time sampling.
See DESIGN.md, Section 4, for the derivations.
"""

from __future__ import annotations

import numpy as np

_TOL = 1e-9


def bp_multiplier(w: np.ndarray) -> np.ndarray:
    """Eigenbasis multiplier ``M[k, l] = phi(w_k - w_l)`` of the belief-propagation
    channel ``Phi``, where ``phi(d) = (2/d) tanh(d/2)`` and ``phi(0) = 1``.

    ``phi`` is the Fourier transform of the high-peak-tent density
    ``p(t) = (2/pi) ln|coth(pi t / 2)|``, so applying ``M`` element-wise in the
    eigenbasis realises ``Phi`` exactly.
    """
    w = np.asarray(w, dtype=float)
    d = w[:, None] - w[None, :]
    small = np.abs(d) < _TOL
    d_safe = np.where(small, 1.0, d)
    M = (2.0 / d_safe) * np.tanh(d_safe / 2.0)
    M[small] = 1.0
    return M


def state_derivative_kernel(w: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Divided-difference kernel ``K`` for ``d_j rho`` in the eigenbasis.

    ``K[k, l] = (p_k - p_l) / (w_k - w_l)`` for ``k != l`` (and in the degenerate
    limit), with the diagonal value ``K[k, k] = -p_k``.  The full eigenbasis
    derivative is ``(d_j rho)_eig = G_j_eig * K + diag(p * <G_j>)``.
    """
    w = np.asarray(w, dtype=float)
    p = np.asarray(p, dtype=float)
    dw = w[:, None] - w[None, :]
    dp = p[:, None] - p[None, :]
    small = np.abs(dw) < _TOL
    dw_safe = np.where(small, 1.0, dw)
    K = dp / dw_safe
    # degenerate / diagonal limit of the divided difference -> dp/dw = -p_k
    K = np.where(small, np.broadcast_to(-p[:, None], K.shape), K)
    return K
