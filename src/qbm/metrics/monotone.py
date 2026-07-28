"""Morozova-Chentsov weight kernels for the monotone metrics.

Every monotone metric is ``g_ij = sum_kl W[k,l] Re( (d_i rho)_kl conj((d_j rho)_kl) )``
with ``W = 1 / c(p_k, p_l)`` for a metric-specific mean ``c``:

============  =================================  ===========================
metric        mean ``c(x, y)``                   weight ``W = 1/c``
============  =================================  ===========================
Fisher-Bures  arithmetic ``(x+y)/2``             ``2/(x+y)``
Kubo-Mori     logarithmic ``(x-y)/(ln x-ln y)``  ``(ln x-ln y)/(x-y)``
Wigner-Yanase ``((sqrt x + sqrt y)/2)^2``        ``4/(sqrt x + sqrt y)^2``
============  =================================  ===========================

These satisfy ``g_FB <= g_WY <= 2 g_FB`` and ``g_KM >= g_FB`` (enforced by tests).
Gibbs states are full rank, so all populations ``p_k`` are strictly positive.
"""

from __future__ import annotations

import numpy as np

_ALIASES = {
    "fisher_bures": "fisher_bures",
    "fisherbures": "fisher_bures",
    "bures": "fisher_bures",
    "sld": "fisher_bures",
    "fb": "fisher_bures",
    "kubo_mori": "kubo_mori",
    "kubomori": "kubo_mori",
    "bkm": "kubo_mori",
    "km": "kubo_mori",
    "wigner_yanase": "wigner_yanase",
    "wigneryanase": "wigner_yanase",
    "wy": "wigner_yanase",
}

_DEGENERATE_TOL = 1e-12
# below this, the alpha-z kernel's 1/(1-alpha) prefactor is numerically singular and we
# use the exact Kubo-Mori limit instead
_ALPHA_TOL = 1e-6
# Boltzmann factors underflow to 0 below ~exp(-745); floor them so the metric weights
# stay finite (1/_POP_FLOOR is representable in float64).
_POP_FLOOR = 1e-300


def canonical_metric_name(kind: str) -> str:
    key = kind.lower().replace("-", "_").replace(" ", "_")
    if key not in _ALIASES:
        raise ValueError(
            f"unknown metric {kind!r}; choose from fisher_bures, kubo_mori, wigner_yanase"
        )
    return _ALIASES[key]


def alpha_z_weight(p: np.ndarray, alpha: float, z: float) -> np.ndarray:
    """Weight kernel of the **alpha-z information matrix** (arXiv:2510.02218, Thm 10).

    The information matrix arising from the alpha-z Renyi relative entropy is

    ``[I_{alpha,z}]_{ij} = sum_{k,l} zeta_{alpha,z}(lam_k, lam_l) Tr[P_k d_i rho P_l d_j rho]``

    with (Eq. 5.6)::

        zeta(x, y) = z/(alpha(1-alpha)) * (x^((1-a)/z) - y^((1-a)/z))/(x - y)
                                        * (x^(a/z) - y^(a/z))/(x^(1/z) - y^(1/z))
        zeta(x, x) = 1/x

    This one kernel contains the whole family: ``alpha -> 1`` gives Kubo-Mori,
    ``(alpha, z) = (1/2, 1/2)`` gives Fisher-Bures, ``(1/2, 1)`` gives Wigner-Yanase,
    ``z = 1`` the Petz-Renyi and ``z = alpha`` the sandwiched-Renyi families.
    """
    p = np.clip(np.asarray(p, dtype=float), _POP_FLOOR, None)
    if abs(alpha - 1.0) < _ALPHA_TOL:  # exact Kubo-Mori limit (the formula is 0/0 there)
        return mc_weight(p, "kubo_mori")
    if z <= 0:
        raise ValueError("z must be positive")
    if alpha <= 0:
        raise ValueError("alpha must be positive")

    x = p[:, None]
    y = p[None, :]
    diff = x - y
    small = np.abs(diff) < _DEGENERATE_TOL
    safe = np.where(small, 1.0, diff)

    e1 = (1.0 - alpha) / z
    e2 = alpha / z
    e3 = 1.0 / z
    with np.errstate(divide="ignore", invalid="ignore"):
        first = (x**e1 - y**e1) / safe
        den = x**e3 - y**e3
        second = (x**e2 - y**e2) / np.where(small, 1.0, den)
        W = (z / (alpha * (1.0 - alpha))) * first * second
    return np.where(small, 1.0 / x, W)


def alpha_z_is_monotone(alpha: float, z: float) -> bool:
    """Whether ``D_{alpha,z}`` obeys the data-processing inequality (Fact 9 / Zhang 2020).

    True iff ``0 < alpha < 1`` and ``z >= max(alpha, 1 - alpha)``, or ``alpha > 1`` and
    ``alpha - 1 <= z <= alpha <= 2 z``.  Outside this region the resulting information
    matrix is still well defined but is *not* guaranteed to be a monotone metric.
    """
    if alpha <= 0 or z <= 0:
        return False
    if alpha < 1:
        return z >= max(alpha, 1.0 - alpha)
    if alpha == 1:
        return True  # Kubo-Mori limit
    return (alpha - 1 <= z) and (z <= alpha) and (alpha <= 2 * z)


def mc_weight(p: np.ndarray, kind) -> np.ndarray:
    """Weight matrix ``W[k, l]`` for a monotone metric.

    ``kind`` is either a registered name (``"kubo_mori"``, ``"fisher_bures"``,
    ``"wigner_yanase"``) or **any callable** ``p -> W``, which is how parameterized
    families (e.g. :class:`~qbm.metrics.AlphaZ`) and user-defined metrics plug in.

    Populations are floored at a tiny positive value: at very low temperature the
    Boltzmann factors underflow to exactly zero, which would make the weights
    ``1/0 -> inf`` and poison the metric with NaNs.  The corresponding state
    derivatives vanish there too, so the floored entries contribute nothing.
    """
    if callable(kind):
        return np.asarray(kind(np.clip(np.asarray(p, dtype=float), _POP_FLOOR, None)))
    name = canonical_metric_name(kind)
    p = np.clip(np.asarray(p, dtype=float), _POP_FLOOR, None)
    x = p[:, None]
    y = p[None, :]
    if name == "fisher_bures":
        return 2.0 / (x + y)
    if name == "wigner_yanase":
        return 4.0 / (np.sqrt(x) + np.sqrt(y)) ** 2
    # kubo_mori: logarithmic-mean weight, with the x≈y limit -> 1/x
    diff = x - y
    small = np.abs(diff) < _DEGENERATE_TOL
    with np.errstate(divide="ignore", invalid="ignore"):
        W = (np.log(x) - np.log(y)) / np.where(small, 1.0, diff)
    return np.where(small, 1.0 / x, W)
