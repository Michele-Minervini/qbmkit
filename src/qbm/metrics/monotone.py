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


def canonical_metric_name(kind: str) -> str:
    key = kind.lower().replace("-", "_").replace(" ", "_")
    if key not in _ALIASES:
        raise ValueError(
            f"unknown metric {kind!r}; choose from "
            "fisher_bures, kubo_mori, wigner_yanase"
        )
    return _ALIASES[key]


def mc_weight(p: np.ndarray, kind: str) -> np.ndarray:
    """Weight matrix ``W[k, l]`` for the requested monotone metric."""
    name = canonical_metric_name(kind)
    p = np.asarray(p, dtype=float)
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
