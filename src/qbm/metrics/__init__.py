"""Quantum-Fisher-information / monotone information metrics for QBMs.

Every metric here is a weight kernel ``W[k, l]`` applied to the state derivatives in
the eigenbasis of ``rho``,

    ``g_ij = sum_{k,l} W[k,l] Re( (d_i rho)_kl conj((d_j rho)_kl) )``,

which is exactly the structure of the general information matrix.  So the library
supports **arbitrary** metrics: pick one of the named ones, take a member of the
two-parameter :class:`AlphaZ` family (arXiv:2510.02218), or supply your own kernel
with :class:`CustomMetric`.

Kubo-Mori is the default for Gibbs-state QBMs (it is the free-energy Hessian);
Fisher-Bures (SLD) is the right choice for pure-state / circuit ansaetze.
"""

from .base import (
    AlphaZ,
    CustomMetric,
    FisherBures,
    KuboMori,
    Metric,
    PetzRenyi,
    SandwichedRenyi,
    WignerYanase,
    get_metric,
)
from .monotone import alpha_z_is_monotone, alpha_z_weight, mc_weight

__all__ = [
    "Metric",
    "KuboMori",
    "FisherBures",
    "WignerYanase",
    "AlphaZ",
    "PetzRenyi",
    "SandwichedRenyi",
    "CustomMetric",
    "get_metric",
    "mc_weight",
    "alpha_z_weight",
    "alpha_z_is_monotone",
]
