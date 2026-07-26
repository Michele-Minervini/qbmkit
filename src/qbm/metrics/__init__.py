"""Quantum-Fisher-information / monotone information metrics for QBMs.

All three metrics share one formula and differ only by the Morozova-Chentsov
weight kernel (see :func:`mc_weight`).  Kubo-Mori is the default for Gibbs-state
QBMs (it is the free-energy Hessian); Fisher-Bures (SLD) is the right choice for
pure-state / circuit ansaetze.
"""

from .base import FisherBures, KuboMori, Metric, WignerYanase, get_metric
from .monotone import mc_weight

__all__ = [
    "Metric",
    "KuboMori",
    "FisherBures",
    "WignerYanase",
    "get_metric",
    "mc_weight",
]
