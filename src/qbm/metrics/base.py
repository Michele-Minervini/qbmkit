"""Metric objects: thin wrappers that ask a ThermalState for its QFI matrix."""

from __future__ import annotations

import numpy as np

from .monotone import canonical_metric_name


class Metric:
    """A monotone information metric identified by ``kind``."""

    def __init__(self, kind: str):
        self.kind = canonical_metric_name(kind)

    def matrix(self, state) -> np.ndarray:
        """Return the QFI matrix for ``state`` (delegates to the backend)."""
        return state.metric(self.kind)

    def __call__(self, state) -> np.ndarray:
        return self.matrix(state)

    def __repr__(self) -> str:
        return f"Metric({self.kind!r})"


class KuboMori(Metric):
    """Kubo-Mori (BKM) metric -- the free-energy Hessian; default for QBMs."""

    def __init__(self):
        super().__init__("kubo_mori")


class FisherBures(Metric):
    """Fisher-Bures (SLD) quantum Fisher information."""

    def __init__(self):
        super().__init__("fisher_bures")


class WignerYanase(Metric):
    """Wigner-Yanase (skew-information) metric."""

    def __init__(self):
        super().__init__("wigner_yanase")


_REGISTRY = {
    "kubo_mori": KuboMori,
    "fisher_bures": FisherBures,
    "wigner_yanase": WignerYanase,
}


def get_metric(metric) -> Metric:
    """Coerce a string or :class:`Metric` into a :class:`Metric` instance."""
    if isinstance(metric, Metric):
        return metric
    return _REGISTRY[canonical_metric_name(metric)]()
