"""Metric objects: thin wrappers that ask a ThermalState for its QFI matrix."""

from __future__ import annotations

import warnings

import numpy as np

from .monotone import alpha_z_is_monotone, alpha_z_weight, canonical_metric_name, mc_weight


class Metric:
    """A monotone information metric identified by ``kind``."""

    def __init__(self, kind: str):
        self.kind = canonical_metric_name(kind)

    def weights(self, p) -> np.ndarray:
        """Morozova-Chentsov weight matrix for populations ``p``."""
        return mc_weight(p, self.kind)

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


class AlphaZ(Metric):
    """The **alpha-z information matrix** (arXiv:2510.02218, Wilde).

    A two-parameter family of quantum Fisher information matrices derived from the
    alpha-z Renyi relative entropies, which contains every metric this library shipped
    before as a special case::

        AlphaZ(alpha -> 1, any z)   == KuboMori
        AlphaZ(0.5, 0.5)            == FisherBures     (sandwiched Renyi 1/2)
        AlphaZ(0.5, 1.0)            == WignerYanase    (Petz-Renyi 1/2)
        AlphaZ(alpha, 1.0)          == Petz-Renyi family
        AlphaZ(alpha, alpha)        == sandwiched-Renyi family

    Parameters
    ----------
    alpha, z : float
        Renyi parameters, both positive.
    check_monotone : bool
        If True (default) warn when ``(alpha, z)`` lies outside the region where the
        data-processing inequality is known to hold (Fact 9 of the paper): the matrix
        is still well defined there, but is not guaranteed to be a monotone metric.
    """

    def __init__(self, alpha: float, z: float = 1.0, check_monotone: bool = True):
        self.alpha = float(alpha)
        self.z = float(z)
        self.kind = f"alpha_z(alpha={self.alpha:g}, z={self.z:g})"
        self.is_monotone = alpha_z_is_monotone(self.alpha, self.z)
        if check_monotone and not self.is_monotone:
            warnings.warn(
                f"alpha={self.alpha:g}, z={self.z:g} lies outside the region where the "
                "alpha-z Renyi relative entropy obeys the data-processing inequality "
                "(0<alpha<1 and z>=max(alpha,1-alpha), or alpha>1 and "
                "alpha-1<=z<=alpha<=2z), so the resulting matrix is not guaranteed to "
                "be a monotone metric.",
                RuntimeWarning,
                stacklevel=2,
            )

    def weights(self, p) -> np.ndarray:
        return alpha_z_weight(p, self.alpha, self.z)

    def matrix(self, state) -> np.ndarray:
        return state.metric(self.weights)

    def __repr__(self) -> str:
        return f"AlphaZ(alpha={self.alpha:g}, z={self.z:g})"


def PetzRenyi(alpha: float, **kw) -> AlphaZ:
    """Petz-Renyi information matrix: the ``z = 1`` slice of the alpha-z family."""
    return AlphaZ(alpha, 1.0, **kw)


def SandwichedRenyi(alpha: float, **kw) -> AlphaZ:
    """Sandwiched-Renyi information matrix: the ``z = alpha`` slice."""
    return AlphaZ(alpha, alpha, **kw)


class CustomMetric(Metric):
    """A metric defined by an arbitrary weight kernel ``f(x, y)``.

    ``kernel`` receives the two population arrays broadcast against each other
    (shapes ``(d, 1)`` and ``(1, d)``) and returns the weight matrix.  For a monotone
    metric this is ``1 / c(x, y)`` for an operator mean ``c`` (Morozova-Chentsov /
    Petz classification); nothing is assumed or checked.

    Examples
    --------
    >>> import qbm, numpy as np
    >>> geometric = qbm.metrics.CustomMetric(lambda x, y: 1 / np.sqrt(x * y), "geometric")
    """

    def __init__(self, kernel, name: str = "custom"):
        self.kernel = kernel
        self.kind = name

    def weights(self, p) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        return np.asarray(self.kernel(p[:, None], p[None, :]))

    def matrix(self, state) -> np.ndarray:
        return state.metric(self.weights)

    def __repr__(self) -> str:
        return f"CustomMetric({self.kind!r})"


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
