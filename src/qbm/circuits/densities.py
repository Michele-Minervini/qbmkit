"""Time-smearing densities for the belief-propagation channels, and samplers for them.

The gradient and information-matrix estimators smear a generator under the model's own
real-time dynamics,

    ``Phi_q(X) = integral dt q(t) e^{-i G t} X e^{+i G t}``,

so a circuit estimator samples ``t ~ q(t)`` and runs ``e^{-iGt}``.  Two densities matter:

``p(t) = (2/pi) ln|coth(pi t / 2)|``
    the "high-peak tent" density behind the Kubo-Mori gradient/metric.
``p_{alpha,z}(t) = z / (2 pi alpha (1-alpha)) * ln(1 + (sin(pi alpha)/sinh(pi z t))^2)``
    the extra factor for the alpha-z family (arXiv:2510.02218, Eq. 1.3).

The alpha-z channel uses their **convolution** ``q_{alpha,z} = p * p_{alpha,z}``, so
sampling is simply ``t = t1 + t2`` with ``t1 ~ p`` and ``t2 ~ p_{alpha,z}`` drawn
independently -- no convolution integral is ever formed.  Both densities decay like
``e^{-pi t}``, so truncation is mild.

In the eigenbasis of ``G`` the channel multiplies matrix element ``(k,l)`` by the
characteristic function ``qhat(w_k - w_l)``; for ``p`` that is the closed form
``(2/D) tanh(D/2)`` used by the dense backend, which is how the circuit and spectral
routes are cross-checked.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def high_peak_tent(t) -> np.ndarray:
    """``p(t) = (2/pi) ln|coth(pi t / 2)|``."""
    t = np.abs(np.asarray(t, dtype=float))
    t = np.where(t < _EPS, _EPS, t)
    return (2.0 / np.pi) * np.log(np.abs(1.0 / np.tanh(np.pi * t / 2.0)))


def alpha_z_density(t, alpha: float, z: float) -> np.ndarray:
    """``p_{alpha,z}(t)`` of Eq. (1.3); the extra smearing for the alpha-z family.

    The paper states this density for ``alpha in (0, 1)``; at ``alpha >= 1`` the
    ``sin(pi alpha)`` factor degenerates (it vanishes identically at integer alpha), so
    the sampling route is restricted to that range.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(
            f"the alpha-z smearing density is defined for alpha in (0, 1); got {alpha}. "
            "Use the dense backend for alpha >= 1."
        )
    t = np.abs(np.asarray(t, dtype=float))
    t = np.where(t < _EPS, _EPS, t)
    s = np.sinh(np.pi * z * t)
    return (z / (2 * np.pi * alpha * (1 - alpha))) * np.log1p((np.sin(np.pi * alpha) / s) ** 2)


class InverseCDFSampler:
    """Sample from a symmetric density on the real line by tabulated inverse CDF.

    Both densities have an integrable logarithmic singularity at ``t = 0``, so the grid
    is geometric near the origin and uniform thereafter; a purely uniform grid
    mis-integrates the peak by ~1%.
    """

    def __init__(self, pdf, t_max=12.0, n_grid=20001):
        n_near = n_grid // 3
        near = np.geomspace(1e-10, 0.05, n_near)
        far = np.linspace(0.05, t_max, n_grid - n_near)[1:]
        grid = np.concatenate([near, far])
        vals = np.asarray(pdf(grid), dtype=float)
        vals = np.clip(np.nan_to_num(vals, nan=0.0, posinf=0.0), 0.0, None)
        cdf = np.concatenate([[0.0], np.cumsum((vals[1:] + vals[:-1]) / 2 * np.diff(grid))])
        self.mass = float(cdf[-1])  # half the total mass (density is symmetric)
        self.grid = grid
        self.cdf = cdf / max(cdf[-1], _EPS)

    def sample(self, size, rng=None) -> np.ndarray:
        rng = np.random.default_rng() if rng is None else rng
        u = rng.random(size)
        t = np.interp(u, self.cdf, self.grid)
        return t * rng.choice([-1.0, 1.0], size=size)  # symmetrise


def tent_sampler(**kw) -> InverseCDFSampler:
    return InverseCDFSampler(high_peak_tent, **kw)


def alpha_z_sampler(alpha: float, z: float, **kw) -> InverseCDFSampler:
    return InverseCDFSampler(lambda t: alpha_z_density(t, alpha, z), **kw)


def sample_times(size, alpha=None, z=None, rng=None, **kw) -> np.ndarray:
    """Draw ``t ~ q(t)``: the tent density, or the alpha-z convolution if given.

    Convolution sampling is exact: ``t = t1 + t2`` for independent draws.
    """
    rng = np.random.default_rng() if rng is None else rng
    t = tent_sampler(**kw).sample(size, rng)
    if alpha is not None and z is not None and abs(alpha - 1.0) > 1e-9:
        t = t + alpha_z_sampler(alpha, z, **kw).sample(size, rng)
    return t


def tent_characteristic(delta) -> np.ndarray:
    """``phat(D) = (2/D) tanh(D/2)`` -- the exact Fourier transform of ``p``."""
    delta = np.asarray(delta, dtype=float)
    small = np.abs(delta) < 1e-9
    safe = np.where(small, 1.0, delta)
    return np.where(small, 1.0, (2.0 / safe) * np.tanh(safe / 2.0))
