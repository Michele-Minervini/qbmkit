"""Automatic differentiation of QBM losses through the Gibbs construction (JAX).

This lets you train a *novel* objective with no hand-derived gradient: write the
loss value as a function of the (JAX) density matrix and let ``jax.grad`` do the
rest.  It is also how we cross-validate the analytic gradients in the test suite.

Requires the optional JAX dependency (``pip install qbmkit[jax]``).
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from .backends.jax_backend import gibbs_density_matrix  # noqa: E402


def value_and_grad(generators, theta, value_fn, offset=None):
    """Differentiate ``value_fn(rho(theta))`` w.r.t. ``theta`` via autodiff.

    Parameters
    ----------
    generators : list of ndarray
        The model generators ``G_j`` (e.g. ``model.ham.generators``).
    theta : array
        Parameter vector.
    value_fn : callable(rho_jax) -> real scalar
        The loss value as a JAX-traceable function of the density matrix.
    offset : ndarray, optional
        Fixed term added to ``G`` (``ParamHamiltonian.offset``).

    Returns
    -------
    (float, ndarray)
        The loss value and its gradient w.r.t. ``theta``.
    """
    Gs = jnp.asarray(np.stack([np.asarray(g, dtype=complex) for g in generators]))
    theta_j = jnp.asarray(np.asarray(theta, dtype=float))
    off = None if offset is None else jnp.asarray(np.asarray(offset, dtype=complex))

    def f(th):
        return value_fn(gibbs_density_matrix(th, Gs, off))

    v, g = jax.value_and_grad(f)(theta_j)
    return float(v), np.asarray(g)


def grad(generators, theta, value_fn, offset=None):
    """Gradient only (see :func:`value_and_grad`)."""
    return value_and_grad(generators, theta, value_fn, offset=offset)[1]
