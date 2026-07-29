"""Hadamard-test estimators: gradients and information matrices from circuits.

These implement the *hardware* route. Both quantities have the same shape -- a
covariance of two operators, one of them smeared by the belief-propagation channel:

**Energy gradient** (arXiv:2410.12935)::

    d_j Tr[O rho] = <O> <G_j> - (1/2) < { O, Phi_p(G_j) } >

**Information matrix** (arXiv:2510.02218, Eq. 1.1)::

    I_{alpha,z}[i,j] = (1/2) < { Phi_q(G_i), G_j } > - <G_i> <G_j>

with ``Phi_q(X) = int dt q(t) e^{-iGt} X e^{+iGt}``.  For Kubo-Mori ``q = p`` (the
high-peak tent); for the alpha-z family ``q = p * p_{alpha,z}``, and sampling a
convolution is just adding two independent draws.

A device evaluates these by sampling ``t ~ q``, applying ``e^{-iGt}`` (Hamiltonian
simulation), and reading an anticommutator expectation off a Hadamard test.  Here the
same estimator runs on the circuit simulator, so the *estimator* is validated
independently of hardware noise: with enough time samples it reproduces the dense
analytic value.
"""

from __future__ import annotations

import numpy as np

from ..metrics.monotone import canonical_metric_name
from . import densities


def channel_multiplier(w, times, chunk=4096) -> np.ndarray:
    """Monte-Carlo estimate of the channel multiplier ``mean_t exp(-i (w_k - w_l) t)``.

    This is the empirical characteristic function of the sampled times; as the sample
    count grows it converges to ``qhat(w_k - w_l)`` -- the closed form the dense backend
    evaluates exactly.  Chunked so the intermediate never exceeds ``chunk * dim^2``.
    """
    dw = np.asarray(w)[:, None] - np.asarray(w)[None, :]
    times = np.asarray(times, dtype=float)
    total = np.zeros(dw.shape, dtype=complex)
    for start in range(0, times.size, chunk):
        block = times[start : start + chunk]
        total += np.exp(-1j * dw[None, :, :] * block[:, None, None]).sum(axis=0)
    return total / times.size


def _smear(state, X, times, eig=None) -> np.ndarray:
    """Monte-Carlo estimate of ``Phi_q(X)`` from sampled evolution times.

    Each sample is ``e^{-iGt} X e^{+iGt}`` -- exactly what a device realises with
    Hamiltonian simulation; here evaluated with the exact evolution operator.
    """
    if eig is None:
        w, V = np.linalg.eigh(state.ham.matrix(state.theta))
        eig = (w, V, channel_multiplier(w, times))
    w, V, mult = eig
    Xe = V.conj().T @ np.asarray(X, dtype=complex) @ V
    return V @ (Xe * mult) @ V.conj().T


def _eigendata(state, times):
    """Shared eigendecomposition and channel multiplier for one estimator call."""
    w, V = np.linalg.eigh(state.ham.matrix(state.theta))
    return w, V, channel_multiplier(w, times)


def _anticommutator_expectation(state, A, B) -> float:
    """``(1/2) < {A, B} >`` -- the quantity a Hadamard test returns."""
    rho = state.density_matrix()
    return float(np.real(np.sum(rho * (0.5 * (A @ B + B @ A)).T)))


def _times_for(state, alpha=None, z=None, n_time_samples=256):
    rng = getattr(state, "rng", None) or np.random.default_rng()
    return densities.sample_times(n_time_samples, alpha=alpha, z=z, rng=rng)


def energy_gradient(state, observable, n_time_samples=256) -> np.ndarray:
    """``d_j Tr[O rho]`` via the belief-propagation channel (arXiv:2410.12935)."""
    O = np.asarray(observable, dtype=complex)
    times = _times_for(state, n_time_samples=n_time_samples)
    exp_O = state.expect(O)
    ge = state.generator_expectations()
    eig = _eigendata(state, times)
    grad = np.empty(state.ham.n_params)
    for j, G_j in enumerate(state.ham.generators):
        smeared = _smear(state, G_j, times, eig)
        grad[j] = exp_O * ge[j] - _anticommutator_expectation(state, O, smeared)
    return grad


def information_matrix(state, kind="kubo_mori", n_time_samples=256) -> np.ndarray:
    """Information matrix from the channel formula (arXiv:2510.02218, Eq. 1.1).

    ``kind`` may be a named metric or an :class:`~qbm.metrics.AlphaZ` instance; only
    Kubo-Mori and the alpha-z family with ``alpha in (0,1)`` have a circuit form here.
    """
    alpha = z = None
    if hasattr(kind, "alpha") and hasattr(kind, "z"):
        alpha, z = float(kind.alpha), float(kind.z)
        if abs(alpha - 1.0) < 1e-9:
            alpha = z = None
    elif isinstance(kind, str):
        name = canonical_metric_name(kind)
        if name != "kubo_mori":
            raise NotImplementedError(
                f"the circuit estimator implements the Kubo-Mori and alpha-z channel "
                f"formulas; {name!r} has no such thermal-state form in the literature. "
                "Use backend='dense' for it, or pass an AlphaZ metric."
            )
    else:
        raise TypeError("kind must be a metric name or an AlphaZ instance")

    times = _times_for(state, alpha=alpha, z=z, n_time_samples=n_time_samples)
    ge = state.generator_expectations()
    gens = state.ham.generators
    J = len(gens)
    eig = _eigendata(state, times)
    smeared = [_smear(state, g, times, eig) for g in gens]
    out = np.empty((J, J))
    for i in range(J):
        for j in range(i, J):
            val = _anticommutator_expectation(state, smeared[i], gens[j]) - ge[i] * ge[j]
            out[i, j] = out[j, i] = val
    return 0.5 * (out + out.T)
