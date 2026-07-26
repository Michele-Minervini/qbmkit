"""Trainability diagnostics: gradient-variance (barren-plateau) scans.

These reuse the *same* gradient and metric estimators as training, so a plateau
study consumes exactly the objects an optimizer would -- no separate code path.
"""

from __future__ import annotations

import numpy as np


def gradient_samples(make_model, loss, n_samples=100, scale=1.0, seed=0):
    """Sample gradients at random parameter points.

    Parameters
    ----------
    make_model : callable() -> Model
        Factory returning a fresh model (fixes architecture, randomises theta).
    loss : Loss
    n_samples : int
        Number of random parameter points.
    scale : float
        Std-dev of the i.i.d. normal parameter initialisation.
    seed : int

    Returns
    -------
    ndarray, shape (n_samples, n_params)
        The sampled gradient vectors.
    """
    rng = np.random.default_rng(seed)
    grads = []
    for _ in range(n_samples):
        model = make_model()
        model.theta = rng.normal(scale=scale, size=model.n_params)
        grads.append(np.asarray(loss.grad(model.state()), dtype=float))
    return np.array(grads)


def gradient_variance(make_model, loss, n_samples=100, scale=1.0, seed=0):
    """Mean over parameters of the per-component gradient variance."""
    g = gradient_samples(make_model, loss, n_samples=n_samples, scale=scale, seed=seed)
    return float(np.mean(np.var(g, axis=0)))


def barren_plateau_scan(make_model_for_n, make_loss_for_n, sizes, n_samples=80, scale=1.0, seed=0):
    """Gradient variance vs system size -- the barren-plateau signature.

    Parameters
    ----------
    make_model_for_n : callable(n) -> (callable() -> Model)
        Given a size ``n``, returns a *factory* of fresh models of that size.
    make_loss_for_n : callable(n) -> Loss
    sizes : iterable of int
    n_samples, scale, seed : sampling controls.

    Returns
    -------
    dict[int, float]
        ``{n: mean gradient variance}``.  An (approximately) exponential decay in
        ``n`` indicates a barren plateau.
    """
    out = {}
    for n in sizes:
        out[int(n)] = gradient_variance(
            make_model_for_n(n), make_loss_for_n(n), n_samples=n_samples, scale=scale, seed=seed
        )
    return out
