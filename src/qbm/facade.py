"""The beginner facade: ``qbm.learn(...)`` wires sensible defaults.

The facade returns the same objects the explicit API uses, so users can drop one
level at any time.
"""

from __future__ import annotations

import numpy as np

from .losses.relative_entropy import RelativeEntropy
from .models.fully_visible import FullyVisibleQBM
from .optim.gradient import Adam
from .train.loop import fit


def _as_distribution(data) -> np.ndarray:
    """Coerce ``data`` to a probability vector over ``2^n`` outcomes.

    Accepts a probability vector (length a power of two) or a 1-D array of
    integer computational-basis samples.
    """
    data = np.asarray(data)
    if data.ndim == 1 and np.issubdtype(data.dtype, np.floating) and np.isclose(data.sum(), 1.0):
        return data.astype(float)
    if data.ndim == 1 and np.issubdtype(data.dtype, np.integer):
        dim = 1 << int(np.ceil(np.log2(data.max() + 1)))
        q = np.bincount(data, minlength=dim).astype(float)
        return q / q.sum()
    raise ValueError(
        "data must be a probability vector (length a power of two) or a 1-D "
        "array of integer basis-state samples"
    )


def learn(
    data,
    steps: int = 300,
    lr: float = 0.1,
    optimizer=None,
    model=None,
    backend=None,
    connectivity: str = "all",
    init_scale: float = 0.05,
    seed: int = 0,
    verbose=False,
    monitor="auto",
):
    """Train a fully-visible QBM to reproduce a classical distribution.

    Parameters
    ----------
    data : array
        Probability vector over ``2^n`` outcomes, or integer samples.
    steps, lr : training budget and learning rate.
    backend : str | Backend, optional
        Engine that builds ``rho(theta)`` each step -- e.g.
        ``CircuitBackend(gibbs_prep="varqite")`` to train on variationally prepared
        thermal states.  Defaults to the exact dense engine.
    connectivity : {"all", "chain"}
        Two-body coupling topology of the default model. ``"all"`` (all-to-all)
        is the strong default for generic targets; ``"chain"`` for 1-D structure.
    optimizer, model : optional overrides.
    monitor : {"auto"} | callable | None
        Recorded into ``model.history.monitor`` each step.  ``"auto"`` tracks the
        measured KL divergence ``D(q || p_model)`` -- the training curve that stays
        available on a measurement backend, where the relative-entropy *value* (which
        needs ``log Z``) does not.

    Returns
    -------
    FullyVisibleQBM
        The trained model (also carries ``model.history``).
    """
    q = _as_distribution(data)
    n = int(round(np.log2(len(q))))
    if (1 << n) != len(q):
        raise ValueError("distribution length must be a power of two")
    if model is None:
        model = FullyVisibleQBM(n=n, connectivity=connectivity, backend=backend)
        # small random init breaks symmetry saddles (e.g. flip-symmetric targets)
        model.theta = np.random.default_rng(seed).normal(scale=init_scale, size=model.n_params)
    if optimizer is None:
        optimizer = Adam(lr=lr)
    if monitor == "auto":

        def monitor(state, mdl):
            return mdl.kl(q)

    model.history = fit(
        model,
        RelativeEntropy(np.diag(q.astype(complex))),
        optimizer,
        steps=steps,
        verbose=verbose,
        monitor=monitor,
    )
    return model
