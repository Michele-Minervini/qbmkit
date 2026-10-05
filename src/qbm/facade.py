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


def _kl_monitor(q: np.ndarray):
    """Monitor recording ``D(q || p_model)`` from the state the step already built.

    Goes quiet -- rather than failing the run -- on a backend that cannot return the
    full distribution at this size.
    """
    support = q > 0
    entropy_term = float(np.sum(q[support] * np.log(q[support])))
    available = True

    def monitor(state, model):
        nonlocal available
        if not available:
            return None
        try:
            p = np.asarray(state.probabilities(), dtype=float)
        except NotImplementedError:
            available = False
            return None
        return entropy_term - float(q[support] @ np.log(np.clip(p[support], 1e-300, None)))

    return monitor


def learn(
    data,
    steps: int = 300,
    lr: float = 0.1,
    optimizer=None,
    model=None,
    backend=None,
    connectivity="all",
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
    connectivity : str or list of (int, int)
        Coupling graph of the default model -- anything :func:`qbm.pauli_pool` accepts:
        ``"all"``, ``"chain"``, ``"ring"``, ``"grid"``, ``"star"`` or an explicit list
        of edges.  ``"all"`` (all-to-all) is the strong default for generic targets;
        use a sparser graph when the data has that structure.
    optimizer, model : optional overrides.
    monitor : {"auto"} | callable | None
        Recorded into ``model.history.monitor`` each step.  ``"auto"`` tracks the
        measured KL divergence ``D(q || p_model)`` -- the training curve that stays
        available on a measurement backend, where the relative-entropy *value* (which
        needs ``log Z``) does not.  A backend that cannot return the whole distribution
        at this size (a tensor network beyond ~14 qubits) records nothing;
        ``model.history.grad_norm`` -- the mismatch between data and model moments --
        is the curve that is always there.

    Notes
    -----
    Nothing here is larger than the ``2^n`` probability vector: the target enters the
    loss through its moments ``<G_j>_data``, read from the Pauli labels.  A backend that
    also works from labels is then limited only by its own cost -- on
    ``"tensor_network"`` this trains a 20-qubit chain, well past the ~13-qubit ceiling
    of a dense density matrix.

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
        monitor = _kl_monitor(q)

    model.history = fit(
        model,
        RelativeEntropy(q),  # a probability vector: diag(q) is never built
        optimizer,
        steps=steps,
        verbose=verbose,
        monitor=monitor,
    )
    return model
