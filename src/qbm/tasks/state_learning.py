"""Quantum-state learning task (match a target density matrix)."""

from __future__ import annotations

import numpy as np

from ..losses.relative_entropy import MarginalRelativeEntropy, RelativeEntropy
from ..models.fully_visible import FullyVisibleQBM
from ..models.visible_hidden import VisibleHiddenQBM
from ..operators import pauli_pool
from ..optim.natural_gradient import NaturalGradient
from ..train.loop import fit
from .base import Result


def learn_state(
    sigma,
    n_hidden: int = 0,
    model=None,
    steps: int = 300,
    lr: float = 0.3,
    metric: str = "kubo_mori",
    reg: float = 1e-6,
    optimizer=None,
    backend=None,
    locality: int = 2,
    init_scale: float = 0.05,
    seed: int = 0,
    verbose: bool = False,
) -> Result:
    """Train a QBM to reproduce a target quantum state by relative-entropy minimisation.

    With ``n_hidden = 0`` the model is fully visible and the exact
    ``<G_j>_data - <G_j>_model`` gradient is used.  With hidden units the loss is the
    reduced-state relative entropy ``D(sigma || Tr_h rho)``.

    Parameters
    ----------
    sigma : ndarray
        Target density matrix on the visible qubits.
    n_hidden : int
        Number of hidden qubits to add to the model.
    locality : int
        Locality of the default Pauli generator pool (2 = every 1- and 2-body Pauli),
        so any 2-local Gibbs target is exactly representable.

    Returns
    -------
    Result
        With ``.relative_entropy`` (the final ``D(sigma || rho)``).

    Examples
    --------
    >>> import qbm
    >>> sigma = qbm.oracles.gibbs(qbm.hamiltonians.heisenberg(3), beta=1.0)
    >>> res = qbm.learn_state(sigma, steps=200)   # doctest: +SKIP
    """
    sigma = np.asarray(sigma, dtype=complex)
    n_visible = int(round(np.log2(sigma.shape[0])))

    if model is None:
        if n_hidden > 0:
            model = VisibleHiddenQBM(n_visible=n_visible, n_hidden=n_hidden, backend=backend)
        else:
            model = FullyVisibleQBM(
                generators=pauli_pool(n_visible, locality=locality), backend=backend
            )
        model.theta = np.random.default_rng(seed).normal(scale=init_scale, size=model.n_params)

    hidden = model.n_qubits - n_visible
    loss = (
        RelativeEntropy(sigma)
        if hidden == 0
        else MarginalRelativeEntropy(sigma, n_visible=n_visible)
    )
    if optimizer is None:
        optimizer = NaturalGradient(metric=metric, lr=lr, reg=reg)

    history = fit(model, loss, optimizer, steps=steps, verbose=verbose)
    return Result(model, history, "state_learning", relative_entropy=history.final_loss)
