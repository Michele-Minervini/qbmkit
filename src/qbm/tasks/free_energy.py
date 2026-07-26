"""Free-energy minimisation task (variational Gibbs-state preparation)."""

from __future__ import annotations

import numpy as np

from ..data.oracles import free_energy as exact_free_energy
from ..losses.free_energy import FreeEnergy
from ..models.fully_visible import FullyVisibleQBM
from ..optim.natural_gradient import NaturalGradient
from ..train.loop import fit
from .base import Result


def free_energy_min(
    H,
    temperature: float = 1.0,
    model=None,
    steps: int = 400,
    lr: float = 0.3,
    metric: str = "kubo_mori",
    reg: float = 1e-4,
    optimizer=None,
    backend=None,
    connectivity: str = "chain",
    init_scale: float = 0.05,
    seed: int = 0,
    compare_exact: bool = True,
    verbose: bool = False,
) -> Result:
    """Minimise ``F = <H> - T S(rho)`` over the QBM parameters.

    The global minimum is the exact free energy ``-T log Tr exp(-H/T)``, attained at
    the Gibbs state of ``H`` at temperature ``T`` -- so this is also *variational
    Gibbs-state preparation*.

    Returns
    -------
    Result
        With ``.free_energy``, and ``.exact_free_energy`` / ``.error`` when available.

    Examples
    --------
    >>> import qbm
    >>> H = qbm.hamiltonians.tfim(3, g=1.0)
    >>> res = qbm.free_energy_min(H, temperature=1.0)   # doctest: +SKIP
    """
    H = np.asarray(H, dtype=complex)
    n = int(round(np.log2(H.shape[0])))
    if model is None:
        model = FullyVisibleQBM(n=n, backend=backend, connectivity=connectivity)
        model.theta = np.random.default_rng(seed).normal(scale=init_scale, size=model.n_params)
    if optimizer is None:
        optimizer = NaturalGradient(metric=metric, lr=lr, reg=reg)

    history = fit(
        model, FreeEnergy(H, temperature=temperature), optimizer, steps=steps, verbose=verbose
    )

    fields = {"free_energy": history.final_loss}
    if compare_exact and H.shape[0] <= 2**14:
        f_exact = exact_free_energy(H, beta=1.0 / temperature)
        fields["exact_free_energy"] = f_exact
        fields["error"] = history.final_loss - f_exact
    return Result(model, history, "free_energy_min", **fields)
