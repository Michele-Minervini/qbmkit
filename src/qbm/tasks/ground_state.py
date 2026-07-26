"""Ground-state energy estimation task."""

from __future__ import annotations

import numpy as np

from ..data.oracles import ground_energy
from ..losses.energy import Energy
from ..models.fully_visible import FullyVisibleQBM
from ..optim.natural_gradient import NaturalGradient
from ..train.loop import fit
from .base import Result


def ground_state(
    H,
    model=None,
    steps: int = 400,
    lr: float = 0.2,
    metric: str = "kubo_mori",
    reg: float = 1e-3,
    optimizer=None,
    backend=None,
    connectivity: str = "chain",
    init_scale: float = 0.05,
    seed: int = 0,
    compare_exact: bool = True,
    verbose: bool = False,
) -> Result:
    """Estimate the ground-state energy of ``H`` by minimising ``Tr[H rho(theta)]``.

    Uses quantum natural gradient with the Kubo-Mori metric by default.

    Parameters
    ----------
    H : ndarray
        Target Hamiltonian (dense Hermitian matrix).
    model : Model, optional
        Defaults to a :class:`~qbm.FullyVisibleQBM` of the right size.
    compare_exact : bool
        If True (and the system is small) also report the exact ground energy.

    Returns
    -------
    Result
        With ``.energy``, and ``.exact_energy`` / ``.error`` when available.

    Examples
    --------
    >>> import qbm
    >>> H = qbm.hamiltonians.tfim(4, J=1.0, g=1.5)
    >>> res = qbm.ground_state(H, steps=200)      # doctest: +SKIP
    >>> print(res.report())                       # doctest: +SKIP
    """
    H = np.asarray(H, dtype=complex)
    n = int(round(np.log2(H.shape[0])))
    if model is None:
        model = FullyVisibleQBM(n=n, backend=backend, connectivity=connectivity)
        model.theta = np.random.default_rng(seed).normal(scale=init_scale, size=model.n_params)
    if optimizer is None:
        optimizer = NaturalGradient(metric=metric, lr=lr, reg=reg)

    history = fit(model, Energy(H), optimizer, steps=steps, verbose=verbose)
    energy = model.energy(H) if hasattr(model, "energy") else model.state().expect(H)

    fields = {"energy": energy}
    if compare_exact and H.shape[0] <= 2**14:
        e0 = ground_energy(H)
        fields["exact_energy"] = e0
        fields["error"] = energy - e0
    return Result(model, history, "ground_state", **fields)
