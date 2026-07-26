"""Semidefinite-programming task, solved through the QBM thermal-state core."""

from __future__ import annotations

import numpy as np

from ..losses.sdp import SDPDual, sdp_hamiltonian
from ..models.base import Model
from ..optim.natural_gradient import NaturalGradient
from ..train.loop import fit
from .base import Result


def solve_sdp(
    C,
    constraints=None,
    b=None,
    beta: float = 20.0,
    steps: int = 300,
    lr=None,
    reg: float = 1e-6,
    optimizer=None,
    backend=None,
    seed: int = 0,
    init_scale: float = 0.0,
    verbose: bool = False,
) -> Result:
    """Solve an entropy-regularised SDP over density matrices.

    ::

        maximise   <C, X> + (1/beta) S(X)
        subject to <A_i, X> = b_i,   X >= 0,   Tr X = 1

    The dual is convex and smooth, and its stationary point is a QBM thermal state
    (see :mod:`qbm.losses.sdp`).  Larger ``beta`` means weaker entropy regularisation
    and a solution closer to the unregularised SDP optimum.

    Parameters
    ----------
    C : ndarray
        Objective matrix (Hermitian).
    constraints : list of ndarray, optional
        The constraint matrices ``A_i`` (Hermitian).  ``None`` for the unconstrained
        problem, whose solution is exactly ``exp(beta C) / Tr exp(beta C)``.
    b : array, optional
        Constraint right-hand sides.  Defaults to zeros.
    beta : float
        Inverse entropy-regularisation strength.
    lr : float, optional
        Natural-gradient step.  Defaults to ``beta``, which makes the update *exactly*
        Newton's method on the dual (the Kubo-Mori metric is the Hessian of log Z).

    Returns
    -------
    Result
        With ``.X`` (the primal density matrix), ``.objective`` = ``<C, X>``,
        ``.entropy``, ``.dual_value``, ``.y`` (multipliers) and
        ``.constraint_violation`` (max ``|<A_i,X> - b_i|``).

    Examples
    --------
    >>> import numpy as np, qbm
    >>> C = np.diag([3.0, 1.0, 0.0, -1.0])
    >>> res = qbm.solve_sdp(C, beta=50)          # doctest: +SKIP
    >>> res.objective                             # close to max eigenvalue of C
    """
    C = np.asarray(C, dtype=complex)
    A = None if constraints is None else [np.asarray(a, dtype=complex) for a in constraints]
    n_con = 0 if A is None else len(A)
    if b is None:
        b = np.zeros(max(n_con, 1))
    b = np.asarray(b, dtype=float)

    ham = sdp_hamiltonian(C, A, beta=beta)
    model = Model(ham, backend=backend)
    model.theta = (
        np.random.default_rng(seed).normal(scale=init_scale, size=model.n_params)
        if init_scale > 0
        else np.zeros(model.n_params)
    )

    loss = SDPDual(b=b, beta=beta)
    if optimizer is None:
        optimizer = NaturalGradient(metric="kubo_mori", lr=beta if lr is None else lr, reg=reg)

    history = fit(model, loss, optimizer, steps=steps, verbose=verbose)

    state = model.state()
    X = state.density_matrix()
    objective = float(np.real(np.trace(C @ X)))
    viol = 0.0
    if A is not None:
        viol = float(max(abs(np.real(np.trace(a @ X)) - bi) for a, bi in zip(A, b)))

    return Result(
        model,
        history,
        "sdp",
        X=X,
        objective=objective,
        entropy=state.entropy(),
        dual_value=history.final_loss,
        y=model.theta.copy(),
        constraint_violation=viol,
        beta=beta,
    )
