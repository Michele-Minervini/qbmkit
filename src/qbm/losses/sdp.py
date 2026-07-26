"""Semidefinite programming through the QBM core (entropy-regularised SDP duality).

Primal (over density matrices, i.e. ``X >= 0`` with ``Tr X = 1``)::

    maximise   <C, X> + (1/beta) S(X)
    subject to <A_i, X> = b_i

The Gibbs variational principle solves the inner maximisation in closed form, so the
Lagrangian dual is a smooth **convex** function of the multipliers ``y``::

    g(y) = (1/beta) log Tr exp(beta (C - sum_i y_i A_i)) + b . y
    X*(y) = exp(beta (C - sum_i y_i A_i)) / Z(y)

and the dual gradient is exactly the constraint violation::

    d g / d y_i = b_i - <A_i, X*(y)> .

Writing ``G(y) = -beta C + sum_i y_i (beta A_i)`` makes ``X*(y) = e^{-G(y)}/Z`` a QBM
thermal state whose generators are ``beta A_i`` and whose fixed offset is ``-beta C``.
So the SDP is *just another loss* over the same parameterized Gibbs state: it reuses
the thermal-state engine, every optimizer, and every backend.

Because the Kubo-Mori metric is the Hessian of ``log Z``, the Hessian of the dual is
``(1/beta) I_KM``; running :class:`~qbm.optim.NaturalGradient` with ``metric="kubo_mori"``
and ``lr = beta`` is therefore *exactly Newton's method* on the SDP dual.
"""

from __future__ import annotations

import numpy as np

from ..operators import ParamHamiltonian
from .base import Loss


def sdp_hamiltonian(C, constraints=None, beta=1.0) -> ParamHamiltonian:
    """Build the dual's parameterized Hamiltonian ``G(y) = -beta C + sum_i y_i (beta A_i)``."""
    C = np.asarray(C, dtype=complex)
    A = [] if constraints is None else [np.asarray(a, dtype=complex) for a in constraints]
    dim = C.shape[0]
    if not A:  # unconstrained: a single zero generator keeps the parameter vector well-defined
        A = [np.zeros((dim, dim), dtype=complex)]
    gens = [beta * a for a in A]
    return ParamHamiltonian(
        gens,
        labels=[f"A{i}" for i in range(len(gens))],
        offset=-beta * C,
    )


class SDPDual(Loss):
    """The convex dual objective ``g(y) = (1/beta) log Z(y) + b . y`` of an entropy-regularised SDP.

    Minimise this over ``y`` (the loss parameters) with any optimizer; the primal
    solution is the model's density matrix ``X = rho(y)``.
    """

    def __init__(self, b=None, beta=1.0, n_constraints=None):
        self.beta = float(beta)
        if b is None:
            b = np.zeros(1 if n_constraints is None else n_constraints)
        self.b = np.asarray(b, dtype=float)

    def value(self, state) -> float:
        return state.log_partition() / self.beta + float(self.b @ state.theta)

    def grad(self, state) -> np.ndarray:
        # d/dy_i = b_i - <A_i, X>, and <G_i> = beta <A_i>
        return self.b - state.generator_expectations() / self.beta
