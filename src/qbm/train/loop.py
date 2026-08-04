"""The generic training loop shared by every task.

``fit`` builds the thermal state once per step, evaluates the loss value and
gradient, and applies the optimizer.  Geometry-aware optimizers receive the
state, so plain GD, Adam, and quantum natural gradient all run through this one
loop unchanged.
"""

from __future__ import annotations

import numpy as np


class History:
    """Records loss values (and optionally parameters) during training."""

    def __init__(self):
        self.loss: list[float] = []
        self.theta: list[np.ndarray] = []
        self.grad_norm: list[float] = []
        self.monitor: list = []

    def record(self, loss, theta, grad, monitor=None):
        self.loss.append(float(loss))
        self.theta.append(np.array(theta, copy=True))
        self.grad_norm.append(float(np.linalg.norm(grad)))
        if monitor is not None:
            self.monitor.append(monitor)

    @property
    def final_loss(self) -> float:
        return self.loss[-1]

    def __len__(self) -> int:
        return len(self.loss)


def fit(
    model,
    loss,
    optimizer,
    steps=300,
    callback=None,
    tol=None,
    verbose=False,
    monitor=None,
    stop=None,
):
    """Train ``model`` to minimise ``loss`` with ``optimizer``.

    Parameters
    ----------
    model : Model
        Mutated in place; ``model.theta`` holds the trained parameters on return.
    loss : Loss
    optimizer : Optimizer
    steps : int
    callback : callable(step, value, model), optional
    tol : float, optional
        Stop early when the gradient norm drops below ``tol``.
    verbose : bool
        Print progress every ~10% of steps.
    monitor : callable(state, model) -> value, optional
        Extra quantity recorded each step into ``history.monitor``.  This exists for
        measurement backends: descent needs only the *gradient*, but some loss
        **values** are not measurable (the relative entropy needs ``log Z``, which
        depends on the spectrum of rho).  When ``loss.value`` raises
        ``NotImplementedError`` the loop keeps training and records ``nan``, so a
        measurable ``monitor`` -- e.g. ``lambda s, m: m.kl(q)`` -- becomes the
        training curve you can actually plot on hardware.
    stop : callable(state, model) -> bool, optional
        Checked each step *before* the update; return ``True`` to halt.  Useful when
        the state itself is approximate: ground-state training drives the QBM to low
        effective temperature, where variational Gibbs preparation degrades and can
        eventually destabilise the very gradient driving it.  Guarding on the
        preparation error stops before that, e.g.
        ``stop=lambda s, m: s.varqite_result().residual > 0.05``.

    Returns
    -------
    History
    """
    history = History()
    for t in range(steps):
        state = model.state()
        try:
            value = loss.value(state)
        except NotImplementedError:
            value = float("nan")  # not measurable on this backend; the gradient still is
        grad = loss.grad(state)
        tracked = None if monitor is None else monitor(state, model)
        history.record(value, model.theta, grad, tracked)
        if verbose and (steps < 10 or t % max(1, steps // 10) == 0):
            shown = f"loss={value:.6f}" if np.isfinite(value) else f"monitor={tracked}"
            print(f"step {t:4d}  {shown}  |grad|={history.grad_norm[-1]:.2e}")
        if callback is not None:
            callback(t, value, model)
        if tol is not None and history.grad_norm[-1] < tol:
            break
        if stop is not None and stop(state, model):
            break
        model.theta = optimizer.step(model.theta, grad, state)
    return history
