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

    def record(self, loss, theta, grad):
        self.loss.append(float(loss))
        self.theta.append(np.array(theta, copy=True))
        self.grad_norm.append(float(np.linalg.norm(grad)))

    @property
    def final_loss(self) -> float:
        return self.loss[-1]

    def __len__(self) -> int:
        return len(self.loss)


def fit(model, loss, optimizer, steps=300, callback=None, tol=None, verbose=False):
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

    Returns
    -------
    History
    """
    history = History()
    for t in range(steps):
        state = model.state()
        value = loss.value(state)
        grad = loss.grad(state)
        history.record(value, model.theta, grad)
        if verbose and (steps < 10 or t % max(1, steps // 10) == 0):
            print(f"step {t:4d}  loss={value:.6f}  |grad|={history.grad_norm[-1]:.2e}")
        if callback is not None:
            callback(t, value, model)
        if tol is not None and history.grad_norm[-1] < tol:
            break
        model.theta = optimizer.step(model.theta, grad, state)
    return history
