"""Result object returned by every task."""

from __future__ import annotations


class Result:
    """The outcome of a task: the trained model, its history, and task-specific fields.

    Extra keyword arguments become attributes, so e.g. a ground-state run exposes
    ``result.energy`` / ``result.exact_energy`` / ``result.error``.
    """

    def __init__(self, model, history, task, **fields):
        self.model = model
        self.history = history
        self.task = task
        self._fields = dict(fields)
        for k, v in fields.items():
            setattr(self, k, v)

    def report(self) -> str:
        """A short human-readable summary of the run."""
        lines = [f"{self.task} result"]
        if self.history is not None and len(self.history):
            lines.append(f"  steps           : {len(self.history)}")
            lines.append(f"  initial loss    : {self.history.loss[0]:.6g}")
            lines.append(f"  final loss      : {self.history.final_loss:.6g}")
            lines.append(f"  final |grad|    : {self.history.grad_norm[-1]:.3g}")
        width = max((len(k) for k in self._fields), default=0)
        for k, v in self._fields.items():
            if isinstance(v, (int, float)):
                lines.append(f"  {k.ljust(max(width, 15))} : {v:.6g}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        n = len(self.history) if self.history is not None else 0
        return f"Result(task={self.task!r}, steps={n})"
