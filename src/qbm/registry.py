"""Lightweight name registries for the plugin system.

Each extensible kind (model, loss, metric, optimizer, backend) has a registry.
A new component is registered by decorator and becomes addressable by name in the
facade and config files, without editing the core::

    @register("loss", "my_loss")
    class MyLoss(Loss):
        ...
"""

from __future__ import annotations

_REGISTRIES: dict[str, dict[str, type]] = {
    "model": {},
    "loss": {},
    "metric": {},
    "optimizer": {},
    "backend": {},
}


def register(kind: str, name: str):
    """Decorator registering a class under ``name`` in the ``kind`` registry."""
    if kind not in _REGISTRIES:
        raise ValueError(f"unknown registry kind {kind!r}")

    def deco(cls):
        _REGISTRIES[kind][name.lower()] = cls
        return cls

    return deco


def get(kind: str, name: str) -> type:
    """Look up a registered class by name."""
    try:
        return _REGISTRIES[kind][name.lower()]
    except KeyError as exc:
        avail = ", ".join(sorted(_REGISTRIES.get(kind, {}))) or "<none>"
        raise KeyError(f"no {kind} named {name!r}; registered: {avail}") from exc


def available(kind: str) -> list[str]:
    return sorted(_REGISTRIES[kind])
