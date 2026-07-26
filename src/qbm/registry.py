"""Name registries for models, losses, metrics, optimizers, backends and tasks.

Every built-in component is registered here, so it is addressable by string from the
facade, from config files, and from the CLI::

    qbm.build("loss", "energy", observable=H)
    qbm.build("optimizer", "natural_gradient", lr=0.1)
    qbm.available("metric")        -> ['fisher_bures', 'kubo_mori', 'wigner_yanase']

Third-party packages extend the library without touching the core::

    @qbm.register("loss", "my_loss")
    class MyLoss(qbm.losses.Loss):
        ...
"""

from __future__ import annotations

KINDS = ("model", "loss", "metric", "optimizer", "backend", "task")

_REGISTRIES: dict[str, dict[str, object]] = {k: {} for k in KINDS}


def register(kind: str, name: str):
    """Decorator registering a class/callable under ``name`` in the ``kind`` registry."""
    if kind not in _REGISTRIES:
        raise ValueError(f"unknown registry kind {kind!r}; choose from {', '.join(KINDS)}")

    def deco(obj):
        _REGISTRIES[kind][name.lower()] = obj
        return obj

    return deco


def add(kind: str, name: str, obj) -> None:
    """Imperatively register ``obj`` (the non-decorator form of :func:`register`)."""
    register(kind, name)(obj)


def get(kind: str, name: str):
    """Look up a registered entry by name."""
    if kind not in _REGISTRIES:
        raise ValueError(f"unknown registry kind {kind!r}; choose from {', '.join(KINDS)}")
    try:
        return _REGISTRIES[kind][name.lower()]
    except KeyError as exc:
        avail = ", ".join(sorted(_REGISTRIES[kind])) or "<none>"
        raise KeyError(f"no {kind} named {name!r}; registered: {avail}") from exc


def build(kind: str, name: str, *args, **kwargs):
    """Construct a registered component: ``build("optimizer", "adam", lr=0.1)``."""
    return get(kind, name)(*args, **kwargs)


def available(kind: str | None = None):
    """List registered names for one kind, or a dict of all kinds."""
    if kind is None:
        return {k: sorted(v) for k, v in _REGISTRIES.items()}
    if kind not in _REGISTRIES:
        raise ValueError(f"unknown registry kind {kind!r}; choose from {', '.join(KINDS)}")
    return sorted(_REGISTRIES[kind])


def register_builtins() -> None:
    """Register every component shipped with the library (called on import)."""
    from . import backends, losses, metrics, models, optim, tasks

    for name, obj in [
        ("fully_visible", models.FullyVisibleQBM),
        ("qbm", models.FullyVisibleQBM),
        ("visible_hidden", models.VisibleHiddenQBM),
        ("sqrbm", models.SemiQuantumRBM),
        ("evolved", models.EvolvedQBM),
    ]:
        add("model", name, obj)

    for name, obj in [
        ("energy", losses.Energy),
        ("relative_entropy", losses.RelativeEntropy),
        ("marginal_relative_entropy", losses.MarginalRelativeEntropy),
        ("nll", losses.NLL),
        ("marginal_nll", losses.MarginalNLL),
        ("sqrbm_nll", losses.SqRBMNLL),
        ("free_energy", losses.FreeEnergy),
        ("sdp_dual", losses.SDPDual),
    ]:
        add("loss", name, obj)

    for name, obj in [
        ("kubo_mori", metrics.KuboMori),
        ("fisher_bures", metrics.FisherBures),
        ("wigner_yanase", metrics.WignerYanase),
    ]:
        add("metric", name, obj)

    for name, obj in [
        ("gradient_descent", optim.GradientDescent),
        ("sgd", optim.GradientDescent),
        ("adam", optim.Adam),
        ("natural_gradient", optim.NaturalGradient),
    ]:
        add("optimizer", name, obj)

    for name, obj in [
        ("dense", backends.DenseBackend),
        ("statevector", backends.StatevectorBackend),
    ]:
        add("backend", name, obj)
    try:  # optional extra
        from .backends.jax_backend import JaxBackend

        add("backend", "jax", JaxBackend)
    except Exception:
        pass

    from .facade import learn

    for name, obj in [
        ("generative", learn),
        ("ground_state", tasks.ground_state),
        ("state_learning", tasks.learn_state),
        ("free_energy_min", tasks.free_energy_min),
        ("sdp", tasks.solve_sdp),
    ]:
        add("task", name, obj)
