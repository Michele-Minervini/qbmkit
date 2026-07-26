"""Backends: the swap seam between the math and concrete linear algebra/hardware.

Built-in backends:

* ``dense``        -- exact NumPy/SciPy density-matrix engine (default).
* ``statevector``  -- thermofield-double purification, with optional shot noise.

Planned optional backends (require extra packages, see DESIGN.md):

* ``tensor_network`` (``pip install qbmkit[tn]``, quimb) -- scalable structured systems.
* ``circuit``        (``pip install qbmkit[circuit]``, PennyLane/Qiskit) -- statevector /
  density-matrix / real hardware with VarQITE/QITE/TFD Gibbs preparation.
* ``jax``            (``pip install qbmkit[jax]``) -- autodiff + GPU.

Any new backend implements the :class:`Backend` / :class:`ThermalState` protocols and
unlocks every model/loss/metric/optimizer with no change to user code.
"""

from .base import Backend, ThermalState
from .dense import DenseBackend, DenseThermalState
from .statevector import StatevectorBackend, StatevectorThermalState

_REGISTRY = {
    "dense": DenseBackend,
    "statevector": StatevectorBackend,
}


def get_backend(backend=None, **kwargs):
    """Resolve a backend.

    ``None`` -> the default :class:`DenseBackend`; a string -> the registered backend
    (constructed with ``kwargs``); a backend instance is returned unchanged.  The
    ``"jax"`` backend is imported lazily so the base install stays JAX-free.
    """
    if backend is None:
        return DenseBackend()
    if isinstance(backend, str):
        if backend in _REGISTRY:
            return _REGISTRY[backend](**kwargs)
        if backend == "jax":
            from .jax_backend import JaxBackend
            return JaxBackend(**kwargs)
        avail = ", ".join(available_backends())
        raise KeyError(f"unknown backend {backend!r}; available: {avail}")
    return backend


def register_backend(name, factory):
    """Register a backend factory under ``name`` (for plugins / external packages)."""
    _REGISTRY[name] = factory


def available_backends():
    names = set(_REGISTRY)
    try:
        import jax  # noqa: F401

        names.add("jax")
    except Exception:
        pass
    return sorted(names)


default_backend = DenseBackend()

__all__ = [
    "Backend",
    "ThermalState",
    "DenseBackend",
    "DenseThermalState",
    "StatevectorBackend",
    "StatevectorThermalState",
    "get_backend",
    "register_backend",
    "available_backends",
    "default_backend",
]
