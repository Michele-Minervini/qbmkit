"""Vendor adapters: emit the internal circuit IR to external SDKs.

These are deliberately **thin leaves**.  All QBM algorithms live in
:mod:`qbm.circuits.builder` in a vendor-neutral IR; an adapter only translates gates.
If an SDK makes a breaking change (Qiskit has removed ``opflow``, moved
``qiskit.algorithms`` out of core, dropped ``execute()`` and revised its primitives
twice), the fix is confined to one file here.

* :mod:`qbm.circuits.adapters.qasm` -- **OpenQASM 3** text. No dependency at all, and
  the most durable target: every SDK and most hardware providers import it.
* :mod:`qbm.circuits.adapters.qiskit_adapter` -- ``qiskit.QuantumCircuit`` (optional).
* :mod:`qbm.circuits.adapters.pennylane_adapter` -- a PennyLane tape/QNode (optional).

Prefer OpenQASM for archival and portability; use a native adapter when you need that
SDK's transpiler or hardware provider.
"""

from .qasm import to_qasm3

__all__ = ["to_qasm3", "to_qiskit", "to_pennylane", "executor", "available_adapters"]


def to_qiskit(circuit):
    """Convert to ``qiskit.QuantumCircuit`` (requires qiskit)."""
    from .qiskit_adapter import to_qiskit as _f

    return _f(circuit)


def to_pennylane(circuit):
    """Return a PennyLane quantum function applying this circuit (requires pennylane)."""
    from .pennylane_adapter import to_pennylane as _f

    return _f(circuit)


def executor(name: str = "builtin", **kwargs):
    """Get an ``executor(circuit) -> statevector`` by name.

    ``"builtin"`` is the library's own simulator (no dependency); ``"qiskit"`` and
    ``"pennylane"`` route execution through those SDKs.  Pass the result to
    ``CircuitBackend(executor=...)`` to swap the execution engine without touching any
    other code.
    """
    if name == "builtin":
        from ..simulator import run

        return run
    if name == "qiskit":
        from .qiskit_adapter import statevector_executor

        return statevector_executor(**kwargs)
    if name == "pennylane":
        from .pennylane_adapter import statevector_executor

        return statevector_executor(**kwargs)
    raise KeyError(f"unknown executor {name!r}; try one of {available_adapters() + ['builtin']}")


def available_adapters() -> list:
    """Adapters usable in this environment (``qasm3`` is always available)."""
    out = ["qasm3"]
    for mod, name in (("qiskit", "qiskit"), ("pennylane", "pennylane")):
        try:
            __import__(mod)
            out.append(name)
        except Exception:
            pass
    return out
