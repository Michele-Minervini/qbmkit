"""Vendor-neutral quantum-circuit layer.

The QBM circuit algorithms live in a small internal IR (:mod:`qbm.circuits.ir`) and are
executed by a dependency-free statevector simulator (:mod:`qbm.circuits.simulator`).
Vendor SDKs are *emitters* of that IR, not dependencies of it -- see
:mod:`qbm.circuits.adapters`.  An SDK breaking change therefore costs one adapter file.
"""

from . import adapters, builder, densities, estimators, ir, simulator, varqite
from .builder import (
    evolution_unitary,
    gibbs_preparation,
    hadamard_test,
    pauli_rotation,
    state_preparation,
    thermofield_double,
    trotter_evolution,
)
from .ir import Circuit, Gate
from .varqite import PauliRotationAnsatz, prepare_gibbs, tfd_ansatz

__all__ = [
    "ir",
    "builder",
    "simulator",
    "adapters",
    "densities",
    "estimators",
    "varqite",
    "Circuit",
    "Gate",
    "PauliRotationAnsatz",
    "tfd_ansatz",
    "prepare_gibbs",
    "state_preparation",
    "thermofield_double",
    "gibbs_preparation",
    "hadamard_test",
    "evolution_unitary",
    "trotter_evolution",
    "pauli_rotation",
]
