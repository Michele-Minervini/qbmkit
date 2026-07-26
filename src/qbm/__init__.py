"""qbm -- the unifying open-source library for Quantum Boltzmann Machines.

Install name: ``qbmkit``.  Import name: ``qbm``.

Quick start
-----------
>>> import qbm
>>> data = qbm.datasets.bars_and_stripes(grid=3)
>>> model = qbm.learn(data)
>>> model.kl(data)            # doctest: +SKIP

The whole library is built on one object -- the parameterized Gibbs state
``rho(theta) = exp(-G(theta)) / Z`` -- and one primitive, the belief-propagation
channel.  See DESIGN.md.
"""

from __future__ import annotations

from . import data, diagnostics, losses, metrics, optim, purification
from .backends import (
    Backend,
    DenseBackend,
    StatevectorBackend,
    ThermalState,
    available_backends,
    get_backend,
)
from .data import datasets, hamiltonians, oracles
from .facade import learn
from .models import EvolvedQBM, FullyVisibleQBM, Model, SemiQuantumRBM, VisibleHiddenQBM
from .operators import ParamHamiltonian, local_pauli_generators, pauli, rbm_generators
from .train import History, fit

# Friendly aliases
QBM = FullyVisibleQBM

__version__ = "0.1.0"

__all__ = [
    "learn",
    "fit",
    "History",
    "QBM",
    "FullyVisibleQBM",
    "VisibleHiddenQBM",
    "SemiQuantumRBM",
    "EvolvedQBM",
    "Model",
    "ParamHamiltonian",
    "pauli",
    "local_pauli_generators",
    "rbm_generators",
    "Backend",
    "DenseBackend",
    "StatevectorBackend",
    "ThermalState",
    "get_backend",
    "available_backends",
    "purification",
    "losses",
    "metrics",
    "optim",
    "diagnostics",
    "data",
    "datasets",
    "hamiltonians",
    "oracles",
    "__version__",
]
