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

from . import data, diagnostics, losses, metrics, optim, purification, sampling, tasks
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
from .operators import (
    ParamHamiltonian,
    local_pauli_generators,
    pauli,
    pauli_pool,
    rbm_generators,
)
from .registry import available, build, register, register_builtins
from .tasks import Result, free_energy_min, ground_state, learn_state, solve_sdp
from .train import History, fit

# Friendly aliases
QBM = FullyVisibleQBM

try:  # keep the version in one place (pyproject.toml)
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("qbmkit")
except Exception:  # running from a source tree without an install
    __version__ = "0.0.0.dev0"

register_builtins()

__all__ = [
    "learn",
    "ground_state",
    "learn_state",
    "free_energy_min",
    "solve_sdp",
    "Result",
    "tasks",
    "register",
    "build",
    "available",
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
    "pauli_pool",
    "rbm_generators",
    "Backend",
    "DenseBackend",
    "StatevectorBackend",
    "ThermalState",
    "get_backend",
    "available_backends",
    "purification",
    "sampling",
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
