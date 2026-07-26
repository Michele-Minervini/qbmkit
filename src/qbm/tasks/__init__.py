"""Task layer: one call per research question, with sensible defaults.

Every task is a thin recipe over the same core (model + loss + optimizer + backend)
and returns a :class:`~qbm.tasks.base.Result`::

    qbm.learn(data)              # generative modelling of a classical distribution
    qbm.ground_state(H)          # ground-state energy estimation
    qbm.learn_state(sigma)       # quantum-state learning
    qbm.free_energy_min(H, T)    # free-energy minimisation / Gibbs preparation
    qbm.solve_sdp(C, A, b)       # semidefinite programming
"""

from .base import Result
from .free_energy import free_energy_min
from .ground_state import ground_state
from .sdp import solve_sdp
from .state_learning import learn_state

__all__ = ["Result", "ground_state", "learn_state", "free_energy_min", "solve_sdp"]
