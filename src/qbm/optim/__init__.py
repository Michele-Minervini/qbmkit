"""Optimizers: parameter-update rules consuming a gradient (and optional metric)."""

from .base import Optimizer
from .gradient import Adam, GradientDescent
from .natural_gradient import NaturalGradient

__all__ = ["Optimizer", "GradientDescent", "Adam", "NaturalGradient"]
