"""Fully-visible QBM: every qubit is visible, no hidden units.

The model state is the Gibbs state of ``G(theta) = sum_j theta_j G_j``.  This is
the Amin et al. QBM and the setting of papers 2410.12935 / 2410.24058: for a
fixed target the relative-entropy gradient is exact even when the generators do
not commute.
"""

from __future__ import annotations

from ..operators import ParamHamiltonian, local_pauli_generators
from .base import Model


class FullyVisibleQBM(Model):
    """A fully-visible QBM over ``n`` qubits.

    Parameters
    ----------
    n : int, optional
        Number of qubits (required unless ``generators``/``ham`` given).
    generators : list of (str | ndarray), optional
        Custom generator set; defaults to local fields ``{Z_i, X_i}`` and
        nearest-neighbour ``{Z_i Z_{i+1}}`` couplings.
    ham : ParamHamiltonian, optional
        Provide a fully-built Hamiltonian directly.
    theta, backend, periodic, fields, couplings : see code.
    """

    def __init__(
        self,
        n=None,
        generators=None,
        ham=None,
        theta=None,
        backend=None,
        fields=("Z", "X"),
        couplings=("ZZ",),
        connectivity="chain",
        periodic=False,
    ):
        if ham is None:
            if generators is None:
                if n is None:
                    raise ValueError("provide n, generators, or ham")
                generators = local_pauli_generators(
                    n,
                    fields=fields,
                    couplings=couplings,
                    connectivity=connectivity,
                    periodic=periodic,
                )
            ham = ParamHamiltonian(generators, n_qubits=n)
        super().__init__(ham, theta=theta, backend=backend)
