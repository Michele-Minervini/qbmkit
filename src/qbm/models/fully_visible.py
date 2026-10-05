"""Fully-visible QBM: every qubit is visible, no hidden units.

The model state is the Gibbs state of ``G(theta) = sum_j theta_j G_j``.  This is
the Amin et al. QBM and the setting of papers 2410.12935 / 2410.24058: for a
fixed target the relative-entropy gradient is exact even when the generators do
not commute.
"""

from __future__ import annotations

from ..operators import ParamHamiltonian, pauli_pool
from .base import Model


class FullyVisibleQBM(Model):
    """A fully-visible QBM over ``n`` qubits.

    Parameters
    ----------
    n : int, optional
        Number of qubits (required unless ``generators``/``ham`` given).
    generators : list of (str | ndarray), optional
        Custom generator set.  If omitted it is built by :func:`~qbm.pauli_pool` from
        the four arguments below.
    ham : ParamHamiltonian, optional
        Provide a fully-built Hamiltonian directly.
    theta : ndarray, optional
        Initial parameters (zeros by default).
    backend : str | Backend, optional
        Engine that builds ``rho(theta)``; the exact dense one by default.
    terms : sequence of str
        Pauli words of the default generator set: one-letter words are fields on every
        qubit, two-letter words couplings on every edge.  The default
        ``("Z", "X", "ZZ")`` is the transverse-field-Ising family.
    connectivity : str or list of (int, int)
        Coupling graph: ``"chain"`` (default), ``"ring"``, ``"grid"``, ``"star"``,
        ``"all"``, or an explicit list of edges.
    periodic : bool
        Close a chain into a ring, or a grid into a torus.
    shape : (int, int), optional
        Rows and columns of a ``"grid"``.
    """

    def __init__(
        self,
        n=None,
        generators=None,
        ham=None,
        theta=None,
        backend=None,
        terms=("Z", "X", "ZZ"),
        connectivity="chain",
        periodic=False,
        shape=None,
    ):
        if ham is None:
            if generators is None:
                if n is None:
                    raise ValueError("provide n, generators, or ham")
                generators = pauli_pool(
                    n, terms, connectivity=connectivity, periodic=periodic, shape=shape
                )
            ham = ParamHamiltonian(generators, n_qubits=n)
        super().__init__(ham, theta=theta, backend=backend)
