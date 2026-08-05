"""Pauli-propagation backend (no external dependency).

Prepares the QBM thermal state as a **sparse sum of Pauli strings** and evolves it under
imaginary time, following *Thermal State Simulation with Pauli and Majorana Propagation*
(arXiv:2602.04878) and *Sampling from Thermal Quantum States via Pauli Propagation*.  The
heavy lifting lives in :mod:`qbm.pauli_prop`; this module wraps it in the backend seam so
that models, losses, and the task layer use it with no change to user code.

**Why it is here.**  It is a third classical engine alongside ``dense`` and
``tensor_network``, with a different sweet spot: the cost is set by the number of retained
Pauli strings, which stays small at high temperature and for near-classical (mostly
commuting) Hamiltonians, and it is *topology-agnostic* -- all-to-all couplings cost no
more than a chain.  It also comes with a principled **sampler** whose bitstring
likelihoods are exact, which is what makes it a natural fit for generative modelling.

**Supported scope.**  Expectation-based training: the relative-entropy / NLL gradient is
``<G_j>_data - <G_j>_model``, read straight off the Pauli coefficients, so generative
modelling and quantum-state learning work.  Sampling and computational-basis
probabilities use the locally normalised sampler.  Quantities that need the belief-
propagation channel or the full spectrum -- the QFI metrics, the energy-loss gradient,
``log Z`` and the entropy -- are not available here (a clear error is raised); use
``backend="dense"`` for those.  Generators must be Pauli-string labels.
"""

from __future__ import annotations

import numpy as np

from .. import pauli_prop


class PauliPropThermalState:
    """Thermal state held as a truncated Pauli sum, read out by coefficient or by sampling."""

    def __init__(
        self,
        ham,
        theta,
        beta=1.0,
        trotter_steps=32,
        coeff_cutoff=1e-10,
        max_weight=np.inf,
        rng=None,
    ):
        if getattr(ham, "offset", None) is not None:
            raise NotImplementedError(
                "ParamHamiltonian.offset is not supported by the pauli-propagation backend"
            )
        self.ham = ham
        self.theta = np.asarray(theta, dtype=float)
        self.n_qubits = ham.n_qubits
        self.dim = ham.dim
        self.beta = beta
        self.trotter_steps = trotter_steps
        self.coeff_cutoff = coeff_cutoff
        self.max_weight = max_weight
        self.rng = np.random.default_rng() if rng is None else rng
        self._psum = None

    # -- the Pauli sum ----------------------------------------------------
    def pauli_sum(self) -> pauli_prop.PauliSum:
        """The propagated, truncated :class:`~qbm.pauli_prop.PauliSum` (cached)."""
        if self._psum is None:
            self._psum = pauli_prop.thermal_state(
                self.ham.labels,
                self.theta,
                beta=self.beta,
                trotter_steps=self.trotter_steps,
                coeff_cutoff=self.coeff_cutoff,
                max_weight=self.max_weight,
            )
        return self._psum

    @property
    def n_terms(self) -> int:
        """Number of Pauli strings retained -- the cost knob of the method."""
        return self.pauli_sum().n_terms

    # -- measurable read-outs --------------------------------------------
    def expect(self, op) -> float:
        return self.pauli_sum().expect(op)

    def generator_expectations(self) -> np.ndarray:
        ps = self.pauli_sum()
        return np.array([ps.expect_pauli(lbl) for lbl in self.ham.labels])

    def density_matrix(self) -> np.ndarray:
        return self.pauli_sum().to_matrix()

    def probabilities(self) -> np.ndarray:
        return self.pauli_sum().probabilities()

    def sample(self, n: int, rng=None) -> np.ndarray:
        return self.pauli_sum().sample(n, rng=self.rng if rng is None else rng)

    def log_likelihood(self, bitstrings) -> np.ndarray:
        """Exact ``log p_hat(x)`` the sampler assigns to each basis-state index."""
        return self.pauli_sum().log_likelihood(bitstrings)

    def spectral_negativity(self) -> float:
        """``N(rho)`` -- the non-physicality truncation induced (small ``n``; bounds fidelity)."""
        return self.pauli_sum().spectral_negativity()

    def resource_estimate(self) -> dict:
        """What the propagation is costing: retained terms and the Trotter schedule."""
        ps = self.pauli_sum()
        return {
            "n_qubits": self.n_qubits,
            "n_parameters": self.ham.n_params,
            "trotter_steps": self.trotter_steps,
            "retained_pauli_terms": ps.n_terms,
            "coeff_cutoff": self.coeff_cutoff,
            "max_weight": None if self.max_weight == np.inf else self.max_weight,
        }

    # -- unsupported (spectrum / channel dependent) ----------------------
    def _unsupported(self, what):
        raise NotImplementedError(
            f"{what} is not available on the pauli-propagation backend: it needs the "
            "belief-propagation channel or the spectrum of rho, which this sparse-Pauli "
            "representation does not expose. Use backend='dense'. This backend supports "
            "expectations, generator expectations (hence relative-entropy / NLL training), "
            "probabilities and sampling."
        )

    def observable_gradient(self, op):
        self._unsupported("the energy-loss gradient")

    def metric(self, kind: str = "kubo_mori"):
        self._unsupported(f"the {kind} metric")

    def belief_prop(self, op):
        self._unsupported("the belief-propagation channel")

    def state_derivatives(self):
        self._unsupported("state derivatives")

    def diagonal_gradient(self):
        self._unsupported("the diagonal gradient")

    def log_partition(self):
        self._unsupported("log Z")

    def entropy(self):
        self._unsupported("the von Neumann entropy")


class PauliPropagationBackend:
    """Builds :class:`PauliPropThermalState` objects (sparse-Pauli imaginary-time engine).

    Parameters
    ----------
    beta : float
        Inverse temperature.  For a QBM leave at ``1.0`` -- the temperature is carried by
        the parameters ``theta`` (``rho(theta) = e^{-G(theta)} / Z``).
    trotter_steps : int
        Imaginary-time layers ``L``.  Error is first order in ``1/L`` and zero for
        commuting Hamiltonians.
    coeff_cutoff : float
        Small-coefficient truncation threshold (relative to ``c_I``).
    max_weight : int
        Pauli-weight truncation cap.
    seed : int, optional
        Seed for the sampler's random generator.
    """

    name = "pauli_propagation"

    def __init__(
        self, beta=1.0, trotter_steps=32, coeff_cutoff=1e-10, max_weight=np.inf, seed=None
    ):
        self.beta = beta
        self.trotter_steps = trotter_steps
        self.coeff_cutoff = coeff_cutoff
        self.max_weight = max_weight
        self._rng = np.random.default_rng(seed)

    def thermal_state(self, ham, theta) -> PauliPropThermalState:
        return PauliPropThermalState(
            ham,
            theta,
            beta=self.beta,
            trotter_steps=self.trotter_steps,
            coeff_cutoff=self.coeff_cutoff,
            max_weight=self.max_weight,
            rng=self._rng,
        )

    def __repr__(self) -> str:
        return (
            f"PauliPropagationBackend(beta={self.beta}, trotter_steps={self.trotter_steps}, "
            f"coeff_cutoff={self.coeff_cutoff})"
        )


__all__ = ["PauliPropagationBackend", "PauliPropThermalState"]
