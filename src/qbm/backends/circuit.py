"""Circuit backend: runs the QBM through actual quantum circuits.

The thermal state is prepared as its thermofield-double purification by a circuit, and
every quantity is obtained the way a device would obtain it -- by **measurement**, with
an optional shot budget.  Executed by default on the library's own dependency-free
statevector simulator; Qiskit / PennyLane are alternative executors of the same IR
(:mod:`qbm.circuits.adapters`), so no vendor SDK is required.

**Scope.** Expectation values, generative gradients (``<G_j>_data - <G_j>_model``),
sampling, and -- via Hadamard-test estimators -- the energy gradient and the
alpha-z / Kubo-Mori family of information matrices.  Quantities that need the *spectrum*
of ``rho`` (entropy, ``log Z``, and hence free energy and the SDP dual) are not
measurable this way and raise a clear error; use ``backend="dense"`` for those.

Gibbs preparation strategies (``gibbs_prep=``):

``"exact"`` (default)
    Synthesise the TFD amplitudes into a state-preparation unitary.  Exact, so the
    circuit *estimators* can be validated independently of preparation error -- but it
    needs the eigendecomposition of ``G(theta)``, and it emits one opaque ``unitary``
    instruction rather than gates.
``"varqite"``
    Variational imaginary-time preparation (:mod:`qbm.circuits.varqite`).  Approximate,
    but built only from expectation values and emitted as an ordinary gate sequence, so
    it is the route that runs on a device.  Tune it with
    ``varqite_options={"depth": ..., "steps": ...}`` and read
    ``state.varqite_result().report()`` for the preparation error.
"""

from __future__ import annotations

import numpy as np

from .. import purification
from ..circuits import builder, simulator
from ..metrics.monotone import mc_weight


class CircuitThermalState:
    """Thermal state realised by a circuit and read out by measurement."""

    #: defaults for ``gibbs_prep="varqite"``; override with ``varqite_options``
    VARQITE_DEFAULTS = {"depth": 3, "steps": 100}

    def __init__(
        self,
        ham,
        theta,
        shots=None,
        rng=None,
        gibbs_prep="exact",
        executor=None,
        varqite_options=None,
    ):
        if gibbs_prep not in ("exact", "varqite"):
            raise ValueError(
                f"unknown Gibbs preparation strategy {gibbs_prep!r}; use 'exact' (TFD "
                "state synthesis) or 'varqite' (variational imaginary-time evolution)"
            )
        self.varqite_options = {**self.VARQITE_DEFAULTS, **(varqite_options or {})}
        self._varqite = None
        self.ham = ham
        self.theta = np.asarray(theta, dtype=float)
        self.n_qubits = ham.n_qubits
        self.dim = ham.dim
        self.shots = shots
        self.rng = np.random.default_rng() if rng is None else rng
        self.gibbs_prep = gibbs_prep
        self.executor = executor or simulator.run
        self._psi = None
        self._rho = None

    # -- circuits ----------------------------------------------------------
    def varqite_result(self):
        """The cached :class:`~qbm.circuits.varqite.VarQITEResult` behind this state.

        Only available with ``gibbs_prep="varqite"``.  Carries the preparation error --
        ``.residual`` (measurable) and ``.trace_distance()`` (simulation-only).
        """
        if self.gibbs_prep != "varqite":
            raise ValueError("varqite_result() needs gibbs_prep='varqite'")
        if self._varqite is None:
            from ..circuits import varqite

            self._varqite = varqite.prepare_gibbs(
                self.ham, self.theta, beta=1.0, **self.varqite_options
            )
        return self._varqite

    def preparation_circuit(self, ancilla_offset=0):
        """The circuit preparing the TFD purification of ``rho(theta)``."""
        if self.gibbs_prep == "varqite":
            return self.varqite_result().circuit(
                offset=ancilla_offset, n_qubits=2 * self.n_qubits + ancilla_offset
            )
        return builder.gibbs_preparation(self.ham, self.theta, ancilla_offset=ancilla_offset)

    def statevector(self):
        if self._psi is None:
            self._psi = self.executor(self.preparation_circuit())
        return self._psi

    def density_matrix(self) -> np.ndarray:
        if self._rho is None:
            self._rho = purification.reduced_system_state(self.statevector(), self.dim)
        return self._rho

    # -- measurement -------------------------------------------------------
    def _hadamard_estimate(self, unitary, imaginary=False) -> float:
        """Run a Hadamard test and return ``Re <U>`` (or ``Im <U>``), shot-aware."""
        prep = self.preparation_circuit(ancilla_offset=1)
        circ = builder.hadamard_test(
            prep, unitary, range(1, self.n_qubits + 1), imaginary=imaginary
        )
        psi = self.executor(circ)
        return simulator.measure_z(psi, 0, circ.n_qubits, shots=self.shots, rng=self.rng)

    def expect(self, op) -> float:
        """``<O>`` by measuring the Hermitian operator ``O`` in its eigenbasis."""
        op = np.asarray(op, dtype=complex)
        if self.shots is None:
            return float(np.real(np.sum(self.density_matrix() * op.T)))
        vals, R = np.linalg.eigh(op)
        rho = self.density_matrix()
        probs = np.real(np.einsum("ik,ij,jk->k", np.conj(R), rho, R))
        probs = np.clip(probs, 0.0, None)
        probs = probs / probs.sum()
        draws = self.rng.choice(len(vals), size=self.shots, p=probs)
        return float(np.mean(vals[draws]))

    def generator_expectations(self) -> np.ndarray:
        return np.array([self.expect(g) for g in self.ham.generators])

    def probabilities(self) -> np.ndarray:
        return np.real(np.diag(self.density_matrix()))

    def sample(self, n: int, rng=None) -> np.ndarray:
        rng = self.rng if rng is None else rng
        idx = simulator.sample_bitstrings(self.statevector(), 2 * self.n_qubits, n, rng)
        return idx // self.dim  # keep the system register (it is the leading factor)

    # -- placeholders filled in by the estimator layer ---------------------
    def observable_gradient(self, op):
        from ..circuits.estimators import energy_gradient

        return energy_gradient(self, np.asarray(op, dtype=complex))

    def metric(self, kind="kubo_mori"):
        from ..circuits.estimators import information_matrix

        return information_matrix(self, kind)

    def state_derivatives(self):
        self._unsupported("state derivatives")

    def diagonal_gradient(self):
        raise NotImplementedError(
            "the diagonal gradient is not a measurable expectation value. For hidden-unit "
            "models use qbm.losses.GibbsMapNLL: its positive phase is an exact classical "
            "computation on the hidden register and its negative phase is just <G_j>, so "
            "it runs on this backend."
        )

    def log_partition(self):
        self._unsupported("log Z")

    def entropy(self):
        self._unsupported("the von Neumann entropy")

    def _unsupported(self, what):
        raise NotImplementedError(
            f"{what} is not a measurable expectation value: it depends on the spectrum "
            "of rho, which a quantum device does not expose. Use backend='dense'. The "
            "circuit backend supports expectations, generative gradients, sampling, the "
            "energy gradient and the information matrices."
        )

    # -- resource accounting ----------------------------------------------
    def resource_estimate(self, kind="kubo_mori", n_time_samples=8) -> dict:
        """Circuit/shot cost of one gradient and one metric evaluation.

        Reports what a hardware run would actually need, so the cost is explicit rather
        than discovered on a queue.
        """
        J = self.ham.n_params
        prep = self.preparation_circuit(ancilla_offset=1)
        shots = self.shots or 0
        out = {
            "n_qubits": 2 * self.n_qubits + 1,
            "prep_gates": len(prep.gates),
            "prep_depth": prep.depth,
            "n_parameters": J,
            "circuits_per_gradient": J * n_time_samples,
            "circuits_per_metric": J * (J + 1) // 2 * n_time_samples,
            "shots_per_gradient": J * n_time_samples * shots,
            "shots_per_metric": J * (J + 1) // 2 * n_time_samples * shots,
        }
        if self.gibbs_prep == "varqite":
            vq = self.varqite_result().resource_estimate(n_hamiltonian_terms=J)
            out["circuits_for_preparation"] = vq["circuits_total"]
        return out


class CircuitBackend:
    """Builds :class:`CircuitThermalState` objects (measurement-based, shot-aware)."""

    name = "circuit"

    def __init__(
        self, shots=None, seed=None, gibbs_prep="exact", executor=None, varqite_options=None
    ):
        self.shots = shots
        self.gibbs_prep = gibbs_prep
        self.executor = executor
        self.varqite_options = varqite_options
        self._rng = np.random.default_rng(seed)

    def thermal_state(self, ham, theta) -> CircuitThermalState:
        return CircuitThermalState(
            ham,
            theta,
            shots=self.shots,
            rng=self._rng,
            gibbs_prep=self.gibbs_prep,
            executor=self.executor,
            varqite_options=self.varqite_options,
        )

    def __repr__(self) -> str:
        return f"CircuitBackend(shots={self.shots}, gibbs_prep={self.gibbs_prep!r})"


__all__ = ["CircuitBackend", "CircuitThermalState", "mc_weight"]
