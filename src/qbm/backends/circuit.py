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

Gibbs preparation strategies:

``"exact"`` (default)
    Synthesise the TFD amplitudes into a state-preparation unitary.  Exact, so the
    circuit *estimators* can be validated independently of preparation error.  This is
    also what a small hardware demonstration would use.
``"varqite"``
    Variational imaginary-time preparation -- the scalable route, and the open research
    problem.  Not implemented; documented as the extension point.
"""

from __future__ import annotations

import numpy as np

from .. import purification
from ..circuits import builder, simulator
from ..metrics.monotone import mc_weight


class CircuitThermalState:
    """Thermal state realised by a circuit and read out by measurement."""

    def __init__(self, ham, theta, shots=None, rng=None, gibbs_prep="exact", executor=None):
        if gibbs_prep != "exact":
            raise NotImplementedError(
                f"Gibbs preparation strategy {gibbs_prep!r} is not implemented. Only "
                "'exact' (TFD state synthesis) is available; variational preparation "
                "(VarQITE/QITE) is the documented extension point and an open research "
                "problem on hardware."
            )
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
    def preparation_circuit(self, ancilla_offset=0):
        """The circuit preparing the TFD purification of ``rho(theta)``."""
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
        self._unsupported("the diagonal gradient")

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
        return {
            "n_qubits": 2 * self.n_qubits + 1,
            "prep_gates": len(prep.gates),
            "prep_depth": prep.depth,
            "n_parameters": J,
            "circuits_per_gradient": J * n_time_samples,
            "circuits_per_metric": J * (J + 1) // 2 * n_time_samples,
            "shots_per_gradient": J * n_time_samples * shots,
            "shots_per_metric": J * (J + 1) // 2 * n_time_samples * shots,
        }


class CircuitBackend:
    """Builds :class:`CircuitThermalState` objects (measurement-based, shot-aware)."""

    name = "circuit"

    def __init__(self, shots=None, seed=None, gibbs_prep="exact", executor=None):
        self.shots = shots
        self.gibbs_prep = gibbs_prep
        self.executor = executor
        self._rng = np.random.default_rng(seed)

    def thermal_state(self, ham, theta) -> CircuitThermalState:
        return CircuitThermalState(
            ham,
            theta,
            shots=self.shots,
            rng=self._rng,
            gibbs_prep=self.gibbs_prep,
            executor=self.executor,
        )

    def __repr__(self) -> str:
        return f"CircuitBackend(shots={self.shots}, gibbs_prep={self.gibbs_prep!r})"


__all__ = ["CircuitBackend", "CircuitThermalState", "mc_weight"]
