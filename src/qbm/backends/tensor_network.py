"""Tensor-network backend (optional, ``pip install qbmkit[tn]``).

Represents the thermal state by its **purification as a matrix-product state**, the
scalable version of the thermofield double used by the statevector backend:

* start from the infinite-temperature purification -- a product of Bell pairs
  between each physical qubit and an ancilla, ``rho = I/d`` when the ancillas are
  traced out;
* apply ``exp(-G(theta)/2)`` to the *physical* legs by second-order (symmetric)
  Trotterised imaginary-time evolution, so that
  ``Tr_anc |psi><psi| = e^{-G/2} (I/d) e^{-G/2} ∝ e^{-G(theta)}``;
* read expectation values as ``<psi| O (x) I |psi>``.

Cost is polynomial in the number of qubits and exponential only in the bond
dimension, so low-entanglement (1-D-like, higher-temperature) systems go far beyond
the dense backend's ~13-qubit ceiling -- 16 qubits run in seconds at bond dimension
4, where a dense density matrix would need ~34 GB.

**Supported scope.** This backend is built for *expectation-based* training: the
relative-entropy / negative-log-likelihood gradient is
``<G_j>_data - <G_j>_model``, which needs only generator expectations, so generative
modelling and state learning work at scale.  Quantities that require the full
spectrum or the belief-propagation channel -- the QFI metrics and the
energy-loss gradient -- are *not* available here; use the dense or jax backend for
those (a clear error is raised rather than a silently wrong number).

Generators must be 1- or 2-body Pauli strings (the labels of the
:class:`~qbm.ParamHamiltonian`), which is what makes the local Trotter gates possible.
"""

from __future__ import annotations

import numpy as np
import quimb as qu
import quimb.tensor as qtn
import scipy.linalg as sla

_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": qu.pauli("X").A,
    "Y": qu.pauli("Y").A,
    "Z": qu.pauli("Z").A,
}


def _decompose(label: str, coeff: float = 1.0):
    """Pauli-string label -> ``(sites, matrix)`` acting on 1 or 2 sites."""
    sites = [i for i, c in enumerate(label) if c != "I"]
    if not sites:
        raise ValueError("identity generators are not supported")
    if len(sites) == 1:
        mat = _PAULI[label[sites[0]]]
    elif len(sites) == 2:
        mat = np.kron(_PAULI[label[sites[0]]], _PAULI[label[sites[1]]])
    else:
        raise ValueError(
            f"generator {label!r} acts on {len(sites)} sites; the tensor-network "
            "backend supports 1- and 2-body Pauli generators only"
        )
    return tuple(sites), coeff * mat


class TensorNetworkThermalState:
    """Purified matrix-product-state thermal state.

    Site ordering is ``[phys_0, anc_0, phys_1, anc_1, ...]`` so each Bell pair is
    adjacent and physical neighbours are two sites apart.
    """

    def __init__(self, ham, theta, max_bond=100, cutoff=1e-14, trotter_steps=100):
        self.ham = ham
        self.theta = np.asarray(theta, dtype=float)
        self.n_qubits = ham.n_qubits
        self.dim = ham.dim
        self.max_bond = max_bond
        self.cutoff = cutoff
        self.trotter_steps = trotter_steps
        self._labels = list(ham.labels)
        self.psi = self._build()

    # -- construction ------------------------------------------------------
    def _build(self):
        n = self.n_qubits
        psi = qtn.MPS_computational_state("0" * (2 * n))
        for i in range(n):  # infinite-temperature purification: Bell pairs
            psi.gate_(qu.hadamard().A, 2 * i, contract=True)
            psi.gate_(qu.CNOT().A, (2 * i, 2 * i + 1), contract="swap+split")

        terms = [_decompose(lbl, t) for lbl, t in zip(self._labels, self.theta)]
        if self.ham.offset is not None:
            raise NotImplementedError("ParamHamiltonian.offset is not supported by the TN backend")
        # exp(-G/2) via symmetric (2nd-order) Trotter: error O(dt^2)
        dt = 0.5 / self.trotter_steps
        half = [(sites, sla.expm(-0.5 * dt * mat)) for sites, mat in terms]
        sweep = half + half[::-1]
        for _ in range(self.trotter_steps):
            for sites, gate in sweep:
                where = tuple(2 * s for s in sites)
                if len(where) == 1:
                    psi.gate_(gate, where[0], contract=True)
                else:
                    psi.gate_(
                        gate,
                        where,
                        contract="swap+split",
                        max_bond=self.max_bond,
                        cutoff=self.cutoff,
                    )
            psi.normalize()
        return psi

    # -- expectation values ------------------------------------------------
    def _expect_label(self, label: str) -> float:
        sites, mat = _decompose(label)
        where = tuple(2 * s for s in sites)
        ket = (
            self.psi.gate(
                mat,
                where,
                contract="swap+split" if len(where) > 1 else True,
                max_bond=self.max_bond,
                cutoff=self.cutoff,
            )
            if len(where) > 1
            else self.psi.gate(mat, where[0], contract=True)
        )
        return float(np.real(self.psi.H @ ket))

    def expect(self, op) -> float:
        """Expectation value of a Pauli-string label, or a dense operator (small ``n`` only)."""
        if isinstance(op, str):
            return self._expect_label(op)
        op = np.asarray(op, dtype=complex)
        if op.shape[0] > 2**14:
            raise ValueError(
                "dense operators are impractical at this size on the tensor-network "
                "backend; pass a Pauli-string label instead"
            )
        rho = self.density_matrix()
        return float(np.real(np.sum(rho * op.T)))

    def generator_expectations(self) -> np.ndarray:
        """``[<G_j>]`` -- the only primitive the relative-entropy/NLL gradient needs."""
        return np.array([self._expect_label(lbl) for lbl in self._labels])

    # -- read-outs ---------------------------------------------------------
    def density_matrix(self) -> np.ndarray:
        """Dense ``rho`` by tracing out the ancillas (small ``n`` only)."""
        if self.n_qubits > 14:
            raise ValueError(f"refusing to build a dense 2^{self.n_qubits} density matrix")
        phys = [2 * i for i in range(self.n_qubits)]
        rho = self.psi.partial_trace_to_mpo(phys).to_dense()
        rho = np.asarray(rho, dtype=complex)
        return rho / np.trace(rho)

    def probabilities(self) -> np.ndarray:
        return np.real(np.diag(self.density_matrix()))

    def sample(self, n: int, rng=None) -> np.ndarray:
        """Sample computational-basis outcomes by sampling the purified MPS.

        Sampling all ``2n`` sites and keeping the physical ones is exactly sampling
        the visible marginal ``<v|rho|v>``.
        """
        seed = (
            None
            if rng is None
            else int(np.random.default_rng(rng).integers(1 << 31))
            if not isinstance(rng, np.random.Generator)
            else int(rng.integers(1 << 31))
        )
        out = []
        for config, _ in self.psi.sample(n, seed=seed):
            bits = [config[2 * i] for i in range(self.n_qubits)]
            out.append(int("".join(str(b) for b in bits), 2))
        return np.array(out)

    # -- deliberately unsupported -----------------------------------------
    def _unsupported(self, what):
        raise NotImplementedError(
            f"{what} needs the full spectrum / belief-propagation channel, which the "
            "tensor-network backend does not provide. Use backend='dense' (exact, "
            "<=13 qubits) or backend='jax'. The TN backend supports expectation-based "
            "training (relative entropy / NLL), which is what scales."
        )

    def observable_gradient(self, op):
        self._unsupported("the energy-loss gradient")

    def metric(self, kind: str = "kubo_mori"):
        self._unsupported(f"the {kind} metric")

    def state_derivatives(self):
        self._unsupported("state derivatives")

    def diagonal_gradient(self):
        # hidden-unit NLL used to die here with a bare AttributeError; the exact
        # alternative is qbm.losses.GibbsMapNLL, whose gradient needs only <G_j>
        raise NotImplementedError(
            "the diagonal gradient needs d_j rho, which the tensor-network backend does "
            "not provide. For hidden-unit models use qbm.losses.GibbsMapNLL, which is "
            "exact (even for non-commuting hidden operators) and needs only generator "
            "expectations, so it runs on this backend."
        )

    def log_partition(self):
        self._unsupported("log Z")

    def entropy(self):
        self._unsupported("the von Neumann entropy")


class TensorNetworkBackend:
    """Builds :class:`TensorNetworkThermalState` objects (purified MPS, scalable)."""

    name = "tensor_network"

    def __init__(self, max_bond=100, cutoff=1e-14, trotter_steps=100):
        self.max_bond = max_bond
        self.cutoff = cutoff
        self.trotter_steps = trotter_steps

    def thermal_state(self, ham, theta) -> TensorNetworkThermalState:
        return TensorNetworkThermalState(
            ham,
            theta,
            max_bond=self.max_bond,
            cutoff=self.cutoff,
            trotter_steps=self.trotter_steps,
        )

    def __repr__(self) -> str:
        return f"TensorNetworkBackend(max_bond={self.max_bond}, trotter_steps={self.trotter_steps})"
