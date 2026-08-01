"""Variational Gibbs preparation by imaginary-time evolution (VarQITE).

The exact TFD synthesis in :mod:`qbm.circuits.builder` needs the *eigendecomposition*
of ``G(theta)`` -- fine for validating estimators, useless as a scaling route.  VarQITE
replaces it with something a device can actually run: a parameterized circuit whose
parameters are transported along the imaginary-time flow, using only expectation values.

**The construction.**  Imaginary-time evolution of a normalised state,

    d|psi>/d tau = -(H - <H>) |psi> ,      |psi(tau)> ∝ e^{-tau H} |psi(0)> ,

is not unitary, so it cannot be run directly.  McLachlan's variational principle instead
projects it onto the tangent space of an ansatz ``|psi(lambda)>``: minimising

    || ( d/d tau + H - <H> ) |psi(lambda(tau))> ||

over ``lambda_dot`` gives the linear system

    A(lambda) lambda_dot = C(lambda),
    A_ij = Re[ <d_i psi|d_j psi> - <d_i psi|psi><psi|d_j psi> ]   (the QGT / QFI-4)
    C_i  = -Re <d_i psi| H |psi>  =  -(1/2) d_i <H> ,

which is exactly **quantum natural gradient flow on the energy** -- the same geometry
this library already uses for training (:class:`qbm.optim.NaturalGradient`), here with
the imaginary time step playing the role of the learning rate.

**Getting a Gibbs state out of it.**  Start from ``n`` Bell pairs on ``2n`` qubits: that
is the infinite-temperature thermofield double, whose system marginal is ``I/2^n``.
Because ``|TFD(beta)> ∝ (e^{-beta H/2} ⊗ I)|TFD(0)>``, running imaginary-time evolution
of ``H ⊗ I`` for ``tau = beta/2`` transports the Bell pairs to ``|TFD(beta)>`` and hence
the system marginal to ``rho = e^{-beta H}/Z``.

**The ansatz.**  Rotations ``exp(-i lambda_k P_k / 2)`` for Pauli strings ``P_k`` on the
doubled register, all zero at ``lambda = 0`` so the circuit starts exactly on the
infinite-temperature TFD.  The strings are not arbitrary: for a Hamiltonian term ``P``
there is a *tilt partner* ``K = A ⊗ B`` with ``K|TFD(0)> = -i (P ⊗ I)|TFD(0)>``, i.e. the
generator that reproduces imaginary-time evolution by ``P`` to first order
(:func:`tilt_partner`).  Including the tilt partner of every Hamiltonian term makes the
ansatz *exact* for commuting Hamiltonians and systematically improvable otherwise; a
generic hardware-efficient ansatz without them can be stuck with zero gradient
(``residual = 1``) from the very first step.

**Honesty.**  Everything here is measurable: ``A`` and ``C`` come from Hadamard tests
with controlled-Pauli insertions (:func:`measured_mclachlan_system`), which the test
suite checks against the exact simulator route.  The cost is real and reported by
:meth:`VarQITEResult.resource_estimate` -- ``O(L^2)`` circuits per time step for ``L``
ansatz parameters.  The diagnostic that matters is the McLachlan **residual**, the
fraction of the imaginary-time flow the ansatz cannot represent; it is computable from
the same measured quantities, so on hardware you still know when you are wrong.

References: McArdle et al., *npj Quantum Inf.* **5**, 75 (2019); Yuan et al., *Quantum*
**3**, 191 (2019); Zoufal, Lucchi & Woerner, *Quantum Mach. Intell.* **3**, 7 (2021).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from ..operators import local_pauli_generators
from . import builder
from .ir import Circuit
from .simulator import measure_z
from .simulator import run as simulator_run


# ---------------------------------------------------------------------------
# fast Pauli action:  P|v> is a signed permutation, so precompute it once
# ---------------------------------------------------------------------------
def pauli_action(label: str):
    """``(coeff, perm)`` with ``P @ v == (coeff * v)[perm]`` for the Pauli string ``P``.

    A Pauli string is a signed/phased permutation of the computational basis, so its
    action costs one gather and one multiply instead of a ``2^n x 2^n`` matrix product.
    Convention matches :func:`qbm.operators.pauli`: qubit 0 is the most significant bit.
    """
    label = label.upper()
    n = len(label)
    dim = 1 << n
    idx = np.arange(dim, dtype=np.int64)
    flip = zmask = 0
    n_y = 0
    for i, p in enumerate(label):
        bit = 1 << (n - 1 - i)
        if p in "XY":
            flip |= bit
        if p in "ZY":
            zmask |= bit
        if p == "Y":
            n_y += 1
        if p not in "IXYZ":
            raise ValueError(f"invalid Pauli string: {label!r}")
    parity = np.zeros(dim, dtype=np.int64)
    for b in range(n):
        if (zmask >> b) & 1:
            parity ^= (idx >> b) & 1
    coeff = (1j**n_y) * (1.0 - 2.0 * parity)
    return coeff.astype(complex), idx ^ flip


def apply_pauli(state, action) -> np.ndarray:
    coeff, perm = action
    return (coeff * state)[perm]


def apply_pauli_rotation(state, action, angle) -> np.ndarray:
    """``exp(-i angle P / 2) |state>``."""
    return np.cos(angle / 2) * state - 1j * np.sin(angle / 2) * apply_pauli(state, action)


# ---------------------------------------------------------------------------
# ansatz
# ---------------------------------------------------------------------------
def tilt_partner(label: str) -> str:
    """The ``2n``-qubit Pauli ``K`` generating imaginary-time evolution by ``P``.

    For an ``n``-qubit Pauli string ``P`` (the system term), returns the Hermitian Pauli
    ``K = A ⊗ B`` on system+ancilla with ``K |TFD(0)> = ±i (P ⊗ I) |TFD(0)>``, so that
    ``exp(-i lambda K / 2)`` moves the maximally entangled state exactly the way
    ``exp(-tau P)`` does.  Example: ``"Z" -> "YX"`` (the single-qubit Gibbs state is
    reached exactly by one ``Y ⊗ X`` rotation).

    Construction: with ``B = X_s`` on one ancilla site ``s`` in the support of ``P``,
    ``A = -i P X_s`` is again a Hermitian Pauli whenever ``P_s`` is ``Z`` or ``Y``; if
    the support is all ``X`` use ``B = Z_s`` instead.
    """
    label = label.upper()
    n = len(label)
    if set(label) - set("IXYZ"):
        raise ValueError(f"tilt_partner needs a Pauli string over I/X/Y/Z; got {label!r}")
    support = [i for i, p in enumerate(label) if p != "I"]
    if not support:
        raise ValueError("the identity has no tilt partner (it does not move the state)")
    site = next((i for i in support if label[i] in "ZY"), None)
    anc_pauli = "X"
    if site is None:  # all-X support
        site = support[0]
        anc_pauli = "Z"
    sys_part = list(label)
    sys_part[site] = {"Z": "Y", "Y": "Z", "X": "Y"}[label[site]]
    anc_part = ["I"] * n
    anc_part[site] = anc_pauli
    return "".join(sys_part) + "".join(anc_part)


def bell_preparation(n: int, offset: int = 0, n_qubits: int | None = None) -> Circuit:
    """``n`` Bell pairs between system qubit ``i`` and ancilla qubit ``n + i``.

    This is the infinite-temperature thermofield double: its system marginal is the
    maximally mixed state ``I / 2^n = e^{-0 H} / Z``.
    """
    total = n_qubits if n_qubits is not None else 2 * n + offset
    circ = Circuit(total, name="tfd-beta0")
    for i in range(n):
        circ.h(offset + i)
        circ.cx(offset + i, offset + n + i)
    return circ


@dataclass
class PauliRotationAnsatz:
    """``|psi(lambda)> = prod_k exp(-i lambda_k P_k / 2) |psi_0>`` on ``2 n_system`` qubits.

    ``labels[k]`` is the Pauli string of rotation ``k`` (length ``2 * n_system``), applied
    in list order.  At ``lambda = 0`` the circuit is the identity, so the ansatz starts
    exactly on the infinite-temperature TFD prepared by :func:`bell_preparation`.
    """

    labels: list[str]
    n_system: int
    name: str = "tfd-ansatz"
    _actions: list = field(default_factory=list, repr=False)

    def __post_init__(self):
        self.labels = [s.upper() for s in self.labels]
        width = 2 * self.n_system
        bad = [s for s in self.labels if len(s) != width]
        if bad:
            raise ValueError(f"ansatz Paulis must have length {width}; got e.g. {bad[0]!r}")
        self._actions = [pauli_action(s) for s in self.labels]

    @property
    def n_params(self) -> int:
        return len(self.labels)

    @property
    def n_qubits(self) -> int:
        return 2 * self.n_system

    @property
    def dim(self) -> int:
        return 1 << self.n_system

    def initial_state(self) -> np.ndarray:
        """The infinite-temperature TFD (``n`` Bell pairs) as a statevector."""
        d = self.dim
        return np.eye(d, dtype=complex).reshape(-1) / np.sqrt(d)

    def state(self, lam) -> np.ndarray:
        psi = self.initial_state()
        for action, a in zip(self._actions, np.asarray(lam, float)):
            psi = apply_pauli_rotation(psi, action, a)
        return psi

    def prefix_states(self, lam) -> list:
        """``[|psi_0>, U_0|psi_0>, U_1 U_0|psi_0>, ...]`` -- one forward pass."""
        out = [self.initial_state()]
        for action, a in zip(self._actions, np.asarray(lam, float)):
            out.append(apply_pauli_rotation(out[-1], action, a))
        return out

    def derivative_states(self, lam):
        """``(|psi>, D)`` with ``D[:, k] = d|psi>/d lambda_k``.

        ``d_k |psi> = U_L..U_{k+1} (-i P_k / 2) U_k..U_1 |psi_0>``; costs ``O(L^2)``
        Pauli applications, the same scaling as the ``O(L^2)`` Hadamard tests a device
        would run for the same matrix.
        """
        lam = np.asarray(lam, float)
        fwd = self.prefix_states(lam)
        L = self.n_params
        D = np.empty((fwd[0].size, L), dtype=complex)
        for k in range(L):
            v = -0.5j * apply_pauli(fwd[k + 1], self._actions[k])
            for m in range(k + 1, L):
                v = apply_pauli_rotation(v, self._actions[m], lam[m])
            D[:, k] = v
        return fwd[-1], D

    # -- circuit emission --------------------------------------------------
    def circuit(self, lam, offset: int = 0, n_qubits: int | None = None) -> Circuit:
        """Gate-level circuit preparing ``|psi(lambda)>`` (Bell pairs + Pauli rotations)."""
        total = n_qubits if n_qubits is not None else self.n_qubits + offset
        circ = bell_preparation(self.n_system, offset=offset, n_qubits=total)
        circ.name = self.name
        for label, a in zip(self.labels, np.asarray(lam, float)):
            builder.pauli_rotation(circ, label, float(a) / 2.0, offset=offset)
        return circ

    def __repr__(self) -> str:
        return (
            f"PauliRotationAnsatz({self.name!r}, n_system={self.n_system}, "
            f"n_qubits={self.n_qubits}, n_params={self.n_params})"
        )


def tfd_ansatz(
    labels=None,
    n: int | None = None,
    depth: int = 2,
    include_rotations: bool = True,
    mirror: bool = False,
    connectivity: str = "all",
) -> PauliRotationAnsatz:
    """Build the TFD ansatz for a Hamiltonian.

    Parameters
    ----------
    labels : list of str, optional
        Pauli strings of the Hamiltonian terms on the ``n`` *system* qubits (e.g.
        ``ParamHamiltonian.labels``).  Each contributes its :func:`tilt_partner` --
        the direction that actually performs imaginary-time evolution -- per layer.
        This *Hamiltonian-adapted* choice is exact for commuting Hamiltonians.
    n : int, optional
        System size; used only when ``labels`` is omitted (a dense Hamiltonian), in
        which case the generic set of all one-body fields and ``XX``/``YY``/``ZZ``
        couplings over ``connectivity`` is used.  That covers the usual 2-local models,
        but it is a guess -- prefer passing ``labels`` when you know the terms, and read
        ``result.residual`` either way.
    depth : int
        Number of repetitions.  Errors fall roughly by an order of magnitude per layer
        until the flow is represented exactly; check ``result.residual``.
    include_rotations : bool
        Also include ``P ⊗ I`` (rotate the system's eigenbasis) alongside each tilt.
    mirror : bool
        Also include ``I ⊗ P`` on the ancilla register.
    """
    if labels is None:
        if n is None:
            raise ValueError("give either the Hamiltonian's Pauli `labels` or the system size `n`")
        labels = local_pauli_generators(
            n, fields=("Z", "X", "Y"), couplings=("ZZ", "XX", "YY"), connectivity=connectivity
        )
    labels = [s.upper() for s in labels]
    bad = [s for s in labels if set(s) - set("IXYZ")]
    if bad:
        raise ValueError(
            f"ansatz labels must be Pauli strings over I/X/Y/Z; got {bad[0]!r}. "
            "A ParamHamiltonian built from dense matrices has no Pauli labels -- pass "
            "`labels=` explicitly, or let `prepare_gibbs` fall back to a generic ansatz."
        )
    n = len(labels[0])
    if any(len(s) != n for s in labels):
        raise ValueError("all Hamiltonian labels must have the same length")
    labels = [s for s in labels if set(s) != {"I"}]
    if not labels:
        raise ValueError("the Hamiltonian has no non-identity terms to build an ansatz from")

    pad = "I" * n
    strings: list[str] = []
    for _ in range(depth):
        for lbl in labels:
            strings.append(tilt_partner(lbl))
            if include_rotations:
                strings.append(lbl + pad)
            if mirror:
                strings.append(pad + lbl)
    return PauliRotationAnsatz(strings, n_system=n, name=f"tfd-ansatz-d{depth}")


# ---------------------------------------------------------------------------
# McLachlan's variational principle
# ---------------------------------------------------------------------------
def mclachlan_system(ansatz, lam, hamiltonian, gauge="qgt") -> dict:
    """``A``, ``C`` and diagnostics at parameters ``lam`` (exact simulator route).

    ``hamiltonian`` is the ``2^n x 2^n`` *system* operator; it acts as ``H ⊗ I`` on the
    doubled register, which for a statevector reshaped to ``(d, d)`` is just ``H @ Psi``.

    ``gauge="qgt"`` uses the quantum geometric tensor (Fubini-Study, ``= QFI / 4``);
    ``gauge="gram"`` uses the plain Gram matrix ``Re<d_i psi|d_j psi>`` of McArdle et al.
    They differ only by the ansatz's global-phase direction.
    """
    H = np.asarray(hamiltonian, dtype=complex)
    d = ansatz.dim
    psi, D = ansatz.derivative_states(lam)
    Psi = psi.reshape(d, d)
    HPsi = (H @ Psi).reshape(-1)

    gram = np.real(D.conj().T @ D)
    phase = np.imag(psi.conj() @ D)  # <psi|d_k psi> = i * phase[k]
    if gauge == "qgt":
        A = gram - np.outer(phase, phase)
    elif gauge == "gram":
        A = gram
    else:
        raise ValueError(f"unknown gauge {gauge!r}; use 'qgt' or 'gram'")
    C = -np.real(D.conj().T @ HPsi)

    energy = float(np.real(np.vdot(psi, HPsi)))
    variance = float(np.real(np.vdot(psi, (H @ H @ Psi).reshape(-1)))) - energy**2
    return {
        "A": 0.5 * (A + A.T),
        "C": C,
        "energy": energy,
        "variance": variance,
        "psi": psi,
        "derivatives": D,
    }


def solve_mclachlan(A, C, regularization=1e-6) -> np.ndarray:
    """Solve ``A x = C`` for the parameter velocity, Tikhonov-regularised.

    ``A`` is positive semi-definite and generically singular (redundant parameter
    directions), so a plain solve is not safe: a ridge plus a least-squares fallback
    picks the minimum-norm velocity, which is also the physically right choice --
    redundant directions should not move.
    """
    A = np.asarray(A, float)
    C = np.asarray(C, float)
    ridge = A + regularization * np.eye(A.shape[0])
    return np.linalg.lstsq(ridge, C, rcond=None)[0]


def mclachlan_residual(variance, C, lam_dot) -> float:
    """Fraction of the imaginary-time flow the ansatz cannot represent, in ``[0, 1]``.

    ``|| (d_tau + H - <H>)|psi> ||^2 = Var(H) - C . lambda_dot`` at the McLachlan
    solution, normalised by ``Var(H)`` (the norm of the flow itself).  ``0`` means the
    ansatz reproduces imaginary-time evolution exactly; ``1`` means it cannot move at
    all.  Every ingredient is measurable, so this diagnostic survives on hardware.
    """
    variance = float(variance)
    if variance <= 1e-300:
        return 0.0
    return float(min(max(variance - float(np.dot(C, lam_dot)), 0.0) / variance, 1.0))


# ---------------------------------------------------------------------------
# the driver
# ---------------------------------------------------------------------------
@dataclass
class VarQITEResult:
    """Outcome of a VarQITE run: the parameters, the circuit, and the diagnostics."""

    parameters: np.ndarray
    ansatz: PauliRotationAnsatz
    tau: float
    beta: float
    history: dict
    hamiltonian: np.ndarray = field(repr=False, default=None)

    # -- the prepared state -------------------------------------------------
    def statevector(self) -> np.ndarray:
        """The prepared purification ``|TFD(beta)>`` on ``2n`` qubits."""
        return self.ansatz.state(self.parameters)

    def density_matrix(self) -> np.ndarray:
        """The prepared system state ``Tr_anc |TFD><TFD| ≈ e^{-beta H} / Z``."""
        from .. import purification

        return purification.reduced_system_state(self.statevector(), self.ansatz.dim)

    def circuit(self, offset: int = 0, n_qubits: int | None = None) -> Circuit:
        """Gate-level preparation circuit (Bell pairs + Pauli rotations)."""
        return self.ansatz.circuit(self.parameters, offset=offset, n_qubits=n_qubits)

    # -- diagnostics --------------------------------------------------------
    @property
    def residual(self) -> float:
        """Largest McLachlan residual along the trajectory (the honest error signal)."""
        return float(np.max(self.history["residual"]))

    @property
    def energy(self) -> float:
        return float(self.history["energy"][-1])

    def infidelity(self, reference=None) -> float:
        """``1 - |<TFD_exact|TFD_varqite>|^2`` against the exact thermofield double."""
        exact = exact_tfd(self.hamiltonian, self.beta) if reference is None else reference
        return float(1.0 - abs(np.vdot(exact, self.statevector())) ** 2)

    def trace_distance(self, reference=None) -> float:
        """Trace distance of the prepared system state from ``e^{-beta H} / Z``."""
        exact = exact_gibbs(self.hamiltonian, self.beta) if reference is None else reference
        diff = self.density_matrix() - exact
        return float(0.5 * np.sum(np.abs(np.linalg.eigvalsh(diff))))

    def resource_estimate(self, n_hamiltonian_terms=None) -> dict:
        """Circuits, depth and qubits a device would need for this run."""
        L = self.ansatz.n_params
        terms = L if n_hamiltonian_terms is None else int(n_hamiltonian_terms)
        circ = self.circuit()
        steps = len(self.history["tau"]) - 1
        per_step = L * (L + 1) // 2 + L * terms + L
        return {
            "n_qubits": self.ansatz.n_qubits + 1,  # + Hadamard-test ancilla
            "n_parameters": L,
            "prep_gates": len(circ.gates),
            "prep_depth": circ.depth,
            "time_steps": steps,
            "circuits_per_step": per_step,
            "circuits_total": per_step * steps,
        }

    def report(self) -> str:
        h = self.history
        lines = [
            "VarQITE Gibbs preparation",
            f"  ansatz       : {self.ansatz.n_params} rotations on {self.ansatz.n_qubits} qubits",
            f"  beta         : {self.beta:g}   (imaginary time tau = {self.tau:g})",
            f"  steps        : {len(h['tau']) - 1}",
            f"  energy       : {self.energy:.6f}",
            f"  max residual : {self.residual:.3e}   (0 = ansatz represents the flow exactly)",
        ]
        if self.hamiltonian is not None:
            lines.append(f"  infidelity   : {self.infidelity():.3e}   (vs the exact TFD)")
            lines.append(
                f"  trace dist.  : {self.trace_distance():.3e}   (vs the exact Gibbs state)"
            )
        return "\n".join(lines)


def exact_tfd(hamiltonian, beta: float = 1.0) -> np.ndarray:
    """Reference ``|TFD(beta)>`` from the eigendecomposition (validation only)."""
    H = np.asarray(hamiltonian, dtype=complex)
    w, V = np.linalg.eigh(H)
    amp = np.exp(-0.5 * beta * (w - w[0]))
    Psi = (V * amp) @ V.conj().T
    psi = Psi.reshape(-1)
    return psi / np.linalg.norm(psi)


def exact_gibbs(hamiltonian, beta: float = 1.0) -> np.ndarray:
    """Reference ``e^{-beta H} / Z`` from the eigendecomposition (validation only)."""
    H = np.asarray(hamiltonian, dtype=complex)
    w, V = np.linalg.eigh(H)
    p = np.exp(-beta * (w - w[0]))
    return (V * (p / p.sum())) @ V.conj().T


def varqite(
    hamiltonian,
    ansatz,
    tau: float,
    steps: int = 100,
    regularization: float = 1e-6,
    gauge: str = "qgt",
    method: str = "euler",
    track_exact: bool = False,
    callback=None,
) -> VarQITEResult:
    """Transport the ansatz along imaginary-time evolution of ``hamiltonian`` to ``tau``.

    Integrates ``A(lambda) lambda_dot = C(lambda)`` from ``lambda = 0`` (the
    infinite-temperature TFD) with fixed steps.  ``method="euler"`` is the literature
    default; ``"rk4"`` costs four solves per step and buys roughly one extra digit at
    equal total work.  ``track_exact=True`` records the infidelity against exact
    imaginary-time evolution at every step -- a simulation-only diagnostic for plots.
    """
    H = np.asarray(hamiltonian, dtype=complex)
    if H.shape != (ansatz.dim, ansatz.dim):
        raise ValueError(
            f"hamiltonian is {H.shape} but the ansatz has {ansatz.n_system} system "
            f"qubits (expected {(ansatz.dim, ansatz.dim)})"
        )
    if steps < 1:
        raise ValueError("steps must be at least 1")

    lam = np.zeros(ansatz.n_params)
    dt = tau / steps
    hist = {k: [] for k in ("tau", "energy", "variance", "residual", "velocity")}
    if track_exact:
        hist["infidelity"] = []

    def velocity(parameters):
        info = mclachlan_system(ansatz, parameters, H, gauge=gauge)
        return solve_mclachlan(info["A"], info["C"], regularization), info

    for step in range(steps + 1):
        v, info = velocity(lam)
        t = step * dt
        hist["tau"].append(t)
        hist["energy"].append(info["energy"])
        hist["variance"].append(info["variance"])
        hist["residual"].append(mclachlan_residual(info["variance"], info["C"], v))
        hist["velocity"].append(float(np.linalg.norm(v)))
        if track_exact:
            ref = exact_tfd(H, 2.0 * t)
            hist["infidelity"].append(float(1.0 - abs(np.vdot(ref, info["psi"])) ** 2))
        if callback is not None:
            callback(step, t, lam, info)
        if step == steps:
            break
        if method == "euler":
            lam = lam + dt * v
        elif method == "rk4":
            k1 = v
            k2 = velocity(lam + 0.5 * dt * k1)[0]
            k3 = velocity(lam + 0.5 * dt * k2)[0]
            k4 = velocity(lam + dt * k3)[0]
            lam = lam + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            raise ValueError(f"unknown method {method!r}; use 'euler' or 'rk4'")

    return VarQITEResult(
        parameters=lam,
        ansatz=ansatz,
        tau=tau,
        beta=2.0 * tau,
        history={k: np.asarray(v) for k, v in hist.items()},
        hamiltonian=H,
    )


def prepare_gibbs(
    hamiltonian,
    theta=None,
    beta: float = 1.0,
    ansatz=None,
    depth: int = 2,
    steps: int = 100,
    **kwargs,
) -> VarQITEResult:
    """Variationally prepare ``rho = e^{-beta H} / Z`` -- the one-call entry point.

    ``hamiltonian`` may be a dense Hermitian matrix or a
    :class:`~qbm.operators.ParamHamiltonian` (with ``theta``), in which case the ansatz
    is built from its Pauli labels automatically -- the *Hamiltonian-adapted* choice,
    which is both smaller and better than the generic fallback.

    When the terms are not known as Pauli strings (dense generators, or a fixed
    ``offset``), a generic 2-local ansatz is used instead and a ``RuntimeWarning`` is
    issued: it may not cover the flow, and ``result.residual`` is how you find out.
    Pass ``labels=``-derived ``ansatz=`` to take control.

    >>> import qbm
    >>> from qbm.circuits import varqite
    >>> H = qbm.hamiltonians.tfim(2, J=1.0, g=0.8)
    >>> res = varqite.prepare_gibbs(H, beta=1.0, depth=3)      # doctest: +SKIP
    >>> res.trace_distance()                                   # doctest: +SKIP
    """
    labels = None
    if hasattr(hamiltonian, "matrix"):  # a ParamHamiltonian
        if theta is None:
            raise ValueError("pass `theta` alongside a ParamHamiltonian")
        labels = list(hamiltonian.labels)
        n_system = hamiltonian.n_qubits
        H = hamiltonian.matrix(theta)
        reason = None
        if any(set(s) - set("IXYZ") or len(s) != n_system for s in labels):
            reason = "its generators are dense matrices, not Pauli strings"
        elif getattr(hamiltonian, "offset", None) is not None:
            reason = "it carries a fixed `offset` whose Pauli content is unknown here"
        if reason is not None and ansatz is None:
            labels = None
            warnings.warn(
                f"building a generic 2-local VarQITE ansatz because {reason}. It may "
                "not represent the imaginary-time flow -- check `result.residual`, and "
                "pass an explicit `ansatz=` if it is large.",
                RuntimeWarning,
                stacklevel=2,
            )
    else:
        H = np.asarray(hamiltonian, dtype=complex)
        n_system = int(round(np.log2(H.shape[0])))
    if ansatz is None:
        ansatz = tfd_ansatz(labels=labels, n=n_system, depth=depth)
    return varqite(H, ansatz, tau=0.5 * beta, steps=steps, **kwargs)


# ---------------------------------------------------------------------------
# the measurable route: Hadamard tests with controlled-Pauli insertions
# ---------------------------------------------------------------------------
def _controlled_pauli(circ: Circuit, label: str, control: int, offset: int, negate=False):
    """Append ``control-P`` gate by gate.

    A controlled Pauli *string* factorises into one controlled single-qubit Pauli per
    site, and ``CY = S CX S^dagger`` on the target -- so this stays inside the
    ``{cx, cz, s, sdg, x}`` gate set every adapter and OpenQASM already support, and no
    opaque ``c-unitary`` ever enters the VarQITE path.  ``negate`` anti-controls it.
    """
    if negate:
        circ.x(control)
    for i, p in enumerate(label):
        q = offset + i
        if p == "X":
            circ.cx(control, q)
        elif p == "Z":
            circ.cz(control, q)
        elif p == "Y":
            circ.sdg(q)
            circ.cx(control, q)
            circ.s(q)
    if negate:
        circ.x(control)
    return circ


def mclachlan_circuit(ansatz, lam, i, j=None, observable=None, imaginary=False) -> Circuit:
    """One Hadamard-test circuit for an entry of ``A`` (``j``) or ``C`` (``observable``).

    Ancilla is qubit 0.  The ansatz runs **once**; a controlled ``P_i`` is inserted after
    rotation ``i`` on the ``|0>`` branch, and either a controlled ``P_j`` after rotation
    ``j`` or a controlled Hamiltonian term at the end on the ``|1>`` branch.  Measuring
    ``<Z>`` on the ancilla returns ``Re<branch0|branch1>`` (or ``Im`` with ``imaginary``).
    """
    lam = np.asarray(lam, float)
    width = ansatz.n_qubits + 1
    circ = bell_preparation(ansatz.n_system, offset=1, n_qubits=width)
    circ.name = "varqite-mclachlan"
    circ.h(0)
    if imaginary:
        circ.sdg(0)
    for k, (label, angle) in enumerate(zip(ansatz.labels, lam)):
        builder.pauli_rotation(circ, label, float(angle) / 2.0, offset=1)
        if k == i:
            _controlled_pauli(circ, ansatz.labels[i], 0, 1, negate=True)
        if j is not None and k == j:
            _controlled_pauli(circ, ansatz.labels[j], 0, 1)
    if observable is not None:
        _controlled_pauli(circ, observable + "I" * ansatz.n_system, 0, 1)
    circ.h(0)
    return circ


def _phase_circuit(ansatz, lam, k) -> Circuit:
    """Hadamard test for ``<P_k>`` in the state after rotation ``k`` (the QGT phase term)."""
    width = ansatz.n_qubits + 1
    circ = bell_preparation(ansatz.n_system, offset=1, n_qubits=width)
    for label, angle in list(zip(ansatz.labels, np.asarray(lam, float)))[: k + 1]:
        builder.pauli_rotation(circ, label, float(angle) / 2.0, offset=1)
    circ.h(0)
    _controlled_pauli(circ, ansatz.labels[k], 0, 1)
    circ.h(0)
    return circ


def measured_mclachlan_system(
    ansatz,
    lam,
    labels,
    coefficients,
    gauge="qgt",
    shots=None,
    rng=None,
    executor=None,
):
    """``(A, C)`` from Hadamard tests only -- the hardware route.

    ``labels``/``coefficients`` are the Pauli decomposition of the *system* Hamiltonian.
    Costs ``L(L+1)/2 + L * n_terms + L`` circuits; the exact simulator route in
    :func:`mclachlan_system` returns the same numbers (the test suite checks it), so this
    exists to prove measurability and to count resources honestly.
    """
    exe = executor or simulator_run
    lam = np.asarray(lam, float)
    L = ansatz.n_params
    width = ansatz.n_qubits + 1

    def z(circ):
        return measure_z(exe(circ), 0, width, shots=shots, rng=rng)

    gram = np.zeros((L, L))
    for i in range(L):
        for j in range(i, L):
            val = 0.25 * z(mclachlan_circuit(ansatz, lam, i, j=j))
            gram[i, j] = gram[j, i] = val

    C = np.zeros(L)
    for i in range(L):
        for label, c in zip(labels, np.asarray(coefficients, float)):
            if c == 0.0:
                continue
            C[i] += 0.5 * c * z(mclachlan_circuit(ansatz, lam, i, observable=label, imaginary=True))

    if gauge == "gram":
        return gram, C
    phase = np.array([-0.5 * z(_phase_circuit(ansatz, lam, k)) for k in range(L)])
    return gram - np.outer(phase, phase), C


__all__ = [
    "PauliRotationAnsatz",
    "VarQITEResult",
    "apply_pauli",
    "apply_pauli_rotation",
    "bell_preparation",
    "exact_gibbs",
    "exact_tfd",
    "mclachlan_circuit",
    "mclachlan_residual",
    "mclachlan_system",
    "measured_mclachlan_system",
    "pauli_action",
    "prepare_gibbs",
    "solve_mclachlan",
    "tfd_ansatz",
    "tilt_partner",
    "varqite",
]
