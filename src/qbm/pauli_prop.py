"""Pauli propagation: thermal-state simulation in the Pauli basis.

A complementary classical engine to the dense and tensor-network backends.  Instead of
storing ``rho`` as a ``2^n x 2^n`` matrix, it tracks the state as a **sparse sum of Pauli
strings**, ``rho = sum_P c_P P``, and evolves that sum under imaginary time.  The key
observation (Rudolph et al., arXiv:2602.04878) is that the thermal state is reached by
evolving the **identity** -- the sparsest possible operator, a single Pauli term -- so at
high temperature the representation stays sparse and the cost is set by the number of
retained Pauli strings, not by ``2^n``.

**Imaginary-time gate.**  The Gibbs state ``rho_beta = e^{-beta H/2} I e^{-beta H/2}/Z``
is built by a Trotterised product of elementary gates, one per Hamiltonian term
``H = sum_k h_k G_k`` (``G_k`` a Pauli string).  Each gate is the symmetric conjugation
``e^{-theta G} (.) e^{-theta G}``, whose action on a Pauli string ``P`` is (Eq. H10 of
that paper; the ``PauliSampling.jl`` reference)::

    e^{-theta G} P e^{-theta G} = { P                              if {P, G} = 0
                                  { cosh(2 theta) P - sinh(2 theta) P G  if [P, G] = 0

so **commuting** terms branch (creating ``P G``) and **anticommuting** terms are left
invariant -- the reverse of real-time propagation, and the reason imaginary time is
denser.  We fold the factor of two into the step angle, so the code uses ``cosh(theta)``
/ ``sinh(theta)`` with ``theta = beta h_k / L`` over ``L`` Trotter layers.

**Truncation.**  The sum is kept sparse by discarding terms whose coefficient falls below
``coeff_cutoff`` (relative to the identity coefficient) or whose Pauli weight exceeds
``max_weight``.  This is exact for commuting (classical) Hamiltonians and controlled by
the truncation for the rest; the error is analysed in arXiv:2602.04878.

**Sampling.**  Truncation can push ``rho`` outside the positive-semidefinite cone, so its
computational-basis "probabilities" may go slightly negative and standard sampling is
ill-defined.  The locally normalised chain-rule sampler of Minervini et al.
(``Sampling from Thermal Quantum States via Pauli Propagation``) resolves this: it draws
bits sequentially, taking absolute values of the quasi-marginals branch by branch, which
is a valid distribution within a controlled total-variation distance of the true Born
distribution.  Implemented in :meth:`PauliSum.sample`.

Everything here is pure NumPy and has no external dependency.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# symplectic Pauli-string algebra
#
# A Pauli string on ``n`` qubits is stored as a pair of integer bitmasks ``(x, z)``:
# bit ``n-1-i`` corresponds to qubit ``i`` (qubit 0 is the most significant, matching
# :func:`qbm.operators.pauli`).  Single-qubit encoding, with ``Y = i X Z``:
#     I=(0,0)  X=(1,0)  Z=(0,1)  Y=(1,1).
# ---------------------------------------------------------------------------
_1Q_TO_XZ = {"I": (0, 0), "X": (1, 0), "Y": (1, 1), "Z": (0, 1)}
_XZ_TO_1Q = {(0, 0): "I", (1, 0): "X", (1, 1): "Y", (0, 1): "Z"}


def label_to_xz(label: str) -> tuple[int, int]:
    """Pack a Pauli-string label (e.g. ``"XIZ"``) into ``(xmask, zmask)``."""
    n = len(label)
    x = z = 0
    for i, p in enumerate(label.upper()):
        xb, zb = _1Q_TO_XZ[p]
        bit = 1 << (n - 1 - i)
        if xb:
            x |= bit
        if zb:
            z |= bit
    return x, z


def xz_to_label(x: int, z: int, n: int) -> str:
    """Unpack ``(xmask, zmask)`` back into an ``n``-qubit Pauli-string label."""
    out = []
    for i in range(n):
        bit = 1 << (n - 1 - i)
        out.append(_XZ_TO_1Q[(int(bool(x & bit)), int(bool(z & bit)))])
    return "".join(out)


def _popcount(v: int) -> int:
    return bin(v).count("1")


def commutes(p: tuple[int, int], q: tuple[int, int]) -> bool:
    """True iff two Pauli strings commute (symplectic inner product is even)."""
    x1, z1 = p
    x2, z2 = q
    return (_popcount(x1 & z2) + _popcount(z1 & x2)) & 1 == 0


def pauli_product(p: tuple[int, int], q: tuple[int, int]) -> tuple[int, int, complex]:
    """``P Q = phase * R``; return ``(xR, zR, phase)`` with ``phase in {1, i, -1, -i}``.

    Derived from the encoding ``P = i^{popcount(x & z)} X^x Z^z``: reordering ``Z^{zP}``
    past ``X^{xQ}`` gives a ``(-1)^{<zP, xQ>}`` sign, and the residual ``i`` powers come
    from the per-string ``Y`` counts.  For commuting ``P, Q`` the phase is real (``+-1``).
    """
    xp, zp = p
    xq, zq = q
    xr, zr = xp ^ xq, zp ^ zq
    exp_i = (_popcount(xp & zp) + _popcount(xq & zq) - _popcount(xr & zr)) % 4
    sign = -1 if _popcount(zp & xq) & 1 else 1
    return xr, zr, (1j**exp_i) * sign


def pauli_weight(p: tuple[int, int]) -> int:
    """Number of non-identity single-qubit factors in a Pauli string."""
    x, z = p
    return _popcount(x | z)


# ---------------------------------------------------------------------------
# the Pauli sum and its imaginary-time evolution
# ---------------------------------------------------------------------------
class PauliSum:
    """A sparse operator ``sum_P coeff[P] P`` over ``n`` qubits (``coeff`` real).

    The coefficients follow the density-matrix convention ``rho = sum_P coeff[P] P``, so a
    normalised thermal state has ``coeff[I] = 2^-n`` and ``expect(G) = coeff[G] * 2^n``.
    Built by :func:`thermal_state`; queried through :meth:`expect`, :meth:`probabilities`,
    :meth:`sample`, and :meth:`to_matrix`.
    """

    def __init__(self, n_qubits: int, coeffs: dict | None = None):
        self.n_qubits = n_qubits
        self.coeffs: dict[tuple[int, int], float] = coeffs if coeffs is not None else {}

    # -- basics -----------------------------------------------------------
    def __len__(self) -> int:
        return len(self.coeffs)

    @property
    def n_terms(self) -> int:
        return len(self.coeffs)

    def get(self, key: tuple[int, int]) -> float:
        return self.coeffs.get(key, 0.0)

    def copy(self) -> PauliSum:
        return PauliSum(self.n_qubits, dict(self.coeffs))

    # -- read-outs --------------------------------------------------------
    def expect_pauli(self, label: str) -> float:
        """``Tr[P rho]`` for a Pauli-string observable, read straight off the coefficient."""
        return self.coeffs.get(label_to_xz(label), 0.0) * (2**self.n_qubits)

    def to_matrix(self) -> np.ndarray:
        """Dense reconstruction ``sum_P coeff[P] P`` (only feasible for small ``n``)."""
        from .operators import pauli

        dim = 2**self.n_qubits
        rho = np.zeros((dim, dim), dtype=complex)
        for (x, z), c in self.coeffs.items():
            rho += c * pauli(xz_to_label(x, z, self.n_qubits))
        return rho

    def expect(self, observable) -> float:
        """``Tr[O rho]`` for a Pauli label (fast) or a dense Hermitian matrix (small ``n``)."""
        if isinstance(observable, str):
            return self.expect_pauli(observable)
        O = np.asarray(observable, dtype=complex)
        return float(np.real(np.sum(O.T * self.to_matrix())))

    def _diagonal_terms(self):
        """``(zmasks, coeffs)`` arrays of the diagonal (``I``/``Z``) Pauli terms.

        Only diagonal strings contribute to computational-basis probabilities.
        """
        zs, cs = [], []
        for (x, z), c in self.coeffs.items():
            if x == 0:
                zs.append(z)
                cs.append(c)
        return np.array(zs, dtype=np.int64), np.array(cs, dtype=float)

    def probabilities(self) -> np.ndarray:
        """Computational-basis quasi-distribution ``p(x) = <x|rho|x>`` over ``2^n`` states.

        These are the diagonal of ``rho``; under truncation a few entries may be slightly
        negative (the "sign problem" the sampler is designed around).  They sum to one
        when the identity coefficient is normalised.
        """
        n = self.n_qubits
        zs, cs = self._diagonal_terms()
        x = np.arange(2**n, dtype=np.int64)
        if zs.size == 0:
            return np.full(2**n, cs.sum() if cs.size else 0.0)
        # parity of (x AND z) for every basis state and every diagonal term
        inter = x[:, None] & zs[None, :]
        parity = _popcount_array(inter) & 1
        signs = 1 - 2 * parity  # (2^n, n_diag)
        return signs @ cs

    def sample(self, n_samples: int, rng=None) -> np.ndarray:
        """Draw computational-basis bitstrings with the locally normalised sampler.

        Implements Algorithm 1 of *Sampling from Thermal Quantum States via Pauli
        Propagation*: bits are chosen one at a time, and each conditional is the absolute
        quasi-marginal of the candidate bit divided by the sum of the two, which is a
        valid distribution branch by branch even when ``rho`` is non-positive.  Returns an
        array of integer basis-state indices (qubit 0 most significant).
        """
        rng = np.random.default_rng() if rng is None else rng
        n = self.n_qubits
        zs, cs = self._diagonal_terms()
        out = np.empty(n_samples, dtype=np.int64)
        for s in range(n_samples):
            prefix = 0
            for j in range(n):
                pos_bit = 1 << (n - 1 - j)
                tail_mask = pos_bit - 1  # positions j+1 .. n-1 (undecided tail)
                active = (zs & tail_mask) == 0  # term is identity on the tail
                za, ca = zs[active], cs[active]
                q = np.empty(2)
                for b in (0, 1):
                    bits = prefix | (b * pos_bit)
                    parity = _popcount_array(za & bits) & 1
                    q[b] = np.dot(1 - 2 * parity, ca)
                a0, a1 = abs(q[0]), abs(q[1])
                total = a0 + a1
                p1 = 0.5 if total == 0.0 else a1 / total
                if rng.random() < p1:
                    prefix |= pos_bit
            out[s] = prefix
        return out

    def log_likelihood(self, bitstrings) -> np.ndarray:
        """Exact ``log p_hat(x)`` the sampler assigns to each basis-state index.

        The locally normalised model has a tractable pointwise likelihood -- the product
        of its chain-rule conditionals -- which most generative models lack.  Returns one
        log-probability per input index.
        """
        n = self.n_qubits
        zs, cs = self._diagonal_terms()
        bitstrings = np.atleast_1d(np.asarray(bitstrings, dtype=np.int64))
        logp = np.zeros(bitstrings.shape[0])
        for idx, x in enumerate(bitstrings):
            prefix = 0
            acc = 0.0
            for j in range(n):
                pos_bit = 1 << (n - 1 - j)
                tail_mask = pos_bit - 1
                active = (zs & tail_mask) == 0
                za, ca = zs[active], cs[active]
                q = np.empty(2)
                for b in (0, 1):
                    bits = prefix | (b * pos_bit)
                    parity = _popcount_array(za & bits) & 1
                    q[b] = abs(np.dot(1 - 2 * parity, ca))
                total = q[0] + q[1]
                chosen = 1 if (x & pos_bit) else 0
                prob = 0.5 if total == 0.0 else q[chosen] / total
                acc += np.log(max(prob, 1e-300))
                if chosen:
                    prefix |= pos_bit
            logp[idx] = acc
        return logp

    def spectral_negativity(self) -> float:
        """``N(rho) = sum_{lambda<0} |lambda|`` -- how far truncation pushed ``rho`` non-physical.

        The certified sampling error is governed by this quantity (Theorem III.1 of the
        sampling paper).  Requires a dense eigendecomposition, so small ``n`` only.
        """
        w = np.linalg.eigvalsh(self.to_matrix())
        return float(-w[w < 0].sum())


def _popcount_array(a: np.ndarray) -> np.ndarray:
    """Vectorised population count for int64 arrays."""
    a = a.astype(np.int64)
    out = np.zeros_like(a)
    while np.any(a):
        out += (a & 1).astype(np.int64)
        a >>= 1
    return out


# ---------------------------------------------------------------------------
# imaginary-time propagation
# ---------------------------------------------------------------------------
def _apply_ite_gate(coeffs, gen, theta):
    """One imaginary-time gate ``e^{-theta G}(.)e^{-theta G}`` applied in place to ``coeffs``.

    Commuting terms ``P`` are scaled by ``cosh(theta)`` and spawn ``-sinh(theta) P G``;
    anticommuting terms are invariant.  ``gen`` is the generator's ``(x, z)`` mask.
    """
    ch, sh = np.cosh(theta), np.sinh(theta)
    additions: dict[tuple[int, int], float] = {}
    for key, c in list(coeffs.items()):
        if not commutes(key, gen):
            continue  # anticommuting Pauli strings pass through unchanged
        coeffs[key] = c * ch
        rx, rz, phase = pauli_product(key, gen)
        additions[(rx, rz)] = additions.get((rx, rz), 0.0) - sh * float(np.real(phase)) * c
    for key, v in additions.items():
        coeffs[key] = coeffs.get(key, 0.0) + v


def _truncate(coeffs, coeff_cutoff, max_weight):
    """Drop terms below ``coeff_cutoff`` (relative to ``c_I``) or above ``max_weight``.

    The identity term is always kept: it carries the trace and the normalisation.
    """
    cI = abs(coeffs.get((0, 0), 1.0))
    thr = coeff_cutoff * cI
    for key in list(coeffs.keys()):
        if key == (0, 0):
            continue
        if abs(coeffs[key]) <= thr or pauli_weight(key) > max_weight:
            del coeffs[key]


def thermal_state(
    labels,
    coeffs,
    beta: float = 1.0,
    trotter_steps: int = 32,
    coeff_cutoff: float = 1e-10,
    max_weight: float = np.inf,
) -> PauliSum:
    """Prepare ``rho = e^{-beta H/2} I e^{-beta H/2} / Z`` as a truncated Pauli sum.

    Parameters
    ----------
    labels : list of str
        Pauli-string labels of the Hamiltonian terms, ``H = sum_k coeffs[k] * labels[k]``.
    coeffs : array
        Their real coefficients (for a QBM, the trainable ``theta``).
    beta : float
        Inverse temperature; the QBM convention absorbs it into ``coeffs`` (``beta = 1``).
    trotter_steps : int
        Number ``L`` of imaginary-time layers.  The Trotter error is first order in
        ``1/L`` and vanishes exactly for commuting Hamiltonians.
    coeff_cutoff : float
        Discard Pauli terms whose coefficient is below this fraction of ``c_I`` after each
        layer (the "small-coefficient" truncation).
    max_weight : float
        Discard Pauli strings acting on more than this many qubits (weight truncation).
    """
    labels = [s.upper() for s in labels]
    bad = [s for s in labels if set(s) - set("IXYZ")]
    if bad:
        raise ValueError(
            f"pauli-propagation needs Pauli-string generators over I/X/Y/Z; got {bad[0]!r}. "
            "A ParamHamiltonian built from dense matrices has no Pauli labels."
        )
    n = len(labels[0])
    if any(len(s) != n for s in labels):
        raise ValueError("all Hamiltonian labels must have the same length")
    if trotter_steps < 1:
        raise ValueError("trotter_steps must be at least 1")

    gens = [(label_to_xz(lbl), float(h)) for lbl, h in zip(labels, np.asarray(coeffs, float))]
    state = {(0, 0): 1.0}  # start from the identity operator (infinite temperature)
    for _ in range(trotter_steps):
        for gen, h in gens:
            if h == 0.0:
                continue
            _apply_ite_gate(state, gen, beta * h / trotter_steps)
        _truncate(state, coeff_cutoff, max_weight)

    cI = state.get((0, 0), 0.0)
    if cI == 0.0:
        raise FloatingPointError("identity coefficient vanished; check the Hamiltonian scale")
    scale = 1.0 / (cI * 2**n)  # normalise c_I = 1, then trace-normalise (rho = 2^-n I + ...)
    return PauliSum(n, {k: v * scale for k, v in state.items()})


__all__ = [
    "PauliSum",
    "thermal_state",
    "label_to_xz",
    "xz_to_label",
    "commutes",
    "pauli_product",
    "pauli_weight",
]
