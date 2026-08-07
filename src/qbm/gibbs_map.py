"""The Gibbs map: exact hidden-unit marginalisation for visible/hidden QBMs.

Block-Gibbs contrastive divergence (:mod:`qbm.sampling`) is biased when the hidden
operators do not commute, because its ``visible | hidden`` step collapses the hidden
register onto a single measured Pauli basis.  This module removes that bias exactly, and
in doing so unlocks hidden-unit training on the backends that cannot form ``d_j rho``.

**The observation.**  In the standard RBM/sqRBM structure every *visible* operator is
diagonal (``I`` or ``Z``), so ``G(theta)`` is block diagonal in the visible computational
basis::

    G(theta) = (+)_v  G_h(v),        G_h(v) = <v| G(theta) |v>   (an operator on the
                                                                  hidden register only)

Consequently the hidden register can be **traced out exactly** rather than sampled.
Writing ``Z_v = Tr_h[exp(-G_h(v))]`` and ``rho_v = exp(-G_h(v)) / Z_v`` for the
*conditional* hidden Gibbs state -- the **Gibbs map** ``v -> rho_v``, an imaginary-time
evolution on the hidden register -- the visible marginal is exact,

    p(v) = Z_v / Z ,      Z = sum_v Z_v ,

and, using ``d_theta Tr[e^{-G}] = -Tr[(d_theta G) e^{-G}]`` (cyclicity makes this hold
even for non-commuting ``G``), so is the negative-log-likelihood gradient:

    d_j  ( -sum_v q(v) log p(v) )  =  sum_v q(v) <G_j>_{sigma_v}  -  <G_j>_rho ,
    sigma_v := |v><v| (x) rho_v .

That is the familiar positive-phase / negative-phase structure of contrastive divergence
-- but **exact for non-commuting hidden operators**, because the positive phase uses the
true conditional Gibbs state instead of a sampled classical conditional.

**Why this matters for scale.**  The negative phase is just ``<G_j>_rho``, which *every*
backend provides via ``generator_expectations()``.  The positive phase touches only the
hidden register (dimension ``2^n_hidden``) and only the visible configurations that
actually occur in the data.  So the cost is ``O(D_distinct * 8^n_hidden)`` and is
**independent of the number of visible units** -- whereas the exact dense route
(:class:`~qbm.losses.MarginalNLL`) needs ``d_j rho``, which the tensor-network,
circuit and Pauli-propagation backends cannot supply.  Use
:class:`~qbm.losses.GibbsMapNLL` to train hidden-unit models on any of them.

Reference: the Gibbs-map / imaginary-time treatment of arXiv:2511.11802.
"""

from __future__ import annotations

import numpy as np

from .operators import pauli

_ENUMERATION_CAP = 1 << 16  # refuse to enumerate all 2^n_visible beyond this


def _as_visible_distribution(data, n_visible):
    """Coerce ``data`` to ``(configs, weights)`` over visible bitstrings.

    Accepts a probability vector of length ``2^n_visible``, or a 1-D array of integer
    basis-state indices (a dataset), in which case only the *distinct* configurations are
    kept -- which is what makes the positive phase independent of ``2^n_visible``.
    """
    data = np.asarray(data)
    dim = 1 << n_visible
    if np.issubdtype(data.dtype, np.floating) and data.ndim == 1 and data.size == dim:
        nz = np.nonzero(data)[0]
        return nz.astype(np.int64), data[nz] / data.sum()
    if np.issubdtype(data.dtype, np.integer) and data.ndim == 1:
        configs, counts = np.unique(data, return_counts=True)
        if configs.max(initial=0) >= dim:
            raise ValueError(f"sample index {configs.max()} out of range for {n_visible} qubits")
        return configs.astype(np.int64), counts / counts.sum()
    raise ValueError(
        "data must be a probability vector of length 2^n_visible or a 1-D integer array "
        "of visible basis-state indices"
    )


class GibbsMap:
    """Exact conditional hidden state ``v -> rho_v`` for a visible/hidden QBM.

    Requires every generator to be a Pauli string that is **diagonal (I or Z) on the
    visible register** -- the RBM / semi-quantum-RBM structure -- which is what makes
    ``G(theta)`` block diagonal in ``v``.  Hidden operators may be arbitrary and
    non-commuting; that is the point.

    Parameters
    ----------
    ham : ParamHamiltonian
        Must carry Pauli-string labels.
    n_visible : int
        Number of leading (most-significant) qubits treated as visible.
    """

    def __init__(self, ham, n_visible: int):
        labels = list(ham.labels)
        bad = [s for s in labels if set(s) - set("IXYZ")]
        if bad:
            raise ValueError(
                f"the Gibbs map needs Pauli-string generators; got {bad[0]!r}. A "
                "ParamHamiltonian built from dense matrices has no Pauli labels."
            )
        if not 0 < n_visible < ham.n_qubits:
            raise ValueError(f"n_visible must be in 1..{ham.n_qubits - 1}, got {n_visible}")
        non_diag = [s for s in labels if set(s[:n_visible]) - set("IZ")]
        if non_diag:
            raise ValueError(
                f"the Gibbs map needs the visible register to be diagonal (I/Z only), but "
                f"generator {non_diag[0]!r} acts with X/Y on a visible qubit. That breaks the "
                "block structure G = (+)_v G_h(v). Use qbm.losses.MarginalNLL on the dense "
                "backend for such models."
            )
        if getattr(ham, "offset", None) is not None:
            raise NotImplementedError("ParamHamiltonian.offset is not supported by the Gibbs map")

        self.ham = ham
        self.n_visible = n_visible
        self.n_hidden = ham.n_qubits - n_visible
        self.dim_hidden = 1 << self.n_hidden
        # visible Z-pattern of each generator (sign s_j(v)) and its hidden Pauli matrix
        self._vis_z = np.array(
            [[1 if c == "Z" else 0 for c in s[:n_visible]] for s in labels], dtype=np.int64
        )
        self._hidden = np.stack([pauli(s[n_visible:]) for s in labels])  # (J, dh, dh)

    # -- the map itself ----------------------------------------------------
    def _signs(self, configs) -> np.ndarray:
        """``s_j(v) = (-1)^{# visible Z's of G_j that sit on a set bit of v}``, shape (B, J)."""
        configs = np.atleast_1d(np.asarray(configs, dtype=np.int64))
        bits = ((configs[:, None] >> np.arange(self.n_visible - 1, -1, -1)) & 1).astype(np.int64)
        return (-1.0) ** (bits @ self._vis_z.T)

    def conditional(self, theta, configs):
        """``(log Z_v, <G_j>_{sigma_v})`` for a batch of visible configurations.

        ``sigma_v = |v><v| (x) rho_v`` is the data-clamped state, so the second return is
        exactly the positive phase's per-configuration contribution.  Batched over a
        single stacked eigendecomposition on the hidden register.
        """
        theta = np.asarray(theta, dtype=float)
        signs = self._signs(configs)  # (B, J)
        # G_h(v) = sum_j theta_j s_j(v) P_j^hidden  -- only 2^n_hidden wide
        Gh = np.einsum("bj,jkl->bkl", signs * theta[None, :], self._hidden)
        w, V = np.linalg.eigh(Gh)  # batched
        shift = w.min(axis=1, keepdims=True)
        p = np.exp(-(w - shift))
        Zv = p.sum(axis=1)
        log_Zv = np.log(Zv) - shift[:, 0]
        rho = np.einsum("bkl,bl,bml->bkm", V, p / Zv[:, None], V.conj())
        # <P_j^hidden>_{rho_v}, then re-apply the visible sign
        exp_h = np.real(np.einsum("jlk,bkl->bj", self._hidden, rho))
        return log_Zv, signs * exp_h

    def conditional_hidden_state(self, theta, config) -> np.ndarray:
        """The conditional hidden Gibbs state ``rho_v`` for one visible configuration."""
        theta = np.asarray(theta, dtype=float)
        Gh = np.einsum("j,jkl->kl", self._signs(config)[0] * theta, self._hidden)
        w, V = np.linalg.eigh(Gh)
        p = np.exp(-(w - w.min()))
        return (V * (p / p.sum())) @ V.conj().T

    # -- marginal and likelihood ------------------------------------------
    def log_unnormalised_marginal(self, theta, configs) -> np.ndarray:
        """``log Z_v`` -- the exact visible free energy, up to the global ``log Z``."""
        return self.conditional(theta, configs)[0]

    def log_partition(self, theta) -> float:
        """``log Z`` by enumerating the visible register (needs ``2^n_visible`` blocks)."""
        dim = 1 << self.n_visible
        if dim > _ENUMERATION_CAP:
            raise NotImplementedError(
                f"log Z needs all {dim} visible configurations, above the enumeration cap "
                f"{_ENUMERATION_CAP}. The gradient does not need it -- only the loss value does."
            )
        log_Zv = self.log_unnormalised_marginal(theta, np.arange(dim))
        m = log_Zv.max()
        return float(m + np.log(np.exp(log_Zv - m).sum()))

    def marginal(self, theta) -> np.ndarray:
        """Exact visible marginal ``p(v)`` over all ``2^n_visible`` configurations."""
        dim = 1 << self.n_visible
        if dim > _ENUMERATION_CAP:
            raise NotImplementedError(
                f"enumerating {dim} visible configurations exceeds the cap {_ENUMERATION_CAP}; "
                "use sample() instead"
            )
        log_Zv = self.log_unnormalised_marginal(theta, np.arange(dim))
        p = np.exp(log_Zv - log_Zv.max())
        return p / p.sum()

    def log_likelihood(self, theta, configs) -> np.ndarray:
        """Exact ``log p(v)`` for the given visible configurations."""
        return self.log_unnormalised_marginal(theta, configs) - self.log_partition(theta)

    # -- the positive phase -------------------------------------------------
    def clamped_expectations(self, theta, data) -> np.ndarray:
        """Positive phase ``sum_v q(v) <G_j>_{sigma_v}``, averaged over the data.

        Only the *distinct* visible configurations present in ``data`` are visited, so the
        cost does not grow with ``2^n_visible``.
        """
        configs, weights = _as_visible_distribution(data, self.n_visible)
        _, exps = self.conditional(theta, configs)
        return weights @ exps

    # -- unbiased model sampling -------------------------------------------
    def sample(self, theta, n_samples: int, sweeps: int = 20, rng=None, burn_in=None):
        """Draw visible configurations from the **exact** model marginal ``p(v)``.

        A single-site heat-bath chain on the exact visible free energy ``-log Z_v``.
        Because the hidden register is traced out exactly rather than sampled, this is
        unbiased for non-commuting hidden operators -- unlike the block-Gibbs chain in
        :mod:`qbm.sampling`.  Returns integer basis-state indices.
        """
        rng = np.random.default_rng() if rng is None else rng
        theta = np.asarray(theta, dtype=float)
        nv = self.n_visible
        burn_in = sweeps // 2 if burn_in is None else burn_in
        bits = rng.integers(0, 2, size=(n_samples, nv), dtype=np.int64)
        weights = 1 << np.arange(nv - 1, -1, -1)

        for _ in range(sweeps + burn_in):
            for i in range(nv):
                trial = np.concatenate([bits, bits])  # branch i=0 then i=1, one batch
                trial[:n_samples, i] = 0
                trial[n_samples:, i] = 1
                log_Zv = self.log_unnormalised_marginal(theta, trial @ weights)
                f0, f1 = log_Zv[:n_samples], log_Zv[n_samples:]
                p1 = 1.0 / (1.0 + np.exp(np.clip(f0 - f1, -700, 700)))
                bits[:, i] = (rng.random(n_samples) < p1).astype(np.int64)
        return bits @ weights

    def __repr__(self) -> str:
        return (
            f"GibbsMap(n_visible={self.n_visible}, n_hidden={self.n_hidden}, "
            f"n_params={self.ham.n_params})"
        )


__all__ = ["GibbsMap"]
