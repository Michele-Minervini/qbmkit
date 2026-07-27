"""Sample-based training: block-Gibbs sampling and contrastive divergence.

Implements the sample-based estimator of arXiv:2511.11802 for semi-quantum restricted
Boltzmann machines.  The bipartite structure makes the two conditionals tractable:

* **hidden given visible.**  Given a visible configuration ``v`` the hidden units
  decouple, and hidden unit ``j`` sits in the single-qubit state
  ``∝ exp(sum_P Phi_j^P(v) sigma^P)``, so measuring Pauli ``P`` gives
  ``<sigma^P> = tanh(||Phi_j||) Phi_j^P / ||Phi_j||`` -- the same closed-form
  activation the exact sqRBM gradient uses.
* **visible given hidden.**  With the hidden units measured in a fixed Pauli basis
  ``P`` with outcomes ``h``, the visible units are diagonal and independent, each
  feeling a field ``f_i = a_i + sum_j w_{ij}^P h_j`` with ``<s_i> = -tanh(f_i)``.

Alternating the two gives a **block-Gibbs chain**; contrastive divergence replaces the
exact model distribution in the gradient by the empirical distribution of a short
chain started from the data.  The cost is then independent of the size of the target's
support, whereas the exact likelihood gradient must sum over it.

Validity
--------
The ``hidden | visible`` conditional above is exact for any sqRBM.  The
``visible | hidden`` conditional treats the measured Pauli basis as if it were the
only hidden term, which is exact when the hidden operators **commute** -- i.e. a
classical RBM (``hidden_paulis=("Z",)``).  There the chain equilibrates to the model
distribution and CD-k converges to the exact gradient as ``k`` and the number of
chains grow (verified in the test suite).

With **non-commuting** hidden Paulis the chain is an *approximation*: measured
total-variation distance to the exact model marginal is ~0.07 for a typical
``("X", "Z")`` sqRBM, and the CD gradient stays positively aligned with the exact one
(a usable descent direction) without converging to it.  Removing that bias needs the
full Gibbs-map treatment (imaginary-time evolution) of arXiv:2511.11802, which is not
implemented here.  For exact non-commuting gradients use
:class:`~qbm.losses.SqRBMNLL` (closed form) or the dense backend.

Choosing an optimizer
---------------------
These gradients are *stochastic*, so prefer plain
:class:`~qbm.optim.GradientDescent`: its step is proportional to the gradient, and the
sampling noise averages out.  Adam rescales by the gradient magnitude, which amplifies
pure noise to a full-size step and makes training a random walk unless the chain count
is large or the learning rate is small.  Measured on 2x2 bars-and-stripes (300 steps,
500 chains, k=5): SGD at ``lr=0.2`` takes KL from 0.97 to 0.05, whereas Adam at
``lr=0.1`` *increases* it.
"""

from __future__ import annotations

import numpy as np


def _bits_to_index(bits: np.ndarray) -> np.ndarray:
    """(n_chains, n_visible) 0/1 array -> computational-basis indices (qubit 0 is MSB)."""
    n = bits.shape[1]
    weights = 1 << np.arange(n - 1, -1, -1)
    return bits @ weights


def block_gibbs_sample(model, v_bits: np.ndarray, k: int = 1, rng=None) -> np.ndarray:
    """Run ``k`` alternating block-Gibbs sweeps of a :class:`~qbm.SemiQuantumRBM`.

    Parameters
    ----------
    model : SemiQuantumRBM
    v_bits : ndarray, shape (n_chains, n_visible)
        Starting visible configurations as 0/1 bits.
    k : int
        Number of alternating (hidden, visible) sweeps.

    Returns
    -------
    ndarray, shape (n_chains, n_visible)
        The visible configurations after ``k`` sweeps.
    """
    rng = np.random.default_rng() if rng is None else rng
    a, b, w = model._unpack(model.theta)
    nP = model.nP
    bits = np.asarray(v_bits, dtype=int).copy()

    for _ in range(k):
        s = 1.0 - 2.0 * bits  # +-1 spins, (n_chains, n_visible)

        # --- hidden given visible: pick a Pauli basis, sample each hidden unit ---
        # rho ∝ exp(-H) and H contains +Phi.sigma, so <sigma^P> = -tanh(|Phi|) Phi^P/|Phi|
        p_idx = rng.integers(nP)
        Phi = b[None, :, :] + np.einsum("ci,ijP->cjP", s, w)  # (chains, nh, nP)
        norm = np.sqrt(np.sum(Phi**2, axis=2))
        safe = np.where(norm > 1e-12, norm, 1.0)
        ratio = np.where(norm > 1e-12, np.tanh(safe) / safe, 1.0)
        expect = -ratio * Phi[:, :, p_idx]  # <sigma^P>_j in the chosen basis
        h = np.where(rng.random(expect.shape) < (1.0 + expect) / 2.0, 1.0, -1.0)

        # --- visible given hidden (diagonal -> independent Bernoulli) ---
        field = a[None, :] + h @ w[:, :, p_idx].T  # f_i = a_i + sum_j w_ij^P h_j
        expect_v = -np.tanh(field)  # <s_i> = -tanh(f_i)
        s_new = np.where(rng.random(field.shape) < (1.0 + expect_v) / 2.0, 1.0, -1.0)
        bits = ((1.0 - s_new) / 2.0).astype(int)

    return bits


def contrastive_divergence_gradient(model, q, k: int = 1, n_chains: int = 200, rng=None):
    """CD-k estimate of the negative-log-likelihood gradient of an sqRBM.

    The exact gradient is ``dlog @ (p_model - q)``; CD-k replaces ``p_model`` by the
    empirical distribution of a ``k``-step block-Gibbs chain started from the data.
    It is therefore consistent: as ``k`` and ``n_chains`` grow it converges to the
    exact gradient.
    """
    rng = np.random.default_rng() if rng is None else rng
    q = np.asarray(q, dtype=float)
    n_visible = model.n_visible

    start = rng.choice(len(q), size=n_chains, p=q / q.sum())
    v0 = ((start[:, None] >> np.arange(n_visible - 1, -1, -1)) & 1).astype(int)

    vk = block_gibbs_sample(model, v0, k=k, rng=rng)
    idx = _bits_to_index(vk)
    p_emp = np.bincount(idx, minlength=len(q)).astype(float) / len(idx)

    return model.state().dlog @ (p_emp - q)
