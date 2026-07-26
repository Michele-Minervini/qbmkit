"""Semi-quantum restricted Boltzmann machine (sqRBM) with a closed-form fast path.

Visible units are diagonal (``Z`` only); hidden units carry non-commuting Pauli
fields; the bipartite structure couples ``Z_i`` to hidden Paulis (arXiv:2502.17562,
arXiv:2511.11802).  Because hidden units decouple given the visible configuration,
the visible marginal has a **closed form** -- no ``2^(n+m)`` diagonalisation:

    p~(v) = exp(-sum_i a_i (-1)^{v_i}) * prod_j cosh(||Phi_j(v)||),
    Phi_j^P(v) = b_j^P + sum_i w_{ij}^P (-1)^{v_i},

with ``||.||`` the Euclidean norm over the hidden Pauli components.  This costs
``O(2^n_visible * n_hidden)`` instead of exponential in the hidden count, so many
hidden units are cheap.  Gradients are closed-form ``tanh`` activations.

The model still exposes :meth:`to_hamiltonian`, so the exact dense Gibbs state can
be cross-checked against the closed form (the test suite asserts they agree).
"""

from __future__ import annotations

import numpy as np

from ..operators import ParamHamiltonian


class SqRBMState:
    """Lightweight closed-form sqRBM state: visible marginal + log-derivative table."""

    def __init__(self, pv: np.ndarray, dlog: np.ndarray):
        self.pv = pv  # (2^n_visible,) visible marginal p(v)
        self.dlog = dlog  # (J, 2^n_visible) d log p~(v) / d theta_k

    def visible_probabilities(self) -> np.ndarray:
        return self.pv


class SemiQuantumRBM:
    """A semi-quantum RBM trained through its closed-form visible marginal.

    Parameters
    ----------
    n_visible, n_hidden : int
    hidden_paulis : tuple of str
        Pauli fields on the hidden units (default ``("X", "Z")``; use ``("Z",)``
        for a classical RBM, ``("X", "Y", "Z")`` for the fully quantum case).
    theta : ndarray, optional
        Flat parameter vector; layout ``[a_i] ++ [b_{j,P}] ++ [w_{i,j,P}]``.
    """

    def __init__(self, n_visible, n_hidden, hidden_paulis=("X", "Z"), theta=None):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.hidden_paulis = tuple(hidden_paulis)
        self.nP = len(self.hidden_paulis)
        self._J = n_visible + n_hidden * self.nP + n_visible * n_hidden * self.nP
        dv = 1 << n_visible
        bits = (np.arange(dv)[:, None] >> (n_visible - 1 - np.arange(n_visible))[None, :]) & 1
        self.S = (1 - 2 * bits).astype(float)  # (dv, n_visible) of +-1, qubit 0 is MSB
        self.theta = np.zeros(self._J) if theta is None else np.asarray(theta, float).copy()

    @property
    def n_params(self) -> int:
        return self._J

    def _unpack(self, theta):
        nv, nh, nP = self.n_visible, self.n_hidden, self.nP
        a = theta[:nv]
        b = theta[nv : nv + nh * nP].reshape(nh, nP)
        w = theta[nv + nh * nP :].reshape(nv, nh, nP)
        return a, b, w

    def state(self) -> SqRBMState:
        a, b, w = self._unpack(self.theta)
        S = self.S  # (dv, nv)
        Phi = b[None, :, :] + np.einsum("vi,ijP->vjP", S, w)  # (dv, nh, nP)
        norm = np.sqrt(np.sum(Phi**2, axis=2))  # (dv, nh)

        log_pt = -(S @ a) + np.sum(np.log(np.cosh(norm)), axis=1)  # (dv,)
        log_pt -= log_pt.max()
        pt = np.exp(log_pt)
        pv = pt / pt.sum()

        # closed-form derivatives of log p~(v)
        ratio = np.where(norm > 1e-12, np.tanh(norm) / np.where(norm > 1e-12, norm, 1.0), 1.0)
        act = ratio[..., None] * Phi  # (dv, nh, nP); d log cosh / d Phi^P

        nv, nh, nP = self.n_visible, self.n_hidden, self.nP
        dv = pv.shape[0]
        dlog = np.empty((self._J, dv))
        dlog[:nv, :] = -S.T  # d/d a_i
        dlog[nv : nv + nh * nP, :] = act.transpose(1, 2, 0).reshape(nh * nP, dv)  # d/d b_{j,P}
        dw = np.einsum("vjP,vi->ijPv", act, S)  # d/d w_{i,j,P}
        dlog[nv + nh * nP :, :] = dw.reshape(nv * nh * nP, dv)
        return SqRBMState(pv, dlog)

    # -- read-outs ---------------------------------------------------------
    def visible_probabilities(self) -> np.ndarray:
        return self.state().pv

    def probabilities(self) -> np.ndarray:
        return self.visible_probabilities()

    def sample(self, n: int, rng=None) -> np.ndarray:
        rng = np.random.default_rng() if rng is None else rng
        pv = np.clip(self.visible_probabilities(), 0.0, None)
        pv = pv / pv.sum()
        return rng.choice(1 << self.n_visible, size=n, p=pv)

    def kl(self, q: np.ndarray) -> float:
        q = np.asarray(q, dtype=float)
        p = self.visible_probabilities()
        mask = q > 0
        return float(np.sum(q[mask] * (np.log(q[mask]) - np.log(np.clip(p[mask], 1e-300, None)))))

    # -- cross-validation against the exact dense Gibbs state --------------
    def to_hamiltonian(self) -> ParamHamiltonian:
        """Build the equivalent dense ``ParamHamiltonian`` (for verification / QFI).

        Generators are listed in the same order as the flat ``theta`` vector, so
        ``ham.matrix(model.theta)`` is exactly ``G(theta)``.  Only feasible for
        small ``n_visible + n_hidden``.
        """
        nv, nh = self.n_visible, self.n_hidden
        n = nv + nh

        def lab(spec):
            s = ["I"] * n
            for q, o in spec.items():
                s[q] = o
            return "".join(s)

        gens = [lab({i: "Z"}) for i in range(nv)]
        for j in range(nh):
            for P in self.hidden_paulis:
                gens.append(lab({nv + j: P}))
        for i in range(nv):
            for j in range(nh):
                for P in self.hidden_paulis:
                    gens.append(lab({i: "Z", nv + j: P}))
        return ParamHamiltonian(gens, n_qubits=n)

    def __repr__(self) -> str:
        return (
            f"SemiQuantumRBM(n_visible={self.n_visible}, n_hidden={self.n_hidden}, "
            f"hidden_paulis={self.hidden_paulis}, n_params={self._J})"
        )
