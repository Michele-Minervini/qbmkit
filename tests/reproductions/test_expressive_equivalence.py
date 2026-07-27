"""Reproduction -- arXiv:2502.17562, "Expressive equivalence of classical and quantum RBMs".

Theorem 2 shows a semi-quantum RBM with m hidden units carrying non-commuting Pauli
fields (W_h = {X, Z}) has the representational capacity of a *classical* RBM with 3m
hidden units.  We reproduce the empirical consequence on the bars-and-stripes
benchmark:

  * per hidden unit, quantum hidden units reach a far lower KL than classical ones;
  * two quantum hidden units match ~six classical ones, while two classical hidden
    units cannot fit the target at all (the ~3x capacity gap).
"""

import numpy as np

import qbm
from qbm.losses import SqRBMNLL

_Q = qbm.datasets.bars_and_stripes(grid=2)  # 4 visible qubits


def _fit_kl(hidden_paulis, m, steps=500, seed=0):
    model = qbm.SemiQuantumRBM(n_visible=4, n_hidden=m, hidden_paulis=hidden_paulis)
    model.theta = np.random.default_rng(seed).normal(scale=0.1, size=model.n_params)
    qbm.fit(model, SqRBMNLL(_Q), qbm.optim.Adam(lr=0.1), steps=steps)
    return model.kl(_Q)


def test_quantum_hidden_units_beat_classical_per_unit():
    # calibrated: m=1 -> 0.556 < 0.811 ; m=2 -> 0.0014 < 0.463
    assert _fit_kl(("X", "Z"), 1) < _fit_kl(("Z",), 1) - 0.1
    assert _fit_kl(("X", "Z"), 2) < _fit_kl(("Z",), 2) - 0.1


def test_two_quantum_hidden_match_six_classical():
    # ~3x capacity: 2 quantum hidden units learn the target as well as 6 classical
    # ones, while 2 classical hidden units cannot.
    kq2 = _fit_kl(("X", "Z"), 2)
    kc6 = _fit_kl(("Z",), 6)
    kc2 = _fit_kl(("Z",), 2)
    assert kq2 < 0.05  # calibrated ~0.0014
    assert kc6 < 0.05  # calibrated ~0.0003
    assert kc2 > 0.3  # calibrated ~0.463  (2 classical cannot fit BAS)
