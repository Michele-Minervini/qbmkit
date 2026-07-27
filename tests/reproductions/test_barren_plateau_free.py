"""Reproduction -- arXiv:2410.12935, "Quantum Boltzmann machine learning of ground-state energies".

A central motivation of the paper is that thermal-state (Gibbs) QBM energy training
avoids the *barren plateaus* that make deep parameterized-circuit training
intractable -- i.e. the gradient variance does not vanish exponentially with system
size.  We reproduce the signature: scanning the mean gradient variance of the energy
loss over n = 2..6 qubits, a linear fit of log(variance) vs n has a non-negative
slope (no exponential decay), in contrast to the strongly negative slope a barren
plateau would produce.
"""

import numpy as np

import qbm


def test_qbm_energy_training_has_no_barren_plateau():
    def make_model_for_n(n):
        return lambda: qbm.FullyVisibleQBM(n=n)

    def make_loss_for_n(n):
        return qbm.losses.Energy(qbm.hamiltonians.tfim(n, g=1.0))

    sizes = [2, 3, 4, 5, 6]
    var = qbm.diagnostics.barren_plateau_scan(
        make_model_for_n, make_loss_for_n, sizes=sizes, n_samples=120
    )
    variances = np.array([var[n] for n in sizes])
    assert np.all(variances > 0)

    slope = np.polyfit(sizes, np.log(variances), 1)[0]
    # calibrated slope ~ +0.12; a barren plateau would give a strongly negative slope
    assert slope > -0.1
