"""Variational free-energy minimisation (= Gibbs-state preparation).

Minimising F = <H> - T S(rho) over the QBM parameters drives rho to the Gibbs state
of H at temperature T, and the minimum value is the exact free energy.

Run:  python examples/04_free_energy_minimization.py
"""

import qbm

H = qbm.hamiltonians.tfim(4, J=1.0, g=1.0)

for T in (2.0, 1.0, 0.5):
    res = qbm.free_energy_min(H, temperature=T, steps=400)
    print(
        f"T = {T:4.1f}   F = {res.free_energy:10.6f}   "
        f"exact = {res.exact_free_energy:10.6f}   error = {res.error:+.2e}"
    )
