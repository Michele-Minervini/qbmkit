"""Estimate the ground-state energy of a TFIM with a QBM + quantum natural gradient.

Run:  python examples/02_ground_state_energy.py
"""

import qbm

n = 6
H = qbm.hamiltonians.tfim(n, J=1.0, g=1.5)

model = qbm.FullyVisibleQBM(n=n)
history = qbm.fit(
    model,
    loss=qbm.losses.Energy(H),
    optimizer=qbm.optim.NaturalGradient(metric="kubo_mori", lr=0.1, reg=1e-3),
    steps=400,
    verbose=True,
)

print(f"\nestimated energy = {model.energy(H):.6f}")
print(f"exact ground energy = {qbm.oracles.ground_energy(H):.6f}")
