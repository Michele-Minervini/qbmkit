"""Training a QBM end-to-end on variationally prepared thermal states.

Every optimisation step rebuilds rho(theta) with VarQITE -- from expectation values,
with no eigendecomposition anywhere in the loop -- and updates theta from a measured
gradient.  See notebooks/12 for the guided version with plots.

Run:  python examples/08_training_with_varqite.py
"""

import time

import numpy as np

import qbm
from qbm.backends.circuit import CircuitBackend
from qbm.losses import Energy


def varqite_backend(depth=3, steps=40, shots=None, seed=0):
    return CircuitBackend(
        seed=seed,
        shots=shots,
        gibbs_prep="varqite",
        varqite_options={"depth": depth, "steps": steps},
    )


# --- 1. generative modelling: VarQITE's home turf -----------------------------
q = np.array([0.50, 0.20, 0.05, 0.25])

t0 = time.time()
model = qbm.learn(q, steps=60, lr=0.15, backend=varqite_backend())
dt = time.time() - t0
ref = qbm.learn(q, steps=60, lr=0.15)  # exact-preparation reference

print("generative modelling of a 2-bit distribution:")
print(
    f"   VarQITE-trained   KL {model.history.monitor[0]:.4f} -> "
    f"{model.history.monitor[-1]:.2e}   ({dt / 60 * 1000:.0f} ms/step)"
)
print(f"   dense reference   KL {ref.history.monitor[0]:.4f} -> {ref.history.monitor[-1]:.2e}")
print(f"   learned p         {np.round(model.probabilities(), 4)}")
print(f"   target  q         {q}")

# The relative-entropy *value* needs log Z, which no device can measure; the *gradient*
# needs only <G_j>.  `fit` records nan for the value and keeps training.
print(f"\n   loss value (needs log Z): {model.history.loss[0]}")
print("   -> train on the gradient, monitor something measurable (here: the KL).")

# --- 2. with a finite shot budget --------------------------------------------
print("\nunder shot noise (plain gradient descent, not Adam):")
for shots in (None, 1000, 250):
    m = qbm.learn(
        q,
        steps=60,
        optimizer=qbm.optim.GradientDescent(lr=0.35),
        backend=varqite_backend(shots=shots, seed=1),
    )
    print(f"   shots={str(shots):>5s}   final KL = {m.history.monitor[-1]:.2e}")

# --- 3. ground state: the hard case, and the guard ---------------------------
# Minimising <H> drives the QBM to low temperature, where a shallow variational
# preparation degrades -- eventually corrupting the gradient that drives it.
H = qbm.hamiltonians.tfim(2, J=1.0, g=1.2)
E0 = np.linalg.eigvalsh(H)[0]


def ground_run(stop=None):
    m = qbm.FullyVisibleQBM(n=2, connectivity="all", backend=varqite_backend())
    m.theta = np.random.default_rng(0).normal(scale=0.05, size=m.n_params)
    return qbm.fit(m, Energy(H), qbm.optim.Adam(lr=0.2), steps=150, stop=stop)


loose = ground_run()
guarded = ground_run(stop=lambda state, mdl: state.varqite_result().residual > 0.05)

print(f"\nground-state energy (exact {E0:.5f}):")
print(
    f"   unguarded, 150 steps   <H> = {loose.loss[-1]:+.5f}   "
    f"error {abs(loose.loss[-1] - E0):.2e}   <-- destabilised"
)
print(
    f"   residual-guarded       <H> = {guarded.loss[-1]:+.5f}   "
    f"error {abs(guarded.loss[-1] - E0):.2e}   (stopped itself at step {len(guarded)})"
)
print("   the preparation error is measurable, so the guard works on hardware too.")
