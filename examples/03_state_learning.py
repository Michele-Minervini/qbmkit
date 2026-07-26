"""Learn a target quantum state (a Gibbs state) by relative-entropy minimisation.

Run:  python examples/03_state_learning.py
"""

import numpy as np

import qbm

n = 4

# Build a realizable target: a Gibbs state in the model's own generator span.
model_true = qbm.FullyVisibleQBM(n=n)
model_true.theta = np.random.default_rng(0).normal(scale=0.4, size=model_true.n_params)
sigma = model_true.density_matrix()

# Train a fresh model to match it.
model = qbm.FullyVisibleQBM(n=n)
history = qbm.fit(
    model,
    loss=qbm.losses.RelativeEntropy(sigma),
    optimizer=qbm.optim.NaturalGradient(metric="kubo_mori", lr=0.3, reg=1e-6),
    steps=300,
    verbose=True,
)

print(f"\nfinal relative entropy D(sigma||rho) = {history.final_loss:.3e}")
