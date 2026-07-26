"""Learn a classical distribution (bars-and-stripes) with a quantum Boltzmann machine.

Run:  python examples/01_generative_modeling.py
"""

import qbm

# Target distribution: 2x2 bars-and-stripes over 2^4 outcomes (snappy demo).
# Try grid=3 too (n=9) -- still exact, but ~1 minute on the dense backend.
data = qbm.datasets.bars_and_stripes(grid=2)

# Five lines to a working simulation: defaults pick a fully-visible QBM with
# all-to-all couplings, negative-log-likelihood loss, Adam, and the dense backend.
model = qbm.learn(data, steps=600, lr=0.1, verbose=True)

print(f"\nfinal KL(data || model) = {model.kl(data):.4f}")
print("10 samples drawn from the trained QBM:", model.sample(10))
