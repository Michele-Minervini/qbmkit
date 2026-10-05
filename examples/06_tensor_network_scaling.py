"""Scaling past the dense ceiling with the tensor-network backend.

The thermal state is represented as a purified matrix-product state, so cost grows
polynomially in the number of qubits (and exponentially only in the bond dimension).
First we check it against the exact dense backend on a small system, then we run sizes
a dense density matrix could never hold, and finally we *train* at such a size: the data
enter the loss through their moments, read from the Pauli labels, so nothing of size
4^n is built anywhere on the way.

Run:  python examples/06_tensor_network_scaling.py     (needs `pip install qbmkit[tn]`)
"""

import time

import numpy as np

import qbm


def chain_generators(n):
    """1- and 2-body Pauli generators on an n-qubit chain."""
    return qbm.pauli_pool(n, terms=("Z", "X", "ZZ"), connectivity="chain")


# --- 1. agreement with the exact dense backend -------------------------------
n = 4
gens = chain_generators(n)
theta = np.random.default_rng(0).normal(scale=0.4, size=len(gens))
ham = qbm.ParamHamiltonian(gens)

dense = qbm.DenseBackend().thermal_state(ham, theta)
tn = qbm.get_backend("tn").thermal_state(ham, theta)
err = np.max(np.abs(dense.generator_expectations() - tn.generator_expectations()))
print(f"n={n}: max |<G_j>_TN - <G_j>_dense| = {err:.2e}  (bond dimension {tn.psi.max_bond()})")

# --- 2. sizes the dense backend cannot reach ---------------------------------
print("\nscaling (dense would need a 2^n x 2^n density matrix):")
for n in (8, 12, 16, 20):
    gens = chain_generators(n)
    ham = qbm.ParamHamiltonian(gens)  # generators stay lazy -- never materialised
    theta = np.random.default_rng(1).normal(scale=0.3, size=len(gens))
    t0 = time.perf_counter()
    tn = qbm.get_backend("tn", trotter_steps=40).thermal_state(ham, theta)
    ge = tn.generator_expectations()
    dt = time.perf_counter() - t0
    dense_gb = (2**n) ** 2 * 16 / 1e9
    print(
        f"  n={n:3d}  {dt:5.2f}s  bond={tn.psi.max_bond():3d}  "
        f"<Z_0>={ge[0]:+.5f}   (dense rho would be {dense_gb:,.1f} GB)"
    )

# --- 3. training at a size the dense backend cannot reach --------------------
# Data: the Boltzmann distribution of a random Ising chain on 16 spins.  The model
# (fields Z and X, couplings ZZ, on the same chain) contains it, so training should
# recover the couplings and switch the transverse fields off.
n = 16
rng = np.random.default_rng(0)
h, J = rng.normal(scale=0.5, size=n), rng.normal(scale=0.5, size=n - 1)
v = np.arange(2**n)
spins = 1 - 2 * ((v[:, None] >> np.arange(n - 1, -1, -1)) & 1)  # qubit 0 = leftmost bit
energy = spins @ h + (spins[:, :-1] * spins[:, 1:]) @ J
data = np.exp(-(energy - energy.min()))
data /= data.sum()

t0 = time.perf_counter()
model = qbm.learn(
    data,
    steps=120,
    backend=qbm.get_backend("tn", max_bond=8, trotter_steps=20),
    connectivity="chain",
)
dt = time.perf_counter() - t0
theta_z, theta_x, theta_zz = model.theta[:n], model.theta[n : 2 * n], model.theta[2 * n :]
history = model.history
print(f"\ntraining at n={n} ({model.n_params} parameters, {len(history)} steps, {dt:.0f}s):")
print(f"  moment mismatch |grad| : {history.grad_norm[0]:.3f} -> {history.grad_norm[-1]:.4f}")
print(f"  max |theta_Z  - h|     : {np.max(np.abs(theta_z - h)):.4f}")
print(f"  max |theta_ZZ - J|     : {np.max(np.abs(theta_zz - J)):.4f}")
print(f"  max |theta_X|          : {np.max(np.abs(theta_x)):.4f}")
print(f"  dense generators built : {model.ham._mats is not None}")
print(f"  (one of them would be {16 * 4**n / 2**30:.0f} GiB)")
