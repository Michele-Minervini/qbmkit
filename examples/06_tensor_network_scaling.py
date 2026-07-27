"""Scaling past the dense ceiling with the tensor-network backend.

The thermal state is represented as a purified matrix-product state, so cost grows
polynomially in the number of qubits (and exponentially only in the bond dimension).
First we check it against the exact dense backend on a small system, then we run sizes
a dense density matrix could never hold.

Run:  python examples/06_tensor_network_scaling.py     (needs `pip install qbmkit[tn]`)
"""

import time

import numpy as np

import qbm


def chain_generators(n):
    """1- and 2-body Pauli generators on an n-qubit chain."""
    return (
        [f"{'I' * i}Z{'I' * (n - i - 1)}" for i in range(n)]
        + [f"{'I' * i}X{'I' * (n - i - 1)}" for i in range(n)]
        + [f"{'I' * i}ZZ{'I' * (n - i - 2)}" for i in range(n - 1)]
    )


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
