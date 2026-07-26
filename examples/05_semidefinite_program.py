"""Solving a semidefinite program with the QBM core.

    maximise   <C, X> + (1/beta) S(X)
    subject to <A_i, X> = b_i,  X >= 0,  Tr X = 1

The dual is convex and its stationary point is a QBM thermal state, so the SDP is
just another loss over the same machinery.  Because the Kubo-Mori metric is the
Hessian of log Z, quantum natural gradient with lr = beta is exactly Newton's method
on the dual.

Run:  python examples/05_semidefinite_program.py
"""

import numpy as np

import qbm

rng = np.random.default_rng(0)


def hermitian(dim):
    Z = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    return (Z + Z.conj().T) / 2


# --- 1. unconstrained: the solution is exactly exp(beta C)/Z, and as beta grows
#        the objective approaches the largest eigenvalue of C -------------------
C = np.diag([3.0, 1.0, 0.0, -1.0]).astype(complex)
print("unconstrained SDP, <C,X> vs lambda_max(C) = 3.0")
for beta in (1.0, 5.0, 20.0, 100.0):
    res = qbm.solve_sdp(C, beta=beta, steps=5)
    print(f"   beta = {beta:6.1f}   <C,X> = {res.objective:.6f}   S(X) = {res.entropy:.4f}")

# --- 2. with linear constraints ------------------------------------------------
dim = 4
C = hermitian(dim)
A = [hermitian(dim), hermitian(dim)]
X_feasible = qbm.oracles.gibbs(hermitian(dim), beta=1.0)  # any density matrix
b = np.array([float(np.real(np.trace(a @ X_feasible))) for a in A])

res = qbm.solve_sdp(C, A, b, beta=4.0, steps=300)
print("\nconstrained SDP")
print(f"   objective <C,X>       = {res.objective:.6f}")
print(f"   entropy   S(X)        = {res.entropy:.6f}")
print(f"   dual value            = {res.dual_value:.6f}")
print(f"   primal <C,X> + S/beta = {res.objective + res.entropy / res.beta:.6f}  (strong duality)")
print(f"   max constraint error  = {res.constraint_violation:.2e}")
print(f"   multipliers y         = {np.round(res.y, 5)}")
