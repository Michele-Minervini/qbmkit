"""Preparing the Gibbs state variationally (VarQITE) instead of by diagonalisation.

The dense backend -- and even the "exact" circuit preparation -- build rho(theta) from
`eigh(G)`.  VarQITE builds it from expectation values only, by transporting a
parameterised circuit along the imaginary-time flow, and emits ordinary gates.

Run:  python examples/07_varqite_gibbs_preparation.py
"""

import numpy as np

import qbm
from qbm.backends.circuit import CircuitBackend
from qbm.circuits import varqite as vq
from qbm.circuits.adapters import to_qasm3

# --- 1. a commuting Hamiltonian: the tilt-partner ansatz is exact -------------
labels, coeffs = ["ZI", "IZ", "ZZ"], [0.7, -0.4, 0.9]
H_cl = sum(c * qbm.pauli(lbl) for c, lbl in zip(coeffs, labels))
res = vq.varqite(H_cl, vq.tfd_ansatz(labels=labels, depth=1), tau=0.5, steps=400)
print("commuting Ising, depth 1 (6 rotations):")
print(f"   McLachlan residual = {res.residual:.1e}   <- 0 means the ansatz is exact")
print(f"   trace distance to exp(-H)/Z = {res.trace_distance():.2e}")

# --- 2. a non-commuting Hamiltonian: systematically improvable ----------------
labels2 = ["ZZ", "XI", "IX"]
H = qbm.hamiltonians.tfim(2, J=1.0, g=0.8)
print("\n2-qubit TFIM -- error falls with ansatz depth:")
for depth in (1, 2, 3, 4):
    r = vq.varqite(H, vq.tfd_ansatz(labels=labels2, depth=depth), tau=0.5, steps=120)
    print(
        f"   depth {depth}: {r.ansatz.n_params:2d} rotations   "
        f"residual = {r.residual:.2e}   trace distance = {r.trace_distance():.2e}"
    )

# --- 3. the diagnostic that works on hardware --------------------------------
# A unitary on the system alone maps I/2^n to itself, so an ansatz without tilt
# partners cannot leave infinite temperature -- and says so, from measurements.
stuck = vq.PauliRotationAnsatz([lbl + "II" for lbl in labels2], n_system=2)
bad = vq.varqite(H, stuck, tau=0.5, steps=50)
print(f"\nansatz with no tilt partners: residual = {bad.residual:.3f} (cannot move at all)")

# --- 4. A and C are measurable: Hadamard tests vs the exact route -------------
small = vq.tfd_ansatz(labels=labels2, depth=1)
lam = np.random.default_rng(5).normal(scale=0.5, size=small.n_params)
exact = vq.mclachlan_system(small, lam, H)
A, C = vq.measured_mclachlan_system(small, lam, labels2, [-1.0, -0.8, -0.8])
print("\nMcLachlan system from Hadamard tests only:")
print(f"   max |A_measured - A_exact| = {np.max(np.abs(A - exact['A'])):.1e}")
print(f"   max |C_measured - C_exact| = {np.max(np.abs(C - exact['C'])):.1e}")

# --- 5. drop it into a QBM: same API, now device-ready -----------------------
ham = qbm.ParamHamiltonian(qbm.local_pauli_generators(2))
theta = np.random.default_rng(0).normal(scale=0.4, size=ham.n_params)
dense = qbm.DenseBackend().thermal_state(ham, theta)
O = qbm.hamiltonians.tfim(2, g=1.2)

state = CircuitBackend(seed=0, gibbs_prep="varqite", varqite_options={"depth": 4}).thermal_state(
    ham, theta
)
prep = state.preparation_circuit()
print("\nQBM on a variationally prepared thermal state:")
print(f"   <O> error vs dense        = {abs(state.expect(O) - dense.expect(O)):.2e}")
print(f"   preparation circuit       = {len(prep.gates)} gates, depth {prep.depth}")
print(f"   gate counts               = {prep.gate_counts()}")
print(f"   OpenQASM 3 export         = {len(to_qasm3(prep).splitlines())} lines")
print(f"   circuits a device needs   = {state.resource_estimate()['circuits_for_preparation']:,}")
