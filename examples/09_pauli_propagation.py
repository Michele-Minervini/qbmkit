"""Preparing and training a QBM with Pauli propagation.

The thermal state is stored as a sparse sum of Pauli strings and evolved under imaginary
time (arXiv:2602.04878); a QBM is trained for generative modelling with the locally
normalised sampler (*Sampling from Thermal Quantum States via Pauli Propagation*).
Everything runs in the Pauli basis, through the unchanged qbm.learn API.  See notebook 13.

The last part goes past the dense ceiling: truncation acts after every gate, so the cost
follows the strings that are kept, and a 16-qubit model trains (about two and a half minutes).

Run:  python examples/09_pauli_propagation.py
"""

import time

import numpy as np

import qbm
from qbm import pauli_prop as pp
from qbm.backends.pauli_propagation import PauliPropagationBackend

# --- 1. exact for commuting Hamiltonians, Trotter-convergent otherwise --------
print("preparing thermal states as sparse Pauli sums:")
rho = pp.thermal_state(["Z"], [0.8], trotter_steps=64, coeff_cutoff=0.0).to_matrix()
exact = (np.eye(2) - np.tanh(0.8) * qbm.pauli("Z")) / 2
print(f"   1 qubit (tanh law):        max|rho_pp - exact| = {np.max(np.abs(rho - exact)):.1e}")

labels, coeffs = ["ZZ", "XI", "IX"], [-1.0, -0.8, -0.8]  # non-commuting TFIM
H = sum(c * qbm.pauli(l) for c, l in zip(coeffs, labels))
w, V = np.linalg.eigh(H)
p = np.exp(-(w - w[0]))
rho_ex = (V * (p / p.sum())) @ V.conj().T
for L in (8, 64):
    rho = pp.thermal_state(labels, coeffs, trotter_steps=L, coeff_cutoff=0.0).to_matrix()
    td = 0.5 * np.sum(np.abs(np.linalg.eigvalsh(rho - rho_ex)))
    print(f"   2q TFIM, L={L:2d}:              trace distance to exact = {td:.2e}")

# --- 2. truncation trades retained terms for accuracy -------------------------
print("\ntruncation on a 4-qubit all-to-all Hamiltonian (full operator = 256 Paulis):")
ham = qbm.ParamHamiltonian(qbm.pauli_pool(4, terms=("Z", "X", "ZZ")))
theta = np.random.default_rng(1).normal(scale=0.5, size=ham.n_params)
dense = qbm.DenseBackend().thermal_state(ham, theta).density_matrix()
for cutoff in (1e-1, 1e-3, 1e-6):
    st = PauliPropagationBackend(trotter_steps=48, coeff_cutoff=cutoff).thermal_state(ham, theta)
    err = 0.5 * np.sum(np.abs(np.linalg.eigvalsh(st.density_matrix() - dense)))
    print(f"   cutoff={cutoff:.0e}:  {st.n_terms:3d} terms   trace distance = {err:.1e}")

# --- 3. the locally normalised sampler (valid samples from a truncated state) --
print("\nsampling with Algorithm 1 (exact likelihoods, sign-problem-safe):")
st = PauliPropagationBackend(trotter_steps=64, coeff_cutoff=1e-8, seed=0).thermal_state(ham, theta)
exact_diag = st.pauli_sum().probabilities()  # the diagonal of rho itself
emp = np.bincount(st.sample(50000), minlength=16) / 50000
print(
    f"   TVD(sampler, exact diagonal) = {0.5 * np.sum(np.abs(emp - exact_diag)):.3e}  (50k shots)"
)
print(f"   exact likelihood of |0000>   = {np.exp(st.log_likelihood([0]))[0]:.4f}  (computed)")

# --- 4. train a QBM for generative modelling, in the Pauli basis --------------
# realizable target: the distribution of a random 4-qubit QBM
target = qbm.FullyVisibleQBM(n=4, connectivity="all")
target.theta = np.random.default_rng(7).normal(scale=0.6, size=target.n_params)
q = target.probabilities()

model = qbm.learn(
    q,
    steps=120,
    lr=0.2,
    backend=PauliPropagationBackend(trotter_steps=32, coeff_cutoff=1e-8, seed=0),
)
ref = qbm.learn(q, steps=120, lr=0.2)  # exact dense engine, same settings
emp = np.bincount(model.state().sample(40000), minlength=16) / 40000

print("\ngenerative training (unchanged qbm.learn API, backend swapped):")
print(f"   Pauli propagation  KL -> {model.history.monitor[-1]:.2e}")
print(f"   dense reference    KL -> {ref.history.monitor[-1]:.2e}  (curves coincide)")
print("   loss value needs log Z (refused) -> nan; training rides the gradient")
print(f"   TVD(model samples, target) = {0.5 * np.sum(np.abs(emp - q)):.3e}")

# --- 5. past the dense ceiling ------------------------------------------------
# Truncation is applied after every gate, so the sum never holds much more than it
# returns ("peak" below) and the cost grows with the system rather than with 4^n.
print("\npast the dense ceiling (chain, coeff_cutoff=1e-3, max_weight=3):")
for n in (12, 16, 20):
    ham = qbm.ParamHamiltonian(qbm.pauli_pool(n, terms=("Z", "X", "ZZ"), connectivity="chain"))
    theta = np.random.default_rng(0).normal(scale=0.3, size=ham.n_params)
    t0 = time.perf_counter()
    st = PauliPropagationBackend(trotter_steps=8, coeff_cutoff=1e-3, max_weight=3).thermal_state(
        ham, theta
    )
    retained, peak = st.n_terms, st.peak_terms
    dt = time.perf_counter() - t0
    print(f"   n={n:2d}  {dt:5.2f}s  retained={retained:5d}  peak={peak:5d}")

# And it trains there.  Same task as examples/06: the Boltzmann distribution of a random
# 16-spin Ising chain, learned by a model on that chain.  The model contains the target,
# so the couplings it recovers measure what the truncation costs.
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
    backend=PauliPropagationBackend(trotter_steps=8, coeff_cutoff=1e-3, seed=0),
    connectivity="chain",
    monitor=lambda state, mdl: state.peak_terms,
)
dt = time.perf_counter() - t0
theta_z, theta_x, theta_zz = model.theta[:n], model.theta[n : 2 * n], model.theta[2 * n :]
history = model.history
print(f"\ntraining at n={n} ({model.n_params} parameters, {len(history)} steps, {dt:.0f}s):")
print(f"   moment mismatch |grad| : {history.grad_norm[0]:.3f} -> {history.grad_norm[-1]:.4f}")
print(f"   max |theta_Z  - h|     : {np.max(np.abs(theta_z - h)):.4f}")
print(f"   max |theta_ZZ - J|     : {np.max(np.abs(theta_zz - J)):.4f}")
print(f"   max |theta_X|          : {np.max(np.abs(theta_x)):.4f}")
print(f"   most strings ever held : {max(history.monitor):,}  (the full operator has 4^16)")
print(f"   dense generators built : {model.ham._mats is not None}")
print("   The cutoff sets that accuracy: 3e-4 brings the couplings to 0.015 at 2.6x the cost.")
print("   For a 1-D problem like this one the tensor-network backend is the better tool")
print("   (examples/06: about half the time, couplings to 1e-3).")
