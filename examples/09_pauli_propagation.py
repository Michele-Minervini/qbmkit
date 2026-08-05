"""Preparing and training a QBM with Pauli propagation.

The thermal state is stored as a sparse sum of Pauli strings and evolved under imaginary
time (arXiv:2602.04878); a QBM is trained for generative modelling with the locally
normalised sampler (*Sampling from Thermal Quantum States via Pauli Propagation*).
Everything runs in the Pauli basis, through the unchanged qbm.learn API.  See notebook 13.

Run:  python examples/09_pauli_propagation.py
"""

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
ham = qbm.ParamHamiltonian(qbm.local_pauli_generators(4, connectivity="all"))
theta = np.random.default_rng(1).normal(scale=0.5, size=ham.n_params)
dense = qbm.DenseBackend().thermal_state(ham, theta).density_matrix()
for cutoff in (1e-1, 1e-3, 1e-6):
    st = PauliPropagationBackend(trotter_steps=48, coeff_cutoff=cutoff).thermal_state(ham, theta)
    err = 0.5 * np.sum(np.abs(np.linalg.eigvalsh(st.density_matrix() - dense)))
    print(f"   cutoff={cutoff:.0e}:  {st.n_terms:3d} terms   trace distance = {err:.1e}")

# --- 3. the locally normalised sampler (valid samples from a truncated state) --
print("\nsampling with Algorithm 1 (exact likelihoods, sign-problem-safe):")
st = PauliPropagationBackend(trotter_steps=64, coeff_cutoff=1e-8, seed=0).thermal_state(ham, theta)
exact_diag = st.probabilities()
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
