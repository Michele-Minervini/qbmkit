# qbmkit — design document

> The unifying, open-source library for **Quantum Boltzmann Machines (QBMs)**.
> Install name: `qbmkit`. Import name: `qbm`.

This document is the authoritative specification: the abstractions, the math
conventions, the public API, and the conventions every contributor must follow.
Code is written *against this document*, and the test suite enforces it.

---

## 1. Philosophy: one object, one primitive

Three independent architecture studies and a survey of the QBM literature all
collapse to the same two facts.

**One object — the parameterized Gibbs state.** Every QBM model in the
literature is a parameterized thermal state

```
rho(theta) = exp(-G(theta)) / Z(theta),   G(theta) = sum_j theta_j G_j,   Z = Tr exp(-G(theta))
```

optionally **wrapped by real-time evolution** `omega = e^{-iH(phi)} rho e^{+iH(phi)}`
(the *Evolved QBM*), and optionally **observed through a visible/hidden partial
trace** `rho_v = Tr_h rho`. A model is therefore nothing but:

- a list of Hermitian **generators** `G_j` (Pauli strings),
- which coefficients `theta_j` are free,
- (optionally) real-time generators `H_k` and parameters `phi_k`,
- (optionally) a visible/hidden partition.

Classical BM, fully-visible QBM, restricted/semi-quantum RBM, deep/semi-restricted,
and the Evolved QBM are all instances of this one class.

**One hard primitive — the belief-propagation channel.** Every gradient and
*every* quantum-Fisher-information metric in the papers is a thin algebraic
recombination of the **high-peak-tent / belief-propagation channel**

```
Phi_theta(X) = integral dt p(t) e^{-iG t} X e^{+iG t},   p(t) = (2/pi) ln|coth(pi t / 2)|
```

and its real-time companion `Psi_phi`. On the dense backend we never sample `t`:
in the eigenbasis of `G` the channel is **exact and diagonal in superoperator
form** (Section 4).

The design consequence: implement the Gibbs state and `Phi_theta` once, and
generative modeling, ground-state energy, state learning, SDP, free-energy
minimization, quantum natural gradient, and barren-plateau diagnostics all
follow as combinations of the same primitives.

---

## 2. Layered architecture

```
facade            qbm.learn(...) / qbm.fit(...)          <- 5-line beginner surface
tasks             generative / ground_state / state_learning / sdp / barren_plateau
primitives        models | losses | metrics (QFI) | optimizers
core              ParamHamiltonian, ThermalState, Phi_theta   <- the unifying substrate
backend seam      Backend Protocol  (dense | tensor-network | circuit/hardware)
```

Rules:

- **Only the backend layer touches concrete linear algebra / hardware.** Losses,
  metrics, optimizers, models, and tasks are written against the `Backend`
  Protocol and the `ThermalState` handle — never against NumPy directly (one
  pragmatic exception: parameter vectors are plain NumPy arrays).
- **Everything pluggable is registered.** A new model/loss/metric/optimizer/
  backend is a small class registered by decorator; the core is never edited.
- **The facade and the explicit API are the same objects underneath.** The
  beginner one-liner just wires sensible defaults.

---

## 3. Conventions

- **Units / temperature.** Inverse temperature `beta` is absorbed into `theta`
  (equivalently `beta = 1`). A scalar `beta` may multiply `G` where convenient
  (e.g. sample-based backends), but the analytic default folds it in.
- **Generators** `G_j` are Hermitian; in v1 they are dense matrices built from
  Pauli strings. `G(theta) = sum_j theta_j G_j`.
- **Model vs target.** `G(theta)` defines the *state*. The *target* of a task is
  separate: an observable Hamiltonian `H` (energy), a classical distribution `q`
  (generative), or a density matrix `sigma` (state learning). They are distinct
  objects and never conflated.
- **Hilbert-space order.** Qubit 0 is the leftmost tensor factor: `pauli("XIZ")`
  is `X ⊗ I ⊗ Z`. Computational-basis index `v` has qubit 0 as the most
  significant bit.
- **Gradients are real vectors** indexed by the free parameters, same length and
  order as `theta`.

---

## 4. The core math (the contract the code implements)

Let `G(theta) = V diag(w) V^dag` be the eigendecomposition, `w0 = min(w)`, and

```
p_k = exp(-(w_k - w0)) / Ztilde,   Ztilde = sum_k exp(-(w_k - w0)),   log Z = -w0 + log Ztilde
rho = V diag(p) V^dag
<O> = Tr(rho O) = sum_k p_k (V^dag O V)_{kk}
```

**Why eigendecomposition, not `expm(-G)/Z`.** `expm(-G)` overflows whenever `G`
has a sufficiently negative eigenvalue; the shift `exp(-(w - w0))` is
overflow-immune. One `eigh(G)` also yields `ln rho`, `sqrt(rho)`, `log Z`, the
channel `Phi`, and all three metrics — `expm` gives only `rho`. This supersedes
the dense `expm` pattern used in earlier QBM code.

**Belief-propagation channel (eigenbasis, exact).** With `Delta = w_k - w_l`,

```
Phi(X)_{kl} = X_{kl} * phi(Delta),   phi(Delta) = (2/Delta) tanh(Delta/2),   phi(0) = 1
```

`phi` is the Fourier transform of the high-peak-tent density `p(t)` — so the dense
backend evaluates `Phi` exactly with no time sampling. (Circuit/TN backends
realize the *same* `Phi` via Monte-Carlo sampling of `t ~ p(t)`.)

**State derivative (the shared kernel).** In the eigenbasis,

```
(d_j rho)_{kl} = (G_j)_{kl} * K_{kl} + delta_{kl} * p_k * <G_j>
K_{kl} = (p_k - p_l) / (w_k - w_l)   for k != l,    K_{kk} = -p_k
```

This `d_j rho` is the single object every gradient and every metric is built
from. (Check: `Tr(d_j rho) = 0`.)

**Gradients.**

- *Energy / observable* `L = Tr(O rho)`:  `d_j L = Tr(O * d_j rho)`.
- *Relative entropy* `L = D(sigma || rho) = Tr(sigma ln sigma) - Tr(sigma ln rho)`:
  `d_j L = <G_j>_sigma - <G_j>_rho` (data minus model — the classic
  positive/negative phase). Exact for the fully-visible case even when the `G_j`
  do not commute, because `sigma` is fixed. Negative log-likelihood of a
  classical distribution `q` is the special case `sigma = diag(q)`.
- *Free energy* `F = <H> - T S`: at the exact Gibbs state, `d_j F = <H_j>` by the
  Gibbs variational identity.
- *Hidden-unit* models replace `<G_j>_sigma` by the modular-flow-lifted
  expectation `<G_j>_{Sigma_{v->vh}(sigma)}` (paper 2512.19819); planned for the
  hidden-unit milestone.

**Quantum Fisher / information metrics (one formula, three kernels).** Every
monotone metric is

```
g^c_{ij} = sum_{kl} W^c_{kl} * Re( (d_i rho)_{kl} * conj((d_j rho)_{kl}) )
```

with the Morozova–Chentsov weight `W^c_{kl} = 1 / c(p_k, p_l)`:

| metric            | mean `c(x,y)`                 | weight `W = 1/c`              |
|-------------------|-------------------------------|-------------------------------|
| Fisher–Bures (SLD)| arithmetic `(x+y)/2`          | `2 / (x+y)`                   |
| Kubo–Mori (BKM)   | logarithmic `(x-y)/(ln x-ln y)`| `(ln x - ln y)/(x - y)`      |
| Wigner–Yanase     | `((sqrt x + sqrt y)/2)^2`     | `4 / (sqrt x + sqrt y)^2`     |

**Arbitrary metrics.** Because the metric enters *only* through `W`, the library is not
limited to those three. The **alpha-z information matrices** of arXiv:2510.02218
(Wilde), derived from the alpha-z Renyi relative entropies, are the kernel (Thm 10)

```
zeta(x,y) = z/(a(1-a)) * (x^((1-a)/z) - y^((1-a)/z))/(x-y)
                       * (x^(a/z) - y^(a/z))/(x^(1/z) - y^(1/z)),   zeta(x,x) = 1/x
```

and the three metrics above are special cases: Kubo-Mori is `a -> 1` (any `z`),
Fisher-Bures is `(a,z) = (1/2, 1/2)`, Wigner-Yanase is `(1/2, 1)`. `z = 1` gives the
Petz-Renyi family and `z = a` the sandwiched-Renyi family. `CustomMetric` takes any
user kernel. `alpha_z_is_monotone` implements the data-processing region (Fact 9 of
the paper); outside it the matrix is still defined but is not a monotone metric, and
`AlphaZ` warns.

These satisfy the Loewner orderings `g_FB <= g_WY <= 2 g_FB` and `g_KM >= g_FB`,
which the test suite asserts. **Kubo–Mori is the default QBM metric** (it is the
Hessian of the free energy, giving Newton-like natural-gradient steps for Gibbs
states); SLD/Fisher–Bures is the right choice for pure-state / circuit ansätze.

**Optimizers.**

- Gradient descent / Adam: `theta <- theta - eta * grad`.
- Quantum natural gradient: `theta <- theta - eta * (g + lambda I)^{-1} grad`,
  with ridge `lambda` (paper prefactors such as 4·eta for FB and 2·eta for KM are
  absorbed into `eta`).
- Newton (free-energy Hessian = Kubo–Mori) is a natural-gradient special case.

---

## 5. Backend Protocol (the swap seam)

A backend turns a `ParamHamiltonian` + parameters into a `ThermalState` handle
and answers a small set of questions in the *language of QBM math*, so dense,
tensor-network, and circuit backends can all satisfy it:

```python
class Backend(Protocol):
    def thermal_state(self, ham: ParamHamiltonian, theta: np.ndarray) -> ThermalState: ...

class ThermalState(Protocol):
    def expect(self, op) -> float: ...                 # <O>
    def grad_rho(self, generators) -> list: ...        # [d_j rho]  (eigenbasis-aware)
    def belief_prop(self, op): ...                     # Phi(op)
    def log_partition(self) -> float: ...              # log Z
    def density_matrix(self): ...                      # rho (dense form, for small n)
    def sample(self, n: int) -> np.ndarray: ...        # computational-basis bitstrings
    # plus eigendata (w, V, p) for analytic losses/metrics
```

v1 ships `DenseBackend` (NumPy/SciPy `eigh`). `TensorNetworkBackend` (quimb) and
`CircuitBackend` (PennyLane/Qiskit; statevector / density-matrix / hardware) are
added later as new implementations of the same Protocol — **zero changes to user
code or to the layers above**. A `verify_backends_agree` test harness asserts
gradients/metrics match across backends on small systems.

---

## 6. Public API

### Task layer — one call per research question

Every task is a thin recipe over the same core and returns a `Result`
(`.model`, `.history`, `.report()`, plus task-specific fields):

```python
qbm.learn(data)                  # generative modelling      -> .kl, .history
qbm.ground_state(H)              # ground-state energy       -> .energy, .exact_energy, .error
qbm.learn_state(sigma)           # quantum-state learning    -> .relative_entropy
qbm.free_energy_min(H, T)        # free energy / Gibbs prep  -> .free_energy, .error
qbm.solve_sdp(C, A, b)           # semidefinite programming  -> .X, .objective, .y
```

Everything is registry-addressable, so components are plugins, not forks:

```python
qbm.available()                       # {'model': [...], 'loss': [...], ...}
qbm.build("optimizer", "adam", lr=0.1)

@qbm.register("loss", "my_loss")
class MyLoss(qbm.losses.Loss): ...
```

### Explicit API

```python
import qbm

# --- beginner: learn a classical distribution in a few lines ---
data  = qbm.datasets.bars_and_stripes(grid=3)      # probability vector over 2^9
model = qbm.learn(data)                             # defaults: FullyVisibleQBM + rel-entropy + Adam + dense
print(model.kl(data))
samples = model.sample(1000)

# --- researcher: ground-state energy via quantum natural gradient (Kubo-Mori) ---
H = qbm.hamiltonians.tfim(n=6, J=1.0, g=1.5)
model = qbm.FullyVisibleQBM(n=6)                    # default local generators {Z_i, X_i, Z_iZ_{i+1}}
hist = qbm.fit(model,
               loss=qbm.losses.Energy(H),
               optimizer=qbm.optim.NaturalGradient(metric="kubo_mori", lr=0.05),
               steps=300)
print("E_est =", model.energy(H), " E0 =", qbm.oracles.ground_energy(H))

# --- researcher: learn a target quantum state ---
sigma = qbm.oracles.gibbs(qbm.hamiltonians.heisenberg(n=5), beta=1.0)
model = qbm.FullyVisibleQBM(n=5)
qbm.fit(model, loss=qbm.losses.RelativeEntropy(sigma), optimizer=qbm.optim.Adam(lr=0.05), steps=300)
```

---

## 7. Repository layout

```
src/qbm/
  operators.py        # PauliString helpers, ParamHamiltonian
  channels.py         # belief-propagation multiplier phi(Delta), kernels
  backends/
    base.py           # Backend + ThermalState Protocols
    dense.py          # DenseBackend, DenseThermalState (eigh)
  models/
    base.py           # Model ABC
    fully_visible.py  # FullyVisibleQBM
  losses/
    base.py · energy.py · relative_entropy.py
  metrics/
    base.py · monotone.py   # KuboMori / FisherBures / WignerYanase (one kernel)
  optim/
    base.py · gradient.py   # GD, Adam · natural_gradient.py
  train/loop.py         # fit(), History
  data/
    datasets.py · hamiltonians.py · oracles.py
  registry.py · facade.py · __init__.py
examples/   tests/   README.md   DESIGN.md   pyproject.toml
```

---

## 8. Correctness is a feature

A *reference* library must be trustworthy. The dense eigendecomposition backend is
an exact oracle, and the suite (100+ tests) gates every change across seven tiers:

1. **Exact / closed form** — eigendecomposition Gibbs state vs `expm`; sqRBM
   closed form vs dense Gibbs; commuting limit = classical Boltzmann machine;
   free energy = `-log Z`; unconstrained SDP = `exp(beta C)/Z`.
2. **Internal identities** — analytic gradient vs finite differences (every loss);
   analytic vs JAX autodiff (~1e-15); metric Loewner orderings
   `g_FB <= g_WY <= 2 g_FB`, `g_KM >= g_FB`; `g_KM = Hessian(log Z)`; variational
   bounds `F >= F_exact`, energy `>= E0`; SDP strong duality and KKT.
3. **Cross-backend** — dense, statevector (TFD) and jax agree to ~1e-9 on states,
   expectations, gradients and metrics.
4. **Convergence** — realizable targets reach machine precision; ground energy vs
   exact diagonalisation; SDP vs an independent scipy-BFGS reference solver.
5. **Paper reproductions** (`tests/reproductions/`) — sqRBM expressivity
   (2502.17562), QNG vs GD (2410.24058), EQBM expressivity (2501.03367),
   barren-plateau-free QBM training (2410.12935).
6. **Property-based** (`tests/test_properties.py`, Hypothesis) — random
   Hamiltonians: valid states, PSD/ordered metrics, correct gradients, cross-backend
   agreement, TFD identities.
7. **Scaling benchmarks** (`benchmarks/scaling.py`) — time/memory vs qubit count
   with the empirical scaling exponent and the documented dense ceiling.

---

## 9. Roadmap

- **v0.1 (done)** — dense+analytic core; `FullyVisibleQBM`; relative-entropy /
  energy losses; GD/Adam + natural gradient; KM/FB/WY metrics; the three
  thin-slice tasks (generative, ground-state, state learning) end-to-end with
  oracle tests.
- **v0.2 (done)** — `FreeEnergy` loss; barren-plateau / gradient-variance
  diagnostics; visible + hidden units (`VisibleHiddenQBM`) with the **exact**
  marginal-likelihood gradient (`MarginalNLL` via the `diagonal_gradient`
  primitive); four guided teaching notebooks; registry scaffolding.
- **v0.3 (done)** — semi-quantum RBM (`SemiQuantumRBM`) with the closed-form
  cosh/tanh fast path and analytic gradients (arXiv:2502.17562), cross-validated
  against the exact dense Gibbs marginal; exact relative-entropy gradient to a
  *quantum* target with hidden units (`MarginalRelativeEntropy`) via the Fréchet
  derivative of `log` — the dense-exact counterpart of the modular-flow lift
  (arXiv:2512.19819); a fifth teaching notebook on expressivity.
- **v0.4 (done)** — Evolved QBM (`EvolvedQBM`): `omega = e^{-iH(phi)} rho(theta) e^{+iH(phi)}`
  with exact `theta` and `phi` gradients (Daleckii–Krein kernel for `d_phi U`) and
  the full `(theta, phi)` quantum Fisher information (all three variants) — all
  reusing the existing `Energy` / `MarginalRelativeEntropy` losses and metric
  machinery via the generic `{d_i omega}` interface. Sixth teaching notebook.
- **v0.5 (done)** — three backends behind one seam + registry
  (`get_backend`/`register_backend`/`available_backends`):
  - `dense` (default), `statevector` (thermofield-double purification, exact ==
    dense, optional **shot noise** for sample-complexity), and `jax`
    (autodiff-powered: gradients via `jax.grad`, QFI metrics via `jax.jacrev`,
    GPU-capable, optional extra). `qbm.purification` (TFD statevector, reduced
    state, entanglement entropy = thermal entropy) and `qbm.autodiff` (differentiate
    novel density-matrix losses) ship alongside.
  - The JAX backend reproduces the analytic engine to ~1e-15, **independently
    validating** the belief-propagation gradient and all three QFI formulas via
    automatic differentiation. Cross-backend agreement is in the test suite; two
    teaching notebooks (07, 08).
- **v0.6 (done)** — the **task layer** (`qbm.ground_state`, `learn_state`,
  `free_energy_min`, `solve_sdp` alongside `learn`), each returning a `Result`;
  **SDP solving** through entropy-regularised duality (`losses/sdp.py`), where the
  dual's stationary point *is* a QBM thermal state and natural gradient with
  `lr = beta` is exactly Newton's method; `ParamHamiltonian.offset`;
  `operators.pauli_pool` (complete k-local generator sets); the **registry wired to
  every built-in** (`qbm.register` / `build` / `available`); CI on Python 3.9–3.13
  with lint, format and notebook-execution gates; `CONTRIBUTING.md`, `CITATION.cff`,
  pre-commit and issue templates.
- **v0.7 (done)** — the verification suite (Phase 3): four paper reproductions
  (`tests/reproductions/`), Hypothesis property-based tests, a scaling benchmark,
  and an examples gallery. 100+ tests; CI green on Python 3.9–3.13 × {core, jax} ×
  {Linux, macOS}.
- **v0.8 (done)** — scaling (Phase 4): a **tensor-network backend** (quimb) that
  represents the thermal state as a purified MPS and reaches 20 qubits in ~1 s at
  bond dimension 4 (a dense `rho` would be ~17 TB), validated against the dense
  engine to ~1e-6; **sample-based training** (`qbm.sampling`) with block-Gibbs
  chains and contrastive divergence, exact for commuting hidden units and a
  documented approximation otherwise; and the lazy-generator fix below.
- **v0.9+ (next)** — circuit/hardware backend (PennyLane/Qiskit) with VarQITE/QITE
  Gibbs preparation and Hadamard-test estimators; the full Gibbs-map (imaginary-time)
  block sampler that removes the non-commuting CD bias; Petz–Tsallis loss;
  metrology / Cramér–Rao module; docs site and PyPI release.

**Scaling note (v0.8).** `ParamHamiltonian` materialises Pauli generators
**lazily**. Building them eagerly costs `n_params * 4^n` complex entries, which caps
*every* backend at ~13 qubits no matter how it represents the state; the
tensor-network backend works from the Pauli labels and so never pays that cost.
Any future backend that scales must likewise avoid touching `.generators`.

**Design note (the unification, validated through v0.4).** Every model exposes a
small generic interface — `density_matrix()`, `expect(O)`, `observable_gradient(O)`,
`state_derivatives()` (`{d_i rho}` in the computational basis), `metric(kind)`,
`diagonal_gradient()`. Losses and metrics are written against *that* interface, so
each new model (fully-visible, visible+hidden, sqRBM, Evolved) reuses the existing
`Energy`, `RelativeEntropy`/`MarginalRelativeEntropy`, NLL, free-energy losses, the
three QFI metrics, and every optimizer with little or no new loss/metric code. The
EQBM in particular added a model only — no new losses or metrics.

**Note on the hidden-unit gradient.** v0.2 implements the exact gradient of the
*measured-distribution* marginal likelihood `p(v) = Tr[(|v><v| (x) I) rho_vh]` by
differentiating the joint Gibbs state and projecting — no Golden–Thompson bound or
modular-flow lift required. The modular-flow lift (arXiv:2512.19819) is still
needed for relative-entropy training to a *quantum* target with hidden units, and
is scheduled for v0.3.
```
