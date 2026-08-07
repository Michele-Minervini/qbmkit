# qbmkit

[![CI](https://github.com/Michele-Minervini/qbmkit/actions/workflows/ci.yml/badge.svg)](https://github.com/Michele-Minervini/qbmkit/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/qbmkit.svg?cacheSeconds=1800)](https://pypi.org/project/qbmkit/)
[![Python](https://img.shields.io/pypi/pyversions/qbmkit.svg?cacheSeconds=1800)](https://pypi.org/project/qbmkit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21844741.svg)](https://doi.org/10.5281/zenodo.21844741)

**The unifying, open-source library for Quantum Boltzmann Machines (QBMs)** — the
reference place to learn, study, research, and run simulations with QBMs.

> Install name: `qbmkit` · Import name: `qbm`

**Every research question is one call**, on the same core:

```python
import qbm

qbm.learn(data)                 # generative modelling of a classical distribution
qbm.ground_state(H)             # ground-state energy estimation
qbm.learn_state(sigma)          # quantum-state learning
qbm.free_energy_min(H, T)       # free-energy minimisation / Gibbs preparation
qbm.solve_sdp(C, A, b)          # semidefinite programming
```

```python
import qbm

H   = qbm.hamiltonians.tfim(4, J=1.0, g=1.5)
res = qbm.ground_state(H)        # quantum natural gradient, Kubo-Mori metric, by default
print(res.report())
# ground_state result
#   final loss   : -6.50374
#   energy       : -6.50374
#   exact_energy : -6.50389
#   error        : 0.000147
```

Every task returns a `Result` (`.model`, `.history`, `.report()`), and every default
is overridable — swap the model, loss, optimizer, metric or backend without leaving
the same API.

## Why this exists

QBM research is scattered across one-off, single-paper repositories. `qbmkit`
provides the missing unifying layer: one homogeneous vocabulary in which
*generative modeling*, *ground-state energy estimation*, *quantum-state learning*,
*free-energy / SDP* problems, and *barren-plateau* studies are all expressed over
the **same core** — and trained with gradient descent, Newton, or **quantum
natural gradient** using the Fisher–Bures, Wigner–Yanase, or Kubo–Mori metrics.

## The idea in one paragraph

Every QBM is a parameterized Gibbs state `ρ(θ) = e^(−G(θ))/Z` (optionally wrapped
by real-time evolution, optionally with hidden units). Every gradient and every
quantum-Fisher-information metric is a thin recombination of one primitive — the
belief-propagation channel `Φ_θ`. Implement those once and the whole field
follows. See [`DESIGN.md`](DESIGN.md).

## Architecture

```
tasks         learn · ground_state · learn_state · free_energy_min · solve_sdp
primitives    models | losses | metrics (QFI) | optimizers      ← all registry-addressable
core          ParamHamiltonian · ThermalState · Φ_θ
backend seam  dense (default) | statevector (TFD) | jax | tensor network | circuit | Pauli propagation
```

Only the backend layer touches concrete linear algebra. The default `DenseBackend`
builds the Gibbs state by **eigendecomposition** of `G(θ)` (overflow-safe, and one
factorization yields ρ, log Z, the channel, all gradients and all metrics).
Tensor-network and circuit/hardware backends plug into the same protocol with no
change to user code.

## Install

```bash
pip install qbmkit
```

Core install is just NumPy and SciPy, and already gives you the **dense**,
**statevector**, **circuit** and **Pauli-propagation** backends. Optional extras add the
rest:

```bash
pip install "qbmkit[jax]"        # + JAX autodiff backend, GPU-capable (pulls NumPy >= 2)
pip install "qbmkit[tn]"         # + tensor-network backend (quimb)
pip install "qbmkit[circuit]"    # + Qiskit emitter   (the circuit backend itself needs no SDK)
pip install "qbmkit[notebooks]"  # + matplotlib/jupyter to run the tutorials
```

Straight from source (latest `main`):

```bash
pip install "git+https://github.com/Michele-Minervini/qbmkit"
```

For development, in a virtual environment:

```bash
git clone https://github.com/Michele-Minervini/qbmkit && cd qbmkit
python3 -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[dev]"          # + pytest, ruff, matplotlib, nbclient
pytest                           # 292 tests
```

> Notes
> - Python 3.9–3.13. The `jax` extra requires NumPy >= 2; the base install runs on
>   NumPy 1.x or 2.x. PennyLane is only installed on Python >= 3.10 (it dropped 3.9).
> - Install name is `qbmkit`, import name is `qbm`. Do not install it alongside the
>   unrelated dormant PyPI package `qbm` — both expose a top-level `qbm` module.

## Tutorials

Guided Jupyter notebooks (theory + code + plots) live in [`notebooks/`](notebooks/):

1. **Theory and your first QBM** — the Gibbs-state model and the belief-propagation primitive.
2. **Generative modelling** — learn bars-and-stripes; fully-visible vs hidden units.
3. **Ground-state energy & quantum natural gradient** — GD vs Adam vs QNG; the QFI metrics.
4. **Barren plateaus** — trainability diagnostics and gradient-variance scaling.
5. **Semi-quantum RBMs** — closed-form fast path; quantum vs classical expressivity; learning quantum states.
6. **Evolved QBM** — real-time evolution on top of the Gibbs state; extra expressivity; the (θ, φ) quantum Fisher information.
7. **Backends & purification** — the backend seam; thermofield-double purification; statevector vs dense; shot noise and sample-complexity.
8. **JAX autodiff** — autodiff gradients/metrics validated against the analytic engine; training a loss with no hand-derived gradient.
9. **Arbitrary QFI metrics** — the α-z family; the map of metrics; natural gradient under different geometries.
10. **Swapping circuit emitters** — the internal IR; the core running with every SDK blocked; QASM 3 / Qiskit / PennyLane.
11. **VarQITE Gibbs preparation** — preparing ρ(θ) on a device from expectation values only, and how to tell when it worked.
12. **Training end-to-end with VarQITE** — the full device-native loop, shot noise, and the residual guard.
13. **Pauli propagation** — thermal states as sparse Pauli sums; the locally normalised sampler; generative QBM training in the Pauli basis.

```bash
pip install -e ".[notebooks]"
jupyter lab notebooks/
```

## Backends

The engine is a swap seam (`qbm.get_backend(...)`); the same models/losses/metrics
run on any backend:

- **`dense`** — exact NumPy/SciPy density matrix (default).
- **`statevector`** — thermofield-double purification, with an optional `shots`
  budget for hardware-like measurement.
- **`jax`** (`pip install qbmkit[jax]`) — autodiff gradients/metrics (`jax.grad`/
  `jax.jacrev`), GPU-capable; reproduces the analytic engine to ~1e-15.
- **`tensor_network`** (`pip install qbmkit[tn]`) — thermal state as a **purified
  matrix-product state**, for expectation-based training (relative entropy / NLL)
  well past the dense ceiling: 20 qubits in ~1 s at bond dimension 4, where a dense
  density matrix would need ~17 TB. Metrics and the energy gradient are not available
  there and raise a clear error.
- **`pauli_propagation`** — thermal state as a **sparse sum of Pauli strings** evolved
  under imaginary time (arXiv:2602.04878). Exact for commuting (classical) Hamiltonians,
  first-order-Trotter otherwise, and **topology-agnostic** (all-to-all costs the same as a
  chain). Ships the locally normalised **sampler** of *Sampling from Thermal Quantum
  States via Pauli Propagation* — valid samples with exact likelihoods even from a
  truncated, non-physical state — which makes it a natural generative-modelling engine.
  Pure NumPy, no dependency; expectation-based training and sampling only.
- **`circuit`** — runs the QBM through **actual quantum circuits**: the thermal state is
  prepared as a thermofield-double purification and every quantity is obtained by
  *measurement*, with an optional shot budget. Includes Hadamard-test estimators for the
  energy gradient and the α-z / Kubo–Mori information matrices, and **VarQITE**
  variational Gibbs preparation — the hardware route. Needs **no vendor SDK**: circuits
  live in an internal IR run by our own simulator, with **OpenQASM 3**, Qiskit and
  PennyLane as thin *emitters* (`pip install qbmkit[circuit]`).

## Running on quantum circuits and hardware

```python
import qbm
from qbm.metrics import AlphaZ

model = qbm.FullyVisibleQBM(n=3, backend="circuit")     # measurement-based
state = model.state()
state.metric(AlphaZ(0.5, 1.0))        # information matrix from Hadamard tests
state.resource_estimate()             # circuits and shots a device would need

from qbm.circuits.adapters import to_qasm3, to_qiskit, executor

# swap the execution engine with one argument -- results are identical
qbm.backends.circuit.CircuitBackend(executor=executor("qiskit"))
qbm.backends.circuit.CircuitBackend(executor=executor("pennylane"))
```

The algorithms are written in a small internal circuit IR, so **no quantum SDK is a
dependency of the core**. Qiskit, PennyLane and OpenQASM 3 are ~100-line adapters — an
SDK breaking change (Qiskit has removed `opflow`, moved `qiskit.algorithms` out of core,
dropped `execute()` and revised its primitives twice) costs one file, not the library.
This is enforced by tests: one asserts no SDK is imported outside `adapters/`, another
that `import qbm` loads no SDK at all. Notebook 10 demonstrates it by blocking every SDK
and running the full circuit pipeline anyway.

### Variational Gibbs preparation (VarQITE)

Exact TFD synthesis needs `eigh(G)` and emits one opaque unitary — useful for validating
the estimators, but not a quantum algorithm. **VarQITE** prepares ρ(θ) from *expectation
values only*, and emits ordinary gates:

```python
from qbm.circuits import varqite

res = varqite.prepare_gibbs(H, beta=1.0, depth=3)   # or (ham, theta)
res.report()                 # energy, McLachlan residual, and (in simulation) the error
res.circuit()                # gate-level -- exports to OpenQASM 3
res.resource_estimate()      # circuits per time step, honestly

model = qbm.FullyVisibleQBM(n=3, backend="circuit")            # exact TFD synthesis
qbm.backends.circuit.CircuitBackend(gibbs_prep="varqite")      # the device route
```

It runs McLachlan's variational principle on the thermofield double — `A λ̇ = C` with
`A` the quantum geometric tensor and `C = −½∇⟨H⟩`, i.e. **quantum natural gradient
flow** — starting from Bell pairs (the β = 0 TFD) and flowing to τ = β/2. The ansatz is
built from *tilt partners* of the Hamiltonian's Pauli terms, which makes it **exact for
commuting Hamiltonians** and systematically improvable otherwise. `A` and `C` are also
implemented as Hadamard tests, checked against the exact route to ~1e-15. The reported
**McLachlan residual** is a genuine error bar: computable from measurements, so it works
where fidelity does not. See [notebook 11](notebooks/11_varqite_gibbs_preparation.ipynb).

The whole training loop runs on it — no eigendecomposition anywhere:

```python
model = qbm.learn(data, backend=CircuitBackend(gibbs_prep="varqite"), steps=60)
```

On classical generative targets that is indistinguishable from exact training (the model
flows to a commuting Hamiltonian, where the ansatz is exact) and it survives shot noise.
Objectives that drive β → ∞, like ground-state search, degrade the preparation as they
go; guard them on the same measurable residual —
`qbm.fit(..., stop=lambda s, m: s.varqite_result().residual > 0.05)`. Worked through in
[notebook 12](notebooks/12_training_a_qbm_with_varqite.ipynb).

Honest scope: expectations, generative gradients, sampling, the energy gradient and the
information matrices are measurable. Entropy, `log Z`, free energy and the SDP dual
depend on the *spectrum* of ρ, which a device does not expose — those raise a clear
error and belong on `backend="dense"`. VarQITE costs `O(L²)` circuits per time step for
`L` ansatz rotations; `resource_estimate()` reports that and the shot cost up front.

## Classical simulation in the Pauli basis (Pauli propagation)

A third classical engine, alongside dense and tensor networks, with a different sweet
spot. Instead of a `2ⁿ × 2ⁿ` matrix it stores ρ as a **sparse sum of Pauli strings** and
reaches the thermal state by evolving the *identity* — the sparsest operator — under
imaginary time ([arXiv:2602.04878](https://arxiv.org/abs/2602.04878)):

```python
from qbm.backends.pauli_propagation import PauliPropagationBackend

backend = PauliPropagationBackend(trotter_steps=32, coeff_cutoff=1e-8)
model   = qbm.learn(data, backend=backend, steps=120)   # generative training, Pauli basis
model.state().sample(40_000)                             # locally normalised sampler
```

Each imaginary-time gate branches a Pauli string only when it **commutes** with the
Hamiltonian term, `P → cosh(2θ)P − sinh(2θ)PG`; anticommuting terms pass through. This is
**exact for commuting (classical) Hamiltonians**, first-order-Trotter otherwise, and
**topology-agnostic** — an all-to-all Hamiltonian costs no more than a chain, where a
tensor network would pay for the entanglement. The sum is kept sparse by discarding
small-coefficient (`coeff_cutoff`) or high-weight (`max_weight`) terms, which trades
retained terms for accuracy — moderate truncation reaches full training accuracy at a
fraction of the terms, and notebook 13 maps the cost/accuracy frontier of both knobs
across a training run (coefficient truncation dominates for densely-connected targets).

Truncation can push ρ outside the positive cone, so its basis "probabilities" may go
slightly negative. The **locally normalised sampler** of *Sampling from Thermal Quantum
States via Pauli Propagation* fixes this: it draws bits sequentially from absolute
quasi-marginals, a valid distribution within a certified total-variation distance of the
true Born distribution, and yields **exact pointwise likelihoods** — a pairing generative
models rarely offer. A QBM trains through the unchanged `qbm.learn` API and matches exact
training (KL to ~4e-5). Same honest scope as the other expectation-based backends: metrics,
the energy gradient, `log Z` and the entropy raise a clear error. Pure NumPy, no
dependency. See [notebook 13](notebooks/13_pauli_propagation_qbm.ipynb); the
high-performance many-qubit implementation is
[`PauliPropagation.jl`](https://github.com/Michele-Minervini/PauliSampling.jl).

## Hidden units at scale: the Gibbs map

Hidden-unit models used to be capped at the dense ceiling, because the exact likelihood
gradient ([`MarginalNLL`](src/qbm/losses/likelihood.py)) needs `d_j ρ` — which no scalable
backend can form. And the block-Gibbs contrastive-divergence alternative is *biased* when
the hidden operators don't commute.

[`qbm.gibbs_map`](src/qbm/gibbs_map.py) removes both problems at once. Because the visible
register is diagonal, `G(θ)` is block diagonal in the visible basis, `G = ⊕ᵥ G_h(v)`, so the
hidden register can be **traced out exactly** instead of sampled:

```python
from qbm.losses import GibbsMapNLL

model = qbm.VisibleHiddenQBM(n_visible=4, n_hidden=2, hidden_paulis=("Z", "X"),
                             backend=qbm.get_backend("pauli_propagation"))
qbm.fit(model, GibbsMapNLL(data, n_visible=4), qbm.optim.Adam(lr=0.12), steps=80)
```

The gradient splits into the familiar positive/negative phases —
`∂ⱼL = Σᵥ q(v)⟨Gⱼ⟩_{σᵥ} − ⟨Gⱼ⟩_ρ` — but the positive phase uses the **exact conditional
Gibbs state** `ρᵥ = e^{−G_h(v)}/Z_v` rather than a sampled classical conditional, so it is
exact for non-commuting hidden operators. The negative phase is just `generator_expectations()`,
which every backend provides — so **hidden-unit training now runs on the tensor-network,
circuit and Pauli-propagation backends** (verified to 4e-16 against the dense route). The
positive phase touches only the hidden register and only the configurations present in the
data, so its cost is independent of the number of visible units.

The same map gives an **unbiased sampler** (`GibbsMap.sample`), a chain on the exact visible
free energy `−log Z_v`. On a non-commuting sqRBM, measured against 40 000 samples:

| sampler | TVD to exact |
|---|---|
| block-Gibbs CD (`qbm.sampling`) | 0.265 — biased |
| **Gibbs map** (`qbm.gibbs_map`) | **0.0031** |
| i.i.d. finite-sample floor | 0.0032 |

It also supplies `log Z` itself, so unlike `RelativeEntropy` the loss *value* is available on
the scalable backends too.

## Arbitrary quantum Fisher information metrics

Every metric is a weight kernel `W[k,l]` on the state derivatives in the eigenbasis of
ρ, so `qbmkit` supports the **whole family** rather than a fixed list — including the
two-parameter **α-z information matrices** of
[arXiv:2510.02218](https://arxiv.org/abs/2510.02218):

```python
from qbm.metrics import AlphaZ, PetzRenyi, SandwichedRenyi, CustomMetric

AlphaZ(0.7, 2.0)                 # any (α, z)
PetzRenyi(2.0)                   # z = 1 slice
SandwichedRenyi(0.5)             # z = α slice
CustomMetric(lambda x, y: 1/np.sqrt(x*y), "geometric")   # your own kernel
```

The three classic metrics are special cases of that one family — verified to machine
precision:

| metric | α-z parameters |
|---|---|
| Kubo–Mori | α → 1 (any z) |
| Fisher–Bures (SLD) | α = ½, z = ½ (sandwiched Rényi ½) |
| Wigner–Yanase | α = ½, z = 1 (Petz–Rényi ½) |

Any of them drops straight into quantum natural gradient
(`NaturalGradient(metric=SandwichedRenyi(0.5))`). `AlphaZ` warns when (α, z) falls
outside the region where the data-processing inequality is known to hold, so a
non-monotone choice is never silent.

## Extending it

Every component is registered by name, so new ones are plugins rather than forks:

```python
@qbm.register("loss", "my_loss")
class MyLoss(qbm.losses.Loss):
    def value(self, state): ...
    def grad(self, state): ...

qbm.build("loss", "my_loss")        # now addressable everywhere
qbm.available()                     # {'model': [...], 'loss': [...], ...}
```

A new **model** needs no new losses or metrics, and a new **backend** needs no
changes anywhere else — see [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Status

v0.12 — one-call **task layer** (generative, ground state, state learning,
free energy, **SDP**); dense + statevector (TFD purification + shots) + **JAX
autodiff** + **tensor-network** + **circuit** + **Pauli-propagation** backends behind a
registry seam, with **VarQITE** variational Gibbs preparation on the circuit route and
imaginary-time **Pauli propagation** (sparse-Pauli engine + locally normalised sampler);
sample-based
training (block-Gibbs / contrastive divergence); fully-visible, visible+hidden,
**semi-quantum RBM** (closed-form) and **Evolved QBM** models; relative-entropy /
energy / marginal-NLL / sqRBM-NLL / free-energy / quantum-target-relative-entropy /
SDP-dual losses, plus autodiff of arbitrary density-matrix objectives; GD / Adam /
quantum natural gradient; **arbitrary QFI metrics** (the α-z family plus user kernels,
with Kubo–Mori / Fisher–Bures / Wigner–Yanase as special cases);
barren-plateau diagnostics. **292 tests** across seven tiers — exact oracles, finite
differences, autodiff (~1e-15), cross-backend agreement, strong duality/KKT with an
independent reference SDP solver, **four paper reproductions**
([`tests/reproductions/`](tests/reproductions)), Hypothesis property-based tests, and
a [scaling benchmark](benchmarks/scaling.py) — green on Python 3.9–3.13 × {core, jax}
× {Linux, macOS}. See the roadmap in [`DESIGN.md`](DESIGN.md).

## How to cite

If `qbmkit` supports your research, please cite the archived software. The DOI below is
the *concept* DOI: it always resolves to the latest version, and each release also has
its own version DOI if you need to pin one.

> Minervini, M. *qbmkit: a unifying open-source library for quantum Boltzmann machines*.
> Zenodo. https://doi.org/10.5281/zenodo.21844741

```bibtex
@software{qbmkit,
  author  = {Minervini, Michele},
  title   = {qbmkit: a unifying open-source library for quantum Boltzmann machines},
  year    = {2026},
  publisher = {Zenodo},
  doi     = {10.5281/zenodo.21844741},
  url     = {https://github.com/Michele-Minervini/qbmkit}
}
```

GitHub's **Cite this repository** button reads [`CITATION.cff`](CITATION.cff) and will
produce the same thing.

## License

MIT.
