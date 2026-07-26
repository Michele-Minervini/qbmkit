# qbmkit

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
backend seam  dense (default) | statevector (TFD) | jax | tensor network | circuit
```

Only the backend layer touches concrete linear algebra. The default `DenseBackend`
builds the Gibbs state by **eigendecomposition** of `G(θ)` (overflow-safe, and one
factorization yields ρ, log Z, the channel, all gradients and all metrics).
Tensor-network and circuit/hardware backends plug into the same protocol with no
change to user code.

## Install

Use an isolated virtual environment (recommended — keeps the project's
dependencies, including the NumPy-2 / JAX stack, off your system Python):

```bash
python3 -m venv .venv            # create the environment
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .                 # core only: numpy, scipy
# or pick extras:
pip install -e ".[dev]"          # + pytest, ruff, matplotlib, nbclient  (run the tests/notebooks)
pip install -e ".[jax]"          # + JAX autodiff backend (pulls NumPy >= 2)
pip install -e ".[dev,jax,notebooks]"   # everything
pytest                           # run the test suite
```

> Notes
> - The `jax` extra requires NumPy >= 2; the venv keeps that isolated. The base
>   install (dense + statevector backends) runs on NumPy 1.x or 2.x.
> - Do not install this alongside the unrelated dormant PyPI package `qbm`; both
>   expose a top-level `qbm` module.

## Tutorials

Guided Jupyter notebooks (theory + code + plots) live in [`notebooks/`](notebooks/):

1. **Theory and your first QBM** — the Gibbs-state model and the belief-propagation primitive.
2. **Generative modelling** — learn bars-and-stripes; fully-visible vs hidden units.
3. **Ground-state energy & quantum natural gradient** — GD vs Adam vs QNG; the QFI metrics.
4. **Barren plateaus** — trainability diagnostics and gradient-variance scaling.
5. **Semi-quantum RBMs** — closed-form fast path; quantum vs classical expressivity; learning quantum states.
6. **Evolved QBM** — real-time evolution on top of the Gibbs state; extra expressivity; the (θ, φ) quantum Fisher information.
7. **Backends & purification** — the backend seam; thermofield-double purification; statevector vs dense; shot noise and sample-complexity.

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
- *planned extras:* `tensor_network` (quimb), `circuit` (PennyLane/Qiskit).

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

v0.6 — one-call **task layer** (generative, ground state, state learning,
free energy, **SDP**); dense + statevector (TFD purification + shots) + **JAX
autodiff** backends behind a registry seam; fully-visible, visible+hidden,
**semi-quantum RBM** (closed-form) and **Evolved QBM** models; relative-entropy /
energy / marginal-NLL / sqRBM-NLL / free-energy / quantum-target-relative-entropy /
SDP-dual losses, plus autodiff of arbitrary density-matrix objectives; GD / Adam /
quantum natural gradient; Kubo–Mori / Fisher–Bures / Wigner–Yanase metrics;
barren-plateau diagnostics. **85 tests** covering exact oracles, finite differences,
autodiff, cross-backend agreement, strong duality/KKT, and an independent reference
SDP solver. See the roadmap in [`DESIGN.md`](DESIGN.md).

## License

MIT.
