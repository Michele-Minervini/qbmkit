# Examples

Short, runnable scripts — each prints a result you can check against an exact value.
Run any of them with `python examples/<file>.py` (from the repo root, in the venv).

| # | Script | What it shows | Typical result |
|---|--------|---------------|----------------|
| 01 | [`01_generative_modeling.py`](01_generative_modeling.py) | Learn a classical distribution (bars-and-stripes) with a QBM | KL ≈ 7e-4 |
| 02 | [`02_ground_state_energy.py`](02_ground_state_energy.py) | Ground-state energy of a TFIM via quantum natural gradient | matches exact to ~1e-4 |
| 03 | [`03_state_learning.py`](03_state_learning.py) | Learn a target quantum state by relative-entropy minimisation | D → machine precision |
| 04 | [`04_free_energy_minimization.py`](04_free_energy_minimization.py) | Variational free energy / Gibbs preparation at several temperatures | matches exact to ~1e-15 |
| 05 | [`05_semidefinite_program.py`](05_semidefinite_program.py) | Solve an SDP through the QBM core (unconstrained + constrained) | strong duality to ~1e-15 |

## Visual gallery (notebooks)

The [`../notebooks/`](../notebooks) directory holds eight guided tutorials that teach
the theory alongside the code, **with plots** (they render directly on GitHub):

theory & first QBM · generative modelling · ground state & natural gradient ·
barren plateaus · semi-quantum RBMs · evolved QBM · backends & purification ·
JAX autodiff.

## Benchmarks

[`../benchmarks/scaling.py`](../benchmarks/scaling.py) measures time and memory of the
core operations vs qubit count and reports the empirical scaling exponent (memory
~ 4ⁿ, the dense ceiling is ~12–13 qubits on a workstation). It writes
`benchmarks/scaling.csv` and `benchmarks/scaling.png`.

## Reproductions

Published results are reproduced as tests in
[`../tests/reproductions/`](../tests/reproductions): the semi-quantum-RBM expressivity
gap (arXiv:2502.17562), quantum natural gradient vs gradient descent
(arXiv:2410.24058), evolved-QBM expressivity (arXiv:2501.03367), and the
barren-plateau-free property of QBM energy training (arXiv:2410.12935).
