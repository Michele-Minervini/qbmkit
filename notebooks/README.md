# qbmkit tutorials

Guided notebooks that teach the theory of quantum Boltzmann machines alongside the
`qbmkit` API, with runnable code and plots. They are written to be read in order.

| # | Notebook | What you learn |
|---|----------|----------------|
| 1 | [`01_theory_and_first_qbm.ipynb`](01_theory_and_first_qbm.ipynb) | What a QBM is (the Gibbs-state model), the belief-propagation primitive `Φ_θ`, building a model and inspecting its thermal state. |
| 2 | [`02_generative_modeling.ipynb`](02_generative_modeling.ipynb) | Learning a classical distribution (bars-and-stripes); relative-entropy vs marginal-NLL objectives; fully-visible vs hidden units. |
| 3 | [`03_ground_state_and_natural_gradient.ipynb`](03_ground_state_and_natural_gradient.ipynb) | Ground-state energy estimation; comparing GD / Adam / quantum natural gradient; the three QFI metrics and their orderings. |
| 4 | [`04_barren_plateaus.ipynb`](04_barren_plateaus.ipynb) | Trainability diagnostics; gradient-variance scaling with system size. |
| 5 | [`05_semiquantum_rbm.ipynb`](05_semiquantum_rbm.ipynb) | Semi-quantum RBMs (closed-form fast path); quantum vs classical hidden-unit expressivity; learning quantum states with hidden units. |
| 6 | [`06_evolved_qbm.ipynb`](06_evolved_qbm.ipynb) | The Evolved QBM (real-time evolution on top of the Gibbs state); extra expressivity; the (θ, φ) quantum Fisher information. |
| 7 | [`07_backends_and_purification.ipynb`](07_backends_and_purification.ipynb) | The backend seam; thermofield-double purification; statevector vs dense; shot noise and sample-complexity. |
| 8 | [`08_jax_autodiff.ipynb`](08_jax_autodiff.ipynb) | The JAX backend; autodiff gradients/metrics validated against the analytic engine; training a novel loss with no hand-derived gradient. (needs `qbmkit[jax]`) |
| 9 | [`09_alpha_z_metrics.ipynb`](09_alpha_z_metrics.ipynb) | **Choosing your geometry**: the α-z family of quantum Fisher information — one kernel containing Kubo–Mori, Fisher–Bures and Wigner–Yanase; the map of metrics and their monotone region; natural gradient under different geometries; custom kernels. |
| 10 | [`10_swapping_circuit_emitters.ipynb`](10_swapping_circuit_emitters.ipynb) | **Not marrying an SDK**: the internal circuit IR, proof the core runs with every SDK blocked, and swapping between the built-in simulator / Qiskit / PennyLane / OpenQASM 3 emitters. |
| 11 | [`11_varqite_gibbs_preparation.ipynb`](11_varqite_gibbs_preparation.ipynb) | **Preparing the Gibbs state on a device**: VarQITE — imaginary time via McLachlan's principle, the tilt-partner ansatz, the measurable residual as an error bar, and a variationally prepared QBM that still does gradients and metrics. |
| 12 | [`12_training_a_qbm_with_varqite.ipynb`](12_training_a_qbm_with_varqite.ipynb) | **Training end-to-end on variational states**: the full device-native loop — what is and isn't measurable, generative modelling under shot noise, and why ground-state search needs a residual guard. |

## Running

```bash
pip install -e ".[notebooks]"
jupyter lab notebooks/
```

The committed notebooks already contain executed output (plots included), so you
can also just read them on GitHub.
