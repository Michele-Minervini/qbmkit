# Contributing to qbmkit

Thanks for considering a contribution. This project aims to be the *reference*
implementation for quantum Boltzmann machines, so correctness and clarity matter
more than speed of merging.

## Setup

```bash
git clone https://github.com/Michele-Minervini/qbmkit
cd qbmkit
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,jax,notebooks]"
pytest
```

## The one design rule

Everything is built on **one object** — the parameterized Gibbs state
`rho(theta) = e^{-G(theta)}/Z` — and **one primitive**, the belief-propagation
channel. Models expose a small generic interface (`density_matrix`, `expect`,
`observable_gradient`, `state_derivatives`, `metric`, `diagonal_gradient`); losses,
metrics and optimizers are written *against that interface only*.

Practical consequence: **a new model should not need new losses or metrics**, and a
new backend should not need changes anywhere else. If your change requires touching
many layers, the abstraction is probably in the wrong place — open an issue first.

See [`DESIGN.md`](DESIGN.md) for the full specification and the exact formulas, and the
[codebase tour](docs/codebase-tour.md) for a dependency-ordered reading path through every
module, with verification exercises — the fastest way to get real ownership of a subsystem.

## Adding something

| You want to add | Do this |
|---|---|
| Model | subclass `models.base.Model`, declare generators; register with `@qbm.register("model", "name")` |
| Loss | subclass `losses.base.Loss` with `value(state)` / `grad(state)`; register it |
| Metric | add a Morozova–Chentsov weight kernel in `metrics/monotone.py` |
| Optimizer | subclass `optim.base.Optimizer` with `step(theta, grad, state=None)` |
| Backend | implement the `Backend` / `ThermalState` protocols in `backends/base.py` |
| Task | add a recipe in `tasks/` returning a `Result` |

## Tests are the deliverable

Every new numerical routine needs at least one test that pins it to something
independent. In order of preference:

1. **Closed form / exact oracle** — compare against an analytic result.
2. **Finite differences** — every new gradient must match `(f(x+h) - f(x-h))/2h`.
3. **Autodiff** — with the `jax` extra, compare against `jax.grad`.
4. **Cross-backend** — the same quantity must agree on `dense`, `statevector`, `jax`.
5. **Identity / inequality** — e.g. metric Loewner orderings, variational bounds.

Run `pytest -q` before opening a PR; CI runs the suite on Python 3.9–3.13 and
executes every notebook.

## Style

- `ruff check` and `ruff format` must pass (`pre-commit install` does this locally).
- NumPy-style docstrings; document the *mathematical* meaning, not just the types.
- If you implement a formula from a paper, cite it (arXiv id) in the docstring.

## Reporting bugs and requesting features

Open an [issue](https://github.com/Michele-Minervini/qbmkit/issues); templates for bug
reports and feature requests are provided. For a bug, please include the qbmkit version
(`qbm.__version__`), the backend, and a minimal reproducer — and for numerical issues the
system size and parameter scale, since bugs in this domain are often regime-specific
(very low temperature, degenerate spectra, aggressive truncation).

## Getting help

For questions about *using* the library — which backend fits your problem, which metric
or loss to choose, how to express a model — please open a
[Discussion](https://github.com/Michele-Minervini/qbmkit/discussions) or an issue
labelled `question` rather than emailing the maintainer, so the answer is searchable by
the next person. The [tutorials](notebooks/) cover most first questions, and
[`DESIGN.md`](DESIGN.md) documents the exact formulas behind every routine.

## Code of conduct

Participation is governed by the [Contributor Covenant](CODE_OF_CONDUCT.md).
