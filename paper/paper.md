---
title: 'qbmkit: one Gibbs state and one primitive for quantum Boltzmann machine research'
tags:
  - Python
  - quantum computing
  - quantum machine learning
  - quantum Boltzmann machines
  - Gibbs states
  - quantum Fisher information
authors:
  - name: Michele Minervini
    orcid: 0009-0000-4739-0075
    affiliation: 1
affiliations:
  - name: School of Electrical and Computer Engineering, Cornell University, Ithaca, NY, USA
    index: 1
date: 7 August 2026
bibliography: paper.bib
---

# Summary

A quantum Boltzmann machine (QBM) is a parameterized thermal state used as a machine
learning model [@amin2018qbm; @kieferova2017tomography]. QBMs are applied to problems
that look unrelated on the surface — generative modelling of classical data,
ground-state energy estimation [@patel2024groundstate], learning an unknown quantum
state, semidefinite programming, and trainability studies — and the literature has
accumulated a correspondingly varied set of gradient formulas, information metrics and
model variants.

`qbmkit` is a Python library built on the observation that this variety is superficial.
Underneath, there is **one object** and **one hard primitive**, and every published QBM
gradient and every monotone quantum Fisher information (QFI) metric is a short algebraic
recombination of them. The library implements the object and the primitive once, derives
everything else from them, and exposes the result through a task layer in which each
research question is a single call. Six interchangeable computational backends — exact
dense, thermofield-double statevector, JAX autodiff, tensor-network, quantum-circuit and
Pauli propagation — sit behind one protocol, so a model written once runs unchanged from
an exact 4-qubit reference to a measurement-based hardware pipeline. The core depends
only on NumPy [@harris2020numpy] and SciPy [@virtanen2020scipy]; every heavier engine is
an optional extra.

# Statement of need

QBM research is distributed across one-off, single-paper repositories. Each re-derives
the thermal state, its gradient and its metric in its own conventions, which makes
results laborious to reproduce and nearly impossible to compare: a natural-gradient
result computed with one paper's metric and one paper's Gibbs-state routine cannot be
cleanly held against another's. General quantum-software frameworks such as PennyLane
[@bergholm2018pennylane] and Qiskit [@javadiabhari2024qiskit] provide circuits and
autodifferentiation but no notion of a parameterized Gibbs state, its belief-propagation
channel, or the QFI metrics that QBM training depends on; tensor-network libraries such
as quimb [@gray2018quimb] provide a representation but not the learning layer. There is
no shared substrate.

The consequence is duplicated, error-prone work. The gradient of a QBM loss is not the
naive derivative of an exponential — `exp(-G)` does not commute with `d G`, and the
correct expression requires an integral representation that is easy to get subtly wrong
and expensive to validate. `qbmkit` targets researchers who want to *use* these
objects — to compare metrics, swap simulation engines, or test a new loss — without
re-deriving and re-validating the substrate each time.

# One object, one primitive

**The object.** Every QBM in the literature is the parameterized Gibbs state

$$\rho(\theta) = e^{-G(\theta)}/Z(\theta), \qquad G(\theta) = \textstyle\sum_j \theta_j G_j,$$

optionally wrapped by real-time evolution $e^{-iH(\phi)}$ — the evolved QBM
[@minervini2026evolved] — and optionally observed through a visible/hidden partial trace
[@wilde2025hidden]. Fully-visible QBMs, classical and semi-quantum restricted Boltzmann
machines [@demidik2025expressive] and evolved QBMs are all instances of this one class,
so a model is fully specified by its generators, which parameters are free, and an
optional partition.

**The primitive.** Every gradient and every metric reduces to the belief-propagation
channel

$$\Phi_\theta(X) = \int \mathrm{d}t\, p(t)\, e^{-iGt} X e^{+iGt}, \qquad
  p(t) = \tfrac{2}{\pi}\ln\lvert\coth(\pi t/2)\rvert .$$

This is the one genuinely hard piece. In the eigenbasis of $G$ it is exact and diagonal,
acting by the multiplier $\varphi(\Delta) = (2/\Delta)\tanh(\Delta/2)$ — the Fourier
transform of $p$ — so the dense backend evaluates it with no time sampling, while
measurement-based backends realise the same channel by sampling $t \sim p(t)$.

**What follows.** From the state derivative built on that channel, the observable
gradient is $\partial_j \mathrm{Tr}[O\rho]$, the relative-entropy gradient collapses to
the positive/negative phase difference $\langle G_j\rangle_\sigma - \langle
G_j\rangle_\rho$, and every monotone metric is a single weighted contraction

$$g^c_{ij} = \sum_{kl} W^c_{kl}\,
  \mathrm{Re}\!\left[(\partial_i\rho)_{kl}\overline{(\partial_j\rho)_{kl}}\right],
  \qquad W^c = 1/c ,$$

with $c$ the Morozova–Chentsov function of the metric [@petz1996monotone]. Because the
metric enters *only* through the weight $W$, `qbmkit` supports the entire monotone family
rather than a fixed list: the two-parameter $\alpha$-$z$ information matrices
[@wilde2025qfi] are implemented directly, with Kubo–Mori, Fisher–Bures and
Wigner–Yanase recovered as special cases and verified to machine precision, and arbitrary
user kernels accepted. Any of them drops into quantum natural gradient
[@stokes2020qng; @patel2025naturalgradient] by name.

The same abstraction makes the backend seam natural. Because losses, metrics and
optimizers are written against the state handle rather than against linear algebra,
adding an engine unlocks the whole library: `qbmkit` ships variational imaginary-time
(VarQITE) Gibbs preparation [@mcardle2019varqite; @zoufal2021vqbm] with Hadamard-test
estimators, requiring no vendor SDK, and an imaginary-time Pauli-propagation engine
[@rudolph2026thermal] with the locally normalised sampler of @minervini2026sampling.
It also exposes the Gibbs map for visible/hidden models, which marginalises the hidden
register exactly and thereby removes the bias that block-Gibbs contrastive divergence
incurs for non-commuting hidden operators [@demidik2025sample].

# Verification

Correctness is treated as a feature. The suite of 292 tests is layered: exact analytic
oracles, finite differences, JAX autodifferentiation agreeing with the analytic engine to
$\sim10^{-15}$, cross-backend agreement, strong duality against an independent
semidefinite-program solver, property-based tests, and four reproductions of published
results [@demidik2025expressive; @patel2025naturalgradient; @minervini2026evolved;
@patel2024groundstate]. Continuous integration runs on Python 3.9–3.13 across Linux and
macOS and executes all thirteen tutorial notebooks, so the documentation cannot rot
silently.

# Acknowledgements

The author thanks the authors of the works this library builds on, whose results form its
test suite.

# References
