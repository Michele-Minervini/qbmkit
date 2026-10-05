# A guided tour of the qbmkit codebase

A structured, 24-week reading path through all 6,947 lines of `src/qbm`, in dependency
order: nothing is read before the thing it rests on. It exists so that a maintainer — or
a contributor who wants real ownership of a subsystem — can work through the library
deliberately rather than by grep.

Each week names the files, states **the claim** the code is making (the mathematical
assertion you should end up believing), gives a **verification** you can run yourself,
and lists **questions to answer**. The questions are the point: if you cannot answer them
without re-reading, you have not finished the week.

**Budget:** 1–3 hours per week. Weeks are sized by difficulty, not by line count — week 19
is 742 lines and week 2 is 110, and they take about the same time.

**How to use a week.** Read the module top to bottom, including docstrings. Then run the
verification *and predict the number before you look*. Then open the matching test file
and check whether the tests would catch the bug you would most fear. Note anything you
would write differently; that list is your future contribution backlog.

> Conventions used throughout: qubit 0 is the leftmost tensor factor and the most
> significant bit; inverse temperature β is absorbed into θ; gradients are real vectors
> ordered like θ. See [`DESIGN.md`](../DESIGN.md) §3.

---

## Phase 1 · The core (weeks 1–4)

Everything else is a recombination of this phase. Do not skip ahead.

### Week 1 — The object: `operators.py` (441)
**Claim.** A QBM model is nothing but a list of Hermitian generators plus which
coefficients are free: `G(θ) = Σⱼ θⱼ Gⱼ`. And every standard list is described the same
way: *which Pauli words*, placed on *which coupling graph*.

**Read.** `pauli`; then the two builders — `pauli_pool` (one register, any graph) and
`rbm_generators` (the bipartite visible/hidden layout) — with the helpers they share;
then `ParamHamiltonian`.

**Verification.**

1. *A Pauli matrix by hand.* `qbm.pauli("XIZ")` is 8 × 8. Label its rows and columns by
   bitstrings with qubit 0 leftmost, so row 0 is `000` and column 7 is `111`. An entry
   of a tensor product is the product of one single-qubit entry per qubit, and `X`, `Y`
   flip their bit while `I`, `Z` keep it — so every row has exactly one nonzero entry.
   Predict, then check: in which column is the nonzero entry of row 0, and what is its
   value? And for row 1 (`001`)?
2. *One builder, two descriptions.* Write the transverse-field-Ising generators of an
   open 4-qubit chain as a `pauli_pool` call and predict how many there are. Then
   predict `len(qbm.pauli_pool(4))` (three Paulis per site, nine per pair), and the
   number of couplings on a 2 × 3 `"grid"`. Which `rbm_generators` arguments give a
   classical RBM, and which a fully quantum one?
3. *Laziness.* "Lazy" means computed on first use rather than at construction. Build a
   20-qubit Hamiltonian,
   `ham = qbm.ParamHamiltonian(qbm.pauli_pool(20, terms=("Z", "X", "ZZ"), connectivity="chain"))`,
   and check that `ham._mats is None`: only the 59 labels are stored, no matrix has been
   built. Before you are tempted to read `ham.generators` at this size, work out how
   many bytes one 2²⁰ × 2²⁰ complex matrix needs.

**Questions.** Why is `generators` a cached property rather than built in `__init__` —
which backends would be capped, and at how many qubits, if it were built eagerly?
Laziness only protects code that never reads `.generators`: list its readers
(`grep -rn "\.generators" src`) and say which of them a tensor-network training run
passes through. Why must a generator never appear twice in the list — what happens to
the metric — and which topology used to produce exactly that? What is `offset` for, and
why does it leave every gradient formula unchanged?

### Week 2 — The primitive: `channels.py` (48) + `linalg.py` (62)
**Claim.** The belief-propagation channel `Φ_θ(X) = ∫dt p(t) e^{-iGt} X e^{iGt}` is
**exact and diagonal** in the eigenbasis of `G`, acting by `φ(Δ) = (2/Δ)tanh(Δ/2)`.

**Read.** `bp_multiplier`, `state_derivative_kernel`, then `linalg.py`.

**Verification.** Confirm `φ(Δ)` is the Fourier transform of the high-peak tent `p(t)`:
integrate `p(t)cos(Δt)` numerically and compare. Check `φ(0) = 1` and that the code
handles `Δ → 0` without dividing by zero.

**Questions.** Why does `K[k,k] = -p_k` rather than the naive limit? Where does
`Tr(∂ⱼρ) = 0` come from, and does the code satisfy it exactly or approximately?

### Week 3 — The reference engine: `backends/base.py` (64) + `backends/dense.py` (155)
**Claim.** One `eigh(G)` yields ρ, ln ρ, √ρ, log Z, Φ, every gradient and every metric —
and never overflows.

**Verification.** Build a Hamiltonian with `λ_min ≈ -30` and compare `expm(-G)/Z` against
the eigendecomposition route. Watch the first overflow and the second not.

**Questions.** Which `ThermalState` methods are *primitive* and which are derived? Why is
`diagonal_gradient` optimised to contract only the diagonal instead of forming `V D V†`?

### Week 4 — Geometry: `metrics/monotone.py` (140) + `metrics/base.py` (153)
**Claim.** Every monotone metric is one weighted contraction, `g_ij = Σ_kl W_kl
Re[(∂_iρ)_kl conj((∂_jρ)_kl)]`, with `W = 1/c` the Morozova–Chentsov function — so the
whole family is one kernel, not three special cases.

**Verification.** Evaluate `AlphaZ(α,z)` at (α→1), (½,½), (½,1) and confirm they equal
Kubo–Mori, Fisher–Bures and Wigner–Yanase to machine precision. Then check the Löwner
ordering between them.

**Questions.** What is `_POP_FLOOR = 1e-300` defending against — construct the low-temperature
state that produces `NaN` without it. Where is the data-processing region enforced, and
what happens outside it? (arXiv:2510.02218 Thm 10, Fact 9.)

---

## Phase 2 · The learning layer (weeks 5–8)

### Week 5 — Objectives I: `losses/base.py` (23) + `energy.py` (24) + `relative_entropy.py` (167)
**Claim.** `∂ⱼ D(σ‖ρ) = ⟨Gⱼ⟩_σ − ⟨Gⱼ⟩_ρ` — exact even when the generators do not commute,
because σ is fixed.

**Verification.** Check that gradient against finite differences for a deliberately
non-commuting generator set.

**Questions.** Why is `RelativeEntropy(q)` *not* the measured-distribution likelihood
for non-commuting generators, and which class is? Why does `value` need `log Z` while
`grad` does not — and which backends does that exclude? The target side `⟨Gⱼ⟩_σ` is
computed from the Pauli *labels*: why does one Walsh–Hadamard transform of `q` give every
`Z`-string moment of a classical target at once, and what would be built instead if the
loss read `ham.generators`?

### Week 6 — Objectives II: `likelihood.py` (75) + `free_energy.py` (32) + `sdp.py` (69)
**Claim.** Entropy-regularised SDP duality: the dual gradient is the constraint violation
`b_i − ⟨A_i, X⟩`, and natural gradient with `lr = β` is exactly Newton on the dual.

**Verification.** Run `qbm.solve_sdp` on a small program and check strong duality against
`tests/test_sdp.py`'s independent scipy reference.

**Questions.** Why does `MarginalNLL.grad` need `diagonal_gradient`, and what does that
cost you on the scalable backends? (Week 15 is the answer.)

### Week 7 — Optimisation: `optim/` (116) + `train/loop.py` (103)
**Claim.** Geometry-aware and plain optimisers share one loop because the optimiser
receives the *state*, not just the gradient.

**Verification.** Reproduce the documented result that Adam *worsens* KL on a stochastic
gradient where plain SGD improves it.

**Questions.** Why does `NaturalGradient` validate metric finiteness and raise
`FloatingPointError` rather than letting `lstsq` return garbage? What do `monitor=` and
`stop=` exist for — which backend limitation forced each?

### Week 8 — Models: `models/base.py` (59) + `fully_visible.py` (64) + `visible_hidden.py` (76)
**Claim.** A model is a parameter vector plus a backend; every read-out is delegated.

**Questions.** Why does `qbm.learn` use a small random init rather than θ=0? Construct the
flip-symmetric target where θ=0 is a saddle with an exactly-zero gradient. A
`VisibleHiddenQBM` with the default `visible_paulis=("Z",)` has a *diagonal* visible
reduced state whatever its hidden units do — prove it from the block structure of `G`,
then check it numerically and watch it fail for `visible_paulis=("Z", "X")`.

---

## Phase 3 · Variants and the public surface (weeks 9–11)

### Week 9 — `models/sqrbm.py` (130)
**Claim.** For a semi-quantum RBM the hidden units decouple, so the visible marginal has
a **closed form** in cosh/tanh — no Gibbs-state diagonalisation, cost independent of the
hidden count.

**Verification.** Compare `SemiQuantumRBM.visible_probabilities()` against the dense Gibbs
state of `to_hamiltonian()`. They must agree to ~1e-16.

**Questions.** Where exactly does the `2cosh(‖Φ‖)` come from? (Re-derive it: it is the
same trace that reappears in week 15.)

### Week 10 — `models/evolved.py` (165)
**Claim.** Real-time evolution on top of the Gibbs state adds expressivity that no θ can
reach, and the (θ,φ) QFI is block-structured. (arXiv:2501.03367 — your own paper.)

**Verification.** `tests/reproductions/test_evolved_expressivity.py`.

**Questions.** Why does `∂exp(-iH)` need the Daleckii–Krein kernel rather than a naive
product rule?

### Week 11 — The public surface: `tasks/` (397) + `facade.py` (137) + `registry.py` (140)
**Claim.** Every research question is one call, and every default is overridable.

**Questions.** How does the registry avoid importing JAX at `import qbm` time — and which
test enforces that it stays that way? Why does `learn_state` switch to a fully quantum
RBM as soon as `n_hidden > 0` (week 8 has the answer), and what is the best relative
entropy the semi-quantum one could reach on a target with coherences? Why does the KL
monitor of `qbm.learn` read the state the step already built instead of calling
`model.kl`?

---

## Phase 4 · Scalable backends (weeks 12–15)

### Week 12 — `backends/statevector.py` (75) + `purification.py` (45)
**Claim.** `|TFD⟩ = Σₖ √pₖ |vₖ⟩|k⟩` purifies ρ, and its system–ancilla entanglement
entropy *is* the thermal entropy.

**Verification.** Confirm `entanglement_entropy(tfd) == S(ρ)` numerically.

### Week 13 — `backends/jax_backend.py` (179) + `autodiff.py` (54)
**Claim.** Autodiff reproduces the analytic engine to ~1e-15 — an independent check on
every formula in Phase 1.

**Questions.** Why does `eigh`'s VJP produce `NaN` at degenerate eigenvalues, and what
does the analytic fallback do? (This was a real CI-only bug — LAPACK-dependent.)

### Week 14 — `backends/tensor_network.py` (240)
**Claim.** A purified MPS reaches 20 qubits at bond dimension 4 where a dense ρ would
need ~17 TB.

**Verification.** Run `examples/06_tensor_network_scaling.py`: its last part trains a
16-qubit model and recovers the couplings of the Ising chain that generated the data.
Confirm that `model.ham._mats` is still `None` afterwards.

**Questions.** Why are only 1- and 2-body Pauli generators supported? Why does this
backend refuse metrics rather than approximating them? At 16 qubits `history.loss` is
all `nan` and `history.monitor` is empty — which two quantities are missing, and what is
the training curve you still have?

### Week 15 — `sampling.py` (133) + `gibbs_map.py` (237)
**Claim.** With a diagonal visible register `G = ⊕ᵥ G_h(v)`, so the hidden register can be
traced out **exactly**; block-Gibbs CD instead *samples* it and is therefore biased for
non-commuting hidden operators.

**Verification.** `examples/10_gibbs_map_hidden_units.py` — CD lands at TVD ≈ 0.26, the
Gibbs map at ≈ 0.003, the i.i.d. floor being ≈ 0.003.

**Questions.** Re-derive `∂ⱼL = Σᵥ q(v)⟨Gⱼ⟩_{σᵥ} − ⟨Gⱼ⟩_ρ`. Which identity makes it exact
for non-commuting `G`? Why is the positive phase independent of the *visible* count but
exponential in the *hidden* count?

---

## Phase 5 · Circuits (weeks 16–19)

### Week 16 — `circuits/ir.py` (130) + `circuits/simulator.py` (108)
**Claim.** The algorithms live in our own IR, so no vendor SDK is a dependency.

**Questions.** In `_apply`, why is `axes = list(qubits)` correct and `[n-1-q ...]` wrong —
and why did the wrong version still pass the Hadamard test? (A consistent permutation
leaves a trace invariant. This was a real bug.)

### Week 17 — `circuits/builder.py` (150) + `circuits/densities.py` (110)
**Claim.** Sampling `t ~ p(t)` realises the same `Φ` the dense backend computes exactly.

**Questions.** Why does the inverse-CDF sampler need a geometric grid near `t=0`? Why is
`p_{α,z}` only defined for α ∈ (0,1)?

### Week 18 — `circuits/estimators.py` (127)
**Claim.** Both the energy gradient and the α-z information matrix are one shape: a
covariance of two operators, one smeared by `Φ`.

### Week 19 — `circuits/varqite.py` (742) — *the biggest module; give it two sittings*
**Claim.** McLachlan's principle on the TFD gives `A λ̇ = C` with `A` the quantum geometric
tensor and `C = −½∇⟨H⟩` — i.e. **quantum natural gradient flow**.

**Verification.** Confirm the *tilt partner* identity `K|TFD(0)⟩ = −i(P⊗I)|TFD(0)⟩` for
`Z → Y⊗X`, then that the Hadamard-test route reproduces the exact route to ~1e-15.

**Questions.** Why does the ansatz branch on *commutation*? Why is the McLachlan residual
measurable when fidelity is not — and why does it cross 0.1 *before* the energy diverges?

---

## Phase 6 · Pauli propagation and closing (weeks 20–24)

### Week 20 — `pauli_prop.py` part 1: algebra + ITE (lines 1–~250)
**Claim.** A Pauli string is a signed permutation `(x,z)`; the imaginary-time gate
branches only on **commutation**, `P → cosh(2θ)P − sinh(2θ)PG`.

**Verification.** Re-derive the product phase `i^{eP+eQ-eR}(-1)^{⟨zP,xQ⟩}` and check it
against dense matrices. Confirm the ITE is *exact* for a commuting Hamiltonian at L=1.

**Questions.** `thermal_state` truncates once per Trotter *layer*, not after every gate.
For a 10-qubit chain with `coeff_cutoff=1e-3, max_weight=3`, count the terms held inside
a layer and compare with the number retained at its end. What does the ratio imply for
the largest size this engine reaches, and what would truncating after every gate change —
in cost, and in the result?

### Week 21 — `pauli_prop.py` part 2: sampler (~250–385) + `backends/pauli_propagation.py` (201)
**Claim.** Truncation can push ρ out of the PSD cone; the locally normalised chain rule
still yields a valid distribution, with exact pointwise likelihoods.

**Questions.** Why do only diagonal strings contribute to a Z-basis marginal? Why is
coefficient truncation more effective than weight truncation on dense Hamiltonians?

### Week 22 — Adapters and periphery: `circuits/adapters/` (259) + `diagnostics.py` (68) + `data/` (135)
**Questions.** If Qiskit 3.0 breaks tomorrow, exactly which files change?

### Week 23 — The test suite: `tests/` (~3,900)
Read the tests as *specification*. For each of the seven tiers, find one test you could
break with a plausible bug — and one you could not.

**Questions.** Which behaviours have **no** test? That list is your highest-value backlog.

### Week 24 — Reconciliation
Re-read [`DESIGN.md`](../DESIGN.md) end to end against the code, and
[`paper/paper.md`](../paper/paper.md) against both. Fix every drift you find. Anything you
still cannot justify: rewrite it, or delete it.

---

## Tracking

| Phase | Weeks | Focus |
|---|---|---|
| 1 · Core | 1–4 | the object, the primitive, dense engine, metrics |
| 2 · Learning | 5–8 | losses, optimisers, training loop, models |
| 3 · Variants | 9–11 | sqRBM, evolved QBM, tasks and registry |
| 4 · Scaling | 12–15 | statevector, JAX, tensor network, Gibbs map |
| 5 · Circuits | 16–19 | IR, estimators, VarQITE |
| 6 · Pauli + close | 20–24 | propagation, adapters, tests, reconciliation |
