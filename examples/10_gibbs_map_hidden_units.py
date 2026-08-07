"""The Gibbs map: exact hidden-unit training, and unbiased sampling.

Two things block-Gibbs contrastive divergence could not do, because it samples the
hidden register instead of tracing it out:

  1. give an *unbiased* model sampler when the hidden operators do not commute;
  2. give a likelihood gradient on backends that cannot form ``d_j rho``.

Both follow from the same observation: with a diagonal visible register,
``G(theta) = (+)_v G_h(v)``, so the hidden units can be marginalised exactly.

Run:  python examples/10_gibbs_map_hidden_units.py
"""

import time

import numpy as np

import qbm
from qbm.gibbs_map import GibbsMap
from qbm.losses import GibbsMapNLL, MarginalNLL
from qbm.sampling import block_gibbs_sample

nv, nh = 3, 2
rng = np.random.default_rng(3)

# --- 1. the exact gradient, for non-commuting hidden operators ----------------
model = qbm.VisibleHiddenQBM(n_visible=nv, n_hidden=nh, hidden_paulis=("Z", "X"))
model.theta = rng.normal(scale=0.6, size=model.n_params)
q = rng.random(2**nv)
q /= q.sum()

exact = MarginalNLL(q, n_visible=nv).grad(model.state())  # dense Frechet-derivative route
gmap_grad = GibbsMapNLL(q, n_visible=nv).grad(model.state())
print("exact likelihood gradient (non-commuting Z/X hidden units):")
print(f"   max |GibbsMapNLL - MarginalNLL| = {np.max(np.abs(gmap_grad - exact)):.2e}")

# --- 2. it needs only <G_j>, so it runs on the scalable backends --------------
print("\nsame gradient, every backend (MarginalNLL cannot run on these at all):")
for tag, be in [
    ("dense", None),
    ("circuit", qbm.get_backend("circuit")),
    (
        "pauli_propagation",
        qbm.get_backend("pauli_propagation", trotter_steps=256, coeff_cutoff=0.0),
    ),
]:
    m = qbm.VisibleHiddenQBM(n_visible=nv, n_hidden=nh, hidden_paulis=("Z", "X"), backend=be)
    m.theta = model.theta
    g = GibbsMapNLL(q, n_visible=nv).grad(m.state())
    print(f"   {tag:20s} max |g - exact| = {np.max(np.abs(g - exact)):.2e}")

# --- 3. the sampler is unbiased where contrastive divergence is not -----------
sq = qbm.SemiQuantumRBM(n_visible=nv, n_hidden=nh, hidden_paulis=("X", "Z"))
sq.theta = np.random.default_rng(3).normal(scale=0.7, size=sq.n_params)
exact_marginal = sq.visible_probabilities()
gmap = GibbsMap(sq.to_hamiltonian(), n_visible=nv)
N = 40000


def tvd(p):
    return 0.5 * float(np.sum(np.abs(p - exact_marginal)))


vk = block_gibbs_sample(sq, rng.integers(0, 2, size=(N, nv)), k=40, rng=np.random.default_rng(0))
p_cd = np.bincount(vk @ (1 << np.arange(nv - 1, -1, -1)), minlength=2**nv) / N
p_map = (
    np.bincount(gmap.sample(sq.theta, N, sweeps=12, rng=np.random.default_rng(0)), minlength=2**nv)
    / N
)
iid = np.bincount(np.random.default_rng(5).choice(2**nv, N, p=exact_marginal), minlength=2**nv) / N

print(f"\nsampling a NON-commuting sqRBM ({N} samples):")
print(f"   block-Gibbs CD chain (qbm.sampling)  TVD = {tvd(p_cd):.4f}   <- biased")
print(f"   Gibbs-map chain      (qbm.gibbs_map) TVD = {tvd(p_map):.4f}")
print(f"   i.i.d. finite-sample floor           TVD = {tvd(iid):.4f}   <- the Gibbs map hits it")

# --- 4. train a hidden-unit QBM on a scalable backend -------------------------
target = qbm.VisibleHiddenQBM(n_visible=nv, n_hidden=nh, hidden_paulis=("Z", "X"))
target.theta = np.random.default_rng(4).normal(scale=0.6, size=target.n_params)
qt = target.state().probabilities().reshape(2**nv, 2**nh).sum(axis=1)

be = qbm.get_backend("pauli_propagation", trotter_steps=48, coeff_cutoff=1e-8)
m = qbm.VisibleHiddenQBM(n_visible=nv, n_hidden=nh, hidden_paulis=("Z", "X"), backend=be)
m.theta = np.random.default_rng(0).normal(scale=0.05, size=m.n_params)
loss = GibbsMapNLL(qt, n_visible=nv)
t0 = time.time()
history = qbm.fit(m, loss, qbm.optim.Adam(lr=0.12), steps=80)
learned = loss.marginal(m.state())
kl = float(np.sum(qt * (np.log(qt) - np.log(np.clip(learned, 1e-300, None)))))

print("\nhidden-unit QBM trained through Pauli propagation:")
print(
    f"   NLL {history.loss[0]:.4f} -> {history.loss[-1]:.4f}   KL -> {kl:.2e}   "
    f"({(time.time() - t0) / 80 * 1000:.0f} ms/step)"
)
print("   (the NLL *value* is available too: the Gibbs map supplies log Z itself)")
