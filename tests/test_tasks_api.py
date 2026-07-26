"""The task layer: every research question is one call, and each converges."""

import numpy as np
import pytest

import qbm


def test_ground_state_task():
    H = qbm.hamiltonians.tfim(3, J=1.0, g=1.2)
    res = qbm.ground_state(H, steps=300)
    assert res.task == "ground_state"
    assert res.error >= -1e-9  # variational: cannot go below E0
    assert res.error < 0.05
    assert np.isclose(res.energy, res.exact_energy + res.error)
    assert "ground_state result" in res.report()


def test_learn_state_task_reaches_zero_on_representable_target():
    # default generator pool is 2-local, so a 2-local Gibbs target is representable
    sigma = qbm.oracles.gibbs(qbm.hamiltonians.heisenberg(3), beta=0.8)
    res = qbm.learn_state(sigma, steps=300)
    assert res.relative_entropy < 1e-8


def test_learn_state_with_hidden_units_runs():
    src = qbm.VisibleHiddenQBM(n_visible=2, n_hidden=2)
    src.theta = np.random.default_rng(5).normal(scale=0.5, size=src.n_params)
    from qbm.linalg import partial_trace_hidden

    target = partial_trace_hidden(src.state().density_matrix(), 2, 2)
    res = qbm.learn_state(target, n_hidden=2, steps=200, optimizer=qbm.optim.Adam(lr=0.1))
    assert res.relative_entropy < res.history.loss[0]


def test_free_energy_task_reaches_exact_value():
    H = qbm.hamiltonians.tfim(3, J=1.0, g=1.0)
    res = qbm.free_energy_min(H, temperature=1.0, steps=400)
    assert res.error > -1e-8  # variational bound: F >= F_exact
    assert abs(res.error) < 1e-2


def test_sdp_task_fields():
    C = np.diag([2.0, 1.0, 0.0, -1.0]).astype(complex)
    res = qbm.solve_sdp(C, beta=30.0, steps=10)
    assert res.task == "sdp"
    assert res.X.shape == (4, 4)
    assert abs(res.objective - 2.0) < 1e-2
    assert res.constraint_violation == 0.0  # unconstrained


def test_generative_task_still_works():
    q = qbm.datasets.parity(n=3)
    model = qbm.learn(q, steps=200)
    assert model.kl(q) < model.history.loss[0]


@pytest.mark.parametrize("task_name", ["ground_state", "state_learning", "free_energy_min", "sdp"])
def test_tasks_are_registered(task_name):
    assert task_name in qbm.available("task")
    assert callable(qbm.registry.get("task", task_name))


def test_result_report_is_informative():
    H = qbm.hamiltonians.tfim(2, g=1.0)
    res = qbm.ground_state(H, steps=50)
    text = res.report()
    for field in ("steps", "final loss", "energy", "exact_energy"):
        assert field in text
