"""Registry wiring: every built-in is addressable by name and constructible."""

import numpy as np
import pytest

import qbm


def test_all_kinds_populated():
    reg = qbm.available()
    assert set(reg) == {"model", "loss", "metric", "optimizer", "backend", "task"}
    for kind, names in reg.items():
        assert names, f"registry {kind!r} is empty"


@pytest.mark.parametrize("name", ["kubo_mori", "fisher_bures", "wigner_yanase"])
def test_build_metric(name):
    m = qbm.build("metric", name)
    assert m.kind == name


@pytest.mark.parametrize("name", ["gradient_descent", "sgd", "adam", "natural_gradient"])
def test_build_optimizer(name):
    opt = qbm.build("optimizer", name, lr=0.1)
    assert hasattr(opt, "step")


@pytest.mark.parametrize("name", ["dense", "statevector"])
def test_build_backend(name):
    b = qbm.build("backend", name)
    assert hasattr(b, "thermal_state")


def test_registered_names_match_classes():
    assert qbm.registry.get("model", "fully_visible") is qbm.FullyVisibleQBM
    assert qbm.registry.get("loss", "energy") is qbm.losses.Energy
    assert qbm.registry.get("task", "ground_state") is qbm.ground_state


def test_user_can_register_a_plugin():
    @qbm.register("loss", "double_energy")
    class DoubleEnergy(qbm.losses.Energy):
        def value(self, state):
            return 2 * super().value(state)

    H = qbm.hamiltonians.tfim(2, g=1.0)
    loss = qbm.build("loss", "double_energy", H)
    model = qbm.FullyVisibleQBM(n=2)
    assert np.isclose(loss.value(model.state()), 2 * qbm.losses.Energy(H).value(model.state()))


def test_unknown_names_and_kinds_raise():
    with pytest.raises(KeyError):
        qbm.build("loss", "does_not_exist")
    with pytest.raises(ValueError):
        qbm.available("not_a_kind")
