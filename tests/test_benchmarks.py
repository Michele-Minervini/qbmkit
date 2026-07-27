"""Smoke test for the scaling benchmark (keeps it runnable, not a performance gate)."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "benchmarks"))

import scaling  # noqa: E402


def test_measure_returns_finite_numbers():
    row = scaling.measure(3)
    assert row["dim"] == 8
    for key in ("t_state", "t_grad", "t_metric", "peak_mb"):
        assert row[key] >= 0
        assert row[key] < 1e6  # finite / sane


def test_run_scaling_and_exponent_fit():
    rows = scaling.run_scaling([2, 3, 4])
    assert [r["n"] for r in rows] == [2, 3, 4]
    # memory should grow with system size
    assert rows[-1]["peak_mb"] >= rows[0]["peak_mb"]
    alpha = scaling._fit_exponent([r["n"] for r in rows], [r["dim"] ** 2 for r in rows])
    assert 1.9 < alpha < 2.1  # dim^2 = 4^n = 2^(2n) exactly
