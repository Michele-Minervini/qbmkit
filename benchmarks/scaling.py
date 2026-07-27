"""Dense-backend scaling benchmark: time and memory vs qubit count.

Measures the cost of the core operations -- building the thermal state
(eigendecomposition), an observable gradient, and a Kubo-Mori metric -- as the
number of qubits grows, and reports the empirical scaling exponent.  The dense
backend is exact and O(4^n) in memory / O(8^n) in time, so this also documents the
practical ceiling (~12-13 qubits on a workstation).

Run:  python benchmarks/scaling.py            # table + benchmarks/scaling.{csv,png}
"""

from __future__ import annotations

import csv
import time
import tracemalloc

import numpy as np

import qbm
from qbm.operators import ParamHamiltonian, local_pauli_generators


def measure(n: int, seed: int = 0) -> dict:
    """Time (seconds) and peak memory (MB) of the core operations at ``n`` qubits."""
    ham = ParamHamiltonian(local_pauli_generators(n))
    theta = np.random.default_rng(seed).normal(scale=0.4, size=ham.n_params)
    O = qbm.hamiltonians.tfim(n, g=1.0)
    backend = qbm.DenseBackend()

    tracemalloc.start()
    t0 = time.perf_counter()
    state = backend.thermal_state(ham, theta)
    t_state = time.perf_counter() - t0

    t0 = time.perf_counter()
    state.observable_gradient(O)
    t_grad = time.perf_counter() - t0

    t0 = time.perf_counter()
    state.metric("kubo_mori")
    t_metric = time.perf_counter() - t0

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "n": n,
        "dim": ham.dim,
        "n_params": ham.n_params,
        "t_state": t_state,
        "t_grad": t_grad,
        "t_metric": t_metric,
        "t_total": t_state + t_grad + t_metric,
        "peak_mb": peak / 1e6,
    }


def run_scaling(sizes) -> list[dict]:
    return [measure(n) for n in sizes]


def _fit_exponent(sizes, values):
    """Fit ``values ~ 2^(alpha * n)`` and return alpha (the base-2 scaling exponent)."""
    sizes = np.asarray(sizes, dtype=float)
    values = np.asarray(values, dtype=float)
    good = values > 0
    return float(np.polyfit(sizes[good], np.log2(values[good]), 1)[0])


def main():
    sizes = list(range(2, 11))
    rows = run_scaling(sizes)

    cols = ("n", "dim", "params", "t_state", "t_grad", "t_metric", "peak_MB")
    header = f"{cols[0]:>3} {cols[1]:>6} {cols[2]:>7} " + " ".join(f"{c:>9}" for c in cols[3:])
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['n']:>3} {r['dim']:>6} {r['n_params']:>7} "
            f"{r['t_state']:>9.4f} {r['t_grad']:>9.4f} {r['t_metric']:>9.4f} {r['peak_mb']:>9.2f}"
        )

    a_time = _fit_exponent(sizes, [r["t_total"] for r in rows])
    a_mem = _fit_exponent(sizes, [r["peak_mb"] for r in rows])
    print(f"\nempirical scaling:  time ~ 2^({a_time:.2f} n)   memory ~ 2^({a_mem:.2f} n)")
    print("(dense is O(8^n)=2^(3n) time, O(4^n)=2^(2n) memory in the asymptotic limit)")

    with open("benchmarks/scaling.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("wrote benchmarks/scaling.csv")

    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
        ns = [r["n"] for r in rows]
        for key, label in [
            ("t_state", "thermal state"),
            ("t_grad", "gradient"),
            ("t_metric", "metric"),
        ]:
            ax1.semilogy(ns, [r[key] for r in rows], "o-", label=label)
        ax1.set_xlabel("qubits n")
        ax1.set_ylabel("time (s)")
        ax1.set_title("time vs system size")
        ax1.legend()
        ax1.grid(alpha=0.3, which="both")
        ax2.semilogy(ns, [r["peak_mb"] for r in rows], "o-", color="#C44E52")
        ax2.set_xlabel("qubits n")
        ax2.set_ylabel("peak memory (MB)")
        ax2.set_title("memory vs system size")
        ax2.grid(alpha=0.3, which="both")
        plt.tight_layout()
        plt.savefig("benchmarks/scaling.png", dpi=110)
        print("wrote benchmarks/scaling.png")
    except ImportError:
        print("(install matplotlib for the plot)")


if __name__ == "__main__":
    main()
