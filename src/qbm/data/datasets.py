"""Benchmark classical distributions, returned as probability vectors of length 2^n.

Bit ordering matches the rest of the library: qubit 0 is the most significant
bit of the computational-basis index (see DESIGN.md, Section 3).
"""

from __future__ import annotations

import itertools

import numpy as np


def _uniform_over(indices: list[int], dim: int) -> np.ndarray:
    q = np.zeros(dim)
    idx = np.unique(np.asarray(indices, dtype=int))
    q[idx] = 1.0 / len(idx)
    return q


def bars_and_stripes(grid: int = 3) -> np.ndarray:
    """Bars-and-stripes distribution on a ``grid x grid`` image (``n = grid^2`` qubits).

    A valid image has either all rows constant ("bars") or all columns constant
    ("stripes"); the all-0 and all-1 images belong to both.  Uniform over the set.
    """
    n = grid * grid
    dim = 1 << n

    def cell_bit(i, j):  # qubit index for cell (row i, col j); qubit 0 is MSB
        return (i * grid + j)

    def index_of(image):  # image[i][j] in {0,1} -> basis index
        v = 0
        for i in range(grid):
            for j in range(grid):
                if image[i][j]:
                    v |= 1 << (n - 1 - cell_bit(i, j))
        return v

    indices: list[int] = []
    for bits in itertools.product([0, 1], repeat=grid):  # bars: each row constant
        indices.append(index_of([[bits[i]] * grid for i in range(grid)]))
    for bits in itertools.product([0, 1], repeat=grid):  # stripes: each col constant
        indices.append(index_of([[bits[j] for j in range(grid)] for _ in range(grid)]))
    return _uniform_over(indices, dim)


def parity(n: int = 4, even: bool = True) -> np.ndarray:
    """Uniform distribution over bitstrings of even (or odd) parity."""
    dim = 1 << n
    target = 0 if even else 1
    indices = [v for v in range(dim) if bin(v).count("1") % 2 == target]
    return _uniform_over(indices, dim)


def cardinality(n: int = 4, k: int | None = None) -> np.ndarray:
    """Uniform distribution over bitstrings of fixed Hamming weight ``k`` (default n//2)."""
    if k is None:
        k = n // 2
    dim = 1 << n
    indices = [v for v in range(dim) if bin(v).count("1") == k]
    return _uniform_over(indices, dim)
