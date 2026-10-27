"""Shared helpers and fixtures for the pylapy test suite.

The equivalence tests rely on the fact that, while several assignments can be optimal (ties),
the *objective* of an assignment problem is unique. We therefore compare costs/objectives rather
than exact link sets so that two equally-optimal assignments do not trigger false failures.
"""

import numpy as np
import pytest

from pylapy import LapSolver


def available_dense_backends() -> list[str]:
    """Return the dense backends that can actually be constructed on this host.

    A backend whose optional dependency is not importable is skipped, so the suite runs anywhere.
    """
    available = []
    for name in LapSolver.implementations:
        try:
            LapSolver(name)
        except ImportError:
            continue
        available.append(name)
    return available


def available_sparse_backends() -> list[str]:
    """Return the sparse backends that can actually be constructed on this host."""
    available = []
    for name in LapSolver.sparse_implementations:
        try:
            LapSolver(sparse_implementation=name)
        except ImportError:
            continue
        available.append(name)
    return available


def make_dist(n: int, m: int) -> np.ndarray:
    """Distinct strictly-positive values so no cell collides with 0 or INF."""
    return (np.arange(1, n * m + 1, dtype=np.float64)).reshape(n, m)


def assignment_cost(dist: np.ndarray, links: np.ndarray) -> float:
    """Sum of ``dist[i, j]`` over the returned links."""
    if len(links) == 0:
        return 0.0
    return float(dist[links[:, 0], links[:, 1]].sum())


def soft_objective(dist: np.ndarray, links: np.ndarray, eta: float) -> float:
    """Canonical soft-threshold objective.

    Each made link costs ``dist[i, j]``; each row and each column left unlinked costs ``eta / 2``.
    This value is invariant across every (equivalent) cost extension and backend, which is what the
    equivalence tests assert.
    """
    n, m = dist.shape
    linked = len(links)
    return assignment_cost(dist, links) + eta * ((n - linked) + (m - linked)) / 2


def assert_valid_links(links: np.ndarray, n: int, m: int) -> None:
    """Assert the returned links form a valid partial assignment of an ``(n, m)`` problem."""
    assert links.dtype == np.uint16
    assert links.ndim == 2
    assert links.shape[1] == 2
    if len(links) == 0:
        return

    rows = links[:, 0]
    cols = links[:, 1]
    assert rows.max() < n
    assert cols.max() < m
    assert len(np.unique(rows)) == len(rows)  # each row linked at most once
    assert len(np.unique(cols)) == len(cols)  # each column linked at most once


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministically seeded random generator."""
    return np.random.default_rng(0)
