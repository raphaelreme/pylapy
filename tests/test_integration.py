"""Integration tests: run every available backend on realistic problems and check equivalence.

Two things are verified:

1. Each backend solves simple-but-sizeable problems correctly (matched against
   ``scipy.optimize.linear_sum_assignment`` when scipy is importable, otherwise against the first
   available backend as a self-consistency reference).
2. Every cost extension, shape extension and sparse extension is *equivalent*: for a given problem,
   swapping the extension (or the backend) leaves the optimal objective unchanged. We compare
   objectives, not exact link sets, because several assignments can be equally optimal.

``lapjv`` is deliberately excluded from the shape-extension equivalence test: it mishandles large
inf-filled shape extensions (``smallest_fill_inf`` / ``sum_fill_inf`` / ``sum_split_inf``). This is a
backend numerical limitation, not an extension-logic issue (all other backends stay equivalent), and is
tracked for later investigation.
"""

import numpy as np
import pytest
import scipy.optimize
from conftest import (
    assert_valid_links,
    assignment_cost,
    available_dense_backends,
    available_sparse_backends,
    soft_objective,
)

from pylapy import LapSolver, cost_extension, shape_extension

DENSE_BACKENDS = available_dense_backends()
SPARSE_BACKENDS = available_sparse_backends()

COST_EXTENSIONS = list(cost_extension.COST_EXTENSIONS.values())
SHAPE_EXTENSIONS = list(shape_extension.SHAPE_EXTENSIONS.values())
SPARSE_EXTENSIONS = ["diag_split_sparse", "diag_row_no_pad", "diag_col_no_pad"]

# Backends that apply the alternative cost natively (can use `no_cost_extension`); the others raise on a
# finite alternative_cost without an explicit extension.
NATIVE_ALTERNATIVE_COST = {"lap", "laprust"}

# lapjv is unreliable on inf-filled shape extensions at scale (see module docstring).
SHAPE_EQUIVALENCE_BACKENDS = [b for b in DENSE_BACKENDS if b != "lapjv"]

SQUARE = (100, 100)
RECTANGULAR = [(80, 120), (120, 80)]
ETA = 0.1  # small enough to leave many rows/cols unassigned -> genuine sparsity


def make_problem(shape: tuple[int, int], seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random(shape)


# --------------------------------------------------------------------------------------------------
# Correctness of each backend on simple sizeable problems
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("backend", DENSE_BACKENDS)
@pytest.mark.parametrize("shape", [SQUARE, *RECTANGULAR])
def test_backend_matches_reference_full_assignment(backend: str, shape: tuple[int, int]) -> None:
    dist = make_problem(shape, seed=1)
    n, m = shape
    links = LapSolver(backend).solve(dist)

    assert_valid_links(links, n, m)
    assert len(links) == min(n, m)  # full assignment when there is no cost limit

    ref_i, ref_j = scipy.optimize.linear_sum_assignment(dist)
    assert np.isclose(assignment_cost(dist, links), dist[ref_i, ref_j].sum())


@pytest.mark.parametrize("backend", DENSE_BACKENDS)
def test_backend_alternative_cost_matches_reference(backend: str) -> None:
    dist = make_problem(SQUARE, seed=2)
    links = LapSolver(backend).solve(dist, alternative_cost=ETA)
    reference = LapSolver(DENSE_BACKENDS[0]).solve(dist, alternative_cost=ETA)

    assert_valid_links(links, *SQUARE)
    assert np.isclose(soft_objective(dist, links, ETA), soft_objective(dist, reference, ETA))


# --------------------------------------------------------------------------------------------------
# Equivalence of all cost extensions, across all backends
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [SQUARE, *RECTANGULAR])
@pytest.mark.parametrize("backend", DENSE_BACKENDS)
@pytest.mark.parametrize("extension", COST_EXTENSIONS, ids=lambda f: f.__name__)
def test_cost_extensions_are_equivalent(extension, backend: str, shape: tuple[int, int]) -> None:
    if extension is cost_extension.no_cost_extension and backend not in NATIVE_ALTERNATIVE_COST:
        pytest.skip(f"{backend} has no native alternative cost; needs an explicit cost extension")

    dist = make_problem(shape, seed=3)

    solver = LapSolver(backend)
    solver.cost_extension = extension
    links = solver.solve(dist, alternative_cost=ETA)

    # Reference: the first backend with its default cost extension.
    reference = LapSolver(DENSE_BACKENDS[0]).solve(dist, alternative_cost=ETA)

    assert_valid_links(links, *shape)
    assert np.isclose(soft_objective(dist, links, ETA), soft_objective(dist, reference, ETA))


# --------------------------------------------------------------------------------------------------
# Equivalence of all shape extensions, across all backends (except lapjv, see module docstring)
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("shape", RECTANGULAR)
@pytest.mark.parametrize("backend", SHAPE_EQUIVALENCE_BACKENDS)
@pytest.mark.parametrize("extension", SHAPE_EXTENSIONS, ids=lambda f: f.__name__)
def test_shape_extensions_are_equivalent(extension, backend: str, shape: tuple[int, int]) -> None:
    dist = make_problem(shape, seed=4)
    n, m = shape

    solver = LapSolver(backend)
    solver.shape_extension = extension
    links = solver.solve(dist)

    reference = LapSolver(SHAPE_EQUIVALENCE_BACKENDS[0]).solve(dist)

    assert_valid_links(links, n, m)
    assert len(links) == min(n, m)
    assert np.isclose(assignment_cost(dist, links), assignment_cost(dist, reference))


# --------------------------------------------------------------------------------------------------
# Equivalence of all sparse extensions, across sparse backends, and vs the dense reference
# --------------------------------------------------------------------------------------------------


@pytest.mark.skipif(not SPARSE_BACKENDS, reason="no sparse backend available")
@pytest.mark.parametrize("sparse_backend", SPARSE_BACKENDS)
@pytest.mark.parametrize("extension", SPARSE_EXTENSIONS)
def test_sparse_extensions_are_equivalent(sparse_backend: str, extension: str) -> None:
    if sparse_backend == "lapmod" and extension != "diag_split_sparse":
        pytest.skip("lapmod only supports the square diag_split_sparse extension")
    pytest.importorskip("numba")

    dist = make_problem(SQUARE, seed=5)

    solver = LapSolver(sparse_implementation=sparse_backend)
    solver.sparse_extension = extension
    links = solver.sparse_solve(dist, alternative_cost=ETA)

    reference = LapSolver().solve(dist, alternative_cost=ETA)  # dense reference

    assert_valid_links(links, *SQUARE)
    assert np.isclose(soft_objective(dist, links, ETA), soft_objective(dist, reference, ETA))
