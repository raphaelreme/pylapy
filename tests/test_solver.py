"""Unit / functional tests for the ``LapSolver`` public API (dense ``solve`` and sparse ``sparse_solve``).

The ``cost_limit`` / ``alternative_cost`` model (see ``LapSolver``):

- ``alternative_cost`` is the former soft threshold ``eta``: a node may be left unmatched at that cost.
  Its canonical objective is ``soft_objective`` (link cost + eta per unlinked row and column), invariant
  across backends / extensions, so equivalence is asserted on that objective, never on exact link sets.
- ``cost_limit`` is a hard discard: no returned link may cost more than it. There the invariant is the
  (maximum) number of links and their total cost, which we compare across backends.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.optimize
from conftest import (
    assert_valid_links,
    assignment_cost,
    available_dense_backends,
    available_sparse_backends,
    make_dist,
    soft_objective,
)

from pylapy import LapSolver, cost_extension, shape_extension

DENSE_BACKENDS = available_dense_backends()
SPARSE_BACKENDS = available_sparse_backends()

# Per-backend default cost extension (see cost_extension.DEFAULT_COST_EXTENSIONS).
COST_DEFAULTS = {
    "scipy": cost_extension.diag_row_pad_zero,
    "lap": cost_extension.diag_row_pad_zero,
    "lapjv": cost_extension.diag_row_pad_zero,
    "lapsolver": cost_extension.diag_split_sparse,
    "laprust": cost_extension.no_cost_extension,
}
# Per-backend default sparse extension name (see solver.LapSolver.__init__).
SPARSE_DEFAULTS = {"csgraph": "diag_row_no_pad", "lapmod": "diag_split_sparse"}

# Backends that apply the alternative cost natively; the others raise without an explicit cost extension.
NATIVE_ALTERNATIVE_COST = {"lap", "laprust"}
NON_NATIVE_BACKENDS = [b for b in DENSE_BACKENDS if b not in NATIVE_ALTERNATIVE_COST]


# --------------------------------------------------------------------------------------------------
# Construction / auto-detection / defaults
# --------------------------------------------------------------------------------------------------


def test_default_construction_picks_a_backend() -> None:
    solver = LapSolver()
    assert solver.implementation in LapSolver.implementations
    # A dense extension strategy is always configured (resolved to a callable).
    assert callable(solver.shape_extension)
    assert callable(solver.cost_extension)


def test_unknown_implementation_raises() -> None:
    with pytest.raises(KeyError):
        LapSolver("does-not-exist")


@pytest.mark.parametrize("backend", DENSE_BACKENDS)
def test_all_backends_default_to_native_shape(backend: str) -> None:
    # Every lapx/scipy backend handles rectangular input natively now, so the default is the no-op.
    assert LapSolver(backend).shape_extension is shape_extension.no_shape_extension


@pytest.mark.parametrize("backend", DENSE_BACKENDS)
def test_cost_extension_default_per_backend(backend: str) -> None:
    assert LapSolver(backend).cost_extension is COST_DEFAULTS[backend]


def test_extensions_settable_by_registry_name() -> None:
    # Extensions accept either a callable or a registry name (str), resolved lazily by the property.
    solver = LapSolver(DENSE_BACKENDS[0])
    solver.shape_extension = "smallest_fill_zero"
    solver.cost_extension = "diag_row_pad_zero"
    assert solver.shape_extension is shape_extension.smallest_fill_zero
    assert solver.cost_extension is cost_extension.diag_row_pad_zero

    # And the named extensions still solve correctly.
    dist = np.random.default_rng(0).random((6, 9))
    links = solver.solve(dist, alternative_cost=0.3)
    assert_valid_links(links, 6, 9)


@pytest.mark.skipif(not SPARSE_BACKENDS, reason="no sparse backend available")
@pytest.mark.parametrize("sparse_backend", SPARSE_BACKENDS)
def test_sparse_extension_default_per_backend(sparse_backend: str) -> None:
    import pylapy.sparse  # noqa: PLC0415

    solver = LapSolver(sparse_implementation=sparse_backend)
    assert solver.sparse_extension is pylapy.sparse.SPARSE_EXTENSIONS[SPARSE_DEFAULTS[sparse_backend]]

    solver.sparse_extension = "diag_col_no_pad"
    assert solver.sparse_extension is pylapy.sparse.diag_col_no_pad


# --------------------------------------------------------------------------------------------------
# Dense solve -- plain assignment
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("backend", DENSE_BACKENDS)
def test_solve_square_unique_optimum(backend: str) -> None:
    dist = np.array([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0], [2.0, 3.0, 1.0]])
    links = LapSolver(backend).solve(dist)
    # Identity is the unique optimum (cost 3).
    assert sorted(map(tuple, links.tolist())) == [(0, 0), (1, 1), (2, 2)]


@pytest.mark.parametrize("backend", DENSE_BACKENDS)
@pytest.mark.parametrize(("n", "m"), [(5, 8), (8, 5)])
def test_solve_rectangular_returns_min_dim_links(backend: str, n: int, m: int) -> None:
    dist = np.random.default_rng(1).random((n, m))
    links = LapSolver(backend).solve(dist)
    assert len(links) == min(n, m)
    assert_valid_links(links, n, m)

    ref_i, ref_j = scipy.optimize.linear_sum_assignment(dist)

    assert_valid_links(links, n, m)
    assert np.isclose(assignment_cost(dist, links), dist[ref_i, ref_j].sum())


# --------------------------------------------------------------------------------------------------
# Dense solve -- cost_limit and alternative_cost
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("backend", DENSE_BACKENDS)
def test_solve_alternative_cost_filter_above_eta(backend: str) -> None:
    dist = np.random.default_rng(3).random((20, 20))
    eta = 0.2
    links = LapSolver(backend).solve(dist, alternative_cost=eta)

    assert (dist[links[:, 0], links[:, 1]] <= eta).all()


@pytest.mark.parametrize("backend", DENSE_BACKENDS)
def test_solve_alternative_cost_drops_a_link_smaller_than_eta(backend: str) -> None:
    # Example where dropping the first row lead to a cost of 0, whereas including it
    # Leave us with a cost of 32. Therefore, with an alternative cost of 10
    # We may drop the first row despite the cost being only 5
    dist = np.array(
        [
            [5.0, 50.0, 50.0],
            [0.0, 7.0, 50.0],
            [25.0, 0.0, 50.0],
        ]
    )
    eta = 10.0
    links = LapSolver(backend).solve(dist, alternative_cost=eta)
    assert sorted(map(tuple, links.tolist())) == [(1, 0), (2, 1)]

    assert soft_objective(dist, links, eta) == eta


@pytest.mark.parametrize("backend", DENSE_BACKENDS)
def test_solve_alternative_cost_matches_reference(backend: str) -> None:
    dist = np.random.default_rng(3).random((20, 20))
    eta = 0.2
    links = LapSolver(backend).solve(dist, alternative_cost=eta)
    reference = LapSolver(DENSE_BACKENDS[0]).solve(dist, alternative_cost=eta)

    assert_valid_links(links, *dist.shape)
    assert np.isclose(soft_objective(dist, links, eta), soft_objective(dist, reference, eta))


@pytest.mark.parametrize("backend", DENSE_BACKENDS)
def test_solve_cost_limit_discards_above_limit(backend: str) -> None:
    dist = np.random.default_rng(4).random((15, 15))
    limit = 0.3
    links = LapSolver(backend).solve(dist, cost_limit=limit)

    assert_valid_links(links, *dist.shape)
    assert all(dist[i, j] <= limit for i, j in links.tolist())  # no discarded link returned

    # Maximum number of feasible links and minimal total cost are invariant across backends.
    reference = LapSolver(DENSE_BACKENDS[0]).solve(dist, cost_limit=limit)
    assert len(links) == len(reference)
    assert np.isclose(assignment_cost(dist, links), assignment_cost(dist, reference))


@pytest.mark.parametrize("backend", DENSE_BACKENDS)
def test_solve_cost_limit_hand_crafted(backend: str) -> None:
    dist = np.array([[0.1, 10.0], [10.0, 10.0]])
    links = LapSolver(backend).solve(dist, cost_limit=1.0)
    assert sorted(map(tuple, links.tolist())) == [(0, 0)]


@pytest.mark.parametrize("backend", DENSE_BACKENDS)
def test_solve_does_not_mutate_input(backend: str) -> None:
    # Mixes inf (replaced internally) and negatives (which drive the shift paths):
    # neither rewrite may leak back to the caller's array.
    dist = np.array([[1.0, np.inf, -2.0], [np.inf, -1.0, 3.0], [-3.0, 2.0, np.inf]])
    original = dist.copy()
    LapSolver(backend).solve(dist, cost_limit=5.0, alternative_cost=5.0)
    assert np.array_equal(dist, original, equal_nan=True)


# --------------------------------------------------------------------------------------------------
# Dense solve - inf / empty / all-inf edge case
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("backend", DENSE_BACKENDS)
def test_solve_filters_inf_links(backend: str) -> None:
    dist = np.array([[1.0, np.inf], [np.inf, np.inf]])
    links = LapSolver(backend).solve(dist)
    # Row 1 can only reach inf cells, so it stays unlinked; no inf link is ever returned.
    assert sorted(map(tuple, links.tolist())) == [(0, 0)]


@pytest.mark.parametrize("backend", DENSE_BACKENDS)
@pytest.mark.parametrize("shape", [(0, 5), (5, 0), (0, 0)])
def test_solve_empty_short_circuits(backend: str, shape: tuple[int, int]) -> None:
    links = LapSolver(backend).solve(np.zeros(shape))
    assert links.shape == (0, 2)
    assert links.dtype == np.uint16


@pytest.mark.parametrize("backend", DENSE_BACKENDS)
def test_solve_all_infeasible_returns_empty(backend: str) -> None:
    # A non-empty matrix whose every entry is infeasible: nothing can be linked.
    dist = np.full((4, 4), np.inf)
    links = LapSolver(backend).solve(dist)
    assert links.shape == (0, 2)
    assert links.dtype == np.uint16


# --------------------------------------------------------------------------------------------------
# Dense solve -- zero-cost links and negative costs edge cases
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("backend", DENSE_BACKENDS)
def test_solve_zero_cost_links(backend: str) -> None:
    # 0 is a perfectly valid cost: the zero diagonal
    # is the unique optimum (total cost 0).
    dist = np.array([[0.0, 5.0, 5.0], [5.0, 0.0, 5.0], [5.0, 5.0, 0.0]])
    links = LapSolver(backend).solve(dist)
    assert sorted(map(tuple, links.tolist())) == [(0, 0), (1, 1), (2, 2)]
    assert assignment_cost(dist, links) == 0.0


@pytest.mark.parametrize("backend", DENSE_BACKENDS)
def test_solve_negative_costs_match_scipy(backend: str) -> None:
    dist = np.array([[-1.0, -5.0, 2.0], [3.0, -2.0, -4.0], [-3.0, 1.0, -1.0]])
    links = LapSolver(backend).solve(dist)

    assert_valid_links(links, *dist.shape)
    assert len(links) == 3  # full assignment, no spurious filtering of (now valid) negative links
    ref_i, ref_j = scipy.optimize.linear_sum_assignment(dist)
    assert np.isclose(assignment_cost(dist, links), dist[ref_i, ref_j].sum())


@pytest.mark.parametrize("backend", DENSE_BACKENDS)
def test_solve_negative_costs_with_alternative_cost(backend: str) -> None:
    dist = np.random.default_rng(6).random((12, 12)) - 0.5  # roughly half negative
    eta = 0.1
    links = LapSolver(backend).solve(dist, alternative_cost=eta)
    reference = LapSolver(DENSE_BACKENDS[0]).solve(dist, alternative_cost=eta)

    assert_valid_links(links, *dist.shape)
    assert np.isclose(soft_objective(dist, links, eta), soft_objective(dist, reference, eta))


# --------------------------------------------------------------------------------------------------
# Error / contract paths
# --------------------------------------------------------------------------------------------------


@pytest.mark.skipif(not NON_NATIVE_BACKENDS, reason="all available backends apply alternative cost natively")
@pytest.mark.parametrize("backend", NON_NATIVE_BACKENDS)
def test_non_native_backend_requires_cost_extension(backend: str) -> None:
    # Without a cost extension, a backend lacking native alternative-cost support must raise (rather than
    # silently ignore the threshold).
    solver = LapSolver(backend)
    solver.cost_extension = cost_extension.no_cost_extension
    with pytest.raises(ValueError, match="alternative cost"):
        solver.solve(np.random.default_rng(0).random((5, 5)), alternative_cost=0.3)


def test_sparse_input_without_backend_raises() -> None:
    scipy_sparse = pytest.importorskip("scipy.sparse")
    solver = LapSolver()
    solver.sparse_implementation = None  # no sparse backend
    sparse_in = scipy_sparse.coo_array(np.eye(4))
    with pytest.raises(ValueError, match="convert to dense"):
        solver.sparse_solve(sparse_in)


# --------------------------------------------------------------------------------------------------
# Sparse solve -- correctness against the dense reference
# --------------------------------------------------------------------------------------------------


@pytest.mark.skipif(not SPARSE_BACKENDS, reason="no sparse backend available")
@pytest.mark.parametrize("sparse_backend", SPARSE_BACKENDS)
def test_sparse_solve_matches_dense_alternative_cost(sparse_backend: str) -> None:
    pytest.importorskip("numba")
    dist = np.random.default_rng(3).random((25, 25)) * 2.0
    eta = 0.15

    sparse_links = LapSolver(sparse_implementation=sparse_backend).sparse_solve(dist, alternative_cost=eta)
    dense_links = LapSolver().solve(dist, alternative_cost=eta)

    assert_valid_links(sparse_links, *dist.shape)
    assert np.isclose(soft_objective(dist, sparse_links, eta), soft_objective(dist, dense_links, eta))


@pytest.mark.skipif(not SPARSE_BACKENDS, reason="no sparse backend available")
@pytest.mark.parametrize("sparse_backend", SPARSE_BACKENDS)
def test_sparse_solve_matches_dense_cost_limit(sparse_backend: str) -> None:
    pytest.importorskip("numba")
    dist = np.random.default_rng(4).random((20, 20)) * 2.0
    limit = 0.3

    links = LapSolver(sparse_implementation=sparse_backend).sparse_solve(dist, cost_limit=limit)

    assert_valid_links(links, *dist.shape)
    assert all(dist[i, j] <= limit for i, j in links.tolist())

    reference = LapSolver().solve(dist, cost_limit=limit)
    assert len(links) == len(reference)
    assert np.isclose(assignment_cost(dist, links), assignment_cost(dist, reference))


@pytest.mark.skipif(not SPARSE_BACKENDS, reason="no sparse backend available")
@pytest.mark.parametrize("sparse_backend", SPARSE_BACKENDS)
@pytest.mark.parametrize("shape", [(15, 15), (12, 18), (18, 12)])
def test_sparse_solve_assume_feasible_matches_optimum(sparse_backend: str, shape: tuple[int, int]) -> None:
    pytest.importorskip("numba")
    if sparse_backend == "lapmod" and shape[0] != shape[1]:
        pytest.skip("lapmod requires a square matrix")

    n, m = shape
    # Fully feasible: every entry is below cost_limit, so no sink nodes are needed.
    dist = np.random.default_rng(9).random(shape) * 0.1
    links = LapSolver(sparse_implementation=sparse_backend).sparse_solve(dist, cost_limit=0.9, assume_feasible=True)

    assert_valid_links(links, n, m)
    assert len(links) == min(n, m)  # fully feasible -> full matching
    ref_i, ref_j = scipy.optimize.linear_sum_assignment(dist)
    assert np.isclose(assignment_cost(dist, links), dist[ref_i, ref_j].sum())


@pytest.mark.skipif(not SPARSE_BACKENDS, reason="no sparse backend available")
@pytest.mark.parametrize("sparse_backend", SPARSE_BACKENDS)
def test_sparse_solve_assume_feasible_matches_handcrafted(sparse_backend: str) -> None:
    pytest.importorskip("numba")
    import pylapy.sparse  # noqa: PLC0415

    dist = np.array(
        [
            [np.inf, 0.0, np.inf],
            [5, np.inf, 7],
            [np.inf, 5, 5],
        ]
    )
    coo = pylapy.sparse.dense_to_coo(dist, 10.0)

    links_sparse = LapSolver(sparse_implementation=sparse_backend).sparse_solve(
        coo, cost_limit=10.0, assume_feasible=True
    )
    links_dense = LapSolver(sparse_implementation=sparse_backend).sparse_solve(
        dist, cost_limit=10.0, assume_feasible=True
    )

    assert_valid_links(links_sparse, 3, 3)

    assert (links_sparse == links_dense).all()
    assert sorted(map(tuple, links_sparse.tolist())) == [(0, 1), (1, 0), (2, 2)]


@pytest.mark.skipif(not SPARSE_BACKENDS, reason="no sparse backend available")
@pytest.mark.parametrize("sparse_backend", SPARSE_BACKENDS)
@pytest.mark.parametrize("extension", ["diag_split_sparse", "diag_row_no_pad", "diag_col_no_pad"])
def test_sparse_solve_scipy_sparse_input_matches_dense(sparse_backend: str, extension: str) -> None:
    # Exercises the scipy-sparse-input branch (the dense-array path goes through the numba builders).
    if sparse_backend == "lapmod" and extension != "diag_split_sparse":
        pytest.skip("lapmod only supports the square diag_split_sparse extension")
    pytest.importorskip("numba")
    import pylapy.sparse  # noqa: PLC0415

    # eta small enough that the feasible graph has no full matching, so the sink nodes are genuinely used.
    eta = 0.1
    dist = np.random.default_rng(11).random((15, 15)) * 0.9

    coo = pylapy.sparse.dense_to_coo(dist, eta)

    solver = LapSolver(sparse_implementation=sparse_backend)
    solver.sparse_extension = extension
    links = solver.sparse_solve(coo, alternative_cost=eta)

    # Dense reference: the same problem with above-eta entries made infeasible (inf), soft threshold eta.
    reference = LapSolver().solve(dist, alternative_cost=eta)

    assert_valid_links(links, *dist.shape)
    assert all(dist[i, j] <= eta for i, j in links.tolist())  # only feasible links returned
    assert np.isclose(soft_objective(dist, links, eta), soft_objective(dist, reference, eta))


@pytest.mark.skipif(not SPARSE_BACKENDS, reason="no sparse backend available")
def test_sparse_solve_with_callable_extension() -> None:
    pytest.importorskip("numba")
    import pylapy.sparse  # noqa: PLC0415

    dist = np.random.default_rng(7).random((20, 20))
    eta = 0.15

    def my_extension(dist, *, cost_limit=float("inf"), alternative_cost=float("inf")):
        # A user-supplied callable must return a scipy sparse matrix; reuse a builtin layout here.
        return pylapy.sparse.diag_split_sparse(dist, cost_limit=cost_limit, alternative_cost=alternative_cost)

    solver = LapSolver(sparse_implementation=SPARSE_BACKENDS[0])
    solver.sparse_extension = my_extension
    links = solver.sparse_solve(dist, alternative_cost=eta)

    reference = LapSolver().solve(dist, alternative_cost=eta)
    assert_valid_links(links, *dist.shape)
    assert np.isclose(soft_objective(dist, links, eta), soft_objective(dist, reference, eta))


def test_sparse_solve_falls_back_to_dense_with_warning() -> None:
    dist = np.random.default_rng(5).random((8, 8))
    eta = 0.3

    solver = LapSolver()
    solver.sparse_implementation = None  # force the dense fallback path

    with pytest.warns(UserWarning, match="Falling back"):
        links = solver.sparse_solve(dist, alternative_cost=eta)

    assert_valid_links(links, *dist.shape)
    assert np.isclose(
        soft_objective(dist, links, eta),
        soft_objective(dist, LapSolver().solve(dist, alternative_cost=eta), eta),
    )


@pytest.mark.skipif(not SPARSE_BACKENDS, reason="no sparse backend available")
@pytest.mark.parametrize("sparse_backend", SPARSE_BACKENDS)
def test_sparse_solve_does_not_mutate_dense_input(sparse_backend: str) -> None:
    # Mixes inf (replaced internally) and negatives (which drive the shift paths):
    # neither rewrite may leak back to the caller's array.
    dist = np.array([[1.0, np.inf, -2.0], [np.inf, -1.0, 3.0], [-3.0, 2.0, np.inf]])
    original = dist.copy()
    LapSolver(sparse_implementation=sparse_backend).sparse_solve(dist, cost_limit=5.0, alternative_cost=5.0)
    assert np.array_equal(dist, original, equal_nan=True)

    LapSolver(sparse_implementation=sparse_backend).sparse_solve(
        dist, cost_limit=5.0, alternative_cost=5.0, assume_feasible=True
    )
    assert np.array_equal(dist, original, equal_nan=True)


@pytest.mark.skipif(not SPARSE_BACKENDS, reason="no sparse backend available")
@pytest.mark.parametrize("sparse_backend", SPARSE_BACKENDS)
def test_sparse_solve_does_not_mutate_sparse_input(sparse_backend: str) -> None:
    import scipy.sparse  # noqa: PLC0415

    # Mixes inf (replaced internally) and negatives (which drive the shift paths):
    # neither rewrite may leak back to the caller's array.
    dist = np.array([[1.0, np.inf, -2.0], [np.inf, -1.0, 3.0], [-3.0, 2.0, np.inf]])
    coo = scipy.sparse.coo_array(dist)  # Will keep all the inf inside
    original = coo.copy()

    LapSolver(sparse_implementation=sparse_backend).sparse_solve(coo, cost_limit=5.0, alternative_cost=5.0)
    assert (original.data == coo.data).all()

    LapSolver(sparse_implementation=sparse_backend).sparse_solve(
        coo, cost_limit=5.0, alternative_cost=5.0, assume_feasible=True
    )
    assert (original.data == coo.data).all()


# --------------------------------------------------------------------------------------------------
# Deprecated v1 API (`eta` / `hard` / `feasible`) — kept as a thin shim over the new parameters
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("call_style", ["positional", "keyword"])
def test_solve_deprecated_eta_maps_to_alternative_cost(call_style: str) -> None:
    # The old positional eta is now the alternative cost.
    dist = make_dist(5, 6)
    eta = 10.0
    solver = LapSolver()

    with pytest.deprecated_call():
        deprecated = solver.solve(dist, eta) if call_style == "positional" else solver.solve(dist, eta=eta)

    reference = solver.solve(dist, alternative_cost=eta)
    assert_valid_links(deprecated, *dist.shape)
    assert np.isclose(soft_objective(dist, deprecated, eta), soft_objective(dist, reference, eta))


@pytest.mark.skipif(not SPARSE_BACKENDS, reason="no sparse backend available")
@pytest.mark.parametrize("sparse_backend", SPARSE_BACKENDS)
@pytest.mark.parametrize("hard", [False, True])
def test_sparse_solve_deprecated_eta_mapping(sparse_backend: str, *, hard: bool) -> None:
    # The old positional eta becomes the cost limit; hard toggles whether an alternative cost is offered.
    dist = make_dist(8, 10)
    eta = float(np.median(dist))
    solver = LapSolver(sparse_implementation=sparse_backend)

    with pytest.deprecated_call():
        deprecated = solver.sparse_solve(dist, eta, hard, False)

    reference = solver.sparse_solve(
        dist, cost_limit=eta, alternative_cost=float("inf") if hard else eta, assume_feasible=False
    )
    assert_valid_links(deprecated, *dist.shape)
    assert len(deprecated) == len(reference)
    assert np.isclose(assignment_cost(dist, deprecated), assignment_cost(dist, reference))
