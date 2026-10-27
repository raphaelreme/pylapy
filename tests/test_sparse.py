"""Tests for the sparse path: ``pylapy.sparse`` helpers/builders and ``LapSolver.sparse_solve``.

Exercises ``LapSolver.sparse_solve`` on scipy-sparse input and covers the edge cases that are specific to
sparse solving:

- empty sparse matrices, ``inf`` entries stored inside, and all-``inf`` matrices;
- 0-cost links (pylapy uses ``inf`` -- not ``0`` -- as the "no link" sentinel, unlike scipy sparse, whose
  canonicalization *drops* stored zeros);
- negative costs (``csgraph`` shifts internally, ``lapmod`` is cost-shifted in ``LapSolver._lapmod``).

The whole module is skipped when ``numba`` (required to import ``pylapy.sparse``) is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.optimize
import scipy.sparse
from conftest import assert_valid_links, assignment_cost, available_sparse_backends, make_dist, soft_objective

from pylapy import LapSolver, cost_extension

pytest.importorskip("numba")  # pylapy.sparse imports numba at module load
import pylapy.sparse

SPARSE_BACKENDS = available_sparse_backends()
pytestmark = pytest.mark.skipif(not SPARSE_BACKENDS, reason="no sparse backend available")


# --------------------------------------------------------------------------------------------------
# pylapy.sparse helpers
# --------------------------------------------------------------------------------------------------


def test_dense_to_coo_filters_inf_and_above_limit_and_keeps_zero() -> None:
    dist = np.array(
        [
            [0.0, 2.0, np.inf],
            [5.0, -5.0, 3.0],
        ]
    )
    coo = pylapy.sparse.dense_to_coo(dist, cost_limit=3.0)

    assert coo.shape == dist.shape
    stored = {(int(i), int(j)): float(v) for i, j, v in zip(coo.row, coo.col, coo.data, strict=True)}
    # Kept: <= 3 and finite (including the 0). Dropped: inf (0, 2) and 5.0 (1, 0).
    assert stored == {(0, 0): 0.0, (0, 1): 2.0, (1, 1): -5.0, (1, 2): 3.0}


def test_dense_to_coo_inf_limit_still_drops_inf() -> None:
    dist = np.array(
        [
            [1.0, np.inf],
            [np.inf, 2.0],
        ]
    )
    coo = pylapy.sparse.dense_to_coo(dist, cost_limit=float("inf"))
    assert sorted(coo.data.tolist()) == [1.0, 2.0]


def test_sparse_to_dense_round_trip() -> None:
    dist = np.array(
        [
            [0.0, 2.0, np.inf],
            [5.0, 1.0, 3.0],
        ]
    )
    coo = pylapy.sparse.dense_to_coo(dist, cost_limit=float("inf"))  # drops the inf cell
    dense = pylapy.sparse.sparse_to_dense(coo, inf=np.inf)

    assert (dense == dist).all()


def test_sparse_to_dense_round_trip_with_cost_limit() -> None:
    dist = np.array(
        [
            [0.0, 2.0, np.inf],
            [5.0, -5.0, 3.0],
        ]
    )
    coo = pylapy.sparse.dense_to_coo(dist, cost_limit=3.0)  # drops above 3.0
    dense = pylapy.sparse.sparse_to_dense(coo, inf=999)

    expected = dist.copy()
    expected[expected > 3.0] = 999  # INF

    assert (dense == expected).all()


def test_to_coo_format_sparse_input_filter_above_limit() -> None:
    dist = np.array([[1.0, np.inf], [np.inf, 7.0]])
    sparse_in = scipy.sparse.coo_array(dist)  # stores the inf entries (inf != 0)
    values, *_ = pylapy.sparse._to_coo_format(sparse_in, cost_limit=3.0)
    assert sorted(values.tolist()) == [1.0]


def test_to_coo_format_sparse_input_limit_inf_still_filters_inf() -> None:
    dist = np.array([[1.0, np.inf], [np.inf, 2.0]])
    sparse_in = scipy.sparse.coo_array(dist)  # stores the inf entries (inf != 0)
    values, *_ = pylapy.sparse._to_coo_format(sparse_in, cost_limit=float("inf"))
    assert sorted(values.tolist()) == [1.0, 2.0]


@pytest.mark.parametrize(
    ("extension", "expected_shape"),
    [
        (pylapy.sparse.diag_split_sparse, (5, 5)),  # (n + m, n + m)
        (pylapy.sparse.diag_row_no_pad, (2, 5)),  # (n, n + m)
        (pylapy.sparse.diag_col_no_pad, (5, 3)),  # (n + m, m)
    ],
)
def test_sparse_extension_shape_and_block(
    extension: pylapy.sparse.SparseExtensionProtocol, expected_shape: tuple[int, int]
) -> None:
    dist = make_dist(2, 3)
    coo = extension(dist, alternative_cost=float("inf"))

    assert scipy.sparse.issparse(coo)
    assert coo.shape == expected_shape

    # The real-link block is preserved exactly.
    dense = pylapy.sparse.sparse_to_dense(coo, inf=np.inf)
    assert np.array_equal(dense[:2, :3], dist)


@pytest.mark.parametrize(
    "extension",
    [pylapy.sparse.diag_split_sparse, pylapy.sparse.diag_row_no_pad, pylapy.sparse.diag_col_no_pad],
)
def test_sparse_extension_sparse_input_equivalent_to_dense_input(
    extension: pylapy.sparse.SparseExtensionProtocol,
) -> None:
    dist = make_dist(2, 3)
    coo_from_dense = extension(dist, alternative_cost=float("inf"))
    coo_from_sparse = extension(pylapy.sparse.dense_to_coo(dist, float("inf")), alternative_cost=float("inf"))

    assert (coo_from_dense.toarray() == coo_from_sparse.toarray()).all()


@pytest.mark.parametrize(("n", "m"), [(5, 5), (5, 4), (3, 6)])
def test_diag_split_sparse_matches_gated_dense_version(n: int, m: int) -> None:
    dist = make_dist(n, m)
    eta = 7.0

    coo = pylapy.sparse.diag_split_sparse(dist, alternative_cost=eta)
    extended = pylapy.sparse.sparse_to_dense(coo)

    dist[dist > eta] = np.inf
    extended_dense = cost_extension.diag_split_sparse(dist, eta)

    assert (extended == extended_dense).all()


@pytest.mark.parametrize(("n", "m"), [(5, 5), (5, 4), (3, 6)])
def test_diag_row_no_pad_matches_gated_dense_version(n: int, m: int) -> None:
    dist = make_dist(n, m)
    eta = 7.0

    coo = pylapy.sparse.diag_row_no_pad(dist, alternative_cost=eta)
    extended = pylapy.sparse.sparse_to_dense(coo)

    dist[dist > eta] = np.inf
    extended_dense = cost_extension.diag_row_pad_inf(dist, eta)

    assert (extended == extended_dense[:n]).all()  # Extended is squared


@pytest.mark.parametrize(("n", "m"), [(5, 5), (5, 4), (3, 6)])
def test_diag_col_no_pad_matches_gated_dense_version(n: int, m: int) -> None:
    dist = make_dist(n, m)
    eta = 7.0

    coo = pylapy.sparse.diag_col_no_pad(dist, alternative_cost=eta)
    extended = pylapy.sparse.sparse_to_dense(coo)

    dist[dist > eta] = np.inf
    extended_dense = cost_extension.diag_col_pad_inf(dist, eta)

    assert (extended == extended_dense[:, :m]).all()  # Extended is squared


# --------------------------------------------------------------------------------------------------
# Sparse solve: Edge case empty / inf-inside / all-inf scipy-sparse input (both backends)
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("sparse_backend", SPARSE_BACKENDS)
@pytest.mark.parametrize("shape", [(0, 5), (5, 0), (0, 0)])
def test_sparse_solve_empty_sparse_input(sparse_backend: str, shape: tuple[int, int]) -> None:
    sparse_in = scipy.sparse.coo_array(np.zeros(shape))
    links = LapSolver(sparse_implementation=sparse_backend).sparse_solve(sparse_in)
    assert links.shape == (0, 2)
    assert links.dtype == np.uint16


@pytest.mark.parametrize("sparse_backend", SPARSE_BACKENDS)
def test_sparse_solve_inf_inside_sparse_input(sparse_backend: str) -> None:
    # A sparse matrix that explicitly stores some inf entries: they must be filtered out, and the
    # finite diagonal (the only feasible assignment) recovered -- same as the dense solve.
    dist = np.array([[1.0, np.inf, 2.0], [np.inf, 1.0, np.inf], [2.0, np.inf, 1.0]])
    sparse_in = scipy.sparse.coo_array(dist)  # inf != 0, so the inf entries are stored
    links = LapSolver(sparse_implementation=sparse_backend).sparse_solve(sparse_in)

    assert_valid_links(links, *dist.shape)
    assert sorted(map(tuple, links.tolist())) == [(0, 0), (1, 1), (2, 2)]


@pytest.mark.parametrize("sparse_backend", SPARSE_BACKENDS)
def test_sparse_solve_all_inf_sparse_input(sparse_backend: str) -> None:
    sparse_in = scipy.sparse.coo_array(np.full((3, 3), np.inf))
    links = LapSolver(sparse_implementation=sparse_backend).sparse_solve(sparse_in)
    assert links.shape == (0, 2)
    assert links.dtype == np.uint16


# --------------------------------------------------------------------------------------------------
# Sparse solve: edge case 0 cost or negative cost links (0 value in sparse array is annoying)
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("sparse_backend", SPARSE_BACKENDS)
def test_sparse_solve_dense_input_keeps_zero_cost_links(sparse_backend: str) -> None:
    # Passing a *dense* array, pylapy keeps 0-cost entries (inf is the sentinel, not 0), so the sparse
    # solve finds the zero diagonal just like the dense solve.
    dist = np.array([[0.0, 5.0, 5.0], [5.0, 0.0, 5.0], [5.0, 5.0, 0.0]])
    links = LapSolver(sparse_implementation=sparse_backend).sparse_solve(dist)

    assert_valid_links(links, *dist.shape)
    assert assignment_cost(dist, links) == 0.0
    assert assignment_cost(dist, links) == assignment_cost(dist, LapSolver().solve(dist))


@pytest.mark.parametrize("sparse_backend", SPARSE_BACKENDS)
def test_sparse_solve_sparse_input_keeps_zero_cost_links(sparse_backend: str) -> None:
    # Passing a *dense* array, pylapy keeps 0-cost entries (inf is the sentinel, not 0), so the sparse
    # solve finds the zero diagonal just like the dense solve.
    dist = np.array([[0.0, 5.0, 5.0], [5.0, 0.0, 5.0], [5.0, 5.0, 0.0]])
    coo = pylapy.sparse.dense_to_coo(dist, 3.0)
    links = LapSolver(sparse_implementation=sparse_backend).sparse_solve(coo)

    assert_valid_links(links, *dist.shape)
    assert assignment_cost(dist, links) == 0.0
    assert assignment_cost(dist, links) == assignment_cost(dist, LapSolver().solve(dist, cost_limit=3.0))


def test_scipy_sparse_from_dense_drops_zeros_but_explicit_coo_keeps_them() -> None:
    # Caveat: building a scipy sparse matrix from a dense array canonicalizes away the stored zeros, so
    # 0-cost links are lost. Constructing the COO explicitly (zeros in `data`) preserves them.
    dense = np.array([[0.0, 5.0, 5.0], [5.0, 0.0, 5.0], [5.0, 5.0, 0.0]])

    dropped = scipy.sparse.coo_array(dense)
    assert dropped.nnz == 6  # the 3 diagonal zeros are gone

    rows = np.repeat(np.arange(3), 3)
    cols = np.tile(np.arange(3), 3)
    explicit = scipy.sparse.coo_array((dense.ravel().copy(), (rows, cols)), shape=(3, 3))
    assert explicit.nnz == 9  # zeros kept

    links = LapSolver(sparse_implementation=SPARSE_BACKENDS[0]).sparse_solve(explicit)
    assert sorted(map(tuple, links.tolist())) == [(0, 0), (1, 1), (2, 2)]
    assert assignment_cost(dense, links) == 0.0


@pytest.mark.parametrize("sparse_backend", SPARSE_BACKENDS)
def test_sparse_solve_negative_costs_match_scipy(sparse_backend: str) -> None:
    dist = np.array([[-1.0, -5.0, 2.0], [3.0, -2.0, -4.0], [-3.0, 1.0, -1.0]])
    links = LapSolver(sparse_implementation=sparse_backend).sparse_solve(dist)

    assert_valid_links(links, *dist.shape)
    assert len(links) == 3
    ref_i, ref_j = scipy.optimize.linear_sum_assignment(dist)
    assert np.isclose(assignment_cost(dist, links), dist[ref_i, ref_j].sum())


@pytest.mark.parametrize("sparse_backend", SPARSE_BACKENDS)
def test_sparse_solve_negative_costs_with_alternative_cost(sparse_backend: str) -> None:
    dist = np.random.default_rng(6).random((15, 15)) - 0.5  # roughly half negative
    eta = 0.1
    links = LapSolver(sparse_implementation=sparse_backend).sparse_solve(dist, alternative_cost=eta)
    reference = LapSolver().solve(dist, alternative_cost=eta)

    assert_valid_links(links, *dist.shape)
    assert np.isclose(soft_objective(dist, links, eta), soft_objective(dist, reference, eta))


# --------------------------------------------------------------------------------------------------
# Contract: lapmod only solves square matrices
# --------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------
# sparse_solve must not mutate the caller's input in place (the negative-cost shift in _csgraph /
# _lapmod is in-place, but it operates on an internal copy -- it must never reach the public input).
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("sparse_backend", SPARSE_BACKENDS)
@pytest.mark.parametrize("shape", [(8, 8), (10, 6)])  # (10, 6) also exercises the internal transpose
def test_sparse_solve_does_not_mutate_dense_input(sparse_backend: str, shape: tuple[int, int]) -> None:
    if sparse_backend == "lapmod" and shape[0] != shape[1]:
        pytest.skip("lapmod requires a square matrix")
    dist = np.random.default_rng(0).random(shape) - 0.5  # negatives trigger the in-place cost shift
    original = dist.copy()
    LapSolver(sparse_implementation=sparse_backend).sparse_solve(dist)
    assert np.array_equal(dist, original)


@pytest.mark.parametrize("sparse_backend", SPARSE_BACKENDS)
@pytest.mark.parametrize("sparse_cls", ["coo_array", "csr_array", "csr_matrix"])
@pytest.mark.parametrize("assume_feasible", [False, True])
def test_sparse_solve_does_not_mutate_sparse_input(sparse_backend: str, sparse_cls: str, assume_feasible: bool) -> None:
    base = np.random.default_rng(0).random((8, 8)) - 0.5  # square + negatives: feasible and shift-prone
    sparse_in = getattr(scipy.sparse, sparse_cls)(base)
    before = sparse_in.toarray()

    solver = LapSolver(sparse_implementation=sparse_backend)
    solver.sparse_solve(sparse_in, cost_limit=0.9, assume_feasible=assume_feasible)

    assert np.array_equal(sparse_in.toarray(), before)


@pytest.mark.skipif("lapmod" not in SPARSE_BACKENDS, reason="lapmod backend unavailable")
def test_lapmod_rejects_rectangular_matrix() -> None:
    # A rectangular problem reaches lapmod only via assume_feasible (no sparse shape extension); lapmod
    # supports square matrices only, so it must raise rather than return a wrong result.
    dist = np.random.default_rng(0).random((4, 6)) * 0.1
    solver = LapSolver(sparse_implementation="lapmod")
    with pytest.raises(ValueError, match="squared matrix"):
        solver.sparse_solve(dist, cost_limit=0.9, assume_feasible=True)
