"""Unit tests for the cost extension functions.

Each function rewrites an ``(n, m)`` distance matrix into an ``(n + m, n + m)`` square one with "sink"
rows/columns encoding the alternative (non-linking) cost ``eta`` -- except ``no_cost_extension``, which
returns the matrix untouched for backends that apply the alternative cost natively. These tests assert the
exact resulting matrix.
"""

import numpy as np
import pytest
from conftest import make_dist

from pylapy import cost_extension

INF = 999.0  # Finite sentinel standing in for the solver's "inf" replacement value.
ETA = 10.0

SHAPES = [(3, 3), (3, 5), (5, 3)]  # square, wide, tall

# The extensions that actually square the matrix (everything but the native no-op).
EXTENDING = [fn for name, fn in cost_extension.COST_EXTENSIONS.items() if name != "none"]


@pytest.mark.parametrize(("n", "m"), SHAPES)
def test_no_cost_extension_returns_input_unchanged(n: int, m: int) -> None:
    dist = make_dist(n, m)
    out = cost_extension.no_cost_extension(dist, ETA, INF)
    assert out.shape == (n, m)
    assert np.array_equal(out, dist)


@pytest.mark.parametrize(("n", "m"), SHAPES)
def test_fill_split(n: int, m: int) -> None:
    dist = make_dist(n, m)
    out = cost_extension.fill_split(dist, ETA, INF)

    expected = np.zeros((n + m, n + m))
    expected[:n, :m] = dist
    expected[:n, m:] = ETA / 2
    expected[n:, :m] = ETA / 2

    assert out.shape == (n + m, n + m)
    assert np.array_equal(out, expected)


@pytest.mark.parametrize(("n", "m"), SHAPES)
def test_diag_split(n: int, m: int) -> None:
    dist = make_dist(n, m)
    out = cost_extension.diag_split(dist, ETA, INF)

    expected = np.full((n + m, n + m), INF)
    expected[:n, :m] = dist
    np.fill_diagonal(expected[:n, m:], ETA / 2)
    np.fill_diagonal(expected[n:, :m], ETA / 2)
    expected[n:, m:] = 0

    assert out.shape == (n + m, n + m)
    assert np.array_equal(out, expected)


@pytest.mark.parametrize(("n", "m"), SHAPES)
def test_diag_split_sparse_dense_dist(n: int, m: int) -> None:
    # With a fully finite dist, the bottom-right mask zeroes the whole block, so the result matches
    # diag_split. The inf-sensitive behaviour is checked separately below.
    dist = make_dist(n, m)
    out = cost_extension.diag_split_sparse(dist, ETA, INF)

    expected = np.full((n + m, n + m), INF)
    expected[:n, :m] = dist
    np.fill_diagonal(expected[:n, m:], ETA / 2)
    np.fill_diagonal(expected[n:, :m], ETA / 2)
    expected[n:, m:] = 0

    assert np.array_equal(out, expected)


def test_diag_split_sparse_keeps_sparsity_on_inf() -> None:
    # Where dist has an inf entry, the mirrored bottom-right cell must stay inf (preserves sparsity).
    n, m = 3, 4
    dist = make_dist(n, m)
    dist[1, 2] = INF  # one "infeasible" link
    out = cost_extension.diag_split_sparse(dist, ETA, INF)

    bottom_right = out[n:, m:]  # shape (m, n)
    # Mirror of dist[1, 2] lives at bottom_right[2, 1]; it must remain INF, the rest 0.
    expected_br = np.zeros((m, n))
    expected_br[2, 1] = INF
    assert np.array_equal(bottom_right, expected_br)


@pytest.mark.parametrize(("n", "m"), SHAPES)
def test_fill_row_pad_zero(n: int, m: int) -> None:
    dist = make_dist(n, m)
    out = cost_extension.fill_row_pad_zero(dist, ETA, INF)

    expected = np.zeros((n + m, n + m))
    expected[:n, :m] = dist
    expected[:n, m:] = ETA

    assert np.array_equal(out, expected)


@pytest.mark.parametrize(("n", "m"), SHAPES)
def test_fill_row_pad_inf(n: int, m: int) -> None:
    dist = make_dist(n, m)
    out = cost_extension.fill_row_pad_inf(dist, ETA, INF)

    expected = np.full((n + m, n + m), INF)
    expected[:n, :m] = dist
    expected[:n, m:] = ETA

    assert np.array_equal(out, expected)


@pytest.mark.parametrize(("n", "m"), SHAPES)
def test_fill_col_pad_zero(n: int, m: int) -> None:
    dist = make_dist(n, m)
    out = cost_extension.fill_col_pad_zero(dist, ETA, INF)

    expected = np.zeros((n + m, n + m))
    expected[:n, :m] = dist
    expected[n:, :m] = ETA

    assert np.array_equal(out, expected)


@pytest.mark.parametrize(("n", "m"), SHAPES)
def test_fill_col_pad_inf(n: int, m: int) -> None:
    dist = make_dist(n, m)
    out = cost_extension.fill_col_pad_inf(dist, ETA, INF)

    expected = np.full((n + m, n + m), INF)
    expected[:n, :m] = dist
    expected[n:, :m] = ETA

    assert np.array_equal(out, expected)


@pytest.mark.parametrize(("n", "m"), SHAPES)
def test_diag_row_pad_zero(n: int, m: int) -> None:
    dist = make_dist(n, m)
    out = cost_extension.diag_row_pad_zero(dist, ETA, INF)

    expected = np.zeros((n + m, n + m))
    expected[:n, :m] = dist
    expected[:n, m:] = INF
    np.fill_diagonal(expected[:n, m:], ETA)

    assert np.array_equal(out, expected)


@pytest.mark.parametrize(("n", "m"), SHAPES)
def test_diag_row_pad_inf(n: int, m: int) -> None:
    dist = make_dist(n, m)
    out = cost_extension.diag_row_pad_inf(dist, ETA, INF)

    expected = np.full((n + m, n + m), INF)
    expected[:n, :m] = dist
    np.fill_diagonal(expected[:n, m:], ETA)

    assert np.array_equal(out, expected)


@pytest.mark.parametrize(("n", "m"), SHAPES)
def test_diag_col_pad_0(n: int, m: int) -> None:
    dist = make_dist(n, m)
    out = cost_extension.diag_col_pad_0(dist, ETA, INF)

    expected = np.zeros((n + m, n + m))
    expected[:n, :m] = dist
    expected[n:, :m] = INF
    np.fill_diagonal(expected[n:, :m], ETA)

    assert np.array_equal(out, expected)


@pytest.mark.parametrize(("n", "m"), SHAPES)
def test_diag_col_pad_inf(n: int, m: int) -> None:
    dist = make_dist(n, m)
    out = cost_extension.diag_col_pad_inf(dist, ETA, INF)

    expected = np.full((n + m, n + m), INF)
    expected[:n, :m] = dist
    np.fill_diagonal(expected[n:, :m], ETA)

    assert np.array_equal(out, expected)


@pytest.mark.parametrize("func", cost_extension.COST_EXTENSIONS.values(), ids=lambda f: f.__name__)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_dtype_preserved(func, dtype) -> None:
    dist = make_dist(4, 6).astype(dtype)
    out = func(dist, ETA, INF)
    assert out.dtype == dtype


@pytest.mark.parametrize("func", EXTENDING, ids=lambda f: f.__name__)
@pytest.mark.parametrize(("n", "m"), SHAPES)
def test_top_left_block_is_dist(func, n: int, m: int) -> None:
    dist = make_dist(n, m)
    out = func(dist, ETA, INF)
    assert out.shape == (n + m, n + m)
    assert np.array_equal(out[:n, :m], dist)


@pytest.mark.parametrize("func", cost_extension.COST_EXTENSIONS.values(), ids=lambda f: f.__name__)
def test_extended_not_gated(func) -> None:
    dist = make_dist(2, 1)
    dist[0, 0] = ETA + 10  # A cost larger than ETA
    out = func(dist, ETA, INF)
    assert out[0, 0] == ETA + 10  # Has not been gated to inf
