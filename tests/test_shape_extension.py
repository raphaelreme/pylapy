"""Unit tests for the shape extension functions.

Each function rewrites a rectangular ``(n, m)`` distance matrix into a square, fully-feasible one (or, for
``no_shape_extension``, leaves it untouched for the backend to handle natively). This asserts the exact
resulting matrix (shape, the preserved top-left block, and the fill regions).
"""

import numpy as np
import pytest
from conftest import make_dist

from pylapy import shape_extension

INF = 999.0
SHAPES = [(3, 3), (3, 5), (5, 3)]  # square, wide, tall

# The explicit-padding extensions (everything but the native no-op) that square the matrix.
PADDING_EXTENSIONS = [fn for name, fn in shape_extension.SHAPE_EXTENSIONS.items() if name != "no_shape_extension"]


@pytest.mark.parametrize(("n", "m"), SHAPES)
def test_no_shape_extension_returns_input_unchanged(n: int, m: int) -> None:
    dist = make_dist(n, m)
    out = shape_extension.no_shape_extension(dist, INF)
    assert out.shape == (n, m)
    assert np.array_equal(out, dist)


@pytest.mark.parametrize(("n", "m"), SHAPES)
def test_smallest_fill_zero(n: int, m: int) -> None:
    dist = make_dist(n, m)
    out = shape_extension.smallest_fill_zero(dist, INF)

    k = max(n, m)
    expected = np.zeros((k, k))
    expected[:n, :m] = dist

    assert out.shape == (k, k)
    assert np.array_equal(out, expected)


@pytest.mark.parametrize(("n", "m"), SHAPES)
def test_smallest_fill_inf(n: int, m: int) -> None:
    dist = make_dist(n, m)
    out = shape_extension.smallest_fill_inf(dist, INF)

    k = max(n, m)
    expected = np.full((k, k), INF)
    expected[:n, :m] = dist

    assert out.shape == (k, k)
    assert np.array_equal(out, expected)


@pytest.mark.parametrize(("n", "m"), SHAPES)
def test_sum_fill_inf(n: int, m: int) -> None:
    dist = make_dist(n, m)
    out = shape_extension.sum_fill_inf(dist, INF)

    expected = np.full((n + m, n + m), INF)
    expected[:n, :m] = dist

    assert out.shape == (n + m, n + m)
    assert np.array_equal(out, expected)


@pytest.mark.parametrize(("n", "m"), SHAPES)
def test_sum_split_inf(n: int, m: int) -> None:
    dist = make_dist(n, m)
    out = shape_extension.sum_split_inf(dist, INF)

    expected = np.full((n + m, n + m), INF)
    expected[:n, :m] = dist
    expected[n:, m:] = 0

    assert out.shape == (n + m, n + m)
    assert np.array_equal(out, expected)


@pytest.mark.parametrize(("n", "m"), SHAPES)
def test_sum_row_inf(n: int, m: int) -> None:
    dist = make_dist(n, m)
    out = shape_extension.sum_row_inf(dist, INF)

    expected = np.zeros((n + m, n + m))
    expected[:n, :m] = dist
    expected[:n, m:] = INF

    assert out.shape == (n + m, n + m)
    assert np.array_equal(out, expected)


@pytest.mark.parametrize(("n", "m"), SHAPES)
def test_sum_col_inf(n: int, m: int) -> None:
    dist = make_dist(n, m)
    out = shape_extension.sum_col_inf(dist, INF)

    expected = np.zeros((n + m, n + m))
    expected[:n, :m] = dist
    expected[n:, :m] = INF

    assert out.shape == (n + m, n + m)
    assert np.array_equal(out, expected)


@pytest.mark.parametrize("func", PADDING_EXTENSIONS, ids=lambda f: f.__name__)
@pytest.mark.parametrize(("n", "m"), SHAPES)
def test_padding_preserves_top_left_block(func, n: int, m: int) -> None:
    dist = make_dist(n, m)
    out = func(dist, INF)
    assert out.shape[0] == out.shape[1]  # squared
    assert np.array_equal(out[:n, :m], dist)


@pytest.mark.parametrize("func", shape_extension.SHAPE_EXTENSIONS.values(), ids=lambda f: f.__name__)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_dtype_preserved(func, dtype) -> None:
    dist = make_dist(4, 6).astype(dtype)
    assert func(dist, INF).dtype == dtype
