"""Sparse cost extensions: build a gated, solvable COO matrix for the sparse backends.

The sparse backends (`scipy.sparse.csgraph.min_weight_full_bipartite_matching` and lapx's `lapmod`) take a
*sparse* cost matrix and need every node to be linkable. A sparse extension turns a dense (or already
sparse) distance matrix into a COO matrix by:

1. dropping every entry above `cost_limit` (this is what makes it sparse / applies the threshold), and
2. adding "sink" edges so a row/column can stay unmatched at `alternative_cost` -- which also guarantees a
   feasible full matching (otherwise a row with no admissible link would break the backends).

`alternative_cost` sets whether a node may be left unmatched: a finite value adds an alternative sink link
at that cost; `inf` offers no alternative (the sink is priced very high and used only as a last resort, to
keep the problem feasible). Links to sink nodes are stripped from the result by `LapSolver`.

Which one to use (see the `benchmark_sparse_extension.py` script and `DEFAULT_SPARSE_EXTENSIONS`):

- `lapmod` only solves square matrices, so it requires `diag_split_sparse` (the `(n+m, n+m)` layout).
- `csgraph` handles rectangular matrices and is fastest with `diag_row_no_pad`.

Sparse solving is usually much faster than a dense cost extension on thresholded problems. Each function
follows `SparseExtensionProtocol` and is registered by name in `SPARSE_EXTENSIONS`.
"""

from typing import Protocol

import numba  # type: ignore[import-untyped]
import numpy as np
import scipy.sparse

sparray = (
    scipy.sparse.dok_array
    | scipy.sparse.coo_array
    | scipy.sparse.bsr_array
    | scipy.sparse.csc_array
    | scipy.sparse.csr_array
    | scipy.sparse.dia_array
    | scipy.sparse.lil_array
)


class SparseExtensionProtocol(Protocol):
    """Callable that builds a gated COO cost matrix with alternative-cost sink edges.

    It receives a dense or sparse distance matrix and returns a `scipy.sparse.coo_array` holding the real
    links (those with cost <= `cost_limit`) plus sink edges priced at `alternative_cost` (`inf` offers no
    alternative).
    """

    __name__: str

    def __call__(
        self,
        dist: np.ndarray | sparray,
        /,
        *,
        cost_limit: float = ...,
        alternative_cost: float = ...,
    ) -> sparray:
        """Return a COO matrix: real links (cost <= `cost_limit`) plus sink edges at `alternative_cost`."""


def sparse_to_dense(sparse: sparray, inf: float = float("inf")) -> np.ndarray:
    """Converts a sparse 2D array into a dense numpy array.

    Args:
        sparse (sparray): Sparse array to convert.
            Shape: (n, m), dtype: dtype
        inf (float): Default value to use for non-filled values.
            Default: inf

    Returns:
        np.ndarray: The converted dense array
            Shape: (n, m), dtype: dtype
    """
    coo = sparse.tocoo(copy=False)
    dist = np.full(coo.shape, inf, dtype=coo.dtype)

    rows, cols = coo.coords  # Expects only 2 dimension

    return _coo_to_dense(coo.data, rows, cols, dist)


@numba.njit(parallel=False)  # Parallel could be slightly faster but fine
def _coo_to_dense(values: np.ndarray, rows: np.ndarray, cols: np.ndarray, out: np.ndarray) -> np.ndarray:
    for i in range(len(values)):
        out[rows[i], cols[i]] = values[i]

    return out


def dense_to_coo(dist: np.ndarray, cost_limit: float) -> scipy.sparse.coo_array:
    """Converts a dense 2D numpy array to a sparse COO array, keeping only values smaller than `cost_limit`.

    If `cost_limit` is inf, it still removes infinite values.

    Args:
        dist (np.ndarray): Dense array to convert.
            Shape: (n, m), dtype: dtype
        cost_limit (float): Max value to keep in the sparse matrix. All values above are filtered

    Returns:
        scipy.sparse.coo_array: The converted sparse array
            Shape: (n, m), dtype: dtype
    """
    values, rows, cols = _dense_to_coo(dist, cost_limit)

    return scipy.sparse.coo_array((values, (rows, cols)), shape=dist.shape, dtype=dist.dtype)


@numba.njit
def _dense_to_coo(dist: np.ndarray, cost_limit: float) -> tuple[list[float], list[int], list[int]]:
    n, m = dist.shape
    values = []
    rows = []
    cols = []

    for i in range(n):
        for j in range(m):
            if dist[i, j] <= cost_limit and not np.isinf(dist[i, j]):
                values.append(dist[i, j])
                rows.append(i)
                cols.append(j)

    return values, rows, cols


def _to_coo_format(dist: np.ndarray | sparray, cost_limit: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shared helper for extension.

    It converts to coo and filter links above cost_limit.
    """
    if scipy.sparse.issparse(dist):
        coo = dist.tocoo(copy=False)

        values = coo.data
        rows = coo.row
        cols = coo.col

        # Ensure there is no values above cost_limit (or inf values)
        valid = (values != np.inf) if np.isinf(cost_limit) else (values <= cost_limit)
        if not valid.all():
            values = values[valid]
            rows = rows[valid]
            cols = cols[valid]
    else:
        values_, rows_, cols_ = _dense_to_coo(dist, cost_limit)
        values, rows, cols = np.array(values_), np.array(rows_), np.array(cols_)

    return values, rows, cols


def diag_split_sparse(
    dist: np.ndarray | sparray, *, cost_limit=float("inf"), alternative_cost=float("inf")
) -> scipy.sparse.coo_array:
    """Square `(n + m, n + m)` sparse extension with sink edges on both diagonals.

    Similar to `pylapy.cost_extension.diag_split_sparse`.

    Keeps the links with cost <= min(`cost_limit`, `alternative_cost`), mirrors each at cost 0 in the
    bottom-right block (so the all-sink solution is never optimal), then adds a diagonal row sink `(i, m+i)`
    and column sink `(n+j, j)`, each priced at `alternative_cost / 2`. With `alternative_cost=inf` the sink
    price becomes `sum(costs) + 1` (no alternative; the sink is used only as a last resort).

    The only square sparse layout, hence the only one `lapmod` can use; it also works with `csgraph`.

    Args:
        dist (np.ndarray | sparray): Distance matrix.
            Shape: (N, M), dtype: float
        cost_limit (float): Links with a cost above this are dropped. Default: inf.
        alternative_cost (float): Cost of an alternative sink link; inf offers no alternative.
            Default: inf.

    Returns:
        scipy.sparse.coo_array: Extended sparse cost matrix.
            Shape: (N + M, N + M), dtype: float
    """
    cost_limit = min(cost_limit, alternative_cost)  # Let's use maximum sparsity (alternative cost also gates links)

    # Convert to coo
    values, rows, cols = _to_coo_format(dist, cost_limit)

    n, m = dist.shape
    alternative_cost = values.sum() + 1 if np.isinf(alternative_cost) else alternative_cost

    # Mirror each real link (i, j) -> (n + j, m + i) at cost 0, so the extended problem stays
    # feasible without collapsing onto the all-sink solution.
    mirror_values = np.zeros_like(values)
    mirror_rows = cols + n
    mirror_cols = rows + m

    # Row sinks (i, m + i) and column sinks (n + j, j), each at alternative_cost / 2.
    sink_values = np.full(n + m, alternative_cost / 2, dtype=values.dtype)
    sink_rows = np.arange(n + m, dtype=rows.dtype)

    sink_cols_1 = np.arange(m, m + n, dtype=cols.dtype)
    sink_cols_2 = np.arange(m, dtype=rows.dtype)

    values = np.concatenate((values, mirror_values, sink_values))
    rows = np.concatenate((rows, mirror_rows, sink_rows))
    cols = np.concatenate((cols, mirror_cols, sink_cols_1, sink_cols_2))

    return scipy.sparse.coo_array((values, (rows, cols)), shape=(n + m, n + m), dtype=dist.dtype)


def diag_row_no_pad(
    dist: np.ndarray | sparray, *, cost_limit=float("inf"), alternative_cost=float("inf")
) -> scipy.sparse.coo_array:
    """Rectangular `(n, n + m)` sparse extension with a diagonal row sink.

    Similar to `pylapy.cost_extension.diag_row_pad_inf`.

    Keeps the links with cost <= min(`cost_limit`, `alternative_cost`) and adds one row sink `(i, m + i)`
    per row, priced at `alternative_cost` (or `sum(costs) + 1` when `alternative_cost=inf`, i.e. no
    alternative). Only rows get a sink (columns may be left unmatched for free), which matches the usual
    alternative-cost objective.

    The fastest layout for `csgraph`. Not square, so `lapmod` cannot use it.

    Args:
        dist (np.ndarray | sparray): Distance matrix.
            Shape: (N, M), dtype: float
        cost_limit (float): Links with a cost above this are dropped. Default: inf.
        alternative_cost (float): Cost of an alternative sink link; inf offers no alternative.
            Default: inf.

    Returns:
        scipy.sparse.coo_array: Extended sparse cost matrix.
            Shape: (N, N + M), dtype: float
    """
    cost_limit = min(cost_limit, alternative_cost)  # Let's use maximum sparsity (alternative cost also gates links)

    # Convert to coo
    values, rows, cols = _to_coo_format(dist, cost_limit)

    n, m = dist.shape
    alternative_cost = values.sum() + 1 if np.isinf(alternative_cost) else alternative_cost

    # Alternative row cost on the diagonal (i, m + i)
    values = np.concatenate((values, np.full(n, alternative_cost, dtype=values.dtype)))
    rows = np.concatenate((rows, np.arange(n, dtype=rows.dtype)))
    cols = np.concatenate((cols, np.arange(m, m + n, dtype=cols.dtype)))

    return scipy.sparse.coo_array((values, (rows, cols)), shape=(n, n + m), dtype=dist.dtype)


def diag_col_no_pad(
    dist: np.ndarray | sparray, *, cost_limit=float("inf"), alternative_cost=float("inf")
) -> scipy.sparse.coo_array:
    """Rectangular `(n + m, m)` sparse extension with a diagonal column sink.

    Similar to `pylapy.cost_extension.diag_col_pad_inf`.

    Mirror of `diag_row_no_pad`: keeps the links with cost <= min(`cost_limit`, `alternative_cost`) and
    adds one column sink `(n + j, j)` per column, priced at `alternative_cost` (or `sum(costs) + 1` when
    `alternative_cost=inf`, i.e. no alternative). Not square, so `lapmod` cannot use it.

    Args:
        dist (np.ndarray | sparray): Distance matrix.
            Shape: (N, M), dtype: float
        cost_limit (float): Links with a cost above this are dropped. Default: inf.
        alternative_cost (float): Cost of an alternative sink link; inf offers no alternative.
            Default: inf.

    Returns:
        scipy.sparse.coo_array: Extended sparse cost matrix.
            Shape: (N + M, M), dtype: float
    """
    cost_limit = min(cost_limit, alternative_cost)  # Let's use maximum sparsity (alternative cost also gates links)

    # Convert to coo
    values, rows, cols = _to_coo_format(dist, cost_limit)

    n, m = dist.shape
    alternative_cost = values.sum() + 1 if np.isinf(alternative_cost) else alternative_cost

    # Alternative column cost on the diagonal (n + j, j)
    values = np.concatenate((values, np.full(m, alternative_cost, dtype=values.dtype)))
    rows = np.concatenate((rows, np.arange(n, n + m, dtype=cols.dtype)))
    cols = np.concatenate((cols, np.arange(m, dtype=cols.dtype)))

    return scipy.sparse.coo_array((values, (rows, cols)), shape=(n + m, m), dtype=dist.dtype)


SPARSE_EXTENSIONS: dict[str, SparseExtensionProtocol] = {
    "diag_split_sparse": diag_split_sparse,
    "diag_row_no_pad": diag_row_no_pad,
    "diag_col_no_pad": diag_col_no_pad,
}

# lapmod only supports diag_split_sparse (squared).
# scipy csgraph is faster with diag_row_no_pad.
DEFAULT_SPARSE_EXTENSIONS: dict[str, SparseExtensionProtocol] = {
    "lapmod": diag_split_sparse,
    "csgraph": diag_row_no_pad,
}
