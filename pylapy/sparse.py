# Hide numba dependency here

from typing import List, Tuple
import warnings

import numba  # type: ignore
import numpy as np
import scipy.sparse  # type: ignore


def sparse_extension(dist, eta: float, hard: bool, method: str) -> scipy.sparse.coo_matrix:
    """Handle sparse extension for our own implemented method"""
    if method not in ("diag_row_cost", "diag_col_cost", "symmetric_cost"):
        raise ValueError(f"Unknown method {method}. Should be one of diag_row_cost, diag_col_cost, symmetric_cost.")

    n, m = dist.shape
    shape = {
        "diag_row_cost": (n, n + m),
        "diag_col_cost": (n + m, m),
        "symmetric_cost": (n + m, n + m),
    }[method]

    values: List[float]
    rows: List[int]
    cols: List[int]

    if isinstance(dist, np.ndarray):
        n, m = dist.shape
        values, rows, cols = {
            "diag_row_cost": sparse_diag_row_cost_dense_to_coo,
            "diag_col_cost": sparse_diag_col_cost_dense_to_coo,
            "symmetric_cost": sparse_symmetric_cost_dense_to_coo,
        }[method](dist, eta, hard)

        return scipy.sparse.coo_matrix((np.array(values, dtype=np.float32), (rows, cols)), shape=shape)

    if scipy.sparse.issparse(dist):
        n, m = dist.shape
        coo: scipy.sparse.coo_matrix = dist.tocoo()

        if (coo.data > eta).any():  # Up to the user to handle this
            warnings.warn("Found some values greater than the cost limit in the sparse format")

        values = coo.data.tolist()
        rows = coo.row.tolist()
        cols = coo.col.tolist()

        if hard:
            eta = eta * len(values)

        if method == "symmetric_cost":
            values.extend((0 for _ in values))
            rows.extend(n + j for j in cols)
            cols.extend(m + i for i in rows)

        if method in ["diag_row_cost", "symmetric_cost"]:
            values.extend(eta for _ in range(n))
            rows.extend(i for i in range(n))
            cols.extend(m + i for i in range(n))

        if method == ["diag_col_cost", "symmetric_cost"]:
            values.extend(eta for _ in range(m))
            rows.extend(n + j for j in range(m))
            cols.extend(j for j in range(m))

        return scipy.sparse.coo_matrix((np.array(values, dtype=np.float32), (rows, cols)), shape=shape)

    raise ValueError(f"Dist should be a numpy array or a scipy sparse matrix. Found {type(dist)}.")


@numba.njit
def dense_to_coo(dense: np.ndarray, eta: float) -> Tuple[List[float], List[int], List[int]]:
    """Converts a dense 2d matrix to COO sparse matrix, by keeping only values smaller than eta

    Args:
        dense (np.ndarray): 2d dense matrix
        eta (float): Max value to keep in the sparse matrix. All values above are filtered

    Returns:
        Tuple[List[float], List[int], List[int]]: COO format (values, rows, cols)
    """
    n, m = dense.shape
    values = []
    rows = []
    cols = []

    for i in range(n):
        for j in range(m):
            if dense[i, j] <= eta:
                values.append(dense[i, j])
                rows.append(i)
                cols.append(j)

    return values, rows, cols


@numba.njit
def sparse_diag_row_cost_dense_to_coo(
    dist: np.ndarray, eta: float, hard=False
) -> Tuple[List[float], List[int], List[int]]:
    """Sparse extension into (n, n + m) by setting the cost limit on the diagonal rows

    dist -> | dist[dist <= eta], Diag(eta) |

    Here Diag(eta) = | eta inf ... inf |
                     | inf eta ... ... |
                     | ... ... ... inf |
                     | inf ... inf eta |

    It removes all links in dist that are greater than eta and add one feasible link for each row
    to a sink node with a cost eta. (The resulting distance is fully linkable)

    By default, the resulting problem is the smooth thresholded one. For hard thresholding,
    we set an inf-like value on the diagonal instead of eta, allowing to link to a sink node only
    as a last resort. (<=> hard thresholding, but sparse and fully linkable)

    This is the fastest implementation for csgraph when the sparsity is standart.

    Args:
        dist (np.ndarray): Dense distance matrix to extend
            Shape: (N, M), dtype: float
        eta (float): Cost limit
        hard (bool): Build an hard thresholded equivalent problem.
            Default: False (will be smooth-thresholded)

    Returns:
        Tuple[List[float], List[int], List[int]]: Extended dist in the COO format (values, rows, cols)
            shape: (n + m, n)
    """
    n, m = dist.shape
    values = []
    rows = []
    cols = []

    for i in range(n):
        for j in range(m):
            if dist[i, j] <= eta:
                values.append(dist[i, j])
                rows.append(i)
                cols.append(j)

    if hard:
        eta = eta * len(values)

    for i in range(n):
        values.append(eta)
        rows.append(i)
        cols.append(m + i)

    return values, rows, cols


@numba.njit
def sparse_diag_col_cost_dense_to_coo(
    dist: np.ndarray, eta: float, hard=False
) -> Tuple[List[float], List[int], List[int]]:
    """Sparse extension into (n, n + m) by setting the cost limit on the diagonal columns

    dist -> | dist[dist <= eta]|
            |      Diag(eta)   |

    Here Diag(eta) = | eta inf ... inf |
                     | inf eta ... ... |
                     | ... ... ... inf |
                     | inf ... inf eta |

    It removes all links in dist that are greater than eta and add one feasible link for each row
    to a sink node with a cost eta. (The resulting distance is fully linkable)

    By default, the resulting problem is the smooth thresholded one. For hard thresholding,
    we set an inf-like value on the diagonal instead of eta, allowing to link to a sink node only
    as a last resort. (<=> hard thresholding, but sparse and fully linkable)

    Args:
        dist (np.ndarray): Dense distance matrix to extend
            Shape: (N, M), dtype: float
        eta (float): Cost limit
        hard (bool): Build an hard thresholded equivalent problem.
            Default: False (will be smooth-thresholded)

    Returns:
        Tuple[List[float], List[int], List[int]]: Extended dist in the COO format (values, rows, cols)
            Shape: (n, n + m)
    """
    n, m = dist.shape
    values = []
    rows = []
    cols = []

    for i in range(n):
        for j in range(m):
            if dist[i, j] <= eta:
                values.append(dist[i, j])
                rows.append(i)
                cols.append(j)

    if hard:
        eta = eta * len(values)

    for j in range(m):
        values.append(eta)
        rows.append(n + j)
        cols.append(j)

    return values, rows, cols


@numba.njit
def sparse_symmetric_cost_dense_to_coo(
    dense: np.ndarray, eta: float, hard=False
) -> Tuple[List[float], List[int], List[int]]:
    """Sparse extension into (n + m, n + m) by setting the cost limit on the diagonal columns

    dist -> | dist[dist <= eta], Diag(eta / 2) |
            |   Diag(eta / 2)  , 0[dist<=eta].T|

    Here Diag(eta) = | eta inf ... inf |
                     | inf eta ... ... |
                     | ... ... ... inf |
                     | inf ... inf eta |

    It removes all links in dist that are greater than eta and add one feasible link for each row
    to a sink node with a cost eta / 2 and for each columns to a sink node with a cost eta / 2.
    In order to have a feasible solution non reduced to all sink nodes, we also need to add 0.0
    in the diagonal block in i, j where dist[j, i] <= eta.

    It is the only sparse implementation that is squared, and thereforre the only one supported by lapmod.

    By default, the resulting problem is the smooth thresholded one. For hard thresholding,
    we set an inf-like value on the diagonals instead of eta, allowing to link to a sink node only
    as a last resort. (<=> hard thresholding, but sparse and fully linkable)

    This is the only implementation for lapmod. Plus this seems to be the fastest for csgraph, with
    smooth thresholding and extreme sparsity. (Up to 3 times faster).

    Args:
        dist (np.ndarray): Dense distance matrix to extend
            Shape: (N, M), dtype: float
        eta (float): Cost limit
        hard (bool): Build an hard thresholded equivalent problem.
            Default: False (will be smooth-thresholded)

    Returns:
        Tuple[List[float], List[int], List[int]]: Extended dist in the COO format (values, rows, cols)
            Shape: (n + m, n + m)
    """
    n, m = dense.shape
    values = []
    rows = []
    cols = []

    for i in range(n):
        for j in range(m):
            if dense[i, j] <= eta:
                values.append(dense[i, j])
                rows.append(i)
                cols.append(j)

                values.append(0.0)
                rows.append(n + j)
                cols.append(m + i)

    if hard:
        eta = eta * len(values) / 2

    for i in range(n):
        values.append(eta / 2)
        rows.append(i)
        cols.append(m + i)

    for j in range(m):
        values.append(eta / 2)
        rows.append(n + j)
        cols.append(j)

    return values, rows, cols
