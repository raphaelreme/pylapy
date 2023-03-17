from typing import Tuple

import numba  # type: ignore
import numpy as np

from pylapy.cost_extension import symmetric_sparse_extension


@numba.jit
def to_csr(array: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert array to csr format (fast)

    Similar scipy.sparse.csr_array but handle inf value as the sparse value. (And faster)

    Args:
        array (np.ndarray): Array to convert to sparse (inf values are ignored)
            Shape: (N, M), dtype: T

    Returns:
        np.ndarray: data
            Shape: (K,), dtype: T
        np.ndarray: row indices
            Shape: (N + 1), dtype: uint32
        np.ndarray: column indices
            Shape: (K,), dtype: uint32
    """
    data = []
    rows = [0]
    columns = []

    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if not np.isinf(array[i, j]):
                data.append(array[i, j])
                columns.append(j)
        rows.append(len(data))

    return np.array(data), np.array(rows), np.array(columns)


def lap_sparse_solve(dist: np.ndarray, eta: float = np.inf) -> np.ndarray:
    """Wraps lap.lapmod for sparse matrix

    Note: It is not stable, lapmod do not support matrices where some rows
    cannot be linked to any columns.

    We also pay the conversion time from non-sparse to sparse.
    # NOTE: Could be added to the LapSolver ?
    """
    import lap  # type: ignore # pylint: disable=import-outside-toplevel

    n, m = dist.shape
    links: np.ndarray

    if not np.isinf(eta) or n != m:  # Extend the matrix if needed or asked
        if np.isinf(eta):  # n != m and full linking => does not work with lap.lapmod implem...
            # TODO: Either go with a high eta, or do smallest fill with a last coef ?
            raise ValueError("lapmod currently do not support n != m without eta")

        extended_dist = symmetric_sparse_extension(dist, eta, np.inf)

        _, x, _ = lap.lapmod(extended_dist.shape[0], *to_csr(extended_dist))
        i = np.arange(x.shape[0])[x != -1]
        j = x[x != -1]
        links = np.array([i, j], dtype=np.uint16).transpose()

        # Filter links that are assigned outside of C
        links = links[links[:, 0] < n]  # i < n
        links = links[links[:, 1] < m]  # j < m

        return links

    # No extension, the implementation can be called directly
    _, x, _ = lap.lapmod(dist.shape[0], *to_csr(dist))
    i = np.arange(x.shape[0])[x != -1]
    j = x[x != -1]
    links = np.array([i, j], dtype=np.uint16).transpose()

    return links


def lap_own_extend(dist: np.ndarray, eta: float = np.inf) -> np.ndarray:
    """Wraps lap.lapjv but using their own implementation for matrix extension

    Faster without extension (same as ours but without the wrapper) in low dimensions
    Slower in high dimension (n > 1000) with eta extension
    Much slower if n != m (They much extend more than required)
    """
    import lap  # type: ignore # pylint: disable=import-outside-toplevel

    n, m = dist.shape
    extend_cost = n != m or not np.isinf(eta)

    _, x, _ = lap.lapjv(dist, extend_cost=extend_cost, cost_limit=eta)
    i = np.arange(x.shape[0])[x != -1]
    j = x[x != -1]
    links = np.array([i, j], dtype=np.uint16).transpose()

    return links
