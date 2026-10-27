from __future__ import annotations

from typing import TYPE_CHECKING

import lap
import numpy as np
import scipy

from pylapy.cost_extension import CostExtensionProtocol, no_cost_extension
from pylapy.shape_extension import ShapeExtensionProtocol, no_shape_extension
from pylapy.sparse import SparseExtensionProtocol, dense_to_coo, sparray

if TYPE_CHECKING:
    from collections.abc import Callable


# SCIPY -> None (internal),                     fill_row_pad_zero
# LAP -> None (internal smallest_fill_0),       fill_row_pad_zero
# LAPJV -> smallest_fill_0,                     fill_row_pad_zero
# Lapsolver -> None (internal smallest_fill_0), Diag split sparse
# LapRust -> None (internal)                  , Internal
# LapMod  -> diag_split_sparse (single choice)
# csgraph -> diag_row_no_pad


def run(
    solver: Callable[[np.ndarray, float], np.ndarray],
    dist: np.ndarray,
    alternative_cost: float = float("inf"),
    inf: float = float("inf"),
    shape_extension: ShapeExtensionProtocol = no_shape_extension,
    cost_extension: CostExtensionProtocol = no_cost_extension,
    *,
    enforce_small_row: bool = False,
) -> np.ndarray:
    # NOTE: Gating with cost_limit is done outside the function

    n, m = dist.shape
    transposed = False

    if n > m and enforce_small_row:  # Transpose to solve with fewer rows
        dist = dist.T
        m, n = n, m
        transposed = True

    if not np.isinf(alternative_cost):
        dist = cost_extension(dist, alternative_cost, inf)
        if cost_extension != no_cost_extension:
            alternative_cost = float("inf")  # No need to extend in solver

    if dist.shape[0] != dist.shape[1]:
        dist = shape_extension(dist, inf)

    links = solver(dist, alternative_cost)

    if dist.shape != (n, m):
        links = links[links[:, 0] < n]  # i < n
        links = links[links[:, 1] < m]  # j < m

    # Filter inf links
    links = links[dist[links[:, 0], links[:, 1]] < inf]

    if transposed:
        links = links[:, ::-1]  # Transposed links

    return links


def solve_scipy(dist: np.ndarray, eta: float = float("inf")) -> np.ndarray:
    if not np.isinf(eta):
        raise ValueError("Scipy solver do not support cost limit")

    return np.stack(scipy.optimize.linear_sum_assignment(dist)).T.astype(np.uint16)


def solve_lap(dist: np.ndarray, eta: float = float("inf")) -> np.ndarray:
    return lap.lapjvxa(dist, extend_cost=True, cost_limit=eta, return_cost=False).astype(np.uint16)


def solve_lapjv(dist: np.ndarray, eta: float = float("inf")) -> np.ndarray:
    if not np.isinf(eta):
        raise ValueError("LapJV solver do not support cost limit")

    return lap.lapjvsa(dist, extend_cost=True, return_cost=False).astype(np.uint16)


def solve_lapsolver(dist: np.ndarray, eta: float = float("inf")) -> np.ndarray:
    if not np.isinf(eta):
        raise ValueError("LapSolver solver do not support cost limit")

    return np.stack(lap.lapjvc(dist, return_cost=False)).T.astype(np.uint16)


def solve_laprust(dist: np.ndarray, eta: float = float("inf")) -> np.ndarray:
    return lap.lapjvr(dist, extend_cost=True, cost_limit=eta, return_cost=False, output="pairs").astype(np.uint16)


def run_sparse(
    solver: Callable[[sparray], np.ndarray],
    dist: np.ndarray | sparray,
    eta: float,
    sparse_extension: SparseExtensionProtocol,
    *,
    hard: bool = False,
    feasible: bool = False,
    enforce_small_row: bool = False,
) -> np.ndarray:
    # Old nomenclature: hard means inf alternative cost
    n, m = dist.shape  # type: ignore[misc]
    transposed = False

    if n > m and enforce_small_row:  # Transpose to solve with fewer rows
        dist = dist.T
        m, n = n, m
        transposed = True

    if hard and feasible:  # No cost extension, let's trust the user
        # XXX: Filter values higher than eta in sparse ?
        dist = dist if scipy.sparse.issparse(dist) else dense_to_coo(dist, eta)  # type: ignore[arg-type]
    elif hard:
        dist = sparse_extension(dist, cost_limit=eta)
    else:
        dist = sparse_extension(dist, cost_limit=eta, alternative_cost=eta)

    links = solver(dist)

    if dist.shape != (n, m):  # Extended, so let's remove sink links
        links = links[links[:, 0] < n]  # i < n
        links = links[links[:, 1] < m]  # j < m

    if transposed:
        links = links[:, ::-1]  # Transposed links

    return links


def solve_csgraph(dist: sparray) -> np.ndarray:
    csr = dist.tocsr()

    # Remove 0 value: scipy does not support it.
    mini = csr.data.min()
    if mini <= 0.0:
        # print(f"WARNING: Found non-positive values ({mini}). The inputed CSR will be modified in place to remove it.")
        csr.data += 0.1 - mini

    return np.stack(scipy.sparse.csgraph.min_weight_full_bipartite_matching(csr)).T.astype(np.uint16)


def solve_lapmod(dist: sparray) -> np.ndarray:
    if dist.shape[0] != dist.shape[1]:  # type: ignore[misc]
        # Do not support feasible rectangular matrices:
        # Shape extension are not sparse (Should be extended hard)
        #
        # Only support the diag_split_sparse extension
        raise ValueError(f"Lapmod only support square distance. Got {dist.shape}.")

    csr = dist.tocsr()

    x, _ = lap.lapmod(csr.shape[0], csr.data, csr.indptr, csr.indices, return_cost=False)
    return np.stack((np.arange(x.shape[0]), x)).T.astype(np.uint16)
