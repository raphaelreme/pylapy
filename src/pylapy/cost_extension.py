"""Cost extension functions to add an alternative non-linking cost.

A cost extension adds "sink" rows and columns to the distance matrix (also squaring it). Linking a true
row/column to a sink means leaving it unassigned, at the alternative cost `eta`. This lets a node stay
unmatched instead of being forced into an expensive link. The many layouts below are equivalent (they
yield the same optimal links) but differ in runtime and numerical robustness.

Which one to use (see the `benchmark_cost_extension.py` script and `DEFAULT_COST_EXTENSIONS`):

- `laprust` has a native alternative cost and is fastest with `no_cost_extension` (no explicit sink
  nodes), by a wide margin.
- `scipy`, `lap`, `lapjv`: `diag_row_pad_zero` is the fast, robust default (its `fill_row_pad_zero`
  twin is equivalent and roughly as fast).
- `lapsolver`: the `diag_split_sparse` is best.

Avoid the `*_pad_inf` variants with `lapjv`: they are slower and make that backend return wrong
assignments (it seems to mishandle the inf sentinel). The `*_pad_zero` variants are safe everywhere.

Each function follows `CostExtensionProtocol` and is registered by name in `COST_EXTENSIONS`.
"""

from typing import Protocol

import numpy as np


class CostExtensionProtocol(Protocol):
    """Callable that adds alternative-cost sink nodes to a distance matrix.

    It receives the distance matrix, the alternative cost `eta` (the cost of leaving a node unmatched) and
    the `inf` sentinel value, and returns a square extended matrix; links to sink nodes are stripped from
    the result by the solver.
    """

    __name__: str

    def __call__(
        self,
        dist: np.ndarray,
        eta: float,
        /,
        inf: float = ...,
    ) -> np.ndarray:
        """Return `dist` extended with sink rows/columns so a node can stay unmatched at cost `eta`."""


def no_cost_extension(dist: np.ndarray, eta: float, inf: float = float("inf")) -> np.ndarray:  # noqa: ARG001
    """Do not extend the distance and let the backend internally solve the problem with alternative cost.

    Only lap and laprust have an internal support for alternative cost.
    """
    return dist


def fill_split(dist: np.ndarray, eta: float, inf: float = np.inf) -> np.ndarray:  # noqa: ARG001
    """Extend to a (n + m, n + m) by splitting the cost limit on rows and columns.

    dist -> |  dist  , eta / 2 |
            | eta / 2,    0    |

    This is the implementation in lap v0.5.0.

    Args:
        dist (np.ndarray): Distance matrix to extend
            Shape: (N, M), dtype: float
        eta (float): Alternative cost.
        inf (float): Current inf value
            Default: np.inf

    Returns:
        np.ndarray: Equivalent extended distance matrix
            Shape: (N + M, N + M), dtype: float
    """
    n, m = dist.shape
    extended_dist = np.full((n + m, n + m), 0, dtype=dist.dtype)
    extended_dist[:n, :m] = dist
    extended_dist[:n, m:] = eta / 2
    extended_dist[n:, :m] = eta / 2

    return extended_dist


def diag_split(dist: np.ndarray, eta: float, inf: float = np.inf) -> np.ndarray:
    """Extend to a (n + m, n + m) by splitting the cost limit in diagonal block.

    dist -> |     dist     , Diag(eta / 2) |
            | Diag(eta / 2),       0       |


    Here Diag(eta) = | eta inf ... inf |
                     | inf eta ... ... |
                     | ... ... ... inf |
                     | inf ... inf eta |

    Args:
        dist (np.ndarray): Distance matrix to extend
            Shape: (N, M), dtype: float
        eta (float): Alternative cost.
        inf (float): Current inf value
            Default: np.inf

    Returns:
        np.ndarray: Equivalent extended distance matrix
            Shape: (N + M, N + M), dtype: float
    """
    n, m = dist.shape
    extended_dist = np.full((n + m, n + m), inf, dtype=dist.dtype)
    extended_dist[:n, :m] = dist
    np.fill_diagonal(extended_dist[:n, m:], eta / 2)
    np.fill_diagonal(extended_dist[n:, :m], eta / 2)
    extended_dist[n:, m:] = 0

    return extended_dist


def diag_split_sparse(dist: np.ndarray, eta: float, inf: float = np.inf) -> np.ndarray:
    """Extend to a (n + m, n + m) by keeping a maximum of sparsity by symmetry.

    dist -> |     dist     , Diag(eta / 2) |
            | Diag(eta / 2), 0[dist.T == inf |

    Here Diag(eta) = | eta inf ... inf |
                     | inf eta ... ... |
                     | ... ... ... inf |
                     | inf ... inf eta |

    Adds very few non-inf cost: it is sparse compatible.

    The only implementation that enable sparsity and a square matrix; and therefore the only choice for lapmod.
    (Note that we rather use the sparse implementation of this method, see `pylapy.sparse`)

    Args:
        dist (np.ndarray): Distance matrix to extend
            Shape: (N, M), dtype: float
        eta (float): Alternative cost.
        inf (float): Current inf value
            Default: np.inf

    Returns:
        np.ndarray: Equivalent extended distance matrix
            Shape: (N + M, N + M), dtype: float
    """
    n, m = dist.shape
    extended_dist = np.full((n + m, n + m), inf, dtype=dist.dtype)
    extended_dist[:n, :m] = dist
    np.fill_diagonal(extended_dist[:n, m:], eta / 2)
    np.fill_diagonal(extended_dist[n:, :m], eta / 2)
    extended_dist[n:, m:][dist.T < inf] = 0  # noqa: SIM300

    return extended_dist


def fill_row_pad_zero(dist: np.ndarray, eta: float, inf: float = np.inf) -> np.ndarray:  # noqa: ARG001
    """Extend to a (n + m, n + m) by setting the cost limit on the rows.

    dist -> | dist, eta |
            |  0  ,  0  |

    One of the fastest choice (except for lapsolver backend).

    Args:
        dist (np.ndarray): Distance matrix to extend
            Shape: (N, M), dtype: float
        eta (float): Alternative cost.
        inf (float): Current inf value
            Default: np.inf

    Returns:
        np.ndarray: Equivalent extended distance matrix
            Shape: (N + M, N + M), dtype: float
    """
    n, m = dist.shape
    extended_dist = np.full((n + m, n + m), 0, dtype=dist.dtype)
    extended_dist[:n, :m] = dist
    extended_dist[:n, m:] = eta

    return extended_dist


def fill_row_pad_inf(dist: np.ndarray, eta: float, inf: float = np.inf) -> np.ndarray:
    """Extend to a (n + m, n + m) by setting the cost limit on the rows and inf everywhere else.

    dist -> | dist, eta |
            | inf , inf |

    Equivalent to the implementation in lap v0.4.0.

    Args:
        dist (np.ndarray): Distance matrix to extend
            Shape: (N, M), dtype: float
        eta (float): Alternative cost.
        inf (float): Current inf value
            Default: np.inf

    Returns:
        np.ndarray: Equivalent extended distance matrix
            Shape: (N + M, N + M), dtype: float
    """
    n, m = dist.shape
    extended_dist = np.full((n + m, n + m), inf, dtype=dist.dtype)
    extended_dist[:n, :m] = dist
    extended_dist[:n, m:] = eta

    return extended_dist


def fill_col_pad_zero(dist: np.ndarray, eta: float, inf: float = np.inf) -> np.ndarray:  # noqa: ARG001
    """Extend to a (n + m, n + m) by setting the cost limit on the columns.

    dist -> | dist, 0 |
            | eta , 0 |

    Args:
        dist (np.ndarray): Distance matrix to extend
            Shape: (N, M), dtype: float
        eta (float): Alternative cost.
        inf (float): Current inf value
            Default: np.inf

    Returns:
        np.ndarray: Equivalent extended distance matrix
            Shape: (N + M, N + M), dtype: float
    """
    n, m = dist.shape
    extended_dist = np.full((n + m, n + m), 0, dtype=dist.dtype)
    extended_dist[:n, :m] = dist
    extended_dist[n:, :m] = eta

    return extended_dist


def fill_col_pad_inf(dist: np.ndarray, eta: float, inf: float = np.inf) -> np.ndarray:
    """Extend to a (n + m, n + m) by setting the cost limit on the columns and inf everywhere else.

    dist -> | dist, inf |
            | eta , inf |

    Args:
        dist (np.ndarray): Distance matrix to extend
            Shape: (N, M), dtype: float
        eta (float): Alternative cost.
        inf (float): Current inf value
            Default: np.inf

    Returns:
        np.ndarray: Equivalent extended distance matrix
            Shape: (N + M, N + M), dtype: float
    """
    n, m = dist.shape
    extended_dist = np.full((n + m, n + m), inf, dtype=dist.dtype)
    extended_dist[:n, :m] = dist
    extended_dist[n:, :m] = eta

    return extended_dist


def diag_row_pad_zero(dist: np.ndarray, eta: float, inf: float = np.inf) -> np.ndarray:
    """Extend to a (n + m, n + m) by setting the cost limit on the diagonal rows.

    dist -> | dist, Diag(eta) |
            |  0  ,     0     |

    Here Diag(eta) = | eta inf ... inf |
                     | inf eta ... ... |
                     | ... ... ... inf |
                     | inf ... inf eta |

    One of the fastest choice (except for lapsolver backend) and the default for scipy/lap/lapjv.

    Args:
        dist (np.ndarray): Distance matrix to extend
            Shape: (N, M), dtype: float
        eta (float): Alternative cost.
        inf (float): Current inf value
            Default: np.inf

    Returns:
        np.ndarray: Equivalent extended distance matrix
            Shape: (N + M, N + M), dtype: float
    """
    n, m = dist.shape
    extended_dist = np.full((n + m, n + m), 0, dtype=dist.dtype)
    extended_dist[:n, :m] = dist
    extended_dist[:n, m:] = inf
    np.fill_diagonal(extended_dist[:n, m:], eta)

    return extended_dist


def diag_row_pad_inf(dist: np.ndarray, eta: float, inf: float = np.inf) -> np.ndarray:
    """Extend to a (n + m, n + m) by setting the cost limit on the diagonal rows and inf everywhere else.

    dist -> | dist, Diag(eta) |
            | inf ,    inf    |

    Here Diag(eta) = | eta inf ... inf |
                     | inf eta ... ... |
                     | ... ... ... inf |
                     | inf ... inf eta |

    Adds very few non-inf cost: it is sparse compatible.

    Args:
        dist (np.ndarray): Distance matrix to extend
            Shape: (N, M), dtype: float
        eta (float): Alternative cost.
        inf (float): Current inf value
            Default: np.inf

    Returns:
        np.ndarray: Equivalent extended distance matrix
            Shape: (N + M, N + M), dtype: float
    """
    n, m = dist.shape
    extended_dist = np.full((n + m, n + m), inf, dtype=dist.dtype)
    extended_dist[:n, :m] = dist
    np.fill_diagonal(extended_dist[:n, m:], eta)

    return extended_dist


def diag_col_pad_0(dist: np.ndarray, eta: float, inf: float = np.inf) -> np.ndarray:
    """Extend to a (n + m, n + m) by setting the cost limit on the diagonal columns.

    dist -> |   dist   , 0 |
            | Diag(eta), 0 |

    Args:
        dist (np.ndarray): Distance matrix to extend
            Shape: (N, M), dtype: float
        eta (float): Alternative cost.
        inf (float): Current inf value
            Default: np.inf

    Returns:
        np.ndarray: Equivalent extended distance matrix
            Shape: (N + M, N + M), dtype: float
    """
    n, m = dist.shape
    extended_dist = np.full((n + m, n + m), 0, dtype=dist.dtype)
    extended_dist[:n, :m] = dist
    extended_dist[n:, :m] = inf
    np.fill_diagonal(extended_dist[n:, :m], eta)

    return extended_dist


def diag_col_pad_inf(dist: np.ndarray, eta: float, inf: float = np.inf) -> np.ndarray:
    """Extend to a (n + m, n + m) by setting the cost limit on the diagonal columns and inf everywhere else.

    dist -> |   dist   , inf |
            | Diag(eta), inf |

    Adds very few non-inf cost: it is sparse compatible.

    Args:
        dist (np.ndarray): Distance matrix to extend
            Shape: (N, M), dtype: float
        eta (float): Alternative cost.
        inf (float): Current inf value
            Default: np.inf

    Returns:
        np.ndarray: Equivalent extended distance matrix
            Shape: (N + M, N + M), dtype: float
    """
    n, m = dist.shape
    extended_dist = np.full((n + m, n + m), inf, dtype=dist.dtype)
    extended_dist[:n, :m] = dist
    np.fill_diagonal(extended_dist[n:, :m], eta)

    return extended_dist


COST_EXTENSIONS: dict[str, CostExtensionProtocol] = {
    "none": no_cost_extension,
    "fill_split": fill_split,
    "diag_split": diag_split,
    "diag_split_sparse": diag_split_sparse,
    "fill_row_pad_zero": fill_row_pad_zero,
    "fill_row_pad_inf": fill_row_pad_inf,
    "fill_col_pad_zero": fill_col_pad_zero,
    "fill_col_pad_inf": fill_col_pad_inf,
    "diag_row_pad_zero": diag_row_pad_zero,
    "diag_row_pad_inf": diag_row_pad_inf,
    "diag_col_pad_0": diag_col_pad_0,
    "diag_col_pad_inf": diag_col_pad_inf,
}

# From the cost extension benchmark. Only lap and laprust have internal cost extension.
# laprust internal is much faster than ours. Otherwise diag_row_pad_zero (or fill_row_pad_zero)
# is the best extension for most methods (outperforms the internal lap)
# lapsolver is faster with diag_split_sparse.
DEFAULT_COST_EXTENSIONS: dict[str, CostExtensionProtocol] = {
    "scipy": diag_row_pad_zero,
    "lap": diag_row_pad_zero,
    "lapjv": diag_row_pad_zero,
    "lapsolver": diag_split_sparse,
    "laprust": no_cost_extension,  # internal cost_limit
}
