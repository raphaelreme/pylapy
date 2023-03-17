"""Implements several cost extension functions

Adds "sink" columns and rows to the cost matrix (also making it square if it is not the case).
When a true row/column is linked to a "sink" column/row, it is considered non linked.

We explore here multiple equivalent ways to set the cost between true nodes and sink nodes.

It seems that the fastest method depends on the sparsity of the initial distance and the
implementation used (scipy/lap/lapjv/lapsolver):

- For lapsolver, `split_cost`, `diag_split_cost` and `symmetric_sparse_extension` are the best
no matter the sparsity. We decided to use `diag_split_cost` by default.
- For scipy, `row_cost` and `row_cost_inf` are the best no matter the sparsity. We decided to use
`row_cost` by default.
- For lap and lapjv, it truly depends on sparsity. Best choices would be `row_cost` or `diag_row_cost`
that are the best with sparsity and great without. But we also emphasize `split_cost`, `diag_split_cost`
and `symmetric_sparse_extension` that are the best without sparsity. You should probably make a choice
depending on your data. We decided to use `row_cost` by default.

In some cases, other methods seems to be a good choice:
- `row_cost`: Very similar performances
- `split_cost`/`symmetric_sparse_extension`: Usually better if the matrix contains very few inf coef (non-sparse)
    but worse with sparse matrix
"""

import numpy as np


def split_cost(dist: np.ndarray, eta: float, inf: float = np.inf) -> np.ndarray:  # pylint: disable=unused-argument
    """Extend to a (n + m, n + m) by splitting the cost limit on rows and columns

    dist -> |  dist  , eta / 2 |
            | eta / 2,    0    |

    This is the implementation in lap v0.5.0.

    One of the fastest choice with non-sparse matrix.

    Args:
        dist (np.ndarray): Distance matrix to extend
            Shape: (N, M), dtype: float
        eta (float): Cost limit
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


def diag_split_cost(dist: np.ndarray, eta: float, inf: float = np.inf) -> np.ndarray:
    """Extend to a (n + m, n + m) by splitting the cost limit in diagonal block

    dist -> |     dist     , Diag(eta / 2) |
            | Diag(eta / 2),       0       |


    Here Diag(eta) = | eta inf ... inf |
                     | inf eta ... ... |
                     | ... ... ... inf |
                     | inf ... inf eta |

    One of the fastest choice with non-sparse matrix (Sligthly faster than `split_cost`).

    Args:
        dist (np.ndarray): Distance matrix to extend
            Shape: (N, M), dtype: float
        eta (float): Cost limit
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


def symmetric_sparse_extension(dist: np.ndarray, eta: float, inf: float = np.inf) -> np.ndarray:
    """Extend to a (n + m, n + m) by keeping a maximum of sparsity by symmetry

    dist -> |     dist     , Diag(eta / 2) |
            | Diag(eta / 2), dist.T == inf |

    Here Diag(eta) = | eta inf ... inf |
                     | inf eta ... ... |
                     | ... ... ... inf |
                     | inf ... inf eta |

    Probably the only choice if you want to use lapmod.
    It is also one of the fastest choice if your matrix is not sparse.

    Args:
        dist (np.ndarray): Distance matrix to extend
            Shape: (N, M), dtype: float
        eta (float): Cost limit
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
    extended_dist[n:, m:][dist.T != np.inf] = 0

    return extended_dist


def row_cost(dist: np.ndarray, eta: float, inf: float = np.inf) -> np.ndarray:  # pylint: disable=unused-argument
    """Extend to a (n + m, n + m) by setting the cost limit on the rows

    dist -> | dist, eta |
            |  0  ,  0  |

    One of the fastest choice when your matrix is sparse (except with lapsolver).
    Also great with non-sparse matrix and scipy.

    Args:
        dist (np.ndarray): Distance matrix to extend
            Shape: (N, M), dtype: float
        eta (float): Cost limit
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


def row_cost_inf(dist: np.ndarray, eta: float, inf: float = np.inf) -> np.ndarray:
    """Extend to a (n + m, n + m) by setting the cost limit on the rows and inf eveywhere else

    dist -> | dist, eta |
            | inf , inf |

    Equivalent to the implementation in lap v0.4.0.
    Usually leads to more computation with lap and lapjv. But one of the fastest with scipy.

    Args:
        dist (np.ndarray): Distance matrix to extend
            Shape: (N, M), dtype: float
        eta (float): Cost limit
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


def col_cost(dist: np.ndarray, eta: float, inf: float = np.inf) -> np.ndarray:  # pylint: disable=unused-argument
    """Extend to a (n + m, n + m) by setting the cost limit on the columns

    dist -> | dist, 0 |
            | eta , 0 |

    Usually leads to more computation, you should rather use another variant. (See module doctring).

    Args:
        dist (np.ndarray): Distance matrix to extend
            Shape: (N, M), dtype: float
        eta (float): Cost limit
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


def col_cost_inf(dist: np.ndarray, eta: float, inf: float = np.inf) -> np.ndarray:
    """Extend to a (n + m, n + m) by setting the cost limit on the columns and inf eveywhere else

    dist -> | dist, inf |
            | eta , inf |

    Usually leads to more computation, you should rather use another variant. (See module doctring).

    Args:
        dist (np.ndarray): Distance matrix to extend
            Shape: (N, M), dtype: float
        eta (float): Cost limit
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


def diag_row_cost(dist: np.ndarray, eta: float, inf: float = np.inf) -> np.ndarray:
    """Extend to a (n + m, n + m) by setting the cost limit on the diagonal rows

    dist -> | dist, Diag(eta) |
            |  0  ,     0     |

    Here Diag(eta) = | eta inf ... inf |
                     | inf eta ... ... |
                     | ... ... ... inf |
                     | inf ... inf eta |

    One of the fastest choice with sparsity (except with lapsolver). Usually faster than `row_cost`.

    Args:
        dist (np.ndarray): Distance matrix to extend
            Shape: (N, M), dtype: float
        eta (float): Cost limit
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


def diag_row_cost_inf(dist: np.ndarray, eta: float, inf: float = np.inf) -> np.ndarray:
    """Extend to a (n + m, n + m) by setting the cost limit on the diagonal rows and inf eveywhere else

    dist -> | dist, Diag(eta) |
            | inf ,    inf    |

    Here Diag(eta) = | eta inf ... inf |
                     | inf eta ... ... |
                     | ... ... ... inf |
                     | inf ... inf eta |

    One of the fastest with scipy, but much slower with other implementations.

    Args:
        dist (np.ndarray): Distance matrix to extend
            Shape: (N, M), dtype: float
        eta (float): Cost limit
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


def diag_col_cost(dist: np.ndarray, eta: float, inf: float = np.inf) -> np.ndarray:
    """Extend to a (n + m, n + m) by setting the cost limit on the diagonal columns

    dist -> |   dist   , 0 |
            | Diag(eta), 0 |

    Usually leads to more computation, you should rather use another variant. (See module doctring).

    Args:
        dist (np.ndarray): Distance matrix to extend
            Shape: (N, M), dtype: float
        eta (float): Cost limit
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


def diag_col_cost_inf(dist: np.ndarray, eta: float, inf: float = np.inf) -> np.ndarray:
    """Extend to a (n + m, n + m) by setting the cost limit on the diagonal columns and inf eveywhere else

    dist -> |   dist   , inf |
            | Diag(eta), inf |

    Usually leads to more computation, you should rather use another variant. (See module doctring).

    Args:
        dist (np.ndarray): Distance matrix to extend
            Shape: (N, M), dtype: float
        eta (float): Cost limit
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
