"""Implements several shape extension functions that convert a non-square assignement
problem into a square one (solvable with Hungarian or JV algorithm)

The shape extension functions should expect a distance matrix (non-square) and the current inf value
and return an equivalent square distance matrix.

They are many equivalent way of extending the distance matrix into a square one. We benchmark the fastest ones using
benchmark_shape_extension.py script. As expected, the two winners are `smallest_fill_inf` and `smallest_fill_0` that
builds that smallest squared matrix by filling by inf or 0 the missing part.

`smallest_fill_0` seems to be the fastest choice, except for scipy where it is quite slow,
and we advise to use `smallest_fill_inf`.
"""

import numpy as np


def smallest_fill_0(dist: np.ndarray, inf: float = np.inf) -> np.ndarray:  # pylint: disable=unused-argument
    """Fill with 0 the smallest dimension

    dist -> |dist, 0| or |dist.T, 0|.T

    This is the fastest shape extension we have tested. Except with scipy solver which is faster with fill_inf version
    (Note that we could also fill with any value)

    Args:
        dist (np.ndarray): Distance matrix to extend
            Shape: (N, M)
        inf (float): Current inf value
            Default: np.inf

    Returns:
        np.ndarray: Equivalent extended distance matrix
            Shape: (max(N, M), max(N, M))
    """
    n, m = dist.shape
    extended_dist = np.full((max(n, m), max(n, m)), 0, dtype=dist.dtype)
    extended_dist[:n, :m] = dist

    return extended_dist


def smallest_fill_inf(dist: np.ndarray, inf: float = np.inf) -> np.ndarray:
    """Fill with inf the smallest dimension

    dist -> |dist, inf| or |dist.T, inf|.T

    Best option for scipy solver and also quite good with others. fill_0 is usually faster.
    (Note that we could also fill with any value)

    This is the option implemented in lap v0.4.0.

    Args:
        dist (np.ndarray): Distance matrix to extend
            Shape: (N, M)
        inf (float): Current inf value
            Default: np.inf

    Returns:
        np.ndarray: Equivalent extended distance matrix
            Shape: (max(N, M), max(N, M))
    """
    n, m = dist.shape
    extended_dist = np.full((max(n, m), max(n, m)), inf, dtype=dist.dtype)
    extended_dist[:n, :m] = dist

    return extended_dist


def sum_fill_inf(dist: np.ndarray, inf: float = np.inf) -> np.ndarray:
    """Extend to a (n + m, n + m) filled with inf

    dist -> | dist, inf |
            | inf , inf |

    Time expensive, you should rather use the `smallest_fill_*` variants.

    Args:
        dist (np.ndarray): Distance matrix to extend
            Shape: (N, M)
        inf (float): Current inf value
            Default: np.inf

    Returns:
        np.ndarray: Equivalent extended distance matrix
            Shape: (N + M, N + M)
    """
    n, m = dist.shape
    extended_dist = np.full((n + m, n + m), inf, dtype=dist.dtype)
    extended_dist[:n, :m] = dist

    return extended_dist


def sum_split_inf(dist: np.ndarray, inf: float = np.inf) -> np.ndarray:
    """Extend to a (n + m, n + m) filled with inf on rows and columns

    dist -> | dist, inf |
            | inf ,  0  |

    Time expensive, you should rather use the `smallest_fill_*` variants.
    This is the option implemented in lap v0.5.0...

    Args:
        dist (np.ndarray): Distance matrix to extend
            Shape: (N, M)
        inf (float): Current inf value
            Default: np.inf

    Returns:
        np.ndarray: Equivalent extended distance matrix
            Shape: (N + M, N + M)
    """
    n, m = dist.shape
    extended_dist = np.full((n + m, n + m), inf, dtype=dist.dtype)
    extended_dist[:n, :m] = dist
    extended_dist[n:, m:] = 0

    return extended_dist


def sum_row_inf(dist: np.ndarray, inf: float = np.inf) -> np.ndarray:
    """Extend to a (n + m, n + m) filled with inf on rows

    dist -> | dist, inf |
            |  0  ,  0  |

    Time expensive, you should rather use the `smallest_fill_*` variants.

    Args:
        dist (np.ndarray): Distance matrix to extend
            Shape: (N, M)
        inf (float): Current inf value
            Default: np.inf

    Returns:
        np.ndarray: Equivalent extended distance matrix
            Shape: (N + M, N + M)
    """
    n, m = dist.shape
    extended_dist = np.full((n + m, n + m), 0, dtype=dist.dtype)
    extended_dist[:n, :m] = dist
    extended_dist[:n, m:] = inf

    return extended_dist


def sum_col_inf(dist: np.ndarray, inf: float = np.inf) -> np.ndarray:
    """Extend to a (n + m, n + m) filled with inf on columns

    dist -> | dist, 0 |
            | inf , 0 |

    Time expensive, you should rather use the `smallest_fill_*` variants.

    Args:
        dist (np.ndarray): Distance matrix to extend
            Shape: (N, M)
        inf (float): Current inf value
            Default: np.inf

    Returns:
        np.ndarray: Equivalent extended distance matrix
            Shape: (N + M, N + M)
    """
    n, m = dist.shape
    extended_dist = np.full((n + m, n + m), 0, dtype=dist.dtype)
    extended_dist[:n, :m] = dist
    extended_dist[:n, m:] = inf

    return extended_dist
