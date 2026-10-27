"""Shape extension functions to convert a rectangular assignment problem into a square one.

A shape extension takes a (rectangular) distance matrix and the current `inf` sentinel value and returns an
equivalent square matrix that a Jonker-Volgenant backend can solve. The added rows/columns are padding;
links landing on them are dropped by `LapSolver` after solving.

Which one to use (see the `benchmark_shape_extension.py` script):

- Every backend we ship (through `lapx`) now handles rectangular input natively, and that internal path
  (`no_shape_extension`) is on par with or faster than any explicit padding. It is therefore the default
  for every backend (see `DEFAULT_SHAPE_EXTENSIONS`).
- If you must pad explicitly, `smallest_fill_zero` (pad the smaller side with 0) is the fastest and the
  only safe choice. The inf-filled variants (`smallest_fill_inf`, `sum_fill_inf`, `sum_split_inf`,
  `sum_row_inf`, `sum_col_inf`) are slower and, worse, make the `lapjv` backend return wrong
  assignments (they mishandle the large inf sentinel). Prefer 0-padding, or let the backend handle it.

Each function follows `ShapeExtensionProtocol` and is registered by name in `SHAPE_EXTENSIONS`.
"""

from typing import Protocol

import numpy as np


class ShapeExtensionProtocol(Protocol):
    """Callable that pads a rectangular distance matrix into an equivalent square one.

    It receives the distance matrix and the `inf` sentinel value and returns a square matrix; the padding
    rows/columns are stripped from the result by the solver.
    """

    __name__: str

    def __call__(
        self,
        dist: np.ndarray,
        /,
        inf: float = ...,
    ) -> np.ndarray:
        """Return a square matrix equivalent to the rectangular `dist`, padding filled using `inf`."""


def no_shape_extension(dist: np.ndarray, inf: float = float("inf")) -> np.ndarray:  # noqa: ARG001
    """Do not extend the distance and let the backend internally solve for rectangular problems.

    lapjv (and old lapsolver) backend have no internal support for rectangular problems.
    (scipy and lapx solvers all support this.)
    """
    return dist


def smallest_fill_zero(dist: np.ndarray, inf: float = np.inf) -> np.ndarray:  # noqa: ARG001
    """Fill the smaller dimension with 0 to reach a square matrix.

    dist -> |dist, 0| or |dist.T, 0|.T

    The fastest and safest explicit shape extension: it works correctly with every backend. (The backends'
    own internal rectangular handling, `no_shape_extension`, is usually as fast or faster and is the
    default - use this one only when you want to pad explicitly. Any constant fill value would work.)

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
    """Fill the smaller dimension with inf to reach a square matrix.

    dist -> |dist, inf| or |dist.T, inf|.T

    Slower than `smallest_fill_zero` and unsafe with the `lapjv` backend, which returns wrong
    assignments when the padding holds the large inf sentinel. Prefer `smallest_fill_zero`.

    This was the option implemented in lap v0.4.0.

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
    """Extend to a (n + m, n + m) filled with inf.

    dist -> | dist, inf |
            | inf , inf |

    Inefficient, and unsafe with the `lapjv` backend (it mishandles the inf sentinel);
    prefer the `smallest_fill_*` variants.

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
    """Extend to a (n + m, n + m) filled with inf on rows and columns.

    dist -> | dist, inf |
            | inf ,  0  |

    Inefficient, and unsafe with the `lapjv` backend (it mishandles the inf sentinel);
    prefer the `smallest_fill_*` variants.
    This was the option implemented in lap v0.5.0.

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
    """Extend to a (n + m, n + m) filled with inf on rows.

    dist -> | dist, inf |
            |  0  ,  0  |

    Inefficient, and unsafe with the `lapjv` backend (it mishandles the inf sentinel);
    prefer the `smallest_fill_*` variants.

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
    """Extend to a (n + m, n + m) filled with inf on columns.

    dist -> | dist, 0 |
            | inf , 0 |

    Inefficient, and unsafe with the `lapjv` backend (it mishandles the inf sentinel);
    prefer the `smallest_fill_*` variants.

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
    extended_dist[n:, :m] = inf

    return extended_dist


SHAPE_EXTENSIONS: dict[str, ShapeExtensionProtocol] = {
    "no_shape_extension": no_shape_extension,
    "smallest_fill_zero": smallest_fill_zero,
    "smallest_fill_inf": smallest_fill_inf,
    "sum_fill_inf": sum_fill_inf,
    "sum_split_inf": sum_split_inf,
    "sum_row_inf": sum_row_inf,
    "sum_col_inf": sum_col_inf,
}


# From the shape extension benchmark. Now all the methods support shape extension.
# And their internal implementation (in lapx: smallest_fill_zero) is always on par or faster than our best one.
DEFAULT_SHAPE_EXTENSIONS: dict[str, ShapeExtensionProtocol] = {
    "scipy": no_shape_extension,
    "lap": no_shape_extension,
    "lapjv": no_shape_extension,
    "lapsolver": no_shape_extension,
    "laprust": no_shape_extension,
}
