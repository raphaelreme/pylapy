"""Shape extension functions to convert a rectangular assignment problem into a square one.

A shape extension takes a (rectangular) distance matrix and the current `inf` sentinel value and returns an
equivalent square matrix that a Jonker-Volgenant backend can solve. The added rows/columns are padding;
links landing on them are dropped by `LapSolver` after solving.

Which one to use (see the `benchmark_shape_extension.py` script):

- Every backend we ship (through `lapx`) now handles rectangular input natively, and that internal path
  (`no_shape_extension`) is on par with or faster than any explicit padding. It is therefore the default
  for every backend (see `DEFAULT_SHAPE_EXTENSIONS`).
- If you must pad explicitly, `smallest_fill_zero` (pad the smaller side with 0) is usually the fastest choice.
  Its inf counterpart (`smallest_fill_inf`) can be fastest for gated rectangular problems with `lap` and `lapjv`
  backends. The `sum_*` variants are only slower, because of their larger (n + m > max(n, m)) extensions.
- Historically `inf` filled variants made the `lapjv` backend return wrong assignments because of a floating-point
  precision bug in its cost comparisons, fixed in `lapx` >= 0.13, still present in `src-d/lapjv`. A genuinely extreme
  sentinel can still lose precision on any backend (except `scipy` and `laprust` that are robust to this).
  pylapy never reaches extreme enough `inf` sentinel values (tested up to n=10^5) to trigger this precision loss.

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

    Slower than `smallest_fill_zero`; prefer `smallest_fill_zero`. (Historically unsafe with the `lapjv`
    backend when the padding held the large inf sentinel -- a floating-point precision bug fixed in
    `lapx` >= 0.13; see the module docstring.)

    Note that for gated rectangular matrices, it is the fastest choice for lap & lapjv.

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

    Inefficient; prefer the `smallest_fill_*` variants. (Historically unsafe with the `lapjv` backend on
    the large inf sentinel -- a floating-point precision bug fixed in `lapx` >= 0.13; see the module
    docstring.)

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

    Inefficient; prefer the `smallest_fill_*` variants. (Historically unsafe with the `lapjv` backend on
    the large inf sentinel -- a floating-point precision bug fixed in `lapx` >= 0.13; see the module
    docstring.)
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

    Inefficient; prefer the `smallest_fill_*` variants. (Historically unsafe with the `lapjv` backend on
    the large inf sentinel -- a floating-point precision bug fixed in `lapx` >= 0.13; see the module
    docstring.)

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

    Inefficient; prefer the `smallest_fill_*` variants. (Historically unsafe with the `lapjv` backend on
    the large inf sentinel -- a floating-point precision bug fixed in `lapx` >= 0.13; see the module
    docstring.)

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
