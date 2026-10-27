"""pylapy: one Pythonic interface over several Linear Assignment Problem (LAP) solvers.

The Linear Assignment Problem matches the rows and columns of a cost matrix one-to-one so as to minimize
the total cost. Several fast solvers exist in the Python ecosystem; `pylapy` wraps them behind a single
class, `LapSolver`, so you can switch backend -- or let it auto-pick the fastest installed one -- without
changing your code. On top of the raw solvers it adds what real problems need: rectangular matrices, `inf`
(impossible) entries, a `cost_limit` to discard over-long links, and an `alternative_cost` for leaving a
row/column unmatched.

Backends
--------

Every solver is an optional dependency -- `pylapy` itself only needs `numpy`:

- dense: `laprust`, `lap`, `lapjv`, `lapsolver` (all from `lapx`) and `scipy`.
- sparse: `lapmod` (`lapx`) and `csgraph` (`scipy`); sparse solving also needs `numba`.

`LapSolver()` auto-selects the first importable backend; pass a name to force one. Backends can differ by
10x or more on a given problem -- see the `benchmark` scripts and `LapSolver`'s docstring for guidance.

Example:
    >>> import numpy as np
    >>> import pylapy
    >>> dist = np.random.default_rng(0).random((2000, 1800))
    >>> solver = pylapy.LapSolver()  # auto-pick the fastest installed backend
    >>> links = solver.solve(dist)  # (L, 2) array of matched (row, col) pairs
    >>> links = solver.solve(dist, alternative_cost=0.1)  # a node may stay unmatched at cost 0.1
    >>> links = solver.sparse_solve(dist, cost_limit=0.1)  # drop links above 0.1 and solve sparsely
    >>> cost = dist[links[:, 0], links[:, 1]].sum()

Both methods return the matching as an `(L, 2)` `uint16` array of `(row, col)` links; row `i` is matched
to column `j` at cost `dist[i, j]`. Each row and each column appears at most once. For a plain assignment
`L = min(N, M)`; a `cost_limit` / `alternative_cost` can leave rows/columns unmatched, in which case they
are simply absent from the output (`L < min(N, M)`).

See `LapSolver` for the full API: the `cost_limit` / `alternative_cost` model, the pluggable
shape/cost/sparse extensions, and per-backend performance notes.
"""

import importlib.metadata

from pylapy.solver import LapSolver

__version__ = importlib.metadata.version("pylapy")
__all__ = ["LapSolver"]
