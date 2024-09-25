import importlib
from typing import Any, Callable, Optional, Union
import warnings

import numpy as np

from .cost_extension import row_cost, diag_split_cost
from .shape_extension import smallest_fill_0, smallest_fill_inf


class LapSolver:  # pylint: disable=too-few-public-methods
    """Solves assignement problem with Hungarian algorithm (JV variants)

    This class is a wrapper around different implementations: lapjv, lap, scipy, lapsolver.
    It also supports sparse implementations: csgraph (scipy) and lapmod (lap).

    It unifies the functionality of each implementation and allows you to use the one which is the fastest
    on your problem.

    It also helps you to handle non square matrices and setting a soft threshold on assignements (usually leads
    to better performances than hard thresholding).

    Handling non square matrices
    ----------------------------

    We provide several shape extension functions to convert a rectangular problem into a solvable square one.
    By default, we use the fastest one for the implementation you use. You can build your own or enforce another one
    by setting `shape_extension` attribute. Please have a look at `shape_extension` module and
    `benchmark_shape_extension` script.

    Handling soft thresholding
    --------------------------

    Rather than applying hard thresholding and cut links that are above a threshold `eta`, it is common and usually
    better to assign a row or a column to "no one" with a cost `eta`. This is done by adding "sink" rows and columns.
    When a true row/column is linked to a "sink" column/row, it is considered non linked.

    Adding these sink nodes can be done multiple ways resulting in equivalent links but different run time.
    We explore several of them in the `cost_extension` module and use the expected fastest for your implementation.
    You can build your own extension or enforce another one by setting the `cost_extension`
    attribute. Please have a look at `cost_extension` module and `benchmark_cost_extension` script.

    Time performances
    -----------------

    From our current benchhmarks, scipy, lap and lapjv are pretty fast and depending on the context,
    one is slightly faster than the others. Lapsolver is usually slower and is much slower with sparsity
    or cost limit.

    Attributes:
        implementations (Dict[str, str]): All implementations supported by the class. Maps the name to the module
        implementation (str): Implementation to use. It not provided, a suitable one is
            found by iterating in order through `implementations`.
        shape_extension (Callable[[np.ndarray, float], np.ndarray]): Called to convert rectangular
            lap problem into square ones (See `Handling non square matrix` and `shape_extension` module)
        cost_extension (Callable[[np.ndarray, float], np.ndarray]): Called to add soft thresholding
            (See `Handling soft thresholding` and `cost_extension` module)

    """

    implementations = {
        "lapjv": "lapjv",
        "lap": "lap",
        "scipy": "scipy.optimize",
        "lapsolver": "lapsolver",
    }
    sparse_implementations = {
        "csgraph": "scipy.sparse.csgraph",
        "lapmod": "lap",
    }

    def __init__(self, implementation: Optional[str] = None, sparse_implementation: Optional[str] = None) -> None:
        self.implementation = implementation
        self.sparse_implementation = sparse_implementation

        if self.implementation:
            importlib.import_module(self.implementations[self.implementation])
        else:
            for imp, module in self.implementations.items():
                try:
                    importlib.import_module(module)
                    self.implementation = imp
                    break
                except ImportError:
                    pass
            if self.implementation is None:
                raise ImportError(
                    f"Unable to import any of the following implementations {tuple(self.implementations.items())}"
                )

        if self.sparse_implementation:
            importlib.import_module(self.sparse_implementations[self.sparse_implementation])
        else:
            for imp, module in self.sparse_implementations.items():
                try:
                    importlib.import_module(module)
                    self.sparse_implementation = imp
                    break
                except ImportError:
                    pass

        self.shape_extension: Callable[[np.ndarray, float], np.ndarray] = smallest_fill_0
        self.cost_extension: Callable[[np.ndarray, float, float], np.ndarray] = row_cost
        self.sparse_extension: Union[str, Callable[[Any, float, bool], Any]] = "symmetric_cost"

        # For scipy and lapsolver the default ones are too slow. See benchmarks.
        if self.implementation == "scipy":
            self.shape_extension = smallest_fill_inf
        if self.implementation == "lapsolver":
            self.cost_extension = diag_split_cost
        if self.sparse_implementation == "csgraph":  # Scipy supports non-square matrices and its faster
            self.sparse_extension = "diag_row_cost"  # Note: symmetric can be faster with extra sparse problems

    def solve(self, dist: np.ndarray, eta=float("inf")) -> np.ndarray:
        """Solve the assignment problem for the given distances

        Handle inf values by replacing them by the sum of all non inf values + 1 so that we only choose
        one when no other choice is possible.

        One can give a soft threshold `eta` which represents the cost of linking row i or column j to "no one".
        The matrix is extended using `cost_extension` attribute: see `Handling soft thresholding`.

        If a non-square matrix is given without `eta` then the matrix is extended into a square one
        using `shape_extension` attribute: see `Handling non square matrix`.

        Links returned are only the valid links (no inf link nor link to "no one").

        Args:
            dist (np.ndarray): Distance matrix
                Shape: (N, M), dtype: float
            eta (float): Soft thresholding (Add the possibility to link with no one with a cost eta)
                Default: inf (No soft thresholding)

        Returns:
            np.ndarray: Links (i, j)
                Shape: (L, 2), dtype: uint16
        """
        dist = dist.copy()
        n, m = dist.shape

        # Handle 0 dimensions (Is almost handled by shape extension or/and backends but not fully)
        if min(n, m) == 0:
            return np.zeros((0, 2), dtype=np.uint16)

        # Replace inf as most lib do not support it well
        inf = dist[dist != np.inf].sum() + 1
        dist[dist == np.inf] = inf

        links: np.ndarray

        if not np.isinf(eta) or n != m:  # Extend the matrix if needed or asked
            if np.isinf(eta):  # n != m and full linking -> shape extension
                extended_dist = self.shape_extension(dist, inf)
            else:  # non full linking -> cost extension (also directly handles rectangular shapes)
                extended_dist = self.cost_extension(dist, eta, inf)

            links = getattr(self, f"_{self.implementation}")(extended_dist)

            # Filter links that are assigned outside of C
            links = links[links[:, 0] < n]  # i < n
            links = links[links[:, 1] < m]  # j < m
        else:  # No extension, the implementation can be called directly
            links = getattr(self, f"_{self.implementation}")(dist)

        # Filter inf links
        links = links[dist[links[:, 0], links[:, 1]] < inf]

        return links

    def sparse_solve(self, dist, eta: float, hard=False, feasible=False) -> np.ndarray:
        """Solve a sparse linear assignment problem for the given distance

        It will remove all links that are greater than `eta` to build a sparse cost matrix.

        By default, `eta` is used as a soft threshold, which represents the cost of linking row i
        or column j to "no one". This can be disabled to maximize the number of links no matter the
        cost, using `hard=True`.

        Typical implementation of sparse lapjv assumes that the matrix is squared, or at least fully
        feasible (the maximum number of links is min(N, M)). This is usually not the case in real life
        problems. In this code, we relax these assumptions, by extending the distance and adding "sink" nodes.

        The distance matrix is extended for soft thresholding, or for hard thresholding if feasible is False.
        By default, feasible is False and the distance is extended. It is up to the user to know if the input distance
        is feasible without adding sink nodes (for instance with scipy.csgraph.maximum_bipartite_matching(csr)).

        Links returned are only the valid links (no inf link nor link to "no one").

        Args:
            dist (Union[np.ndarray, scipy.sparse.spmatrix]): Distance matrix. If a sparse matrix is given
                we do not check that elements are all below eta.
                Shape: (N, M), dtype: float
            eta (float): Cost limit (soft thresholding or hard thresholding)
                No links will be made above this value. In soft thresholding, a link with a smaller value
                can also be avoided if it allows to save more than eta in cost in thefull graph.
            hard (bool): Do hard thresholding
                Default: False (Soft thresholding with sink nodes)
            feasible (bool): Whether the provided distance is fully feasible or not. (maximum number of links
                is min(N, M)).
                Default: False

        Returns:
            np.ndarray: Links (i, j)
                Shape: (L, 2), dtype: uint16

        """
        if not self.sparse_implementation:
            warnings.warn("No implementation found for sparse solving. Falling back to dense solving...")

            if hard:
                dist = np.array(dist)
                dist[dist < eta] = np.inf
                return self.solve(dist)

            return self.solve(np.asarray(dist), eta)

        # Scipy and numba are required
        import scipy.sparse  # type: ignore  # pylint: disable=import-outside-toplevel
        from . import sparse  # pylint: disable=import-outside-toplevel

        n, m = dist.shape

        if not hard or not feasible:
            if callable(self.sparse_extension):
                coo = self.sparse_extension(dist, eta, hard)  # pylint: disable=not-callable
                if not scipy.sparse.issparse(coo):
                    raise ValueError("The provided sparse_extension did not return a sparse matrix")

                coo = coo.tocoo()
            else:
                coo = sparse.sparse_extension(dist, eta, hard, self.sparse_extension)
        else:
            if isinstance(dist, np.ndarray):
                values, cols, rows = sparse.dense_to_coo(dist, eta)
                coo = scipy.sparse.coo_matrix((np.array(values, dtype=np.float32), (rows, cols)), shape=(n, m))
            elif scipy.sparse.issparse(dist):
                coo = dist.tocoo()
                if (coo.data > eta).any():  # Up to the user to handle this
                    warnings.warn("Found some values greater than the cost limit in the sparse format")
            else:
                raise ValueError(f"Dist should be a numpy array or a scipy sparse matrix. Found {type(dist)}.")

        coo.data += 0.1  # Add 0.1 to all values because 0 can be a tricky value

        if len(coo.data) == 0:
            return np.zeros((0, 2), dtype=np.uint16)

        links = getattr(self, f"_{self.sparse_implementation}")(coo.tocsr())

        # Filter links that are assigned outside of C
        links = links[links[:, 0] < n]  # i < n
        links = links[links[:, 1] < m]  # j < m

        return links

    @staticmethod
    def _lap(dist: np.ndarray) -> np.ndarray:
        """Solve with lap (https://github.com/gatagat/lap)

        Note: We are now using the lapx distribution from https://github.com/rathaROG/lapx.

        You should not call this method directly. To enforce using lap,
        rather set `implementation` attribute to "lap" and call `solve`.

        Args:
            dist (np.ndarray): Dist matrix
                Shape: (N, N), dtype: float

        Returns:
            np.ndarray: Links (i, j)
                Shape: (L, 2), dtype: uint16
        """
        import lap  # type: ignore # pylint: disable=import-outside-toplevel

        x: np.ndarray

        _, x, _ = lap.lapjv(dist)

        # Note that -1 are returned by lap when no link is made
        # In our case as we handle it ourselves, no -1 should be found
        if (x == -1).any():
            warnings.warn("-1 found in _lap. Should not occur except if you have called yourself the function.")

        i = np.arange(x.shape[0])[x != -1]
        j = x[x != -1]

        return np.array([i, j], dtype=np.uint16).transpose()

    @staticmethod
    def _lapjv(dist: np.ndarray) -> np.ndarray:
        """Solve with lapjv (https://github.com/src-d/lapjv)

        You should not call this method directly. To enforce using lapjv,
        rather set `implementation` attribute to "lapjv" and call `solve`.

        Args:
            dist (np.ndarray): Dist matrix
                Shape: (N, N), dtype: float

        Returns:
            np.ndarray: Links (i, j)
                Shape: (L, 2), dtype: uint16
        """
        import lapjv  # type: ignore # pylint: disable=import-outside-toplevel

        x, _, _ = lapjv.lapjv(dist)  # pylint: disable=c-extension-no-member
        i = np.arange(x.shape[0])
        j = x

        return np.array([i, j], np.uint16).transpose()

    @staticmethod
    def _scipy(dist: np.ndarray) -> np.ndarray:
        """Solve with scipy (scipy.optimize.linear_sum_assignment)

        You should not call this method directly. To enforce using scipy,
        rather set `implementation` attribute to "scipy" and call `solve`.

        Args:
            dist (np.ndarray): Dist matrix
                Shape: (N, N), dtype: float

        Returns:
            np.ndarray: Links (i, j)
                Shape: (L, 2), dtype: uint16
        """
        import scipy.optimize  # type: ignore # pylint: disable=import-outside-toplevel

        i, j = scipy.optimize.linear_sum_assignment(dist)

        return np.array([i, j], dtype=np.uint16).transpose()

    @staticmethod
    def _lapsolver(dist: np.ndarray) -> np.ndarray:
        """Solve with lapsolver (https://github.com/cheind/py-lapsolver)

        You should not call this method directly. To enforce using lapsolver,
        rather set `implementation` attribute to "lapsolver" and call `solve`.

        Args:
            dist (np.ndarray): Dist matrix
                Shape: (N, N), dtype: float

        Returns:
            np.ndarray: Links (i, j)
                Shape: (L, 2), dtype: uint16
        """
        import lapsolver  # type: ignore # pylint: disable=import-outside-toplevel

        i, j = lapsolver.solve_dense(dist)

        return np.array([i, j], np.uint16).transpose()

    @staticmethod
    def _csgraph(dist) -> np.ndarray:
        """Sparse solving with scipy (scipy.sparse.csgraph.min_weight_full_bipartite_matching)

        You should not call this method directly. To enforce using csgraph,
        rather set `sparse_implementation` attribute to "csgraph" and call `sparse_solve`.

        Args:
            dist (scipy.sparse.csr_matrix): Sparse dist matrix in the CSR format
                Shape: (N, N), dtype: float

        Returns:
            np.ndarray: Links (i, j)
                Shape: (L, 2), dtype: uint16
        """
        import scipy.sparse.csgraph  # type: ignore # pylint: disable=import-outside-toplevel

        i, j = scipy.sparse.csgraph.min_weight_full_bipartite_matching(dist)

        return np.array([i, j], np.uint16).transpose()

    @staticmethod
    def _lapmod(dist) -> np.ndarray:
        """Sparse solving with lapmod from lap (https://github.com/gatagat/lap)

        Note: We are now using the lapx distribution from https://github.com/rathaROG/lapx.

        You should not call this method directly. To enforce using lapmod,
        rather set `sparse_implementation` attribute to "lapmod" and call `sparse_solve`.

        Args:
            dist (scipy.sparse.csr_matrix): Sparse dist matrix in the CSR format
                Shape: (N, N), dtype: float

        Returns:
            np.ndarray: Links (i, j)
                Shape: (L, 2), dtype: uint16
        """
        import lap  # type: ignore # pylint: disable=import-outside-toplevel

        x: np.ndarray

        _, x, _ = lap.lapmod(dist.shape[0], dist.data, dist.indptr, dist.indices)

        # Note that -1 are returned by lap when no link is made
        # In our case as we handle it ourselves, no -1 should be found
        if (x == -1).any():
            warnings.warn("-1 found in _lap. Should not occur except if you have called yourself the function.")

        i = np.arange(x.shape[0])[x != -1]
        j = x[x != -1]

        return np.array([i, j], dtype=np.uint16).transpose()
