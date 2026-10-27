from __future__ import annotations

import importlib
import warnings
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from pylapy.cost_extension import COST_EXTENSIONS, DEFAULT_COST_EXTENSIONS
from pylapy.shape_extension import DEFAULT_SHAPE_EXTENSIONS, SHAPE_EXTENSIONS

if TYPE_CHECKING:
    from pylapy.cost_extension import CostExtensionProtocol
    from pylapy.shape_extension import ShapeExtensionProtocol
    from pylapy.sparse import SparseExtensionProtocol, sparray


class LapSolver:
    """Unified solver for the Linear Assignment Problem (Jonker-Volgenant and friends).

    A thin wrapper over several LAP backends, all currently provided by `lapx` and `scipy`:

    - dense: `laprust` (lapx), `lap` (lapx), `lapjv` (lapx), `scipy` (scipy), `lapsolver` (lapx)
    - sparse: `lapmod` (lapx), `csgraph` (scipy)

    It exposes a single interface (`solve` / `sparse_solve`), lets you pick (or auto-detect) the backend,
    and transparently handles rectangular matrices and `inf` entries.

    Cost limit and alternative cost
    -------------------------------

    Both `solve` and `sparse_solve` take two optional, independent parameters (both default to `inf`):

    - `cost_limit`: thresholds the distances. Any link with a cost above `cost_limit` is discarded; a node
      with no remaining link is left unmatched.
    - `alternative_cost`: the cost of an alternative "sink" link, i.e. of leaving a row/column unmatched. A
      node is matched only when its best remaining link is cheaper than this. With `alternative_cost=inf`
      no alternative is offered (a node is left unmatched only when it has no remaining link).

    By definition, `alternative_cost` superseds `cost_limit` as it prevents any link with higher cost.

    So `solve(dist)` is a plain full assignment, `solve(dist, cost_limit=eta)` discards links above `eta`
    (matching as many of the rest as possible), and `solve(dist, alternative_cost=eta)` lets a node stay
    unmatched at cost `eta`. Backends that cannot apply these (or take rectangular input) natively are
    handled by rewriting the cost matrix via the extensions below.

    Extensions
    ----------

    `shape_extension`, `cost_extension` and `sparse_extension` can each be set to a registry name (str) or
    to your own callable (see the matching module and `benchmark` scripts). Sensible defaults are chosen
    per backend; where a backend applies a cost limit, alternative cost or rectangular input natively,
    that native path is used by default.
    `enforce_less_row_than_col` (default True) transposes the problem internally so there are fewer rows
    than columns, which is faster for several backends and never changes the result.

    Time performances
    -----------------

    From the `benchmark` studies:

    - `laprust` is the best general-purpose default: fastest (or tied) both for plain assignment and with
      an alternative cost (its native cost limit avoids any dense rewrite).
    - Its one weak spot is a dense problem that only sets a cost limit (`alternative_cost=inf`), where it
      degrades at scale. For such discarded-link / sparse problems prefer a sparse backend (`csgraph` or
      `lapmod`); `scipy` is a decent dense fallback.
    - `lap` and `scipy` are competitive for plain assignment but much slower once a cost extension
      is involved; `lapsolver` is consistently the slowest; `lapjv` is slow and can return wrong
      assignments on cost-limited / inf-heavy problems.
    - Sparse backends carry a small constant overhead, so they only pay off from a few hundred nodes up.

    Attributes:
        implementations (dict[str, str]): Supported dense backends, mapping name -> importable module.
            Iterated in order to auto-select a backend when none is given.
        sparse_implementations (dict[str, str]): Supported sparse backends, likewise.
        implementation (str): The selected dense backend.
        sparse_implementation (str | None): The selected sparse backend (None if none is importable).
        shape_extension (ShapeExtensionProtocol): Rectangular -> square rewrite (name or callable).
        cost_extension (CostExtensionProtocol): Alternative-cost sink rewrite (name or callable).
        sparse_extension (SparseExtensionProtocol): Sparse gated/sink rewrite (name or callable).
        enforce_less_row_than_col (bool): Transpose internally to keep rows <= columns. Default: True.

    """

    # We now rely on lapx (rathaROG/lapx) for a stable implementation of:
    # 1. lap & lapmod (gatagat/lap)
    # 2. lapjv (src-d/lapjv)
    # 3. lapsolver (cheind/py-lapsolver)
    # 4. laprust (rathaROG/lapx)
    implementations: ClassVar = {
        "laprust": "lap",
        "lap": "lap",
        "scipy": "scipy.optimize",
        "lapsolver": "lap",  # The slowest option
        "lapjv": "lap",  # Let's put lapjv last: it is unreliable.
    }
    sparse_implementations: ClassVar = {
        "lapmod": "lap",
        "csgraph": "scipy.sparse.csgraph",
    }

    def __init__(self, implementation: str | None = None, sparse_implementation: str | None = None) -> None:
        self.implementation = implementation
        self.sparse_implementation = sparse_implementation
        self.enforce_less_row_than_col = True

        if self.implementation:
            importlib.import_module(self.implementations[self.implementation])
        else:  # pragma: no cover  # Covered but coverage fails to get it
            for imp, module in self.implementations.items():
                try:
                    importlib.import_module(module)
                    self.implementation = imp
                    break
                except ImportError:  # pragma: no cover
                    pass
            if self.implementation is None:
                raise ImportError(  # pragma: no cover
                    f"Unable to import any of the following implementations {tuple(self.implementations.items())}"
                )

        if self.sparse_implementation:
            importlib.import_module(self.sparse_implementations[self.sparse_implementation])
        else:  # pragma: no cover  # Covered but coverage fails to get it
            for imp, module in self.sparse_implementations.items():
                try:
                    importlib.import_module(module)
                    self.sparse_implementation = imp
                    break
                except ImportError:  # pragma: no cover
                    pass

        self._shape_extension: str | ShapeExtensionProtocol = DEFAULT_SHAPE_EXTENSIONS.get(
            self.implementation, "smallest_fill_zero"
        )
        self._cost_extension: str | CostExtensionProtocol = DEFAULT_COST_EXTENSIONS.get(
            self.implementation, "diag_row_pad_zero"
        )

        # Let's not rely on the DEFAULT_SPARSE_EXTENSION to avoid import
        self._sparse_extension: str | SparseExtensionProtocol = (
            "diag_row_no_pad" if self.sparse_implementation == "csgraph" else "diag_split_sparse"
        )

    @property
    def shape_extension(self) -> ShapeExtensionProtocol:
        """Shape extension callable (rectangular -> square), resolving a registry name to its function."""
        if isinstance(self._shape_extension, str):
            return SHAPE_EXTENSIONS[self._shape_extension]

        return self._shape_extension

    @shape_extension.setter
    def shape_extension(self, shape_extension: str | ShapeExtensionProtocol) -> None:
        """Set the shape extension, as a `SHAPE_EXTENSIONS` name or a `ShapeExtensionProtocol` callable."""
        self._shape_extension = shape_extension

    @property
    def cost_extension(self) -> CostExtensionProtocol:
        """Cost extension callable (alternative-cost sinks), resolving a registry name to its function."""
        if isinstance(self._cost_extension, str):
            return COST_EXTENSIONS[self._cost_extension]

        return self._cost_extension

    @cost_extension.setter
    def cost_extension(self, cost_extension: str | CostExtensionProtocol) -> None:
        """Set the cost extension, as a `COST_EXTENSIONS` name or a `CostExtensionProtocol` callable."""
        self._cost_extension = cost_extension

    @property
    def sparse_extension(self) -> SparseExtensionProtocol:
        """Sparse extension callable (gated COO + sink edges), resolving a registry name to its function."""
        import pylapy.sparse  # noqa: PLC0415

        if isinstance(self._sparse_extension, str):
            return pylapy.sparse.SPARSE_EXTENSIONS[self._sparse_extension]

        return self._sparse_extension

    @sparse_extension.setter
    def sparse_extension(self, sparse_extension: str | SparseExtensionProtocol) -> None:
        """Set the sparse extension, as a `SPARSE_EXTENSIONS` name or a `SparseExtensionProtocol` callable."""
        self._sparse_extension = sparse_extension

    def solve(self, dist: np.ndarray, *, cost_limit=float("inf"), alternative_cost=float("inf")) -> np.ndarray:
        """Solve the (optionally thresholded) assignment problem for a dense distance matrix.

        Two independent parameters decide what happens to a row/column that is not matched (see the class
        docstring):

        - `cost_limit`: links with a cost above it are forbidden; a node with no admissible link is left
          unmatched.
        - `alternative_cost`: the cost of leaving a node unmatched. A node is matched only when its best
          admissible link is cheaper than this. Nodes with admissible links lower than this alternative cost
          can also be skipped if they induce larger cost in the global assignement.

        By definition, `alternative_cost` superseds `cost_limit` as it prevents any link with higher cost.

        Common uses: `solve(dist)` (full assignment), `solve(dist, cost_limit=eta)` (discard links above
        `eta`), `solve(dist, alternative_cost=eta)` (a node may stay unmatched at cost `eta`).

        Rectangular matrices and `inf` entries are handled transparently: `inf` (and above-`cost_limit`)
        entries are replaced internally by a large finite sentinel, the matrix is squared with
        `shape_extension` (or by the backend) and/or `cost_extension` as needed, and the links are mapped
        back. The returned links are the real feasible ones only (never a padding/sink node, never a cost
        above `cost_limit`).

        Args:
            dist (np.ndarray): Distance matrix.
                Shape: (N, M), dtype: float
            cost_limit (float): Maximum admissible link cost. Default: inf (no cutoff).
            alternative_cost (float): Cost of leaving a row/column unmatched. Default: inf (a node is left
                unmatched only when it has no admissible link).

        Returns:
            np.ndarray: Links (i, j).
                Shape: (L, 2), dtype: uint16
        """
        dist = dist.copy()
        n, m = dist.shape
        transposed = False

        # Edge case for empty matrices
        if dist.size == 0:
            return np.zeros((0, 2), dtype=np.uint16)

        # inf defined as a large enough value (inf is not supported by many backend).
        # We use the sum of absolute values (not the signed sum): Even with negative costs
        # it is >= any single |cost| and >= the magnitude of any assignment,
        inf = np.abs(dist[dist != np.inf]).sum() + 1

        if not np.isinf(cost_limit):
            dist[dist > cost_limit] = inf
        else:
            dist[dist == np.inf] = inf  # Replace pre-existing inf in dist

        if n > m and self.enforce_less_row_than_col:  # Transpose to solve with fewer rows
            dist = dist.T
            m, n = n, m
            transposed = True

        if not np.isinf(alternative_cost):
            dist = self.cost_extension(dist, alternative_cost, inf)
            if dist.shape != (n, m):  # Externally handled
                alternative_cost = float("inf")  # so the solver should not extend it.

        if dist.shape[0] != dist.shape[1]:  # If still not squared
            dist = self.shape_extension(dist, inf)

        links: np.ndarray = getattr(self, f"_{self.implementation}")(dist, alternative_cost)

        if dist.shape != (n, m):  # Dist was extended, so let's remove the links to sink nodes.
            links = links[links[:, 0] < n]  # i < n
            links = links[links[:, 1] < m]  # j < m

        # Filter links with inf cost
        links = links[dist[links[:, 0], links[:, 1]] < inf]

        if transposed:  # Transpose back the links
            links = links[:, ::-1]

        return links

    def sparse_solve(
        self,
        dist: np.ndarray | sparray,
        *,
        cost_limit=float("inf"),
        alternative_cost=float("inf"),
        assume_feasible: bool = False,
    ) -> np.ndarray:
        """Sparse analogue of `solve`: same `cost_limit` / `alternative_cost` parameters, sparse backend.

        `cost_limit` and `alternative_cost` mean exactly what they do in `solve`. Entries above
        min(`cost_limit`, `alternative_cost`) are dropped to build the sparse matrix, so a finite limit
        is what makes the problem sparse (or pass an already-sparse `dist`, whose stored entries are taken
        as the remaining links). A finite `alternative_cost` adds an alternative sink link at that cost.
        `alternative_cost=inf` offers no alternative.

        Without an alternative cost, a sparse problem can be infeasible (a row with no remaining link),
        which breaks the backends; by default sink edges are added to keep it solvable (these edges are
        priced very high and used only as a last resort, for feasibility). If you already know
        a full matching exists within `cost_limit` (e.g. via
        `scipy.sparse.csgraph.maximum_bipartite_matching`), pass `assume_feasible=True` to skip the sink
        extension and solve the gated matrix directly (faster). It is ignored when `alternative_cost` is
        finite (sinks are then required by design).

        Falls back to the dense `solve` (with a warning) if no sparse backend is available and `dist` is
        dense.

        Args:
            dist (np.ndarray | sparray): Distance matrix (dense or scipy sparse).
                Shape: (N, M), dtype: float
            cost_limit (float): Links above this are discarded.
                Default: inf.
            alternative_cost (float): Cost of an alternative sink link (leaving a node unmatched); inf
                offers no alternative. Default: inf.
            assume_feasible (bool): Skip the feasibility sink extension; valid only when `alternative_cost`
                is inf and a full matching exists within `cost_limit`. Default: False.

        Returns:
            np.ndarray: Links (i, j).
                Shape: (L, 2), dtype: uint16
        """
        # Edge case for empty matrices
        if dist.size == 0:
            return np.zeros((0, 2), dtype=np.uint16)

        if not self.sparse_implementation:
            if isinstance(dist, np.ndarray):
                warnings.warn("No sparse implementation available. Falling back to dense solving...", stacklevel=2)
                return self.solve(dist, cost_limit=cost_limit, alternative_cost=alternative_cost)

            # Importing sparse_to_dense won't work, as scipy and numba are probably not available
            raise ValueError("No sparse implementation available. Please convert to dense matrix.")

        # Scipy and numba are required
        import scipy.sparse  # noqa: PLC0415

        import pylapy.sparse  # noqa: PLC0415

        n, m = dist.shape  # type: ignore[misc]
        transposed = False

        if n > m and self.enforce_less_row_than_col:  # Transpose to solve with fewer rows
            dist = dist.T
            m, n = n, m
            transposed = True

        if np.isinf(alternative_cost) and assume_feasible:  # Let's trust the user and try to solve the direct problem
            # Let's convert to coo and removing invalid values (> cost_limit or inf).
            values, rows, cols = pylapy.sparse._to_coo_format(dist, cost_limit)  # noqa: SLF001
            dist = scipy.sparse.coo_array((values, (rows, cols)), dist.shape, dist.dtype)
        else:  # Let's do a cost_extension
            # alternative_cost=inf: no alternative sink, only the cost_limit threshold on the distances.
            dist = self.sparse_extension(dist, cost_limit=cost_limit, alternative_cost=alternative_cost)

        links: np.ndarray = getattr(self, f"_{self.sparse_implementation}")(dist)

        if dist.shape != (n, m):  # Dist was extended, so let's remove the links to sink nodes.
            links = links[links[:, 0] < n]  # i < n
            links = links[links[:, 1] < m]  # j < m

        if transposed:  # Transpose back the links
            links = links[:, ::-1]

        return links

    @staticmethod
    def _lap(dist: np.ndarray, alternative_cost: float = float("inf")) -> np.ndarray:
        """Solve with lap (https://github.com/gatagat/lap).

        Note: We are now using the lapx distribution from https://github.com/rathaROG/lapx, which
              provides the equivalent `lap.lapjvxa` method.

        You should not call this method directly. To enforce using lap,
        rather set `implementation` attribute to "lap" and call `solve`.

        This backend internally supports for rectangular (N != M) matrices and alternative_cost.

        Args:
            dist (np.ndarray): Distance matrix.
                Shape: (N, M), dtype: float
            alternative_cost (float): Passed to the backend as its native cost limit (inf = no limit).

        Returns:
            np.ndarray: Links (i, j)
                Shape: (L, 2), dtype: uint16
        """
        import lap  # noqa: PLC0415

        return lap.lapjvxa(dist, extend_cost=True, cost_limit=alternative_cost, return_cost=False).astype(np.uint16)

    @staticmethod
    def _lapjv(dist: np.ndarray, alternative_cost: float = float("inf")) -> np.ndarray:
        """Solve with lapjv (https://github.com/src-d/lapjv).

        Note: We are now using the lapx distribution from https://github.com/rathaROG/lapx, which
              provides the equivalent `lap.lapjvxa` method.

        You should not call this method directly. To enforce using lapjv,
        rather set `implementation` attribute to "lapjv" and call `solve`.

        This backend internally supports for rectangular (N != M) matrices but not alternative_cost.

        Args:
            dist (np.ndarray): Distance matrix
                Shape: (N, M), dtype: float
            alternative_cost (float): Must be inf; this backend has no native alternative cost (raises
                otherwise). Set a `cost_extension` to support one.

        Returns:
            np.ndarray: Links (i, j)
                Shape: (L, 2), dtype: uint16
        """
        if not np.isinf(alternative_cost):
            raise ValueError("Lapjv backend do not support internal alternative cost. Please set a `cost_extension`.")

        import lap  # noqa: PLC0415

        return lap.lapjvsa(dist, extend_cost=True, return_cost=False).astype(np.uint16)

    @staticmethod
    def _scipy(dist: np.ndarray, alternative_cost: float = float("inf")) -> np.ndarray:
        """Solve with scipy (scipy.optimize.linear_sum_assignment).

        You should not call this method directly. To enforce using scipy,
        rather set `implementation` attribute to "scipy" and call `solve`.

        This backend internally supports for rectangular (N != M) matrices but not alternative_cost.

        Args:
            dist (np.ndarray): Distance matrix
                Shape: (N, M), dtype: float
            alternative_cost (float): Must be inf; this backend has no native alternative cost (raises
                otherwise). Set a `cost_extension` to support one.

        Returns:
            np.ndarray: Links (i, j)
                Shape: (L, 2), dtype: uint16
        """
        if not np.isinf(alternative_cost):
            raise ValueError("Scipy backend do not support internal alternative cost. Please set a `cost_extension`.")

        import scipy.optimize  # noqa: PLC0415

        i, j = scipy.optimize.linear_sum_assignment(dist)

        return np.array([i, j], dtype=np.uint16).transpose()

    @staticmethod
    def _lapsolver(dist: np.ndarray, alternative_cost: float = float("inf")) -> np.ndarray:
        """Solve with lapsolver (https://github.com/cheind/py-lapsolver).

        Note: We are now using the lapx distribution from https://github.com/rathaROG/lapx, which
              provides the equivalent `lap.lapjvc` method.

        You should not call this method directly. To enforce using lapsolver,
        rather set `implementation` attribute to "lapsolver" and call `solve`.

        This backend internally supports for rectangular (N != M) matrices but not alternative_cost.

        Args:
            dist (np.ndarray): Distance matrix
                Shape: (N, M), dtype: float
            alternative_cost (float): Must be inf; this backend has no native alternative cost (raises
                otherwise). Set a `cost_extension` to support one.

        Returns:
            np.ndarray: Links (i, j)
                Shape: (L, 2), dtype: uint16
        """
        if not np.isinf(alternative_cost):
            raise ValueError(
                "LapSolver backend do not support internal alternative cost. Please set a `cost_extension`."
            )

        import lap  # noqa: PLC0415

        return np.stack(lap.lapjvc(dist, return_cost=False)).T.astype(np.uint16)

    @staticmethod
    def _laprust(dist: np.ndarray, alternative_cost: float = float("inf")) -> np.ndarray:
        """Solve with laprust (https://github.com/rathaROG/lapx).

        You should not call this method directly. To enforce using laprust,
        rather set `implementation` attribute to "laprust" and call `solve`.

        This backend internally supports for rectangular (N != M) matrices and alternative_cost.

        Args:
            dist (np.ndarray): Distance matrix
                Shape: (N, M), dtype: float
            alternative_cost (float): Passed to the backend as its native cost limit (inf = no limit).

        Returns:
            np.ndarray: Links (i, j)
                Shape: (L, 2), dtype: uint16
        """
        import lap  # noqa: PLC0415

        return lap.lapjvr(
            dist, extend_cost=True, cost_limit=alternative_cost, return_cost=False, output="pairs"
        ).astype(np.uint16)

    @staticmethod
    def _csgraph(dist: sparray) -> np.ndarray:
        """Sparse solving with scipy (scipy.sparse.csgraph.min_weight_full_bipartite_matching).

        You should not call this method directly. To enforce using csgraph,
        rather set `sparse_implementation` attribute to "csgraph" and call `sparse_solve`.

        Warning: This method modify inplace the input, if it contains negative values.

        This backend supports rectangular (N != M) sparse matrices.

        Args:
            dist (sparray): Sparse dist matrix.
                Shape: (N, M), dtype: float

        Returns:
            np.ndarray: Links (i, j)
                Shape: (L, 2), dtype: uint16
        """
        import scipy.sparse.csgraph  # noqa: PLC0415

        csr = dist.tocsr()

        # Remove 0 value: scipy does not support it.
        mini = csr.data.min()
        if mini <= 0.0:
            csr.data += 0.1 - mini

        return np.stack(scipy.sparse.csgraph.min_weight_full_bipartite_matching(csr)).T.astype(np.uint16)

    @staticmethod
    def _lapmod(dist: sparray) -> np.ndarray:
        """Sparse solving with lapmod from lap (https://github.com/gatagat/lap).

        Note: We are now using the lapx distribution from https://github.com/rathaROG/lapx, which
              provides the equivalent `lap.lapmod` method.

        You should not call this method directly. To enforce using lapmod,
        rather set `sparse_implementation` attribute to "lapmod" and call `sparse_solve`.

        Warning: This method modify inplace the input, if it contains negative values.

        This backend only support square matrices.

        Args:
            dist (sparray): Sparse dist matrix.
                Shape: (N, N), dtype: float

        Returns:
            np.ndarray: Links (i, j)
                Shape: (L, 2), dtype: uint16
        """
        import lap  # noqa: PLC0415

        if dist.shape[0] != dist.shape[1]:  # type: ignore[misc]
            # It only supports square matrices:
            # 1. Only diag_split_sparse cost extension
            # 2. Feasible rectangular matrices are not supported (no sparse shape extension)
            raise ValueError(f"Lapmod only support squared matrix. Got {dist.shape}.")

        csr = dist.tocsr()

        # lapmod rejects negative costs. Shifting every stored value by a constant is objective-invariant
        # for the square perfect matching lapmod solves, so we shift into the positives (like _csgraph).
        mini = csr.data.min()
        if mini < 0.0:
            csr.data += 0.1 - mini

        x, _ = lap.lapmod(csr.shape[0], csr.data, csr.indptr, csr.indices, return_cost=False)
        return np.stack((np.arange(x.shape[0]), x)).T.astype(np.uint16)
