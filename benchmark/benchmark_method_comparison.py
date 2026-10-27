"""Method Comparison Benchmark: race every backend with its chosen extension on all use cases.

Steps 1-3 picked, per backend, the best shape / cost / sparse extension. Here each method is pinned to its
choice (see METHOD_EXTENSIONS / SPARSE_METHOD_EXTENSIONS -- edit those to change the picks) and all methods
are timed against each other across the four cost configs x three squareness ratios.

Cost configs: ``full`` (no threshold), ``gated`` (hard threshold ``dist[dist>eta]=inf``), ``alternative``
(soft, finite ``eta`` + cost extension), ``gated_alternative`` (gated + soft). Sparse methods only run when
there is a cost limit (never on ``full``); ``hard = gated and not alternative`` (so ``alternative`` and
``gated_alternative`` are identical from the sparse point of view). Sparse methods receive the dense matrix,
so the dense->COO conversion is included in their timing. Everything runs with ``enforce_small_row=True``.

Correctness: each method's ``(link count, achieved cost)`` is compared to a scipy reference for that config.

Results are written to JSON for ``plot_method_comparison.py``.

Run:
    uv run python -m benchmark_v2.benchmark_method_comparison
    uv run python -m benchmark_v2.benchmark_method_comparison --sizes 5,10,50 --samples 2 --repeats 3
"""

from __future__ import annotations

import argparse
import functools
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

import pylapy.cost_extension as cost_ext
import pylapy.shape_extension as shape_ext
import pylapy.sparse as sparse_ext
from benchmark import core, methods

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import scipy.sparse

    from benchmark.methods import sparray

    ShapeExt = Callable[[np.ndarray, float], np.ndarray]
    CostExt = Callable[[np.ndarray, float, float], np.ndarray]
    SparseExt = Callable[[np.ndarray | sparray, float, bool], scipy.sparse.coo_array]
    SparseSolver = Callable[[sparray], np.ndarray]
    Solve = Callable[[np.ndarray, float, float], np.ndarray]  # (dist, eta, inf) -> links

# Dense: method -> (shape_extension, cost_extension). run() uses the shape one when eta=inf (full/gated)
# and the cost one when eta is finite (alternative). Edit to change a backend's chosen extension.
METHOD_EXTENSIONS: dict[str, tuple[shape_ext.ShapeExtensionProtocol, cost_ext.CostExtensionProtocol]] = {
    "scipy": (shape_ext.no_shape_extension, cost_ext.diag_row_pad_zero),
    "lap": (shape_ext.no_shape_extension, cost_ext.diag_row_pad_zero),
    "lapjv": (shape_ext.smallest_fill_zero, cost_ext.diag_row_pad_zero),
    "lapsolver": (shape_ext.no_shape_extension, cost_ext.diag_split_sparse),
    "laprust": (shape_ext.no_shape_extension, cost_ext.no_cost_extension),  # internal cost_limit
}

# Sparse: method -> sparse_extension.
SPARSE_METHOD_EXTENSIONS: dict[str, sparse_ext.SparseExtensionProtocol] = {
    "csgraph": sparse_ext.diag_row_no_pad,
    "lapmod": sparse_ext.diag_split_sparse,
}

SPARSE_METHODS: dict[str, SparseSolver] = {
    "csgraph": methods.solve_csgraph,
    "lapmod": methods.solve_lapmod,
}

SQUARENESS = [0.9, 1.0, 1.1]

# Trusted dense cost extension used to build the scipy reference (inf-free, scipy-safe, exact).
REFERENCE_COST_EXTENSION = cost_ext.fill_row_pad_zero

# Cost-match tolerance. Equally-optimal matchings differ only by a few ULPs (~1e-15); genuine suboptimal
# results differ by >=1e-3. This silences the former and catches the latter.
RTOL = 1e-6
ATOL = 1e-9


class Config(NamedTuple):
    """A cost-limit use case: whether the matrix is gated and whether a soft alternative cost is used."""

    name: str
    gated: bool
    alternative: bool


COST_CONFIGS = [
    Config("full", gated=False, alternative=False),
    Config("gated", gated=True, alternative=False),
    Config("alternative", gated=False, alternative=True),
    Config("gated_alternative", gated=True, alternative=True),
]


def make_dense_solve(
    solver: Callable[[np.ndarray, float], np.ndarray],
    shape: shape_ext.ShapeExtensionProtocol,
    cost: cost_ext.CostExtensionProtocol,
    *,
    alternative: bool,
) -> Solve:
    """Build a ``(dist, eta, inf) -> links`` solve for a dense backend pinned to its extensions."""

    def solve(dist: np.ndarray, eta: float, inf: float) -> np.ndarray:
        return methods.run(
            solver,
            dist,
            alternative_cost=eta if alternative else float("inf"),
            inf=inf,
            shape_extension=shape,
            cost_extension=cost,
            enforce_small_row=True,
        )

    return solve


def make_sparse_solve(
    solver: SparseSolver, sparse_extension: sparse_ext.SparseExtensionProtocol, *, hard: bool
) -> Solve:
    """Build a ``(dist, eta, inf) -> links`` solve for a sparse backend pinned to its extension."""

    def solve(dist: np.ndarray, eta: float, _inf: float) -> np.ndarray:
        return methods.run_sparse(
            solver, dist, eta, sparse_extension, hard=hard, feasible=False, enforce_small_row=True
        )

    return solve


def reference_metric(dist: np.ndarray, eta: float, inf: float, *, alternative: bool) -> tuple[int, float]:
    """Scipy reference ``(link count, achieved cost)`` for a config (``dist`` is already gated if needed).

    ``alternative`` -> finite ``eta`` + soft sink extension; otherwise scipy solves the (gated-or-not)
    matrix directly (eta=inf), which covers both ``full`` and the hard-threshold ``gated`` case.
    """
    ref_links = methods.run(
        methods.solve_scipy,
        dist,
        alternative_cost=eta if alternative else float("inf"),
        inf=inf,
        cost_extension=REFERENCE_COST_EXTENSION,
        enforce_small_row=True,
    )
    return len(ref_links), core.achieved_cost(dist, ref_links)


def measure_combo(
    solve: Solve,
    instances: Sequence[tuple[np.ndarray, float, float]],
    references: list[tuple[int, float]],
    *,
    repeats: int,
) -> dict:
    """Time one pinned method over every instance, comparing ``(link count, achieved cost)`` to its reference.

    Reports the worst per-instance deviation (link-count mismatches first) as ``cost``/``ref``/``diff``, or
    ``{"skipped": True, "error": ...}`` if the solve raised.
    """
    means: list[float] = []
    stds: list[float] = []
    cost_ok = True
    worst: tuple[float, float, float] | None = None  # (severity, cost, reference) -- severity = inf on count mismatch
    try:
        for (dist, eta, inf), (ref_n, ref_cost) in zip(instances, references, strict=True):
            fn = functools.partial(solve, dist, eta, inf)
            mean, std, result = core.time_callable(fn, repeats)
            means.append(mean)
            stds.append(std)
            n_links = len(result)
            cost = core.achieved_cost(dist, result)
            severity = abs(cost - ref_cost) if n_links == ref_n else float("inf")
            if worst is None or severity > worst[0]:
                worst = (severity, cost, ref_cost)
            if n_links != ref_n or not np.isclose(cost, ref_cost, rtol=RTOL, atol=ATOL):
                cost_ok = False
    except Exception as err:  # noqa: BLE001 - a backend that cannot handle a case is recorded, not fatal
        return {"skipped": True, "error": f"{type(err).__name__}: {err}"}
    assert worst is not None  # instances is never empty
    return {
        "skipped": False,
        "error": None,
        "mean_time": float(np.mean(means)),
        "std_time": float(np.mean(stds)),
        "cost_ok": cost_ok,
        "cost": float(worst[1]),
        "ref": float(worst[2]),
        "diff": float(worst[1] - worst[2]),
    }


class Cell(NamedTuple):
    """One (size, dataset, squareness, config) sweep cell with its instances and reference metrics."""

    data_name: str
    squareness: float
    config: Config
    n: int
    instances: Sequence[tuple[np.ndarray, float, float]]
    references: list[tuple[int, float]]


def run_combo(
    records: list[dict],
    stopped: set[tuple],
    cell: Cell,
    method_name: str,
    extension: str,
    solve: Solve,
    repeats: int,
    max_time: float,
) -> None:
    """Measure one pinned method for a cell, append its record, and apply per-combo early-stop."""
    key = (cell.data_name, method_name, cell.config.name, cell.squareness)
    if key in stopped:
        return

    rows, cols = cell.instances[0][0].shape
    ref_cost = float(np.mean([cost for _, cost in cell.references]))
    outcome = measure_combo(solve, cell.instances, cell.references, repeats=repeats)
    records.append(
        {
            "study": "method_comparison",
            "data": cell.data_name,
            "method": method_name,
            "config": cell.config.name,
            "extension": extension,
            "squareness": cell.squareness,
            "n": cell.n,
            "rows": rows,
            "cols": cols,
            "samples": len(cell.instances),
            "repeats": repeats,
            "ref_cost": ref_cost,
            "mean_time": None,
            "std_time": None,
            "cost_ok": None,
            "cost": None,
            "ref": None,
            "diff": None,
            **outcome,
        }
    )

    if outcome["skipped"]:
        return
    if not outcome["cost_ok"]:
        print(
            f"WARN cost mismatch: {key} n={cell.n} "
            f"worst cost={outcome['cost']:.6g} vs ref={outcome['ref']:.6g} (diff={outcome['diff']:.3g})"
        )
    if outcome["mean_time"] > max_time:
        stopped.add(key)
        print(f"stop  {key} after n={cell.n} ({outcome['mean_time']:.3g}s > {max_time}s)")


def sweep_cell(records: list[dict], stopped: set[tuple], cell: Cell, repeats: int, max_time: float) -> None:
    """Run every pinned method for one cell: the 6 dense always, the 2 sparse only when there is a cost limit."""
    for method_name, (shape, cost) in METHOD_EXTENSIONS.items():
        solve = make_dense_solve(core.METHODS[method_name], shape, cost, alternative=cell.config.alternative)
        extension = f"{shape.__name__}|{cost.__name__}"
        run_combo(records, stopped, cell, method_name, extension, solve, repeats, max_time)

    if cell.config.gated or cell.config.alternative:  # sparse needs a cost limit; never runs on "full"
        hard = cell.config.gated and not cell.config.alternative
        for method_name, sparse_extension in SPARSE_METHOD_EXTENSIONS.items():
            solve = make_sparse_solve(SPARSE_METHODS[method_name], sparse_extension, hard=hard)
            run_combo(records, stopped, cell, method_name, sparse_extension.__name__, solve, repeats, max_time)


def run_study(data_names: list[str], sizes: list[int], samples: int, repeats: int, max_time: float) -> list[dict]:
    """Sweep sizes x datasets x squareness x cost configs, returning flat JSON records.

    ``n`` is the outermost loop so each combo's previous-size timing is meaningful. Early-stop is
    *per combo*: the ``stopped`` set is keyed by the full configuration, so a cheap combo keeps
    climbing to larger ``n`` even after an expensive sibling has stopped.
    """
    records: list[dict] = []
    stopped: set[tuple] = set()

    for n in sizes:
        print("=" * 50)
        print(f"Running for n={n}")
        print("=" * 50)
        for data_name in data_names:
            data_fn = core.DATASETS[data_name]
            for squareness in SQUARENESS:
                for config in COST_CONFIGS:
                    instances = [core.make_instance(data_fn, n, squareness, gated=config.gated) for _ in range(samples)]
                    references = [
                        reference_metric(dist, eta, inf, alternative=config.alternative) for dist, eta, inf in instances
                    ]
                    cell = Cell(data_name, squareness, config, n, instances, references)
                    sweep_cell(records, stopped, cell, repeats, max_time)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", choices=["experimental", "uniform", "both"], default="experimental")
    parser.add_argument("--sizes", default=",".join(str(size) for size in core.SIZES))
    parser.add_argument("--samples", type=int, default=3, help="Random matrices averaged per size/config")
    parser.add_argument("--repeats", type=int, default=3, help="Timed repeats per matrix")
    parser.add_argument("--max-time", type=float, default=0.1, help="Stop a combo once its mean exceeds this (s)")
    parser.add_argument("--out", default=str(core.RESULTS_DIR / "method_comparison.json"))
    args = parser.parse_args()

    data_names = ["experimental", "uniform"] if args.data == "both" else [args.data]
    sizes = [int(size) for size in args.sizes.split(",")]

    records = run_study(data_names, sizes, args.samples, args.repeats, args.max_time)
    core.save_results(args.out, records)
    print(f"Saved {len(records)} records to {args.out}")


if __name__ == "__main__":
    main()
