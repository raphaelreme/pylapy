"""Sparse Extension Benchmark: What is the best sparse-extension strategy per sparse backend?

The two sparse LAP solvers (``scipy.sparse.csgraph.min_weight_full_bipartite_matching`` and lapx's
``lap.lapmod``) take a gated/sparse cost matrix plus "sink" edges so a node can stay unmatched. pylapy
offers a few sparse extensions to build that COO matrix; this compares them per sparse backend.

For every sparse backend we measure, across a size sweep, every sparse extension (the 3 in
``pylapy.sparse``), over the configs ``squareness in {0.9, 1.0, 1.1}`` x ``hard in {False, True}``. Matrices
are always gated at ``eta`` (that is what makes them sparse) and always oriented to have fewer rows than
columns (``enforce_small_row=True``). ``hard`` chooses hard thresholding (sink cost inf-like) vs an
alternative cost (sink = ``eta``). Only ``diag_split_sparse`` is square, so ``lapmod`` skips the other two.
We check every extension reaches the right (link-count, cost) against a scipy reference.

Results are written to JSON for ``plot_sparse_extension.py``.

Run:
    uv run python -m benchmark_v2.benchmark_sparse_extension
    uv run python -m benchmark_v2.benchmark_sparse_extension --sizes 5,10,50 --samples 2 --repeats 3
"""

from __future__ import annotations

import argparse
import functools
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import scipy.sparse

import pylapy.cost_extension as cost_ext
import pylapy.sparse as sparse_ext
from benchmark import core, methods

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from benchmark.methods import sparray

# The two sparse solvers (study-local; core.METHODS is the dense set).
SPARSE_METHODS: dict[str, Callable[[sparray], np.ndarray]] = {
    "csgraph": methods.solve_csgraph,
    "lapmod": methods.solve_lapmod,
}

# The 3 sparse extensions. Only ``diag_split_sparse`` is square -> the only one lapmod accepts.
SPARSE_EXTENSIONS = sparse_ext.SPARSE_EXTENSIONS

SQUARENESS = [0.9, 1.0, 1.1]
HARD = [False, True]

# Trusted dense cost extension for the alternative-cost (soft) reference.
REFERENCE_COST_EXTENSION = cost_ext.fill_row_pad_zero

# Cost-match tolerance. Equally-optimal matchings differ only by a few ULPs in float (~1e-15); genuine
# suboptimal results (e.g. lapmod can return a non-optimal matching) differ by >=1e-3.
RTOL = 1e-6
ATOL = 1e-9


def reference_metric(dist: np.ndarray, eta: float, inf: float, *, hard: bool) -> tuple[int, float]:
    """Scipy reference ``(link count, achieved cost)`` for a gated instance.

    ``hard`` -> scipy on the gated dense matrix (hard-threshold min-cost maximum matching); otherwise ->
    scipy with a dense soft sink extension (alternative cost ``eta``).
    """
    if hard:
        ref_links = methods.run(methods.solve_scipy, dist, inf=inf, enforce_small_row=True)
    else:
        ref_links = methods.run(
            methods.solve_scipy,
            dist,
            alternative_cost=eta,
            inf=inf,
            cost_extension=REFERENCE_COST_EXTENSION,
            enforce_small_row=True,
        )
    return len(ref_links), core.achieved_cost(dist, ref_links)


def measure_combo(
    solver: Callable[[sparray], np.ndarray],
    extension: sparse_ext.SparseExtensionProtocol,
    instances: Sequence[tuple[np.ndarray | sparray, float, float]],
    references: list[tuple[int, float]],
    *,
    hard: bool,
    repeats: int,
) -> dict:
    """Time one (sparse-solver, extension) combo over every instance.

    Always solved with ``enforce_small_row=True`` and the instance's own ``eta``/``hard``. Each instance's
    ``(link count, achieved cost)`` is compared to *its own* scipy reference -- a mismatch is either a
    different number of feasible links or a cost gap beyond RTOL/ATOL (tie-break rounding is silenced).
    Reports the worst per-instance deviation (link-count mismatches first) as ``cost``/``ref``/``diff``.

    Returns the usual success dict, or ``{"skipped": True, "error": ...}`` for unsupported combos
    (e.g. lapmod on a non-square extension).
    """
    means: list[float] = []
    stds: list[float] = []
    cost_ok = True
    worst: tuple[float, float, float] | None = None  # (severity, cost, reference) -- severity = inf on count mismatch
    try:
        for (dist, eta, _inf), (ref_n, ref_cost) in zip(instances, references, strict=True):
            fn = functools.partial(
                methods.run_sparse, solver, dist, eta, extension, hard=hard, feasible=False, enforce_small_row=True
            )
            mean, std, result = core.time_callable(fn, repeats)
            means.append(mean)
            stds.append(std)
            n_links = len(result)
            cost = core.achieved_cost(sparse_ext.sparse_to_dense(dist) if scipy.sparse.issparse(dist) else dist, result)  # type: ignore[arg-type]
            severity = abs(cost - ref_cost) if n_links == ref_n else float("inf")
            if worst is None or severity > worst[0]:
                worst = (severity, cost, ref_cost)
            if n_links != ref_n or not np.isclose(cost, ref_cost, rtol=RTOL, atol=ATOL):
                cost_ok = False
    except Exception as err:  # noqa: BLE001 - unsupported combos are expected and recorded, not fatal
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
    """One (size, dataset, config) sweep cell with its pre-generated instances and reference metrics."""

    data_name: str
    squareness: float
    hard: bool
    n: int
    instances: Sequence[tuple[np.ndarray | sparray, float, float]]
    references: list[tuple[int, float]]


def sweep_cell(records: list[dict], stopped: set[tuple], cell: Cell, repeats: int, max_time: float) -> None:
    """Run every (sparse method, extension) combo for one cell, appending records in place.

    Skips combos already in ``stopped`` and, after each run, warns on a cost mismatch and adds the
    combo's key to ``stopped`` once its mean time exceeds ``max_time`` (per-combo early-stop).
    """
    rows, cols = cell.instances[0][0].shape  # type: ignore[misc]
    ref_cost = float(np.mean([cost for _, cost in cell.references]))
    for method_name, solver in SPARSE_METHODS.items():
        for ext_name, extension in SPARSE_EXTENSIONS.items():
            key = (cell.data_name, method_name, ext_name, cell.squareness, cell.hard)
            if key in stopped:
                continue

            outcome = measure_combo(solver, extension, cell.instances, cell.references, hard=cell.hard, repeats=repeats)
            records.append(
                {
                    "study": "sparse_extension",
                    "data": cell.data_name,
                    "method": method_name,
                    "extension": ext_name,
                    "squareness": cell.squareness,
                    "hard": cell.hard,
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
                continue
            if not outcome["cost_ok"]:
                print(
                    f"WARN cost mismatch: {key} n={cell.n} "
                    f"worst cost={outcome['cost']:.6g} vs ref={outcome['ref']:.6g} "
                    f"(diff={outcome['diff']:.3g})"
                )
            if outcome["mean_time"] > max_time:
                stopped.add(key)
                print(f"stop  {key} after n={cell.n} ({outcome['mean_time']:.3g}s > {max_time}s)")


def run_study(
    data_names: list[str],
    sizes: list[int],
    samples: int,
    repeats: int,
    max_time: float,
    sparse_dist: bool,  # noqa: FBT001
) -> list[dict]:
    """Sweep sizes x datasets x configs, returning flat JSON records.

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
                for hard in HARD:
                    # Always gated at eta: that is what makes the matrix sparse.
                    instances = [core.make_instance(data_fn, n, squareness, gated=True) for _ in range(samples)]
                    references = [reference_metric(dist, eta, inf, hard=hard) for dist, eta, inf in instances]

                    if sparse_dist:
                        cell = Cell(
                            data_name,
                            squareness,
                            hard,
                            n,
                            [(sparse_ext.dense_to_coo(dist, eta), eta, inf) for dist, eta, inf in instances],
                            references,
                        )
                    else:
                        cell = Cell(data_name, squareness, hard, n, instances, references)

                    sweep_cell(records, stopped, cell, repeats, max_time)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", choices=["experimental", "uniform", "both"], default="experimental")
    parser.add_argument("--sizes", default=",".join(str(size) for size in core.SIZES))
    parser.add_argument("--samples", type=int, default=3, help="Random matrices averaged per size/config")
    parser.add_argument("--repeats", type=int, default=3, help="Timed repeats per matrix")
    parser.add_argument("--max-time", type=float, default=0.1, help="Stop a combo once its mean exceeds this (s)")
    parser.add_argument("--sparse-array", action="store_true", help="Pre-convert dist into sparse array.")
    parser.add_argument("--out", default=str(core.RESULTS_DIR / "sparse_extension.json"))
    args = parser.parse_args()

    data_names = ["experimental", "uniform"] if args.data == "both" else [args.data]
    sizes = [int(size) for size in args.sizes.split(",")]

    records = run_study(data_names, sizes, args.samples, args.repeats, args.max_time, args.sparse_array)
    core.save_results(args.out, records)
    print(f"Saved {len(records)} records to {args.out}")


if __name__ == "__main__":
    main()
