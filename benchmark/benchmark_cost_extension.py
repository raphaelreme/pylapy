"""Cost Extension Benchmark: What is the best cost-extension strategy per backend?

A soft threshold lets a row/column stay *unmatched* at cost ``eta``. Backends can't do that directly, so
pylapy rewrites the cost matrix by adding "sink" rows/cols (a cost extension) -- or, for some backends,
relies on a native ``cost_limit``. This compares pylapy's cost extensions (plus the native option when it
exists) per backend.

For every backend we measure, across a size sweep, every cost extension (``none`` + the 11 in
``pylapy.cost_extension``), over the configs ``squareness in {0.9, 1.0, 1.1}`` x ``gated in {False, True}``
with a finite ``eta`` (alternative cost). The matrix is always oriented to have fewer rows than columns
(``enforce_small_row=True``). The 11 extensions all build an ``(n+m, n+m)`` square matrix, so no shape
extension is needed. We check every extension reaches the same soft objective as a scipy reference.

Results are written to JSON for ``plot_cost_extension.py``.

Run:
    uv run python -m benchmark_v2.benchmark_cost_extension
    uv run python -m benchmark_v2.benchmark_cost_extension --sizes 5,10,50 --samples 2 --repeats 3
"""

from __future__ import annotations

import argparse
import functools
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

import pylapy.cost_extension as cost_ext
from benchmark import core, methods

if TYPE_CHECKING:
    from collections.abc import Callable

# ``none`` (no extension, relies on the backend's native cost_limit -- only lap/laprust) + the 11 pylapy ones.
COST_EXTENSIONS = cost_ext.COST_EXTENSIONS


# Square included: a cost extension adds sinks to any shape.
SQUARENESS = [0.9, 1.0, 1.1]
GATED = [False, True]

# Trusted reference cost extension.
REFERENCE_EXTENSION = cost_ext.fill_row_pad_zero

# Cost-match tolerance. Equally-optimal assignments reached via a different extension sum to the same soft
# objective but differ by a few ULPs in float (~1e-15); genuine suboptimal results (e.g. lapjv on
# inf-containing extensions) differ by >=1e-3. This silences the former, catches the latter.
RTOL = 1e-6
ATOL = 1e-9


def measure_combo(
    solver: Callable[[np.ndarray, float], np.ndarray],
    extension: cost_ext.CostExtensionProtocol,
    instances: list[tuple[np.ndarray, float, float]],
    references: list[float],
    *,
    repeats: int,
) -> dict:
    """Time one (solver, cost-extension) combo over every instance.

    The matrix is always solved with ``enforce_small_row=True`` and the instance's own ``eta``. Each
    instance's soft objective is compared to *its own* reference with a tolerance (see RTOL/ATOL):
    equally-optimal assignments differ only by float rounding, so exact equality would warn constantly.
    Reports the single worst per-instance deviation (``cost``/``ref``/``diff``) so the warning shows a real
    offending pair rather than mixing one instance's objective with the mean reference.

    Returns a dict with ``mean_time``/``std_time``/``cost_ok`` (+ worst-deviation fields) on success, or
    ``{"skipped": True, "error": ...}`` when the combo is unsupported (the solver raised, e.g. ``none`` on
    a backend without a native cost limit).
    """
    means: list[float] = []
    stds: list[float] = []
    cost_ok = True
    worst: tuple[float, float, float] | None = None  # (diff, cost, reference)
    try:
        for (dist, eta, inf), reference in zip(instances, references, strict=True):
            fn = functools.partial(
                methods.run,
                solver,
                dist,
                alternative_cost=eta,
                inf=inf,
                cost_extension=extension,
                enforce_small_row=True,
            )
            mean, std, result = core.time_callable(fn, repeats)
            means.append(mean)
            stds.append(std)
            cost = core.soft_objective(dist, result, eta)
            diff = cost - reference
            if worst is None or abs(diff) > abs(worst[0]):
                worst = (diff, cost, reference)
            if not np.isclose(cost, reference, rtol=RTOL, atol=ATOL):
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
        "diff": float(worst[0]),
    }


class Cell(NamedTuple):
    """One (size, dataset, config) sweep cell with its pre-generated instances and reference objectives."""

    data_name: str
    squareness: float
    gated: bool
    n: int
    instances: list[tuple[np.ndarray, float, float]]
    references: list[float]


def sweep_cell(records: list[dict], stopped: set[tuple], cell: Cell, repeats: int, max_time: float) -> None:
    """Run every (method, extension) combo for one cell, appending records in place.

    Skips combos already in ``stopped`` and, after each run, warns on a cost mismatch and adds the
    combo's key to ``stopped`` once its mean time exceeds ``max_time`` (per-combo early-stop).
    """
    rows, cols = cell.instances[0][0].shape
    ref_cost = float(np.mean(cell.references))
    for method_name, solver in core.METHODS.items():
        for ext_name, extension in COST_EXTENSIONS.items():
            key = (cell.data_name, method_name, ext_name, cell.squareness, cell.gated)
            if key in stopped:
                continue

            outcome = measure_combo(solver, extension, cell.instances, cell.references, repeats=repeats)
            records.append(
                {
                    "study": "cost_extension",
                    "data": cell.data_name,
                    "method": method_name,
                    "extension": ext_name,
                    "squareness": cell.squareness,
                    "gated": cell.gated,
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
                for gated in GATED:
                    instances = [core.make_instance(data_fn, n, squareness, gated=gated) for _ in range(samples)]
                    # Reference soft objective: scipy + a trusted inf-free cost extension.
                    references = [
                        core.soft_objective(
                            dist,
                            methods.run(
                                methods.solve_scipy,
                                dist,
                                alternative_cost=eta,
                                inf=inf,
                                cost_extension=REFERENCE_EXTENSION,
                                enforce_small_row=True,
                            ),
                            eta,
                        )
                        for dist, eta, inf in instances
                    ]
                    cell = Cell(data_name, squareness, gated, n, instances, references)
                    sweep_cell(records, stopped, cell, repeats, max_time)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", choices=["experimental", "uniform", "both"], default="experimental")
    parser.add_argument("--sizes", default=",".join(str(size) for size in core.SIZES))
    parser.add_argument("--samples", type=int, default=3, help="Random matrices averaged per size/config")
    parser.add_argument("--repeats", type=int, default=3, help="Timed repeats per matrix")
    parser.add_argument("--max-time", type=float, default=0.1, help="Stop a combo once its mean exceeds this (s)")
    parser.add_argument("--out", default=str(core.RESULTS_DIR / "cost_extension.json"))
    args = parser.parse_args()

    data_names = ["experimental", "uniform"] if args.data == "both" else [args.data]
    sizes = [int(size) for size in args.sizes.split(",")]

    records = run_study(data_names, sizes, args.samples, args.repeats, args.max_time)
    core.save_results(args.out, records)
    print(f"Saved {len(records)} records to {args.out}")


if __name__ == "__main__":
    main()
