"""Shared helpers for the pylapy ``benchmark_v2`` studies.

Timing, problem-instance construction, JSON IO, and a generic runtime-vs-size plotting helper.
Every study (shape extension, cost extension, method comparison) builds on this module so the
timing/averaging/plotting is identical and early-stop bookkeeping stays consistent.

Nothing here imports ``pylapy``: the whole point of ``benchmark_v2`` is to measure the raw LAP
backends (via :mod:`benchmark_v2.methods`) so the results can *decide* pylapy's own defaults.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from benchmark import dataset, methods

if TYPE_CHECKING:
    from collections.abc import Callable

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
IMAGES_DIR = HERE / "images"

# Default data-size sweep (log-spaced from 5 to 5000).
SIZES = [5, 10, 20, 50, 100, 250, 500, 1000, 2000, 5000]

# Name -> solver callable (direct backend wrappers, no pylapy involved).
METHODS: dict[str, Callable[[np.ndarray, float], np.ndarray]] = {
    "scipy": methods.solve_scipy,
    "lap": methods.solve_lap,
    "lapjv": methods.solve_lapjv,
    "lapsolver": methods.solve_lapsolver,
    "laprust": methods.solve_laprust,
}

# Name -> data generator returning (cost, eta).
DATASETS: dict[str, Callable[[int, float], tuple[np.ndarray, float]]] = {
    "experimental": dataset.experimental_cost,
    "uniform": dataset.uniform_cost,
}


def make_instance(
    data_fn: Callable[[int, float], tuple[np.ndarray, float]],
    n: int,
    squareness: float,
    *,
    gated: bool,
) -> tuple[np.ndarray, float, float]:
    """Generate one problem instance and the ``inf`` sentinel expected by :func:`methods.run`.

    Reproduces the notebook idiom: ``inf`` is larger than any feasible assignment cost, and gating
    pushes every above-threshold cost onto that sentinel so the solver pipeline drops it.

    Returns:
        (dist, eta, inf): the cost matrix, its threshold, and the finite inf sentinel.
    """
    dist, eta = data_fn(n, squareness)
    inf = float(dist[dist != np.inf].sum() + 1)
    if gated:
        dist[dist > eta] = inf
    return dist, eta, inf


def achieved_cost(matrix: np.ndarray, links: np.ndarray) -> float:
    """Total cost of ``links`` on ``matrix`` (0.0 for an empty assignment).

    Computed on whatever matrix was actually solved, so a transposed run yields the same total as
    the direct run and both stay comparable to the reference cost.
    """
    if len(links) == 0:
        return 0.0
    return float(matrix[links[:, 0], links[:, 1]].sum())


def soft_objective(dist: np.ndarray, links: np.ndarray, eta: float) -> float:
    """Soft-threshold objective: made-link costs + ``eta`` per unmatched row and per unmatched column.

    Invariant across every equivalent cost extension (they all return the same optimal "link iff
    ``dist <= eta``" set), so it is the correctness check for the cost-extension study. Mirrors
    ``tests/conftest.py::soft_objective``.
    """
    n, m = dist.shape
    linked = len(links)
    return achieved_cost(dist, links) + eta * ((n - linked) + (m - linked))


def time_callable(fn: Callable[[], np.ndarray], repeats: int) -> tuple[float, float, np.ndarray]:
    """Warm up ``fn`` once, then time ``repeats`` calls with :func:`time.perf_counter`.

    Returns:
        (mean_seconds, std_seconds, warmup_result): the warm-up result is returned so the caller
        can run its correctness check without an extra solve.
    """
    result = fn()  # Warm-up; also returned for the correctness check.
    timings = np.empty(repeats)
    for i in range(repeats):
        start = time.perf_counter()
        fn()
        timings[i] = time.perf_counter() - start
    return float(timings.mean()), float(timings.std()), result


def save_results(path: Path | str, records: list[dict]) -> None:
    """Write ``records`` to ``path`` as indented JSON, creating parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def load_results(path: Path | str) -> list[dict]:
    """Load the JSON records written by :func:`save_results`."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def plot_curves(
    records: list[dict],
    series_key: str,
    title: str,
    out_path: Path | str,
    *,
    time_scale: float = 1000.0,
) -> None:
    """Plot mean runtime vs ``n`` (log-log), one line per distinct ``series_key`` value.

    Records sharing a (series value, ``n``) are averaged together, so the caller controls what each
    line aggregates purely through the subset it passes in. ``time_scale`` converts seconds to the
    plotted unit (default: milliseconds). Records without a numeric ``mean_time`` are ignored.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415 - optional at run time; only needed for plotting

    # series value -> {n: [mean_time, ...]}, insertion order preserved for a logical legend order.
    grouped: dict[object, dict[int, list[float]]] = {}
    for record in records:
        if record.get("mean_time") is None:
            continue
        grouped.setdefault(record[series_key], {}).setdefault(record["n"], []).append(record["mean_time"])

    fig, ax = plt.subplots()
    for series, by_size in grouped.items():
        sizes = sorted(by_size)
        values = [float(np.mean(by_size[size])) * time_scale for size in sizes]
        ax.plot(sizes, values, marker="*", label=str(series))

    ax.set_title(title)
    ax.set_xlabel("Size (n)")
    ax.set_ylabel("Execution time (ms)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(SIZES, [str(size) for size in SIZES])
    ax.set_xbound(3 * 1e0, 1e4)
    ax.set_ybound(3e-3, 2e3)
    ax.grid(visible=True, which="both", alpha=0.3)
    ax.legend()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
