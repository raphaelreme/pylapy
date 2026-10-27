"""Plot the results of :mod:`benchmark_v2.benchmark_method_comparison`.

Reads the JSON results and writes one PNG **per config** under ``images/method_comparison/<data>/``, each
overlaying every method's runtime-vs-``n`` curve:

- ``<config>_sq<squareness>.png`` - all methods compared, one (cost config, squareness).

Run:
    uv run python -m benchmark_v2.plot_method_comparison
    uv run python -m benchmark_v2.plot_method_comparison --data experimental
"""

from __future__ import annotations

import argparse

from benchmark import core


def plot_for_data(records: list[dict], data_name: str) -> None:
    """Write one per-config figure (methods overlaid) for a single dataset."""
    out_dir = core.IMAGES_DIR / "method_comparison" / data_name
    configs = sorted({(record["config"], record["squareness"]) for record in records})

    for config, squareness in configs:
        subset = [record for record in records if record["config"] == config and record["squareness"] == squareness]
        if subset:
            core.plot_curves(
                subset,
                "method",
                title=f"{config} - sq{squareness}",
                out_path=out_dir / f"{config}_sq{squareness}.svg",
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=str(core.RESULTS_DIR / "method_comparison.json"))
    parser.add_argument("--data", choices=["experimental", "uniform", "both"], default="both")
    args = parser.parse_args()

    records = [record for record in core.load_results(args.results) if not record.get("skipped")]

    data_names = sorted({record["data"] for record in records})
    if args.data != "both":
        data_names = [name for name in data_names if name == args.data]

    for data_name in data_names:
        data_records = [record for record in records if record["data"] == data_name]
        plot_for_data(data_records, data_name)
        msg = f"Plotted {data_name}: {len(data_records)} records -> {core.IMAGES_DIR / 'method_comparison' / data_name}"
        print(msg)


if __name__ == "__main__":
    main()
