"""Plot the results of :mod:`benchmark_v2.benchmark_cost_extension`.

Reads the JSON results and writes PNGs under ``images/cost_extension/<data>/``:

- ``<method>__sq<squareness>_gated<bool>.png`` - cost extensions compared, one config.

Run:
    uv run python -m benchmark_v2.plot_cost_extension
    uv run python -m benchmark_v2.plot_cost_extension --data experimental
"""

from __future__ import annotations

import argparse

from benchmark import core


def config_label(squareness: float, gated: bool) -> str:  # noqa: FBT001 - plain label formatting
    return f"sq{squareness}_gated{gated}"


def plot_for_data(records: list[dict], data_name: str) -> None:
    """Write every per-method figure for a single dataset."""
    out_dir = core.IMAGES_DIR / "cost_extension" / data_name
    seen_methods = sorted({record["method"] for record in records})
    configs = sorted({(record["squareness"], record["gated"]) for record in records})

    for method in seen_methods:
        method_records = [record for record in records if record["method"] == method]

        # One figure per config: compare extensions.
        for squareness, gated in configs:
            subset = [
                record for record in method_records if record["squareness"] == squareness and record["gated"] == gated
            ]
            if subset:
                core.plot_curves(
                    subset,
                    "extension",
                    title=f"{method} - {config_label(squareness, gated)}",
                    out_path=out_dir / f"{method}__{config_label(squareness, gated)}.svg",
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=str(core.RESULTS_DIR / "cost_extension.json"))
    parser.add_argument("--data", choices=["experimental", "uniform", "both"], default="both")
    args = parser.parse_args()

    records = [record for record in core.load_results(args.results) if not record.get("skipped")]

    data_names = sorted({record["data"] for record in records})
    if args.data != "both":
        data_names = [name for name in data_names if name == args.data]

    for data_name in data_names:
        data_records = [record for record in records if record["data"] == data_name]
        plot_for_data(data_records, data_name)
        print(f"Plotted {data_name}: {len(data_records)} records -> {core.IMAGES_DIR / 'cost_extension' / data_name}")


if __name__ == "__main__":
    main()
