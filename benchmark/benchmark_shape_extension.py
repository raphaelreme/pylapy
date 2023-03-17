import argparse
import time
from typing import Dict, List

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import tqdm

import pylapy
from pylapy import shape_extension

from additional_lap import lap_own_extend
from data import generate


def main(rows: List[int], sparsity: float, ratio: float, repeat: int):
    solvers = {solver: pylapy.LapSolver(solver) for solver in ["lap", "lapjv", "scipy", "lapsolver"]}
    extensions = [
        shape_extension.smallest_fill_0,
        shape_extension.smallest_fill_inf,
        shape_extension.sum_col_inf,
        shape_extension.sum_fill_inf,
        shape_extension.sum_row_inf,
        shape_extension.sum_split_inf,
    ]

    # Warmup each solver, method and check that they are coherent
    dist = generate(50, int(50 * ratio), sparsity)
    print(dist.shape, (dist == np.inf).sum() / dist.size)
    costs = []
    hps = []
    for solver in solvers:
        for extension in extensions:
            hps.append((solver, extension.__name__))
            solvers[solver].shape_extension = extension

            links = solvers[solver].solve(dist)
            cost = dist[links[:, 0], links[:, 1]].sum()
            costs.append(cost)

            rev_links = solvers[solver].solve(dist.T)

            sorted_links = sorted([(link[0], link[1]) for link in links])
            sorted_revlinks = sorted([(link[1], link[0]) for link in rev_links])
            assert sorted_links == sorted_revlinks, f"{solver}{extension.__name__}"

    if (np.array(costs) != costs[0]).any():  # Check that all methods have similar results
        print("Warning a method has yield a different cost:")
        print([(*hp, cost) for hp, cost in zip(hps, costs)])

    # Compute ref time
    # if lap version is 0.5 then it is slow (but by default 0.4.0 is installed)
    ref = []
    for row in rows:
        total_time = 0.0
        for _ in range(repeat):
            dist = generate(row, int(row * ratio), sparsity)
            t = time.time()

            links = lap_own_extend(dist)
            total_time += time.time() - t

        ref.append(total_time / repeat)

    for solver in tqdm.tqdm(solvers, desc="Solver"):
        timings: Dict[str, List[float]] = {extension.__name__: [] for extension in extensions}
        for extension in tqdm.tqdm(extensions, desc="Extension", leave=False):
            solvers[solver].shape_extension = extension

            for row in tqdm.tqdm(rows, desc="Rows", miniters=1, leave=False):
                if solver == "lapsolver" and row > 2000:  # Lapsolver is too slow
                    continue

                total_time = 0.0
                for _ in range(repeat):
                    dist = generate(row, int(row * ratio), sparsity)
                    t = time.time()
                    links = solvers[solver].solve(dist)
                    total_time += time.time() - t

                timings[extension.__name__].append(total_time / repeat)

        plt.figure()
        plt.plot(rows, np.array(ref) * 1000, label="ref", marker="*")
        for extension_name in timings:
            plt.plot(
                rows[: len(timings[extension_name])],
                np.array(timings[extension_name]) * 1000,
                label=extension_name,
                marker="*",
            )

        plt.title(solver)
        plt.xlabel("Size (n_rows)")
        plt.ylabel("Execution time (ms)")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlim(1, 1e4)
        plt.ylim(1e-2, 1e5)
        plt.grid()
        plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmarking lap")
    parser.add_argument("--rows", default="5,10,20,50,100,250,500,1000,2000,5000", help="Number of rows to try")
    parser.add_argument("--sparsity", default=0.0, type=float, help="Cost sparsity")
    parser.add_argument("--ratio", default=0.8, type=float, help="Columns/rows ratio")
    parser.add_argument("--repeat", default=3, type=int, help="Repeat n times")

    args = parser.parse_args()

    main([int(row) for row in args.rows.split(",")], args.sparsity, args.ratio, args.repeat)
