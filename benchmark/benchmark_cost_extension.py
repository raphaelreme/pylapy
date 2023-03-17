import argparse
import time
from typing import Dict, List

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import tqdm

import pylapy
from pylapy import cost_extension

from additional_lap import lap_own_extend
from data import generate


def main(rows: List[int], sparsity: float, eta: float, ratio: float, repeat: int):
    solvers = {solver: pylapy.LapSolver(solver) for solver in ["lap", "lapjv", "scipy", "lapsolver"]}
    extensions = [
        # You can uncomment them to see how they perform. But they are all usually worse
        # cost_extension.col_cost,
        # cost_extension.col_cost_inf,
        # cost_extension.diag_col_cost,
        # cost_extension.diag_col_cost_inf,
        cost_extension.diag_row_cost,
        cost_extension.diag_row_cost_inf,
        cost_extension.diag_split_cost,
        cost_extension.row_cost,
        cost_extension.row_cost_inf,
        cost_extension.split_cost,
        cost_extension.symmetric_sparse_extension,
    ]

    # Add a cmap if too many extensions
    # cmap = plt.get_cmap("hsv")
    # colors = {extension.__name__: cmap(i / len(extensions))[:3] for i, extension in enumerate(extensions)}

    # Warmup each solver, extension and check that they are coherent
    dist = generate(50, int(50 * ratio), sparsity)
    print(dist.shape, (dist == np.inf).sum() / dist.size)
    costs = []
    hps = []
    for solver in solvers:
        for extension in extensions:
            hps.append((solver, extension.__name__))
            solvers[solver].cost_extension = extension

            links = solvers[solver].solve(dist, eta)
            cost = dist[links[:, 0], links[:, 1]].sum()
            costs.append(cost)

    if (np.array(costs) != costs[0]).any():  # Check that all methods have similar results
        print("Warning a method has yield a different cost:")
        print([(*hp, cost) for hp, cost in zip(hps, costs)])

    # Compute ref time
    ref = []
    for row in rows:
        total_time = 0.0
        for _ in range(repeat):
            dist = generate(row, int(row * ratio), sparsity)
            t = time.time()

            links = lap_own_extend(dist, eta)
            total_time += time.time() - t

        ref.append(total_time / repeat)

    for solver in tqdm.tqdm(solvers, desc="Solver"):
        timings: Dict[str, List[float]] = {extension.__name__: [] for extension in extensions}
        for extension in tqdm.tqdm(extensions, desc="Extension", leave=False):
            solvers[solver].cost_extension = extension

            for row in tqdm.tqdm(rows, desc="Rows", miniters=1, leave=False):
                if solver == "lapsolver" and row > 2000:  # Lapsolver is too slow
                    continue

                total_time = 0.0
                for _ in range(repeat):
                    dist = generate(row, int(row * ratio), sparsity)
                    t = time.time()
                    links = solvers[solver].solve(dist, eta)
                    total_time += time.time() - t

                timings[extension.__name__].append(total_time / repeat)

        plt.figure()
        plt.plot(rows, np.array(ref) * 1000, label="ref", marker="*")
        for extension_name in timings:
            plt.plot(
                rows[: len(timings[extension_name])],
                np.array(timings[extension_name]) * 1000,
                # color=colors[extension_name],
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
    parser.add_argument("--eta", default=0.1, type=float, help="Cost limit")
    parser.add_argument("--ratio", default=1.0, type=float, help="Columns/rows ratio")
    parser.add_argument("--repeat", default=3, type=int, help="Repeat n times")

    args = parser.parse_args()

    main([int(row) for row in args.rows.split(",")], args.sparsity, args.eta, args.ratio, args.repeat)
