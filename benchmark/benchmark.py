import argparse
import time
from typing import Dict, List

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import tqdm

import pylapy

from additional_lap import lap_own_extend, lap_sparse_solve
from data import generate


ADD_REF = False
CORRECT_LINK_PROPORTION = False


def main(rows: List[int], sparsity: float, eta: float, ratio: float, repeat: int, lapmod: bool):
    # Check that all can be imported (and import all so that it does not imped the run time)
    # pylint: disable=import-outside-toplevel, unused-import
    import lap  # type: ignore
    import lapjv  # type: ignore
    import lapsolver  # type: ignore
    import scipy.optimize  # type: ignore

    # pylint: enable=import-outside-toplevel, unused-import

    solvers = {solver: pylapy.LapSolver(solver).solve for solver in pylapy.LapSolver.implementations}

    if lapmod:
        solvers["lapmod (sparse)"] = lap_sparse_solve
        lap_sparse_solve(generate(5, 5, 0))  # Compile to_csr

    if ADD_REF:
        solvers["ref"] = lap_own_extend

    timings: Dict[str, List[float]] = {solver: [] for solver in solvers}

    for row in tqdm.tqdm(rows, miniters=1):
        total_time = {solver: 0.0 for solver in solvers}
        for _ in tqdm.trange(repeat, leave=False):
            dist = generate(row, int(row * ratio), sparsity)

            costs = []
            for solver in solvers:
                if solver == "lapsolver" and row > 2000:  # Lap solver is too slow
                    continue
                if solver == "lapmod (sparse)" and eta == np.inf and links.shape[0] != row:  # type: ignore
                    continue
                t = time.time()

                if CORRECT_LINK_PROPORTION:
                    # Eta is more used with small rows because statistically there are less small distances
                    # We can correct this by dividing it by the expected non inf element on each row
                    links = solvers[solver](dist, eta / (row * (1 - sparsity)))
                else:
                    links = solvers[solver](dist, eta)

                total_time[solver] += time.time() - t
                costs.append(dist[links[:, 0], links[:, 1]].sum())

            if (np.array(costs) != costs[0]).any():  # Check that all method have similar results
                tqdm.tqdm.write("Warning a method has yield a different cost:")
                tqdm.tqdm.write(str(list(zip(solvers, costs))))

        for solver in solvers:
            if solver == "lapsolver" and row > 2000:
                continue
            if solver == "lapmod (sparse)" and eta == np.inf and links.shape[0] != row:
                timings[solver].append(float("nan"))
                continue
            timings[solver].append(total_time[solver] / repeat)

        tqdm.tqdm.write(f"Links proportion: {links.shape[0]/row}")

    plt.title(f"Sparsity: {sparsity}, Eta: {eta}, ratio: {ratio}")
    for solver in solvers:
        plt.plot(rows[: len(timings[solver])], np.array(timings[solver]) * 1000, label=solver, marker="*")

    plt.xlabel("Size (n_rows)")
    plt.ylabel("Execution time (ms)")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(1, 1e4)
    plt.ylim(1e-2, 1e4)
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmarking lap")
    parser.add_argument("--rows", default="5,10,20,50,100,250,500,1000,2000,5000", help="Number of rows to try")
    parser.add_argument("--eta", default=np.inf, type=float, help="Cost limit")
    parser.add_argument("--sparsity", default=0.0, type=float, help="Cost sparsity")
    parser.add_argument("--ratio", default=1.0, type=float, help="Columns/rows ratio")
    parser.add_argument("--repeat", default=10, type=int, help="Repeat n times")
    parser.add_argument(
        "--lapmod",
        action="store_true",
        help="Try lapmod (WARNING: yields segfault for most unfeasible solution. To use with eta or low sparsity)",
    )

    args = parser.parse_args()

    main([int(row) for row in args.rows.split(",")], args.sparsity, args.eta, args.ratio, args.repeat, args.lapmod)
