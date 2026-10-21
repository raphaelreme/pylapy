import argparse
import time

import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import numpy as np
import tqdm
from additional_lap import lap_own_extend
from data import generate

import pylapy

ADD_REF = True
CORRECT_LINK_PROPORTION = False


def main(rows: list[int], sparsity: float, eta: float, ratio: float, repeat: int):  # noqa: C901
    solvers = {solver: pylapy.LapSolver(solver).solve for solver in pylapy.LapSolver.implementations}

    if ADD_REF:
        solvers["ref"] = lap_own_extend

    timings: dict[str, list[float]] = {solver: [] for solver in solvers}

    for row in tqdm.tqdm(rows, miniters=1):
        total_time = {solver: 0.0 for solver in solvers}  # noqa: C420
        for _ in tqdm.trange(repeat, leave=False):
            dist = generate(row, int(row * ratio), sparsity)

            costs = []
            for solver in solvers:  # noqa: PLC0206
                if solver == "lapsolver" and row > 2000:  # Lap solver is too slow  # noqa: PLR2004
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
                tqdm.tqdm.write(str(list(zip(solvers, costs, strict=True))))

        for solver in solvers:
            if solver == "lapsolver" and row > 2000:  # noqa: PLR2004
                continue
            timings[solver].append(total_time[solver] / repeat)

        tqdm.tqdm.write(f"Links proportion: {links.shape[0] / row}")

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

    args = parser.parse_args()

    main([int(row) for row in args.rows.split(",")], args.sparsity, args.eta, args.ratio, args.repeat)
