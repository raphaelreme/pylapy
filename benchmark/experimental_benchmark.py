import argparse
import time
from typing import Dict, List

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import tqdm

import pylapy

from additional_lap import lap_own_extend
from data import experimental_generate


ADD_REF = True


def main(rows: List[int], fa: float, md: float, smooth: bool, repeat: int):
    solvers = {solver: pylapy.LapSolver(solver).solve for solver in pylapy.LapSolver.implementations}

    if ADD_REF:
        solvers["ref"] = lap_own_extend

    timings: Dict[str, List[float]] = {solver: [] for solver in solvers}

    for row in tqdm.tqdm(rows, miniters=1):
        total_time = {solver: 0.0 for solver in solvers}
        for _ in tqdm.trange(repeat, leave=False):
            eta = 2 * np.sqrt(0.5 / row / np.pi)
            dist = experimental_generate(row, fa, md, space_size=1.0, hard_thresh=eta)

            costs = []
            for solver in solvers:
                if solver == "lapsolver" and row > 2000:  # Lap solver is too slow
                    continue
                t = time.time()

                if smooth:
                    links = solvers[solver](dist, eta)
                else:
                    links = solvers[solver](dist)

                total_time[solver] += time.time() - t
                costs.append(dist[links[:, 0], links[:, 1]].sum())

            if (np.array(costs) != costs[0]).any():  # Check that all method have similar results
                tqdm.tqdm.write("Warning a method has yield a different cost:")
                tqdm.tqdm.write(str(list(zip(solvers, costs))))

        for solver in solvers:
            if solver == "lapsolver" and row > 2000:
                continue
            timings[solver].append(total_time[solver] / repeat)

        tqdm.tqdm.write(
            f"Links proportion: {links.shape[0]/row}, Sparsity: {np.isinf(dist).sum() / dist.size}, Shape: {dist.shape}"
        )

    plt.title(f"Fa/Md: {fa}/{md}, {'smooth' if smooth else 'hard'} thresholding")
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
    # By default: non-square matrix, (let's have more detections than expected)
    parser = argparse.ArgumentParser(description="Benchmarking lap on experimental data")
    parser.add_argument("--rows", default="5,10,20,50,100,250,500,1000,2000,5000", help="Number of rows to try")
    parser.add_argument("--fa", default=0.15, type=float, help="Proportion of false alarms")
    parser.add_argument("--md", default=0.1, type=float, help="Proportion of missed detections")
    parser.add_argument("--smooth", action="store_true", help="Use smooth thresholding rather than hard")
    parser.add_argument("--repeat", default=10, type=int, help="Repeat n times")
    args = parser.parse_args()

    main([int(row) for row in args.rows.split(",")], args.fa, args.md, args.smooth, args.repeat)
