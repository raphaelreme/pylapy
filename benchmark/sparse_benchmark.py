import argparse
import time
from typing import Dict, List

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import tqdm

import pylapy

from data import experimental_generate


K_SIGMA = 3  # Threshold at 3sigma


def main(rows: List[int], fa: float, md: float, smooth: bool, repeat: int):
    solvers = {"lapmod_symmetric": pylapy.LapSolver(sparse_implementation="lapmod").sparse_solve}
    solver_ = pylapy.LapSolver(sparse_implementation="csgraph")
    solver_.sparse_extension = "diag_row_cost"  # diag_col_cost is equivalent
    solvers["csgraph_diag_row"] = solver_.sparse_solve
    solver_ = pylapy.LapSolver(sparse_implementation="csgraph")
    solver_.sparse_extension = "symmetric_cost"
    solvers["csgraph_symmetric"] = solver_.sparse_solve

    # Warm up (compile numba)
    for solver in solvers:  # pylint: disable=consider-using-dict-items
        solvers[solver](np.random.rand(5, 5), 0.1, hard=not smooth)

    # Let's compare with scipy ?
    dense_implementation = "scipy"
    dense_solver = pylapy.LapSolver(implementation=dense_implementation)

    timings: Dict[str, List[float]] = {solver: [] for solver in solvers}
    timings[dense_implementation] = []

    for row in tqdm.tqdm(rows, miniters=1):
        total_time = {solver: 0.0 for solver in solvers}
        total_time[dense_implementation] = 0.0
        for _ in tqdm.trange(repeat, leave=False):
            eta = K_SIGMA * np.sqrt(0.5 / row / np.pi)
            dist = experimental_generate(row, fa, md, space_size=1.0, hard_thresh=eta)

            costs = []
            for solver in solvers:  # pylint: disable=consider-using-dict-items
                t = time.time()

                links = solvers[solver](dist, eta, hard=not smooth)

                total_time[solver] += time.time() - t
                costs.append(dist[links[:, 0], links[:, 1]].sum())

            t = time.time()

            if smooth:
                links = dense_solver.solve(dist, eta)
            else:
                links = dense_solver.solve(dist)

            total_time[dense_implementation] += time.time() - t
            costs.append(dist[links[:, 0], links[:, 1]].sum())

            if (np.array(costs) != costs[0]).any():  # Check that all method have similar results
                tqdm.tqdm.write("Warning a method has yield a different cost:")
                tqdm.tqdm.write(str(list(zip(list(solvers) + [dense_solver], costs))))

        for solver in solvers:
            timings[solver].append(total_time[solver] / repeat)

        timings[dense_implementation].append(total_time[dense_implementation] / repeat)

        tqdm.tqdm.write(
            f"Links proportion: {links.shape[0]/row}, Sparsity: {np.isinf(dist).sum() / dist.size}, Shape: {dist.shape}"
        )

    plt.title(f"Fa/Md: {fa}/{md}, {'smooth' if smooth else 'hard'} thresholding")
    for solver in solvers:
        plt.plot(rows[: len(timings[solver])], np.array(timings[solver]) * 1000, label=solver, marker="*")

    plt.plot(
        rows[: len(timings[dense_implementation])],
        np.array(timings[dense_implementation]) * 1000,
        label=dense_implementation,
        marker="*",
    )

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
