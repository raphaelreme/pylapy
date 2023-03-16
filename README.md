# Pylapy

[![Lint and Test](https://github.com/raphaelreme/pylapy/actions/workflows/tests.yml/badge.svg)](https://github.com/raphaelreme/pylapy/actions/workflows/tests.yml)

Solves assignement problem with Hungarian algorithm (Jonker-Volgenant variants [1])

This class is a wrapper around different implementations you can find in python: lap, lapjv, scipy, lapsolver [2, 3, 4, 5].

It unifies the functionality of each implementation and allows you to use the one which is the fastest
on your problem.

It also helps you to handle non square matrices and setting a soft threshold on assignements (usually leads
to better performances than hard thresholding).


## Install

```bash
$ pip install pylapy
$ # By default it does not install any backend solver
$ # You can either install by hand your favorite solver (scipy, lap, lapjv, lapsolver)
$ pip install pylapy[scipy]  # or pylapy[lap] etc
$ # Note that some backend requires numpy to be installed correctly
$ # You may need to install numpy before
$ pip install numpy
```


## Getting started

```python

import numpy as np
import pylapy

# Simulate data
n, m = (2000, 2000)
sparsity = 0.5

dist = np.random.uniform(0, 1, (2000, 2000))
dist[np.random.uniform(0, 1, (2000, 2000)) < sparsity] = np.inf

# Create the solver and solves

solver = pylapy.LapSolver()  # Choose the current most efficient method that is installed
# solver = pylapy.LapSolver("scipy"|"lap"|"lapjv"|"lapsolver")  # You can choose which method you rather use

links = solver.solve(dist)

# Find the final cost

print(dist[links[:, 0], links[:, 1]])
```

## References

* [1] R. Jonker and A. Volgenant, "A Shortest Augmenting Path Algorithm for Dense and Sparse Linear Assignment Problems", Computing 38, 325-340 (1987)
* [2] "lap: Linear Assignment Problem solver", https://github.com/gatagat/lap
* [3] "lapjv: Linear Assignment Problem solver using Jonker-Volgenant algorithm", https://github.com/src-d/lapjv
* [4] "scipy: linear_sum_assignment", https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html#scipy.optimize.linear_sum_assignment
* [5] "py-lapsolver", https://github.com/cheind/py-lapsolver
