import numpy as np


def generate(n: int, m: int, sparsity=0.0) -> np.ndarray:
    """Generate a (n, m) dist matrix with optional sparsity

    Args:
        n (int): Rows
        m (int): Columns
        sparsity (float): Proportion of unfeasible distances. (Set to np.inf in the dist matrix)

    Returns:
        np.ndarray: Distance matrix
            Shape: (n, m), dtype: float64
    """
    dist = np.random.uniform(0, 1, (n, m))
    dist[np.random.uniform(0, 1, (n, m)) < sparsity] = np.inf
    return dist


# Could use a more faithful generate
# def generate(n: int, m: int, sparsity=0.0) -> np.ndarray:
#     """Generate a (n, m) dist matrix with optional sparsity

#     Args:
#         n (int): Rows
#         m (int): Columns
#         sparsity (float): Proportion of unfeasible distances. (Set to np.inf in the dist matrix)

#     Returns:
#         np.ndarray: Distance matrix
#             Shape: (n, m), dtype: float64
#     """
#     points = np.random.uniform(0, 1, n)
#     shifted = points + np.random.randn(n)

#     if m < n:
#         shifted = shifted[:m]
#     elif m > n:
#         shifted = np.concatenate((shifted, np.random.uniform(0, 1, m - n)))

#     dist = np.abs(points[:, None] - shifted[None, :])
#     dist[dist > np.quantile(dist, 1-sparsity)] = np.inf

#     return dist
