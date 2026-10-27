import numpy as np


def experimental_cost(n: int, squareness: float = 1.0) -> tuple[np.ndarray, float]:
    space_size = 1.0  # ~ Image height (or width), here normalized to 1.0

    rows = round(n * squareness)
    cols = round(n / squareness)
    if rows <= cols:  # Few rows, let's miss around 10% and complete with false
        missed = round(0.1 * rows)  # 10% Missed
        false = cols - (rows - missed)  # The remaining (10% or more)
    if rows > cols:  # Many rows, let's miss enough to have at least 10% completion
        missed = round(rows - cols * 0.9)
        false = cols - (rows - missed)  # The remaining (at least 10%)

    # Original points uniform in space
    points = np.random.rand(rows, 2) * space_size

    # We sample the std such that the half the total area matches with the sum of the area of disks with stds as radii
    # ie: space_size**2 = n * pi * r**2 => r = space_size / sqrt(n * pi)
    new_points = points + np.random.randn(rows, 2) * space_size / np.sqrt(n * np.pi / 0.5)

    # Some points are lost (missed)
    new_points = new_points[: rows - missed]

    # New points are added (false)
    new_points = np.concatenate((new_points, np.random.rand(false, 2) * space_size))

    # Shuffle new_points order
    np.random.shuffle(new_points)

    # Cost is simply the Euclidean distance
    cost = np.sqrt(((points[:, None] - new_points[None, :]) ** 2).sum(axis=-1))

    # As threshold, we use a typical 3 std
    eta = 3 * space_size / np.sqrt(n * np.pi / 0.5)

    return cost, eta


def uniform_cost(n: int, squareness: float = 1.0) -> tuple[np.ndarray, float]:
    rows = round(n * squareness)
    cols = round(n / squareness)

    cost = np.random.rand(rows, cols)

    # As threshold, we use something that keeps roughly expected_match per row/col
    expected_match = n ** (1 / 5)  # From 1 to 5-6 at 5000
    eta = expected_match / (n + 1)

    return cost, eta
