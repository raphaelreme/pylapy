import numpy as np
import cv2
import matplotlib.pyplot as plt  # type: ignore


PLOT = False


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
def experimental_generate(n: int, fa=0.0, md=0.0, space_size=1, area_ratio=0.5, hard_thresh=float("inf")) -> np.ndarray:
    """Generate a realistic (n, n - md + fa) dist matrix with some sparsity

    Generated from 2D points in a confined space that moves under Brownian assumption (Gaussian noise)
    Some points are lost, some are wrongly detected (False alarms / missed detections)

    The motion amplitude (std of the Gaussian) is set from the area_ratio so that
    space_size**2 * area_ratio = n * pi * std**2

    Distance is set to inf when it is beyond hard_thresh

    Args:
        n (int): Number of points in the 2D space
        fa (float): Number of false alarms
            Default: 0.0
        md (float): Number of missed detections
            Default: 0.0
        space_size (int): Size of the 2D space
            Default: 1000
        area_ratio (float): Controls how much particles cover the 2d space
            Default: 0.3
        hard_thresh (float): Distance is inf if it is beyond hard_thresh
            Default: inf

    Returns:
        np.ndarray: Distance matrix
            Shape: (n, n - md + fa), dtype: float64
    """
    points = np.random.uniform(0, space_size, (n, 2))  # random points on the 2d space

    # Let's move them by a random deplacement
    # We fix the area of space (space **2) to be equal to the area of particles motions (~ n * pi* std**2) times a ratio
    new_points = points + np.random.randn(n, 2) * space_size * np.sqrt(area_ratio / n / np.pi)

    if PLOT:
        image = np.zeros((1000, 1000, 3), dtype=np.uint8)

        scale = 1000 / space_size

        for i in range(n):
            source = (points[i] * scale).round().astype(np.int32)
            dest = (new_points[i] * scale).round().astype(np.int32)
            cv2.circle(image, source, 5, (255, 255, 255), -1)
            cv2.circle(image, dest, 3, (255, 0, 0), -1)
            cv2.line(image, source, dest, (0, 0, 255), 2)

        plt.figure(figsize=(24, 16))
        plt.imshow(image)
        plt.show()

    # Lost points:
    new_points = new_points[: None if -int(n * md) >= 0 else -int(n * md)]

    # False alarm (Random points in the 2d space)
    new_points = np.concatenate((new_points, np.random.uniform(0, space_size, (int(n * fa), 2))))

    # Shuffle points
    np.random.shuffle(new_points)

    dist = np.sqrt(((points[:, None] - new_points[None, :]) ** 2).sum(axis=-1))
    dist[dist > hard_thresh] = np.inf
    # dist[dist > np.quantile(dist, 1 - sparsity)] = np.inf

    return dist
