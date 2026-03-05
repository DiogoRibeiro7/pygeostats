"""Point process simulation tools."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def simulate_poisson_process(
    intensity: float,
    bounds: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate a homogeneous 2D Poisson point process.

    Parameters
    ----------
    intensity : float
        Event intensity per unit area (lambda).
    bounds : tuple of float
        (xmin, xmax, ymin, ymax) simulation window.
    random_state : int, optional
        Seed for reproducible simulations.

    Returns
    -------
    ndarray, shape (n_points, 2)
        Simulated point coordinates.
    """

    if intensity < 0.0:
        raise ValueError("intensity must be non-negative")

    xmin, xmax, ymin, ymax = bounds
    if xmax <= xmin or ymax <= ymin:
        raise ValueError("Invalid bounds; expected xmin < xmax and ymin < ymax")

    area = (xmax - xmin) * (ymax - ymin)
    mean_count = intensity * area

    rng = np.random.default_rng(random_state)
    n_points = int(rng.poisson(mean_count))
    if n_points == 0:
        return np.empty((0, 2), dtype=np.float64)

    x = rng.uniform(xmin, xmax, size=n_points)
    y = rng.uniform(ymin, ymax, size=n_points)
    return np.column_stack([x, y]).astype(np.float64)
