"""Point process simulation tools."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter


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


def simulate_cox_process(
    mean_intensity: float,
    bounds: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
    grid_size: int = 50,
    field_sigma: float = 1.0,
    field_smoothness: float = 2.0,
    random_state: Optional[int] = None,
    return_intensity: bool = False,
) -> np.ndarray | Dict[str, np.ndarray]:
    """
    Simulate a simple log-Gaussian Cox process on a regular grid.

    Parameters
    ----------
    mean_intensity : float
        Target average intensity per unit area.
    bounds : tuple of float
        (xmin, xmax, ymin, ymax) simulation window.
    grid_size : int
        Number of cells per axis used for latent field simulation.
    field_sigma : float
        Standard deviation of latent Gaussian field.
    field_smoothness : float
        Gaussian filter sigma (in grid cells) controlling spatial correlation.
    random_state : int, optional
        Seed for reproducible simulations.
    return_intensity : bool
        If True, return a dictionary with points and intensity surface.
    """

    if mean_intensity < 0.0:
        raise ValueError("mean_intensity must be non-negative")
    if grid_size < 2:
        raise ValueError("grid_size must be >= 2")
    if field_sigma < 0.0:
        raise ValueError("field_sigma must be non-negative")
    if field_smoothness <= 0.0:
        raise ValueError("field_smoothness must be positive")

    xmin, xmax, ymin, ymax = bounds
    if xmax <= xmin or ymax <= ymin:
        raise ValueError("Invalid bounds; expected xmin < xmax and ymin < ymax")

    rng = np.random.default_rng(random_state)
    area = (xmax - xmin) * (ymax - ymin)
    cell_area = area / float(grid_size * grid_size)

    # Latent Gaussian field with spatial smoothness.
    white = rng.normal(0.0, 1.0, size=(grid_size, grid_size))
    latent = gaussian_filter(white, sigma=field_smoothness, mode="reflect")
    latent = latent - np.mean(latent)
    latent_std = float(np.std(latent))
    if latent_std > 0.0:
        latent = latent / latent_std
    latent = latent * field_sigma

    # Convert to intensity field and re-scale to hit target mean intensity.
    intensity = np.exp(latent)
    intensity *= mean_intensity / float(np.mean(intensity))

    counts = rng.poisson(intensity * cell_area)

    x_edges = np.linspace(xmin, xmax, grid_size + 1)
    y_edges = np.linspace(ymin, ymax, grid_size + 1)
    xs = []
    ys = []

    for i in range(grid_size):
        for j in range(grid_size):
            k = int(counts[i, j])
            if k == 0:
                continue
            xs.append(rng.uniform(x_edges[i], x_edges[i + 1], size=k))
            ys.append(rng.uniform(y_edges[j], y_edges[j + 1], size=k))

    if xs:
        points = np.column_stack([np.concatenate(xs), np.concatenate(ys)]).astype(
            np.float64
        )
    else:
        points = np.empty((0, 2), dtype=np.float64)

    if return_intensity:
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
        grid_x, grid_y = np.meshgrid(x_centers, y_centers, indexing="ij")
        return {
            "points": points,
            "intensity": intensity,
            "x": grid_x,
            "y": grid_y,
        }
    return points
