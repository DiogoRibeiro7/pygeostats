"""Spatial clustering and hot spot analysis tools."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.spatial import distance_matrix
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from pygeostats.utils.validation import validate_coordinates, validate_values


def spatial_dbscan(
    coords: np.ndarray,
    eps: float,
    min_samples: int = 5,
) -> np.ndarray:
    """Run DBSCAN on spatial coordinates and return cluster labels."""

    if eps <= 0.0:
        raise ValueError("eps must be positive")
    if min_samples < 1:
        raise ValueError("min_samples must be >= 1")

    arr = validate_coordinates(coords)
    model = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    return model.fit_predict(arr[:, :2])


def getis_ord_gi_star(
    coords: np.ndarray,
    values: np.ndarray,
    distance_threshold: float,
    include_self: bool = True,
) -> np.ndarray:
    """
    Compute local Getis-Ord Gi* z-scores using binary distance weights.

    Notes
    -----
    This implementation uses a fixed distance band and does not apply
    multiple-testing corrections.
    """

    if distance_threshold <= 0.0:
        raise ValueError("distance_threshold must be positive")

    arr = validate_coordinates(coords)
    vals = validate_values(values)
    if len(arr) != len(vals):
        raise ValueError("Coordinates and values must have the same length")
    if len(arr) < 3:
        raise ValueError("At least 3 points are required")

    dists = distance_matrix(arr[:, :2], arr[:, :2])
    weights = (dists <= distance_threshold).astype(np.float64)
    if not include_self:
        np.fill_diagonal(weights, 0.0)

    n = len(vals)
    x_bar = np.mean(vals)
    s = np.sqrt(np.mean(vals**2) - x_bar**2)
    if s == 0.0:
        return np.zeros(n, dtype=np.float64)

    w_sum = np.sum(weights, axis=1)
    w_sq_sum = np.sum(weights**2, axis=1)

    numerator = weights @ vals - x_bar * w_sum
    denominator = s * np.sqrt((n * w_sq_sum - w_sum**2) / (n - 1))

    gi = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator > 0.0,
    )
    return gi


def kernel_density_estimate(
    coords: np.ndarray,
    bandwidth: Optional[float] = None,
    grid_size: int = 100,
    bounds: Optional[Tuple[float, float, float, float]] = None,
) -> Dict[str, np.ndarray]:
    """Estimate spatial intensity surface with Gaussian KDE on a regular grid."""

    arr = validate_coordinates(coords)
    if len(arr) < 2:
        raise ValueError("At least 2 points are required")
    if grid_size < 2:
        raise ValueError("grid_size must be >= 2")

    x = arr[:, 0]
    y = arr[:, 1]

    if bounds is None:
        xmin, xmax = float(x.min()), float(x.max())
        ymin, ymax = float(y.min()), float(y.max())
    else:
        xmin, xmax, ymin, ymax = bounds

    if xmax <= xmin or ymax <= ymin:
        raise ValueError("Invalid bounds; expected xmin < xmax and ymin < ymax")

    grid_x, grid_y = np.meshgrid(
        np.linspace(xmin, xmax, grid_size),
        np.linspace(ymin, ymax, grid_size),
    )

    samples = np.vstack([x, y])
    bw_method = None if bandwidth is None else bandwidth
    kde = gaussian_kde(samples, bw_method=bw_method)
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()])
    density = kde(grid_points).reshape(grid_x.shape)

    return {"x": grid_x, "y": grid_y, "density": density}


def cluster_validation_metrics(coords: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Compute clustering quality metrics for non-noise clusters.

    Returns NaN metrics when fewer than two clusters are present.
    """

    arr = validate_coordinates(coords)
    labs = np.asarray(labels)
    if len(arr) != len(labs):
        raise ValueError("coords and labels must have the same length")

    non_noise_mask = labs != -1
    arr_eval = arr[non_noise_mask, :2]
    labs_eval = labs[non_noise_mask]

    unique_clusters = np.unique(labs_eval)
    n_clusters = int(len(unique_clusters))
    if n_clusters < 2 or len(arr_eval) <= n_clusters:
        return {
            "n_clusters": float(n_clusters),
            "silhouette": np.nan,
            "calinski_harabasz": np.nan,
            "davies_bouldin": np.nan,
        }

    return {
        "n_clusters": float(n_clusters),
        "silhouette": float(silhouette_score(arr_eval, labs_eval)),
        "calinski_harabasz": float(calinski_harabasz_score(arr_eval, labs_eval)),
        "davies_bouldin": float(davies_bouldin_score(arr_eval, labs_eval)),
    }
