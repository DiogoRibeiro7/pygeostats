"""Spatial weights matrix construction utilities."""

from __future__ import annotations

from typing import Optional

import numpy as np

from pyspatialstats.utils.validation import validate_coordinates


def row_standardize_weights(weights: np.ndarray) -> np.ndarray:
    """Row-standardize a square weights matrix."""

    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 2 or w.shape[0] != w.shape[1]:
        raise ValueError("weights must be a square 2D array")
    row_sums = w.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return w / row_sums


def spatial_weights_knn(
    coords: np.ndarray,
    k: int = 8,
    row_standardize: bool = True,
) -> np.ndarray:
    """Build a binary k-nearest-neighbor spatial weights matrix."""

    arr = validate_coordinates(coords)
    n = len(arr)
    if n < 2:
        raise ValueError("At least two points are required")
    if k < 1 or k >= n:
        raise ValueError("k must satisfy 1 <= k < n")

    distances = np.linalg.norm(arr[:, None, :2] - arr[None, :, :2], axis=2)
    np.fill_diagonal(distances, np.inf)
    neighbors = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]

    weights = np.zeros((n, n), dtype=np.float64)
    rows = np.repeat(np.arange(n), k)
    weights[rows, neighbors.ravel()] = 1.0
    if row_standardize:
        weights = row_standardize_weights(weights)
    return weights


def spatial_weights_distance_band(
    coords: np.ndarray,
    threshold: float,
    binary: bool = True,
    row_standardize: bool = True,
) -> np.ndarray:
    """
    Build a distance-band spatial weights matrix.

    If `binary` is False, uses inverse distance weights inside the threshold.
    """

    if threshold <= 0.0:
        raise ValueError("threshold must be positive")

    arr = validate_coordinates(coords)
    d = np.linalg.norm(arr[:, None, :2] - arr[None, :, :2], axis=2)
    np.fill_diagonal(d, np.inf)

    mask = d <= threshold
    if binary:
        w = mask.astype(np.float64)
    else:
        w = np.zeros_like(d, dtype=np.float64)
        w[mask] = 1.0 / d[mask]

    np.fill_diagonal(w, 0.0)
    if row_standardize:
        w = row_standardize_weights(w)
    return w


def spatial_weights_inverse_distance(
    coords: np.ndarray,
    power: float = 1.0,
    max_distance: Optional[float] = None,
    row_standardize: bool = True,
) -> np.ndarray:
    """Build an inverse-distance spatial weights matrix."""

    if power <= 0.0:
        raise ValueError("power must be positive")

    arr = validate_coordinates(coords)
    d = np.linalg.norm(arr[:, None, :2] - arr[None, :, :2], axis=2)
    np.fill_diagonal(d, np.inf)

    if max_distance is not None and max_distance <= 0.0:
        raise ValueError("max_distance must be positive when provided")

    mask = np.isfinite(d)
    if max_distance is not None:
        mask &= d <= max_distance

    w = np.zeros_like(d, dtype=np.float64)
    w[mask] = 1.0 / np.power(d[mask], power)
    np.fill_diagonal(w, 0.0)
    if row_standardize:
        w = row_standardize_weights(w)
    return w
