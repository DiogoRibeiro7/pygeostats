"""Basic point pattern analysis tools."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from pyspatialstats.utils.validation import validate_coordinates


def _infer_area(coords: np.ndarray) -> float:
    """Infer window area from the coordinate bounding box."""

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    extents = maxs[:2] - mins[:2]
    area = float(extents[0] * extents[1])
    if area <= 0.0:
        raise ValueError("Unable to infer a positive area from coordinates")
    return area


def nearest_neighbor_distances(
    coords: np.ndarray,
    k: int = 1,
    return_indices: bool = False,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """Compute distance to the k-th nearest neighbor for each point."""

    if k < 1:
        raise ValueError("k must be >= 1")

    arr = validate_coordinates(coords)
    if len(arr) <= k:
        raise ValueError("Number of points must be greater than k")

    tree = cKDTree(arr[:, :2])
    distances, indices = tree.query(arr[:, :2], k=k + 1)
    kth_distances = distances[:, k]
    kth_indices = indices[:, k]
    if return_indices:
        return kth_distances, kth_indices
    return kth_distances


def ripley_k_function(
    coords: np.ndarray,
    radii: np.ndarray,
    area: Optional[float] = None,
) -> np.ndarray:
    """
    Estimate Ripley's K function (without edge correction).

    Notes
    -----
    This implementation is intended as a baseline and does not include
    edge correction terms.
    """

    arr = validate_coordinates(coords)
    r = np.asarray(radii, dtype=np.float64)
    if r.ndim != 1:
        raise ValueError("radii must be a 1D array")
    if np.any(r < 0.0):
        raise ValueError("radii must be non-negative")

    n = len(arr)
    if n < 2:
        raise ValueError("At least two points are required")

    if area is None:
        area = _infer_area(arr)
    if area <= 0.0:
        raise ValueError("area must be positive")

    tree = cKDTree(arr[:, :2])
    all_counts = np.empty_like(r)

    for i, radius in enumerate(r):
        # query_pairs counts unordered pairs with distance <= radius
        pair_count = len(tree.query_pairs(radius))
        all_counts[i] = pair_count

    # K(r) = (A / n^2) * sum_{i != j} I(d_ij <= r) = (2A / n^2) * pairs
    return (2.0 * area / (n * n)) * all_counts


def ripley_l_function(
    coords: np.ndarray,
    radii: np.ndarray,
    area: Optional[float] = None,
) -> np.ndarray:
    """Estimate Ripley's L function from Ripley's K."""

    k_values = ripley_k_function(coords, radii, area=area)
    return np.sqrt(k_values / np.pi)


def g_function(coords: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Estimate nearest-neighbor CDF G(r)."""

    r = np.asarray(radii, dtype=np.float64)
    if r.ndim != 1:
        raise ValueError("radii must be a 1D array")
    if np.any(r < 0.0):
        raise ValueError("radii must be non-negative")

    nnd = nearest_neighbor_distances(coords, k=1)
    return np.array([np.mean(nnd <= radius) for radius in r], dtype=np.float64)


def f_function(
    coords: np.ndarray,
    radii: np.ndarray,
    bounds: Tuple[float, float, float, float],
    n_random: int = 2000,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Estimate empty-space CDF F(r) via random probe points."""

    arr = validate_coordinates(coords)
    r = np.asarray(radii, dtype=np.float64)
    if r.ndim != 1:
        raise ValueError("radii must be a 1D array")
    if np.any(r < 0.0):
        raise ValueError("radii must be non-negative")
    if n_random < 1:
        raise ValueError("n_random must be >= 1")

    xmin, xmax, ymin, ymax = bounds
    if xmax <= xmin or ymax <= ymin:
        raise ValueError("Invalid bounds; expected xmin < xmax and ymin < ymax")

    rng = np.random.default_rng(random_state)
    probes = np.column_stack(
        [
            rng.uniform(xmin, xmax, size=n_random),
            rng.uniform(ymin, ymax, size=n_random),
        ]
    )
    tree = cKDTree(arr[:, :2])
    distances, _ = tree.query(probes, k=1)
    return np.array([np.mean(distances <= radius) for radius in r], dtype=np.float64)


def pair_correlation_function(
    coords: np.ndarray,
    radii: np.ndarray,
    area: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """Estimate pair correlation g(r) from finite differences of K(r)."""

    r = np.asarray(radii, dtype=np.float64)
    if r.ndim != 1:
        raise ValueError("radii must be a 1D array")
    if len(r) < 2:
        raise ValueError("radii must contain at least two values")
    if np.any(r <= 0.0):
        raise ValueError("radii must be positive for pair correlation")

    k_values = ripley_k_function(coords, r, area=area)
    dk = np.gradient(k_values, r)
    g = dk / (2.0 * np.pi * r)
    return {"r": r, "g": g}
