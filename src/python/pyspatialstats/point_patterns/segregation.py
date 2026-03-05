"""Spatial segregation index utilities for marked point patterns."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from pyspatialstats.utils.validation import validate_coordinates


def _entropy(probabilities: np.ndarray) -> float:
    valid = probabilities > 0.0
    if not np.any(valid):
        return 0.0
    p = probabilities[valid]
    return float(-np.sum(p * np.log(p)))


def compute_spatial_segregation_indices(
    coords: np.ndarray,
    marks: np.ndarray,
    n_cells: int = 10,
    bounds: Optional[Tuple[float, float, float, float]] = None,
) -> Dict[str, float]:
    """
    Compute baseline spatial segregation metrics over a regular grid.

    Returns
    -------
    dict
        `entropy_segregation` : Theil H in [0, 1] (or 0 when undefined).
        `mean_cell_entropy` : Weighted mean local entropy.
        `global_entropy` : Entropy of global mark proportions.
        `dissimilarity_index` : Binary dissimilarity index in [0, 1], NaN if >2 groups.
        `n_groups` : Number of unique mark groups.
    """

    if n_cells < 2:
        raise ValueError("n_cells must be >= 2")

    arr = validate_coordinates(coords)
    mark_arr = np.asarray(marks)
    if mark_arr.ndim != 1:
        raise ValueError("marks must be a 1D array-like")
    if len(arr) != len(mark_arr):
        raise ValueError("coords and marks must have the same length")
    if len(arr) == 0:
        raise ValueError("coords and marks must be non-empty")

    if bounds is None:
        xmin, xmax = float(arr[:, 0].min()), float(arr[:, 0].max())
        ymin, ymax = float(arr[:, 1].min()), float(arr[:, 1].max())
    else:
        xmin, xmax, ymin, ymax = bounds

    if xmax <= xmin or ymax <= ymin:
        raise ValueError("Invalid bounds; expected xmin < xmax and ymin < ymax")

    unique_marks, inverse = np.unique(mark_arr, return_inverse=True)
    n_groups = len(unique_marks)
    n = len(mark_arr)

    x_bin = np.floor((arr[:, 0] - xmin) / (xmax - xmin) * n_cells).astype(int)
    y_bin = np.floor((arr[:, 1] - ymin) / (ymax - ymin) * n_cells).astype(int)
    x_bin = np.clip(x_bin, 0, n_cells - 1)
    y_bin = np.clip(y_bin, 0, n_cells - 1)
    cell_id = x_bin * n_cells + y_bin
    n_total_cells = n_cells * n_cells

    counts = np.zeros((n_total_cells, n_groups), dtype=np.float64)
    np.add.at(counts, (cell_id, inverse), 1.0)
    cell_totals = counts.sum(axis=1)

    global_counts = counts.sum(axis=0)
    global_probs = global_counts / float(n)
    global_entropy = _entropy(global_probs)

    local_entropies = np.zeros(n_total_cells, dtype=np.float64)
    non_empty = cell_totals > 0.0
    if np.any(non_empty):
        local_probs = counts[non_empty] / cell_totals[non_empty, None]
        local_entropies[non_empty] = np.array([_entropy(p) for p in local_probs])

    mean_cell_entropy = float(np.sum((cell_totals / n) * local_entropies))
    if global_entropy > 0.0:
        entropy_segregation = float(
            (global_entropy - mean_cell_entropy) / global_entropy
        )
    else:
        entropy_segregation = 0.0

    dissimilarity_index = np.nan
    if n_groups == 2:
        a = counts[:, 0]
        b = counts[:, 1]
        a_total = float(np.sum(a))
        b_total = float(np.sum(b))
        if a_total > 0.0 and b_total > 0.0:
            dissimilarity_index = float(
                0.5 * np.sum(np.abs(a / a_total - b / b_total))
            )

    return {
        "entropy_segregation": entropy_segregation,
        "mean_cell_entropy": mean_cell_entropy,
        "global_entropy": global_entropy,
        "dissimilarity_index": dissimilarity_index,
        "n_groups": float(n_groups),
    }
