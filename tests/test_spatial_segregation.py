"""Tests for spatial segregation indices."""

import numpy as np

from pyspatialstats.point_patterns import compute_spatial_segregation_indices


def test_entropy_segregation_higher_for_separated_pattern() -> None:
    rng = np.random.default_rng(42)

    mixed_coords = rng.uniform(0.0, 1.0, size=(400, 2))
    mixed_marks = np.array(["A"] * 200 + ["B"] * 200)
    rng.shuffle(mixed_marks)
    mixed = compute_spatial_segregation_indices(mixed_coords, mixed_marks, n_cells=10)

    left = np.column_stack([rng.uniform(0.0, 0.45, 200), rng.uniform(0.0, 1.0, 200)])
    right = np.column_stack([rng.uniform(0.55, 1.0, 200), rng.uniform(0.0, 1.0, 200)])
    sep_coords = np.vstack([left, right])
    sep_marks = np.array(["A"] * 200 + ["B"] * 200)
    separated = compute_spatial_segregation_indices(sep_coords, sep_marks, n_cells=10)

    assert separated["entropy_segregation"] > mixed["entropy_segregation"]
    assert separated["dissimilarity_index"] > mixed["dissimilarity_index"]


def test_segregation_index_output_ranges() -> None:
    rng = np.random.default_rng(0)
    coords = rng.uniform(0.0, 2.0, size=(300, 2))
    marks = rng.choice(np.array(["A", "B", "C"]), size=300, replace=True)
    result = compute_spatial_segregation_indices(coords, marks, n_cells=8)

    assert set(result.keys()) == {
        "entropy_segregation",
        "mean_cell_entropy",
        "global_entropy",
        "dissimilarity_index",
        "n_groups",
    }
    assert 0.0 <= result["entropy_segregation"] <= 1.0
    assert result["mean_cell_entropy"] >= 0.0
    assert result["global_entropy"] >= 0.0
    assert result["n_groups"] == 3.0
    assert np.isnan(result["dissimilarity_index"])
