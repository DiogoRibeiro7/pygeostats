"""Tests for Moran's I spatial autocorrelation statistics."""

import numpy as np

from pyspatialstats.spatial_autocorrelation import (
    local_morans_i,
    morans_i,
    spatial_weights_knn,
)


def _grid_coords(n: int = 12) -> np.ndarray:
    x = np.linspace(0.0, 1.0, n)
    xx, yy = np.meshgrid(x, x)
    return np.column_stack([xx.ravel(), yy.ravel()])


def test_global_morans_i_positive_for_smooth_field() -> None:
    coords = _grid_coords(12)
    values = coords[:, 0] + coords[:, 1]  # spatially smooth gradient
    w = spatial_weights_knn(coords, k=8)
    result = morans_i(values, w, permutations=99, random_state=42)

    assert result["I"] > 0.0
    assert result["p_value"] <= 0.05


def test_global_morans_i_near_zero_for_random_field() -> None:
    coords = _grid_coords(12)
    rng = np.random.default_rng(7)
    values = rng.normal(0.0, 1.0, size=len(coords))
    w = spatial_weights_knn(coords, k=8)
    result = morans_i(values, w, permutations=99, random_state=7)

    assert abs(result["I"]) < 0.2


def test_local_morans_i_output_shape() -> None:
    coords = _grid_coords(10)
    values = np.sin(coords[:, 0] * np.pi) + np.cos(coords[:, 1] * np.pi)
    w = spatial_weights_knn(coords, k=6)
    result = local_morans_i(values, w)

    assert set(result.keys()) == {"I_local", "z"}
    assert result["I_local"].shape == values.shape
    assert result["z"].shape == values.shape
    assert np.all(np.isfinite(result["I_local"]))
