"""Tests for Moran's I spatial autocorrelation statistics."""

import numpy as np
from pygeostats.spatial_autocorrelation import (
    gearys_c,
    global_getis_ord_g,
    local_gearys_c,
    local_getis_ord_g,
    local_morans_i,
    morans_i,
    row_standardize_weights,
    spatial_weights_distance_band,
    spatial_weights_inverse_distance,
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


def test_global_gearys_c_less_than_one_for_smooth_field() -> None:
    coords = _grid_coords(12)
    values = coords[:, 0] + coords[:, 1]
    w = spatial_weights_knn(coords, k=8)
    result = gearys_c(values, w, permutations=99, random_state=10)

    assert result["C"] < 1.0
    assert result["p_value"] <= 0.05


def test_global_gearys_c_near_one_for_random_field() -> None:
    coords = _grid_coords(12)
    rng = np.random.default_rng(123)
    values = rng.normal(0.0, 1.0, size=len(coords))
    w = spatial_weights_knn(coords, k=8)
    result = gearys_c(values, w, permutations=99, random_state=12)

    assert abs(result["C"] - 1.0) < 0.3


def test_local_gearys_c_output_shape() -> None:
    coords = _grid_coords(10)
    values = np.sin(coords[:, 0] * np.pi) + np.cos(coords[:, 1] * np.pi)
    w = spatial_weights_knn(coords, k=6)
    result = local_gearys_c(values, w)

    assert set(result.keys()) == {"C_local", "z"}
    assert result["C_local"].shape == values.shape
    assert result["z"].shape == values.shape
    assert np.all(np.isfinite(result["C_local"]))


def test_local_getis_ord_detects_hot_area() -> None:
    coords = _grid_coords(12)
    values = np.ones(len(coords))
    hotspot = (coords[:, 0] > 0.7) & (coords[:, 1] > 0.7)
    values[hotspot] = 10.0

    w = spatial_weights_knn(coords, k=8)
    result = local_getis_ord_g(values, w, include_self=True)
    z = result["z_score"]

    assert z.shape == values.shape
    assert np.mean(z[hotspot]) > np.mean(z[~hotspot])


def test_global_getis_ord_returns_stat_and_pvalue() -> None:
    coords = _grid_coords(10)
    values = coords[:, 0] + coords[:, 1]
    w = spatial_weights_knn(coords, k=6)
    result = global_getis_ord_g(values, w, permutations=99, random_state=3)

    assert set(result.keys()) >= {"G", "expected_G", "p_value"}
    assert np.isfinite(result["G"])
    assert 0.0 <= result["p_value"] <= 1.0


def test_distance_band_weights_shape_and_row_sums() -> None:
    coords = _grid_coords(8)
    w = spatial_weights_distance_band(coords, threshold=0.25, binary=True)
    assert w.shape == (len(coords), len(coords))
    row_sums = w.sum(axis=1)
    assert np.all((np.isclose(row_sums, 1.0)) | (np.isclose(row_sums, 0.0)))


def test_inverse_distance_weights_are_nonnegative() -> None:
    coords = _grid_coords(6)
    w = spatial_weights_inverse_distance(coords, power=1.5, max_distance=0.5)
    assert w.shape == (len(coords), len(coords))
    assert np.all(w >= 0.0)
    assert np.allclose(np.diag(w), 0.0)


def test_row_standardize_weights_handles_zero_rows() -> None:
    w = np.array([[0.0, 2.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    ws = row_standardize_weights(w)
    assert np.isclose(np.sum(ws[0]), 1.0)
    assert np.isclose(np.sum(ws[2]), 1.0)
    assert np.isclose(np.sum(ws[1]), 0.0)
