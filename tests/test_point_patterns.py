"""Tests for point pattern analysis tools."""

import numpy as np

from pygeostats.point_patterns import (
    f_function,
    g_function,
    nearest_neighbor_distances,
    pair_correlation_function,
    ripley_k_function,
    ripley_l_function,
)


def _sample_points(n: int = 200, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, size=(n, 2))


def test_nearest_neighbor_distances_shape_and_positive() -> None:
    coords = _sample_points()
    d = nearest_neighbor_distances(coords, k=1)
    assert d.shape == (len(coords),)
    assert np.all(d > 0.0)


def test_ripley_k_monotonic_non_decreasing() -> None:
    coords = _sample_points()
    radii = np.linspace(0.01, 0.25, 25)
    k = ripley_k_function(coords, radii, area=1.0)
    assert k.shape == radii.shape
    assert np.all(np.diff(k) >= -1e-12)


def test_ripley_l_non_decreasing() -> None:
    coords = _sample_points()
    radii = np.linspace(0.01, 0.25, 25)
    l = ripley_l_function(coords, radii, area=1.0)
    assert l.shape == radii.shape
    assert np.all(np.diff(l) >= -1e-12)


def test_g_and_f_are_valid_cdfs() -> None:
    coords = _sample_points()
    radii = np.linspace(0.0, 0.3, 30)

    g = g_function(coords, radii)
    f = f_function(coords, radii, bounds=(0.0, 1.0, 0.0, 1.0), random_state=123)

    assert np.all((g >= 0.0) & (g <= 1.0))
    assert np.all((f >= 0.0) & (f <= 1.0))
    assert np.all(np.diff(g) >= -1e-12)
    assert np.all(np.diff(f) >= -1e-12)


def test_pair_correlation_output_shape_and_finite() -> None:
    coords = _sample_points()
    radii = np.linspace(0.02, 0.25, 20)
    result = pair_correlation_function(coords, radii, area=1.0)

    assert set(result.keys()) == {"r", "g"}
    assert result["r"].shape == radii.shape
    assert result["g"].shape == radii.shape
    assert np.all(np.isfinite(result["g"]))
