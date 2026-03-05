"""Tests for point process simulation tools."""

import numpy as np

from pyspatialstats.point_patterns import simulate_poisson_process


def test_poisson_process_reproducible_with_seed() -> None:
    bounds = (-2.0, 2.0, 1.0, 3.0)
    pts1 = simulate_poisson_process(10.0, bounds=bounds, random_state=42)
    pts2 = simulate_poisson_process(10.0, bounds=bounds, random_state=42)
    assert np.array_equal(pts1, pts2)


def test_poisson_process_points_inside_bounds() -> None:
    bounds = (0.0, 5.0, -1.0, 2.0)
    pts = simulate_poisson_process(8.0, bounds=bounds, random_state=7)
    xmin, xmax, ymin, ymax = bounds
    if len(pts) > 0:
        assert np.all((pts[:, 0] >= xmin) & (pts[:, 0] <= xmax))
        assert np.all((pts[:, 1] >= ymin) & (pts[:, 1] <= ymax))


def test_poisson_process_expected_count_close_in_large_window() -> None:
    bounds = (0.0, 10.0, 0.0, 10.0)
    intensity = 5.0
    pts = simulate_poisson_process(intensity, bounds=bounds, random_state=1)
    expected = intensity * (bounds[1] - bounds[0]) * (bounds[3] - bounds[2])
    # Poisson standard deviation is sqrt(expected); 5 sigma gives a robust bound.
    assert abs(len(pts) - expected) <= 5.0 * np.sqrt(expected)
