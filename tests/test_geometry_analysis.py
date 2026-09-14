"""Tests for spatial geometry diagnostics.

These assertions previously ran at import time inside
``pygeostats.variogram.geometry_analysis`` under ``if __debug__:``, which
made the whole package unimportable when they failed -- and they did fail.
They are real behavioural expectations, so they belong here.
"""

import numpy as np
import pytest
from pygeostats.variogram.directional import DirectionalResult
from pygeostats.variogram.geometry_analysis import SpatialGeometryAnalyzer

REGULAR_GRID = np.array(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, 1.0],
        [3.0, 1.0],
    ]
)


def _grid(nx, ny):
    xs, ys = np.meshgrid(np.arange(float(nx)), np.arange(float(ny)))
    return np.column_stack([xs.ravel(), ys.ravel()])


def _pattern(coords):
    return SpatialGeometryAnalyzer(coords).analyze().sampling_pattern


def test_regular_grid_is_classified_regular():
    # Classification used distances between consecutive rows. Listed row by row,
    # this grid has a jump of sqrt(10) between its rows and came out irregular.
    assert _pattern(REGULAR_GRID) == "regular"


def test_sampling_pattern_does_not_depend_on_point_order():
    # A 10x10 grid came out irregular in its row order, shuffled, and sorted by x.
    grid = _grid(10, 10)
    shuffled = grid[np.random.default_rng(0).permutation(len(grid))]

    assert _pattern(grid) == "regular"
    assert _pattern(shuffled) == "regular"


def test_hexagonal_lattice_is_classified_regular():
    lattice = np.array(
        [
            [i + 0.5 * (j % 2), j * np.sqrt(3.0) / 2.0]
            for i in range(8)
            for j in range(8)
        ]
    )
    assert _pattern(lattice) == "regular"


@pytest.mark.parametrize(
    ("jitter", "expected"), [(0.1, "regular"), (0.3, "quasi-regular")]
)
def test_jittered_grid(jitter, expected):
    # Uniform jitter of up to jitter * spacing in each coordinate. Over 20 seeds the
    # nearest-neighbour coefficient of variation was 0.05 at 0.1 and 0.17 to 0.20 at
    # 0.3, against thresholds of 0.15 and 0.35.
    grid = _grid(20, 20)
    offsets = np.random.default_rng(0).uniform(-jitter, jitter, size=grid.shape)
    assert _pattern(grid + offsets) == expected


def test_random_cloud_is_classified_irregular():
    rng = np.random.default_rng(4)
    coords = rng.uniform(-1.0, 1.0, size=(40, 2))
    assert SpatialGeometryAnalyzer(coords).analyze().sampling_pattern == "irregular"


def test_repeated_location_is_counted_once():
    # A duplicate's nearest neighbour is at distance 0. Counted twice, this one
    # location would put the grid's coefficient of variation at 0.54.
    assert _pattern(np.vstack([REGULAR_GRID, REGULAR_GRID[:1]])) == "regular"


def test_coincident_points_are_insufficient():
    assert _pattern(np.zeros((3, 2))) == "insufficient"


def test_primary_angle_is_averaged_as_an_axis():
    # Points along 175 degrees, and the longest directional range at 0 degrees. As
    # axes these are 5 degrees apart, but their arithmetic mean put the primary
    # angle at 87.5, perpendicular to both.
    along = np.linspace(-1.0, 1.0, 30)
    theta = np.radians(175.0)
    coords = np.column_stack([along * np.cos(theta), along * np.sin(theta)])
    bin_centers = np.linspace(0.1, 1.0, 10)
    results = {}
    for angle, length in ((0.0, 0.5), (45.0, 0.2), (90.0, 0.1), (135.0, 0.2)):
        gamma = 1.0 - np.exp(-bin_centers / length)
        results[angle] = DirectionalResult(
            angle=angle,
            bin_centers=bin_centers,
            gamma=gamma,
            counts=np.full(bin_centers.size, 5),
            ci_lower=gamma,
            ci_upper=gamma,
        )

    diag = SpatialGeometryAnalyzer(coords, results).analyze()

    assert diag.primary_angle_deg == pytest.approx(177.5)
    assert diag.secondary_angle_deg == pytest.approx(87.5)


def test_regular_grid_primary_axis_is_along_the_long_side():
    """The grid is 4 wide and 2 tall, so the principal axis is near 0/180 deg."""
    diag = SpatialGeometryAnalyzer(REGULAR_GRID).analyze()
    angle = diag.primary_angle_deg
    assert 0.0 <= angle <= 1.0 or 179.0 <= angle <= 180.0


def test_analyze_rejects_degenerate_input():
    with pytest.raises(ValueError):
        SpatialGeometryAnalyzer(np.zeros((2, 2)))
    with pytest.raises(ValueError):
        SpatialGeometryAnalyzer(np.zeros((5, 3)))
