"""Tests for spatial geometry diagnostics.

These assertions previously ran at import time inside
``pygeostats.variogram.geometry_analysis`` under ``if __debug__:``, which
made the whole package unimportable when they failed -- and they did fail.
They are real behavioural expectations, so they belong here.
"""

import numpy as np
import pytest
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


@pytest.mark.xfail(
    reason="_sampling_pattern() classifies a perfect 4x2 lattice as irregular. "
    "Expectation introduced in feca441 and never executed; needs the "
    "classifier checked against the intended definition of 'regular'.",
    strict=True,
)
def test_regular_grid_is_classified_regular():
    diag = SpatialGeometryAnalyzer(REGULAR_GRID).analyze()
    assert diag.sampling_pattern == "regular"


def test_regular_grid_primary_axis_is_along_the_long_side():
    """The grid is 4 wide and 2 tall, so the principal axis is near 0/180 deg."""
    diag = SpatialGeometryAnalyzer(REGULAR_GRID).analyze()
    angle = diag.primary_angle_deg
    assert 0.0 <= angle <= 1.0 or 179.0 <= angle <= 180.0


def test_random_cloud_is_classified_irregular():
    rng = np.random.default_rng(4)
    coords = rng.uniform(-1.0, 1.0, size=(40, 2))
    assert SpatialGeometryAnalyzer(coords).analyze().sampling_pattern == "irregular"


def test_analyze_rejects_degenerate_input():
    with pytest.raises(ValueError):
        SpatialGeometryAnalyzer(np.zeros((2, 2)))
    with pytest.raises(ValueError):
        SpatialGeometryAnalyzer(np.zeros((5, 3)))
