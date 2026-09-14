"""Parameter-recovery tests for anisotropy initialization.

These assertions previously ran at import time inside
``pygeostats.variogram.initialization`` under ``if __debug__:``. A failure
there made the entire package unimportable rather than reporting a failing
test, so they have been moved into the suite.
"""

from typing import Dict

import numpy as np
import pytest
from pygeostats.variogram.directional import DirectionalResult
from pygeostats.variogram.initialization import (
    InitializationEnsemble,
    RangeInitializer,
    estimate_rotation_angle,
)

BASE_MAJOR = 0.45
BASE_MINOR = 0.22
SILL = 1.05
NUGGET = 0.05
ANGLES = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0]

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

# Orientations of an exact ellipse for estimate_rotation_angle, in degrees.
ELLIPSE_AXES = [0.0, 20.0, 45.0, 70.0, 90.0, 110.0, 135.0, 160.0]


def _synthetic_directional_results() -> Dict[float, DirectionalResult]:
    """Exponential directional variograms with a known major/minor range."""
    results: Dict[float, DirectionalResult] = {}
    for angle in ANGLES:
        if angle % 90 == 0:
            rng = BASE_MAJOR
        elif angle in (30.0, 150.0):
            rng = 0.5 * (BASE_MAJOR + BASE_MINOR)
        else:
            rng = BASE_MINOR
        bin_centers = np.linspace(0.05, 0.5, 12)
        gamma = SILL - (SILL - NUGGET) * np.exp(-bin_centers / rng)
        results[angle] = DirectionalResult(
            angle=angle,
            bin_centers=bin_centers,
            gamma=gamma,
            counts=np.full(bin_centers.size, 6, dtype=int),
            ci_lower=gamma - 0.05,
            ci_upper=gamma + 0.05,
        )
    return results


def _ellipse_ranges(axis_deg, major, minor, directions):
    delta = np.radians(np.asarray(directions, dtype=float) - axis_deg)
    return 1.0 / np.sqrt(np.cos(delta) ** 2 / major**2 + np.sin(delta) ** 2 / minor**2)


def _axial_difference(a, b):
    difference = abs(a - b) % 180.0
    return min(difference, 180.0 - difference)


@pytest.fixture
def directional_results() -> Dict[float, DirectionalResult]:
    return _synthetic_directional_results()


@pytest.mark.xfail(
    reason="RangeInitializer.estimate() does not recover the major/minor range "
    "of a synthetic exponential field within the 0.03 tolerance asserted in "
    "feca441. Expectation was never executed; needs the estimator checked "
    "against the tolerance, or the tolerance justified.",
    strict=True,
)
def test_range_initializer_recovers_major_and_minor_range(directional_results):
    result = RangeInitializer(directional_results).estimate()
    assert abs(result.range_major - BASE_MAJOR) < 0.03
    assert abs(result.range_minor - BASE_MINOR) < 0.03


def test_range_initializer_keeps_sill_above_nugget(directional_results):
    result = RangeInitializer(directional_results).estimate()
    assert result.sill >= result.nugget


def test_ensemble_reports_candidates_on_a_regular_grid(directional_results):
    result = InitializationEnsemble(REGULAR_GRID, directional_results).run()
    assert result.quality > 0.2
    assert result.top_candidates


def test_ensemble_ratio_is_at_least_one_for_scattered_coordinates(directional_results):
    coords = np.random.default_rng(0).uniform(-1.0, 1.0, size=(40, 2))
    result = InitializationEnsemble(coords, directional_results).run()
    assert result.ratio >= 1.0


@pytest.mark.parametrize(
    "directions",
    [
        np.arange(0.0, 180.0, 15.0),
        np.arange(0.0, 180.0, 30.0),
        np.arange(0.0, 180.0, 45.0),
    ],
    ids=["12-directions", "6-directions", "4-directions"],
)
@pytest.mark.parametrize("axis", ELLIPSE_AXES)
def test_estimate_rotation_angle_reports_the_major_axis(axis, directions):
    # Exact ranges of a 3:1 ellipse. The fit at an angle and at that angle plus 90
    # degrees is the same ellipse with its axes swapped and has the same error, and
    # the grid search reported whichever it met first: 3 to 5 of these 8
    # orientations came back as the minor axis, with a ratio of 1/3.
    ranges = _ellipse_ranges(axis, 0.6, 0.2, directions)
    result = estimate_rotation_angle(directions, ranges)

    assert _axial_difference(result.angle_deg, axis) < 1e-6
    assert result.ratio == pytest.approx(3.0)
    assert result.diagnostics["major_range"] == pytest.approx(0.6)
    assert result.diagnostics["minor_range"] == pytest.approx(0.2)


def test_estimate_rotation_angle_interval_wraps_through_zero():
    # A 3:1 ellipse along the x-axis with 5% noise on its ranges. The near-best fits
    # lie either side of 0 degrees, so the interval runs from just below 180 round
    # to just above 0 and low is greater than high.
    directions = np.arange(0.0, 180.0, 15.0)
    noise = 1.0 + 0.05 * np.random.default_rng(9).standard_normal(directions.size)
    ranges = _ellipse_ranges(0.0, 0.6, 0.2, directions) * noise
    result = estimate_rotation_angle(directions, ranges)

    low, high = result.angle_confidence
    assert low > high
    assert (high - low) % 180.0 < 5.0
    assert result.angle_deg >= low or result.angle_deg <= high
    assert _axial_difference(result.angle_deg, 0.0) < 5.0


def _ellipse_directional_results(axis_deg, major=0.5, minor=0.15):
    """Exact exponential directional variograms whose ranges trace an ellipse."""
    bin_centers = np.linspace(0.02, 1.2, 60)
    results: Dict[float, DirectionalResult] = {}
    for angle in np.arange(0.0, 180.0, 15.0):
        practical_range = _ellipse_ranges(axis_deg, major, minor, [angle])[0]
        gamma = 1.0 - np.exp(-3.0 * bin_centers / practical_range)
        results[float(angle)] = DirectionalResult(
            angle=float(angle),
            bin_centers=bin_centers,
            gamma=gamma,
            counts=np.full(bin_centers.size, 10, dtype=int),
            ci_lower=gamma,
            ci_upper=gamma,
        )
    return results


@pytest.mark.parametrize("axis", [178.0, 2.0, 60.0])
def test_ensemble_angle_is_blended_as_an_axis(axis):
    # Exact directional variograms, and sampling locations elongated along the same
    # axis. The ensemble averaged its angles arithmetically: near 180 degrees the
    # geometry angle and the vote came out near 0, and an axis at 178 was reported
    # as 90.
    rng = np.random.default_rng(0)
    along = rng.uniform(-1.0, 1.0, 60)
    across = rng.uniform(-0.15, 0.15, 60)
    theta = np.radians(axis)
    coords = np.column_stack(
        [
            along * np.cos(theta) - across * np.sin(theta),
            along * np.sin(theta) + across * np.cos(theta),
        ]
    )

    result = InitializationEnsemble(coords, _ellipse_directional_results(axis)).run()

    assert _axial_difference(result.angle_deg, axis) < 5.0
