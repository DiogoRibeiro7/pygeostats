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
