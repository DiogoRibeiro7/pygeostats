"""Tests for directional variogram analysis."""

import numpy as np
import pytest

pytest.importorskip("pygeostats._core", reason="requires compiled Rust extension")

from pygeostats.variogram.directional import DirectionalVariogram

from tests.data_generation import VariogramParameters, generate_anisotropic_field


def test_directional_variogram_handles_basic_geometry():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    values = np.array([0.0, 1.0, 2.0, 3.0])
    dv = DirectionalVariogram(
        coords, values, directions=[0.0, 90.0], tolerance=10.0, n_bins=5
    )
    dv.compute()
    summary = dv.directional_summary()
    assert 0.0 in summary and 90.0 in summary
    assert summary[90.0]["counts"].sum() == 0


DETECTION_PARAMS = VariogramParameters(nugget=0.02, sill=1.0, range=0.25)
DETECTION_DIRECTIONS = list(np.arange(0.0, 180.0, 15.0))


def _axial_difference(a, b):
    difference = abs(a - b) % 180.0
    return min(difference, 180.0 - difference)


def _detect(seed, stretch, axis_deg=0.0):
    coords, values = generate_anisotropic_field(
        150,
        "exponential",
        DETECTION_PARAMS,
        stretch=stretch,
        rotation=np.radians(axis_deg),
        seed=seed,
    )
    dv = DirectionalVariogram(
        coords, values, directions=DETECTION_DIRECTIONS, tolerance=15.0, n_bins=10
    )
    return dv.compute().detect_anisotropy()


def test_detect_anisotropy_recovers_the_major_axis():
    # Thirty independent fields with a 2.9:1 ratio and major axes drawn between the
    # sampled directions. Reporting the direction with the longest range, ties going
    # to the first direction listed, missed the axis by 30 degrees in the median on
    # fields like these. A single realisation is noisy: over 60 such fields the
    # fitted ellipse came within 15 degrees for 54 and was 52 degrees off at worst,
    # so the set is held to that rate with margin instead of pinning each field.
    axes = np.random.default_rng(2024).uniform(0.0, 180.0, size=30)
    results = [_detect(1000 + i, 0.35, axis) for i, axis in enumerate(axes)]
    errors = np.array(
        [
            _axial_difference(result.major_direction, axis)
            for result, axis in zip(results, axes, strict=True)
        ]
    )

    assert np.sum(errors <= 15.0) >= 21
    assert np.median(errors) <= 10.0
    assert sum(result.is_anisotropic for result in results) >= 27


def test_detect_anisotropy_ratio_separates_isotropic_fields():
    # The ratio runs low -- about 1.4 for these 2.9:1 fields -- but isotropic fields
    # sit lower still, around 1.06, and no ratio is below 1.
    anisotropic = [_detect(3000 + i, 0.35).anisotropy_ratio for i in range(10)]
    isotropic = [_detect(4000 + i, 1.0).anisotropy_ratio for i in range(10)]

    assert min(anisotropic + isotropic) >= 1.0
    assert np.median(isotropic) < np.median(anisotropic)


def test_anisotropy_rose_data_matches_directions():
    params = VariogramParameters(nugget=0.02, sill=0.8, range=0.3)
    coords, values = generate_anisotropic_field(
        50, "gaussian", params, stretch=0.5, rotation=np.pi / 6, seed=5
    )
    directions = DirectionalVariogram.automatic_direction_set(coords, n_directions=4)
    dv = DirectionalVariogram(
        coords, values, directions=directions, tolerance=20.0, n_bins=9
    )
    dv.compute()
    angles, ranges = dv.anisotropy_rose_data()
    assert len(angles) == len(directions)
    assert len(ranges) == len(directions)
    assert np.all(np.isfinite(ranges))


def test_directional_variogram_requires_fitted_before_anisotropy():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    values = np.array([1.0, 2.0, 3.0])
    dv = DirectionalVariogram(coords, values)
    with pytest.raises(RuntimeError):
        dv.detect_anisotropy()


def _pairs_along_x(coords, values, bandwidth):
    dv = DirectionalVariogram(
        coords,
        values,
        directions=[0.0],
        tolerance=20.0,
        bandwidth=bandwidth,
        max_distance=2.0,
        n_bins=4,
    ).compute()
    summary = dv.directional_summary()[0.0]
    populated = summary["counts"] > 0
    return int(summary["counts"].sum()), summary["gamma"][populated]


def test_directional_variogram_bandwidth_filters_pairs():
    # A-B and A-C both lie within 20 degrees of the x-axis, 0.1 and 0.3 off it, and
    # share a lag bin; every other pair lies outside the sector. The data this test
    # used before put every in-range pair 0.2 or more off the axis, so a bandwidth
    # of 0.15 rightly kept none.
    coords = np.array([[0.0, 0.0], [1.0, 0.1], [1.0, -0.3], [0.0, 2.0]])
    values = np.array([0.0, 1.0, 3.0, 10.0])

    count, gamma = _pairs_along_x(coords, values, bandwidth=None)
    assert count == 2
    np.testing.assert_allclose(gamma, [(0.5 + 4.5) / 2])

    count, gamma = _pairs_along_x(coords, values, bandwidth=0.2)
    assert count == 1
    np.testing.assert_allclose(gamma, [0.5])

    count, _ = _pairs_along_x(coords, values, bandwidth=0.05)
    assert count == 0
