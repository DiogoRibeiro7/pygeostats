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


@pytest.mark.xfail(
    reason="detect_anisotropy() reports a major direction of 45 deg for a field stretched along 90 deg. Either an angle-convention mismatch (major axis vs its normal) or a genuine detection error; needs the convention pinned down.",
    strict=True,
)
def test_directional_variogram_detects_anisotropy():
    params = VariogramParameters(nugget=0.05, sill=1.2, range=0.4)
    coords, values = generate_anisotropic_field(
        60, "exponential", params, stretch=0.4, rotation=0.0, seed=12
    )
    dv = DirectionalVariogram(
        coords, values, directions=[0.0, 45.0, 90.0, 135.0], tolerance=15.0, n_bins=10
    )
    dv.compute()
    result = dv.detect_anisotropy(ratio_threshold=1.1, range_difference=0.02)
    assert result.is_anisotropic
    assert pytest.approx(result.major_direction, abs=15.0) == 90.0
    assert result.anisotropy_ratio > 1.5


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


@pytest.mark.xfail(
    reason="Bandwidth filtering discards every pair -- all bin counts come back zero instead of a reduced but non-empty set.",
    strict=True,
)
def test_directional_variogram_bandwidth_filters_pairs():
    coords = np.array([[0.0, 0.0], [1.0, 0.2], [2.0, 0.4], [3.0, 1.0]])
    values = np.array([1.0, 1.5, 2.0, 2.5])
    dv = DirectionalVariogram(
        coords, values, directions=[0.0], tolerance=20.0, bandwidth=0.15, n_bins=6
    )
    dv.compute()
    summary = dv.directional_summary()[0.0]
    assert summary["counts"].sum() > 0
    dv_wide = DirectionalVariogram(
        coords, values, directions=[0.0], tolerance=20.0, bandwidth=None, n_bins=6
    )
    dv_wide.compute()
    summary_wide = dv_wide.directional_summary()[0.0]
    assert summary_wide["counts"].sum() >= summary["counts"].sum()
