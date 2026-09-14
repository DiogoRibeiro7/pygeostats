"""Tests for averaging axial directions."""

import pytest
from pygeostats.variogram._axial import axial_mean


def _axial_difference(a, b):
    difference = abs(a - b) % 180.0
    return min(difference, 180.0 - difference)


def test_axes_either_side_of_zero_average_to_zero():
    # 175 and 5 degrees are 10 degrees apart as axes; their arithmetic mean is 90.
    assert _axial_difference(axial_mean([(175.0, 1.0), (5.0, 1.0)]), 0.0) < 1e-9


def test_axes_away_from_the_wrap_average_as_numbers():
    assert axial_mean([(30.0, 1.0), (50.0, 1.0)]) == pytest.approx(40.0)


def test_weights_pull_towards_the_heavier_axis():
    # 178 and 0 degrees are 2 degrees apart as axes. Weighted 0.6 and 0.4, the mean
    # lies about 0.8 degrees from 178 towards 0.
    assert axial_mean([(178.0, 0.6), (0.0, 0.4)]) == pytest.approx(178.8, abs=0.01)


def test_perpendicular_axes_of_equal_weight_have_no_mean():
    assert axial_mean([(0.0, 1.0), (90.0, 1.0)]) == 0.0


@pytest.mark.parametrize(
    ("angle", "axis"), [(-30.0, 150.0), (200.0, 20.0), (359.0, 179.0)]
)
def test_result_is_the_same_axis_in_zero_to_180(angle, axis):
    result = axial_mean([(angle, 1.0)])
    assert 0.0 <= result < 180.0
    assert result == pytest.approx(axis)
