"""Averaging axial directions, which repeat every 180 degrees."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Tuple

import numpy as np


def axial_mean(weighted_angles: Iterable[Tuple[float, float]]) -> float:
    """Weighted mean of axial directions, given as ``(angle_deg, weight)`` pairs.

    A direction and its opposite are the same axis, so 175 and 5 degrees are 10
    degrees apart rather than 170, and their arithmetic mean of 90 is perpendicular
    to both. Doubling each angle turns every axis into a single point on the circle,
    where a vector mean is well defined; halving the result turns it back.

    Returns degrees in [0, 180). When the weighted axes cancel, as two perpendicular
    axes of equal weight do, there is no mean and 0 is returned.
    """
    sin_sum = 0.0
    cos_sum = 0.0
    total_weight = 0.0
    for angle_deg, weight in weighted_angles:
        doubled = 2.0 * np.deg2rad(angle_deg)
        sin_sum += weight * np.sin(doubled)
        cos_sum += weight * np.cos(doubled)
        total_weight += abs(weight)
    if np.hypot(sin_sum, cos_sum) <= 1e-12 * total_weight:
        return 0.0
    angle = float(np.degrees(0.5 * np.arctan2(sin_sum, cos_sum))) % 180.0
    # A result a hair below 0 wraps to a value that rounds to exactly 180.
    return 0.0 if angle >= 180.0 else angle
