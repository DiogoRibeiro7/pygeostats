"""Tests for anisotropic initialization heuristics."""

import numpy as np
import pytest

from pyspatialstats.variogram.directional import DirectionalVariogram
from pyspatialstats.variogram.initialization import (
    AnisotropyInitializationCandidate,
    AnisotropyInitializationSummary,
)
from tests.data_generation import VariogramParameters, generate_anisotropic_field, generate_isotropic_field


def _angular_distance(a: float, b: float) -> float:
    diff = abs(a - b) % 180.0
    return min(diff, 180.0 - diff)


def test_initialization_identifies_major_direction():
    params = VariogramParameters(nugget=0.05, sill=1.0, range=0.45)
    coords, values = generate_anisotropic_field(
        n_points=80,
        model="exponential",
        params=params,
        stretch=0.4,
        rotation=np.pi / 6,
        seed=24,
    )
    directions = np.linspace(0.0, 157.5, 8)
    dv = DirectionalVariogram(coords, values, directions=directions, tolerance=15.0, n_bins=11)
    dv.compute()

    summary = dv.estimate_initial_parameters(min_weight=4)
    assert isinstance(summary, AnisotropyInitializationSummary)
    assert isinstance(summary.best_candidate, AnisotropyInitializationCandidate)
    assert len(summary.candidates) >= 3

    detection = dv.detect_anisotropy(ratio_threshold=1.1)
    best = summary.best_candidate
    assert best.ratio > 1.2
    if detection.is_anisotropic and np.isfinite(detection.major_direction):
        assert _angular_distance(best.angle_deg, detection.major_direction) <= 20.0


def test_initialization_returns_isotropic_when_data_isotropic():
    params = VariogramParameters(nugget=0.02, sill=0.9, range=0.35)
    coords, values = generate_isotropic_field(60, "gaussian", params, seed=11)
    directions = np.linspace(0.0, 135.0, 6)
    dv = DirectionalVariogram(coords, values, directions=directions, tolerance=20.0, n_bins=9)
    dv.compute()

    summary = dv.estimate_initial_parameters(min_weight=3)
    best = summary.best_candidate
    assert abs(best.ratio - 1.0) <= 0.3
    assert np.isfinite(best.angle_deg)
    assert best.range_major > 0
    assert summary.diagnostics["num_candidates"] >= 1
