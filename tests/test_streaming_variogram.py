"""The streaming variogram builder, and optional arguments of the Rust core.

StreamingVariogramBuilder.add_pairs documents ``weights`` as optional, but the Rust
accumulator it calls required the argument, so leaving it out raised TypeError:
PyO3 treats a trailing Option argument as required unless its signature gives it a
default. haversine_distances and fit_anisotropic_variogram had the same gap.
"""

import numpy as np
import pytest
from pygeostats import _core
from pygeostats.variogram import StreamingVariogramBuilder, streaming_variogram

BIN_EDGES = np.linspace(0.0, 5.0, 6)


def test_add_pairs_without_weights_counts_each_pair_once():
    distances = np.array([0.5, 0.7, 2.2, 4.9])
    semivariances = np.array([0.1, 0.3, 0.8, 1.2])

    unweighted = StreamingVariogramBuilder(BIN_EDGES)
    unweighted.add_pairs(distances, semivariances)
    weighted = StreamingVariogramBuilder(BIN_EDGES)
    weighted.add_pairs(distances, semivariances, weights=np.ones(len(distances)))

    result, expected = unweighted.finalize(), weighted.finalize()
    np.testing.assert_array_equal(result.weights, expected.weights)
    np.testing.assert_array_equal(result.gamma, expected.gamma)
    np.testing.assert_array_equal(result.weights, [2, 0, 1, 0, 1])
    assert result.gamma[0] == pytest.approx(0.2)


def test_add_pairs_matches_streaming_variogram():
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 5, size=(40, 2))
    values = rng.normal(size=len(coords))
    first, second = np.triu_indices(len(coords), k=1)
    distances = np.linalg.norm(coords[first] - coords[second], axis=1)
    semivariances = 0.5 * (values[first] - values[second]) ** 2

    builder = StreamingVariogramBuilder(BIN_EDGES)
    builder.add_pairs(distances, semivariances)
    result = builder.finalize()

    expected = streaming_variogram(coords, values, BIN_EDGES)
    np.testing.assert_array_equal(result.weights, expected.weights)
    np.testing.assert_allclose(result.gamma, expected.gamma, rtol=1e-12)


def test_haversine_distances_defaults_to_the_earth_radius():
    # Latitude and longitude in degrees: one degree of longitude on the equator.
    coords = np.array([[0.0, 0.0], [0.0, 1.0]])

    default = _core.haversine_distances(coords)

    np.testing.assert_array_equal(default, _core.haversine_distances(coords, 6371.0))
    assert default[0, 1] == pytest.approx(6371.0 * np.pi / 180.0, rel=1e-9)


def test_fit_anisotropic_variogram_works_without_weights():
    directions = np.radians(np.arange(0, 180, 30))
    lags = np.linspace(0.25, 9.0, 12)
    lag_grid, direction_grid = np.meshgrid(lags, directions)
    distances, angles = lag_grid.ravel(), direction_grid.ravel()
    nugget, sill, range_major, range_minor, rotation = 0.1, 1.0, 3.0, 1.0, 0.6
    scaled = np.hypot(
        distances * np.cos(angles - rotation) / range_major,
        distances * np.sin(angles - rotation) / range_minor,
    )
    gamma = nugget + (sill - nugget) * (1.0 - np.exp(-scaled))
    start = np.array([0.05, 0.8, 2.0, 1.5, 0.3])

    unweighted = _core.fit_anisotropic_variogram(
        distances, gamma, angles, "exponential", start
    )
    weighted = _core.fit_anisotropic_variogram(
        distances, gamma, angles, "exponential", start, weights=np.ones_like(distances)
    )

    assert len(unweighted.parameters) == 5
    np.testing.assert_allclose(unweighted.parameters, weighted.parameters, rtol=1e-8)
