"""Tests for anisotropic variogram fitting.

The anisotropic path shares its Levenberg-Marquardt loop with the isotropic fit,
including the projected optimality test and the active-set step, but nothing
exercised it. Nothing in the Python package calls it either:
create_anisotropic_variogram_from_directional sets parameters without fitting.
These tests therefore call the Rust core directly.

The data is a noise-free anisotropic variogram sampled on a grid of directions
and lags, so the generating parameters are exactly recoverable -- unlike a
random field, where a single realisation does not determine them.

Parameters are compared in a canonical form. An ellipse with its axes swapped
and its angle rotated by 90 degrees is the same model, and the angle is only
defined modulo 180 degrees.
"""

import numpy as np
import pytest
from pygeostats._core import fit_anisotropic_variogram

MODELS = ["exponential", "gaussian", "spherical"]

# nugget, sill, range_major, range_minor, rotation angle in radians
TRUTH = (0.1, 1.0, 3.0, 1.0, 0.6)
MODEST_START = (0.05, 0.8, 2.0, 1.5, 0.3)
# From this start the fit returns the swapped-axes form of TRUTH.
FAR_START = (0.0, 1.5, 6.0, 0.5, 1.4)

# The optimiser caps at 250 iterations. Every fit measured for these tests
# converged in under 30.
ITERATION_BUDGET = 100
# Noise-free fits recovered TRUTH to 4e-7 or better across 90 random starts.
TOLERANCE = 1e-5

DIRECTIONS = np.radians(np.arange(0, 180, 15))
LAGS = np.linspace(0.25, 9.0, 18)
_LAG_GRID, _DIRECTION_GRID = np.meshgrid(LAGS, DIRECTIONS)
DISTANCES = _LAG_GRID.ravel()
ANGLES = _DIRECTION_GRID.ravel()


def _anisotropic_gamma(params, model):
    nugget, sill, range_major, range_minor, rotation = params
    delta = ANGLES - rotation
    h = np.hypot(
        DISTANCES * np.cos(delta) / range_major,
        DISTANCES * np.sin(delta) / range_minor,
    )
    if model == "exponential":
        shape = 1.0 - np.exp(-h)
    elif model == "gaussian":
        shape = 1.0 - np.exp(-(h**2))
    else:
        shape = np.where(h < 1.0, 1.5 * h - 0.5 * h**3, 1.0)
    return nugget + (sill - nugget) * shape


def _fit(truth, start, model, directions=ANGLES):
    return fit_anisotropic_variogram(
        DISTANCES,
        _anisotropic_gamma(truth, model),
        directions,
        model,
        np.asarray(start, dtype=float),
        weights=np.ones_like(DISTANCES),
    )


def _canonical(params):
    nugget, sill, range_major, range_minor, rotation = params
    if range_minor > range_major:
        range_major, range_minor = range_minor, range_major
        rotation += np.pi / 2
    return np.array([nugget, sill, range_major, range_minor, rotation % np.pi])


def _canonical_error(fitted, truth):
    error = np.abs(_canonical(fitted) - _canonical(truth))
    error[4] = min(error[4], np.pi - error[4])
    return error


def _assert_clean_anisotropic_fit(result):
    assert result.status == "succeeded"
    assert result.converged
    assert not result.fallback_used
    assert len(result.parameters) == 5
    assert result.iterations < ITERATION_BUDGET


@pytest.mark.parametrize("model", MODELS)
def test_recovers_known_parameters(model):
    result = _fit(TRUTH, MODEST_START, model)

    _assert_clean_anisotropic_fit(result)
    assert np.all(_canonical_error(result.parameters, TRUTH) < TOLERANCE)


@pytest.mark.parametrize("model", MODELS)
def test_recovers_known_parameters_from_random_starts(model):
    rng = np.random.default_rng(12345)
    for _ in range(30):
        start = (
            rng.uniform(0.0, 0.3),
            rng.uniform(0.5, 1.5),
            rng.uniform(0.5, 6.0),
            rng.uniform(0.5, 6.0),
            rng.uniform(0.0, np.pi),
        )
        result = _fit(TRUTH, start, model)

        _assert_clean_anisotropic_fit(result)
        assert np.all(_canonical_error(result.parameters, TRUTH) < TOLERANCE), start


def test_far_start_reaches_an_equivalent_model():
    result = _fit(TRUTH, FAR_START, "exponential")

    _assert_clean_anisotropic_fit(result)
    assert np.all(_canonical_error(result.parameters, TRUTH) < TOLERANCE)


def test_optimum_with_nugget_on_its_bound():
    # The zero nugget sits on its lower bound. The anisotropic path uses the same
    # active-set step as the isotropic fit, which holds a parameter pinned against
    # its bound out of the solve; without it such optima ran to the iteration cap.
    truth = (0.0, 1.0, 3.0, 1.0, 0.6)
    result = _fit(truth, MODEST_START, "exponential")

    _assert_clean_anisotropic_fit(result)
    assert result.parameters[0] == pytest.approx(0.0, abs=1e-10)
    assert np.all(_canonical_error(result.parameters, truth) < TOLERANCE)


def test_isotropic_data_recovers_equal_ranges():
    # With equal ranges the rotation angle has no effect on the model, so it is
    # not checked.
    truth = (0.1, 1.0, 2.0, 2.0, 0.0)
    result = _fit(truth, (0.05, 0.8, 3.0, 1.0, 0.3), "exponential")

    _assert_clean_anisotropic_fit(result)
    nugget, sill, range_major, range_minor, _ = result.parameters
    assert (nugget, sill, range_major, range_minor) == pytest.approx(
        (0.1, 1.0, 2.0, 2.0), abs=TOLERANCE
    )


def test_directions_in_degrees_match_radians():
    # The core treats direction values larger than 2*pi as degrees.
    in_radians = _fit(TRUTH, MODEST_START, "exponential")
    in_degrees = _fit(TRUTH, MODEST_START, "exponential", directions=np.degrees(ANGLES))

    np.testing.assert_allclose(
        in_degrees.parameters, in_radians.parameters, rtol=0, atol=1e-12
    )


@pytest.mark.xfail(
    reason=(
        "The fit does not order the axes: from some starts it returns the same "
        "ellipse with range_major < range_minor and the angle rotated by 90 "
        "degrees. Measured over 30 random starts per model, this happened in "
        "11-17 of them. AnisotropicKriging.get_anisotropy_info divides "
        "range_major by range_minor, so for a 3:1 field it then reports a ratio "
        "of 0.33 and is_anisotropic=False."
    ),
    strict=True,
)
def test_fitted_range_major_is_the_larger_range():
    # Kept separate from test_far_start_reaches_an_equivalent_model so that this
    # xfail can only ever be hiding the axis ordering.
    result = _fit(TRUTH, FAR_START, "exponential")

    assert result.parameters[2] >= result.parameters[3]
