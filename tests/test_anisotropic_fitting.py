"""Tests for anisotropic variogram fitting.

The anisotropic path shares its Levenberg-Marquardt loop with the isotropic fit,
including the projected optimality test and the active-set step, but nothing
exercised it. Nothing in the Python package calls it either:
create_anisotropic_variogram_from_directional sets parameters without fitting.
These tests therefore call the Rust core directly.

The data is a noise-free anisotropic variogram sampled on a grid of directions
and lags, so the generating parameters are exactly recoverable -- unlike a
random field, where a single realisation does not determine them.

An ellipse with its axes swapped and its angle rotated by 90 degrees is the same
model, and the angle is only defined modulo 180 degrees. The fit reports the
longer axis as range_major. Recovery is checked in a canonical form and the
ordering separately, so a failure to recover and a failure to order show up as
different assertions.
"""

import numpy as np
import pytest
from pygeostats._core import fit_anisotropic_variogram

MODELS = ["exponential", "gaussian", "spherical"]

# nugget, sill, range_major, range_minor, rotation angle in radians
TRUTH = (0.1, 1.0, 3.0, 1.0, 0.6)
MODEST_START = (0.05, 0.8, 2.0, 1.5, 0.3)
# From this start the optimiser converges to the swapped labelling of TRUTH.
FAR_START = (0.0, 1.5, 6.0, 0.5, 1.4)
# A 60:1 field, past the ratio of 50 above which anisotropy is treated as
# unreliable.
EXTREME_TRUTH = (0.05, 1.0, 6.0, 0.1, 0.6)

# The optimiser caps at 250 iterations. Every fit of TRUTH measured for these
# tests converged in under 30.
ITERATION_BUDGET = 100
# Noise-free fits recovered TRUTH to 4e-7 or better across 90 random starts.
TOLERANCE = 1e-5
# The 60:1 field converges less tightly: the worst kept fit measured was 7e-6.
EXTREME_TOLERANCE = 1e-4
# A range on its floor, as returned by an isotropic fit that collapsed.
COLLAPSED_RANGE = 1e-6 * (1 + 1e-9)

RATIO_WARNING = "exceeds recommended limits"
REJECTION_WARNING = "Isotropic fallback rejected"

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


def _fit_gamma(gamma, start, model="exponential", directions=ANGLES):
    return fit_anisotropic_variogram(
        DISTANCES,
        gamma,
        directions,
        model,
        np.asarray(start, dtype=float),
        weights=np.ones_like(DISTANCES),
    )


def _fit(truth, start, model, directions=ANGLES):
    return _fit_gamma(_anisotropic_gamma(truth, model), start, model, directions)


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


def _extreme_starts():
    # Before axes were ordered, 9 of these 12 starts fell back and 3 converged to
    # the swapped labelling, reported a ratio of 0.017 and skipped both the warning
    # and the fallback. Once ordered, all 12 fell back -- to an isotropic model
    # whose range had collapsed to the floor, reported as converged.
    rng = np.random.default_rng(7)
    starts = [
        MODEST_START,
        FAR_START,
        (0.1, 1.0, 0.5, 4.0, 0.2),
        (0.1, 1.0, 1.0, 1.0, 0.0),
    ]
    for _ in range(8):
        starts.append(
            (
                rng.uniform(0.0, 0.2),
                rng.uniform(0.5, 1.5),
                rng.uniform(0.1, 8.0),
                rng.uniform(0.1, 8.0),
                rng.uniform(0.0, np.pi),
            )
        )
    return starts


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
        assert result.parameters[2] >= result.parameters[3], start


def test_far_start_reaches_an_equivalent_model():
    result = _fit(TRUTH, FAR_START, "exponential")

    _assert_clean_anisotropic_fit(result)
    assert np.all(_canonical_error(result.parameters, TRUTH) < TOLERANCE)


def test_longer_axis_is_reported_as_range_major():
    # FAR_START converges to the swapped labelling of TRUTH. The fit used to return
    # it as found -- range_major 1, range_minor 3 -- so
    # AnisotropicKriging.get_anisotropy_info reported the 3:1 field as a ratio of
    # 0.33 with is_anisotropic=False.
    result = _fit(TRUTH, FAR_START, "exponential")

    _assert_clean_anisotropic_fit(result)
    np.testing.assert_allclose(result.parameters, TRUTH, rtol=0, atol=TOLERANCE)


def test_parameter_std_follows_the_axis_swap():
    # Noise gives the standard errors a nonzero size. Both starts reach the same
    # optimum, FAR_START in the swapped labelling, so once the axes are reordered
    # the standard errors must match too. They are computed where the Jacobian was
    # evaluated, before reordering, and have to move with the ranges.
    noise = 0.02 * np.random.default_rng(3).standard_normal(DISTANCES.size)
    gamma = _anisotropic_gamma(TRUTH, "exponential") + noise
    from_modest = _fit_gamma(gamma, MODEST_START)
    from_far = _fit_gamma(gamma, FAR_START)

    for result in (from_modest, from_far):
        _assert_clean_anisotropic_fit(result)
        assert result.parameters[2] >= result.parameters[3]
    np.testing.assert_allclose(
        from_far.parameters, from_modest.parameters, rtol=0, atol=TOLERANCE
    )
    np.testing.assert_allclose(
        from_far.parameter_std, from_modest.parameter_std, rtol=0, atol=TOLERANCE
    )


def test_extreme_anisotropy_is_never_reported_as_a_clean_fit():
    # Ratios above 50 are treated as unreliable. The fit warns, then tries an
    # isotropic fallback and uses it only if that fit is itself valid. On this field
    # it usually is not: the isotropic fit collapses its range to the floor, giving
    # a constant model. In that case the anisotropic fit is kept, with a warning,
    # and not reported as converged.
    #
    # Which branch a start takes depends on the local optimum it reaches, so each
    # result is held to the rule rather than pinned to a branch. Measured: 11 of the
    # 12 starts reach the true 60:1 model and keep it; the twelfth reaches a
    # different optimum, which a valid isotropic fit replaces.
    gamma = _anisotropic_gamma(EXTREME_TRUTH, "exponential")
    kept = 0
    for start in _extreme_starts():
        result = _fit_gamma(gamma, start)

        assert any(RATIO_WARNING in message for message in result.warnings), start
        if result.fallback_used:
            assert result.status == "fallback_to_isotropic", start
            assert result.converged, start
            assert not any(REJECTION_WARNING in m for m in result.warnings), start
            assert result.parameters[2] > COLLAPSED_RANGE, start
        else:
            kept += 1
            assert result.status == "succeeded", start
            assert not result.converged, start
            assert any(REJECTION_WARNING in m for m in result.warnings), start
            assert result.parameters[2] >= result.parameters[3], start
            error = _canonical_error(result.parameters, EXTREME_TRUTH)
            assert np.all(error < EXTREME_TOLERANCE), start
    assert kept >= 1


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
