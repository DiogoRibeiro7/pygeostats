"""Property-based tests for empirical variogram behaviour."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pygeostats.variogram.empirical import EmpiricalVariogram
from pygeostats.variogram.models import Variogram
from scipy.stats import spearmanr

from .data_generation import VariogramParameters, generate_isotropic_field

# too_slow is suppressed because the slow draws are not this strategy's: its
# examples generate in milliseconds. Hypothesis scans the source of modules it
# considers local for constants, and it does so inside a draw whenever new
# modules have been imported. It recognises installed packages by a
# "/site-packages/" substring, which never matches a Windows path, so there it
# scans numpy, scipy and every other third-party module the earlier tests
# imported. Locally this stalled a single draw for 86-101 s and failed the run.
_PROPERTY_SETTINGS = settings(
    deadline=None, suppress_health_check=[HealthCheck.too_slow]
)


@st.composite
def coordinate_value_sets(draw):
    n_points = draw(st.integers(min_value=3, max_value=12))
    coords = np.array(
        draw(
            st.lists(
                st.tuples(
                    st.floats(
                        min_value=-1.0,
                        max_value=1.0,
                        allow_nan=False,
                        allow_infinity=False,
                    ),
                    st.floats(
                        min_value=-1.0,
                        max_value=1.0,
                        allow_nan=False,
                        allow_infinity=False,
                    ),
                ),
                min_size=n_points,
                max_size=n_points,
            )
        ),
        dtype=float,
    )
    values = np.array(
        draw(
            st.lists(
                st.floats(
                    min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False
                ),
                min_size=n_points,
                max_size=n_points,
            )
        ),
        dtype=float,
    )
    return coords, values


@settings(_PROPERTY_SETTINGS, max_examples=25)
@given(data=coordinate_value_sets())
def test_empirical_variogram_non_negative(data):
    coords, values = data
    ev = EmpiricalVariogram(coords, values, n_bins=8).compute()
    mask = ev.counts_ > 0
    assert np.all(ev.gamma_[mask] >= 0.0)


def test_empirical_variogram_zero_distance_duplicate_points():
    coords = np.array([[0.0, 0.0], [0.0, 0.0], [0.5, 0.0]])
    values = np.array([1.0, 1.0, 2.0])
    ev = EmpiricalVariogram(coords, values, n_bins=4).compute()
    zero_bin = ev.counts_ > 0
    assert np.isclose(ev.gamma_[zero_bin][0], 0.0, atol=1e-10)


def test_empirical_variogram_handles_single_point():
    # A single point has no pairs. With default bin edges compute() used to raise,
    # taking the maximum of an empty distance array.
    coords = np.array([[0.2, 0.4]])
    values = np.array([5.0])
    ev = EmpiricalVariogram(coords, values, n_bins=2).compute()
    assert np.all(ev.counts_ == 0)
    assert np.all(ev.gamma_ == 0.0)


def test_empirical_variogram_collinear_data():
    coords = np.column_stack((np.linspace(0, 1, 10), np.zeros(10)))
    values = np.linspace(0, 5, 10)
    ev = EmpiricalVariogram(coords, values, n_bins=6).compute()
    assert ev.is_fitted_


def test_empirical_variogram_extreme_values():
    coords = np.array([[0.0, 0.0], [1e6, 0.0], [0.0, 1e6]])
    values = np.array([1e9, -1e9, 5e8])
    ev = EmpiricalVariogram(coords, values, n_bins=5).compute()
    mask = ev.counts_ > 0
    assert np.all(np.isfinite(ev.gamma_[mask]))


def test_variogram_rises_from_short_to_long_lags():
    # An empirical variogram is a noisy estimate and dips freely: for this field it
    # was non-decreasing on 1 seed in 100, so that is not a property to test. The
    # trend is. Over 100 seeds the first third of populated bins averaged below the
    # last third every time, and the rank correlation between lag and semivariance
    # was at least 0.28, with a median of 0.86. The set of seeds is held to that
    # with margin rather than each seed.
    params = VariogramParameters(nugget=0.05, sill=1.0, range=0.4)
    rises = 0
    correlations = []
    for seed in range(20):
        coords, values = generate_isotropic_field(40, "exponential", params, seed=seed)
        ev = EmpiricalVariogram(coords, values, n_bins=10).compute()
        populated = ev.counts_ > 0
        gamma = ev.gamma_[populated]
        third = max(len(gamma) // 3, 1)
        rises += int(gamma[:third].mean() < gamma[-third:].mean())
        correlations.append(spearmanr(ev.distances_[populated], gamma).correlation)

    assert rises >= 18
    assert np.median(correlations) > 0.5


@pytest.mark.parametrize("model", ["exponential", "spherical", "gaussian"])
def test_variogram_parameter_boundaries(model):
    params = VariogramParameters(nugget=0.03, sill=0.9, range=0.35)
    coords, values = generate_isotropic_field(30, model, params, seed=11)
    ev = EmpiricalVariogram(coords, values, n_bins=12).compute()

    variogram = Variogram(model=model)
    variogram.fit(ev.distances_, ev.gamma_, weights=ev.counts_)

    assert variogram.nugget_ >= 0.0
    assert variogram.sill_ >= variogram.nugget_
    assert variogram.range_ > 0.0


@settings(_PROPERTY_SETTINGS, max_examples=10)
@given(data=coordinate_value_sets())
def test_variogram_symmetry_on_random_data(data):
    coords, values = data
    ev = EmpiricalVariogram(coords, values, n_bins=5).compute()
    mask = ev.counts_ > 0
    distances = ev.distances_[mask]
    gamma = ev.gamma_[mask]
    order = np.argsort(distances)
    distances = distances[order]
    gamma = gamma[order]
    assert np.all(np.diff(distances) >= 0)
    assert np.all(gamma >= 0)
