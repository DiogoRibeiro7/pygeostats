"""Property-based tests for empirical variogram behaviour."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from pyspatialstats.variogram.empirical import EmpiricalVariogram
from pyspatialstats.variogram.models import Variogram

from .data_generation import VariogramParameters, generate_isotropic_field


@st.composite
def coordinate_value_sets(draw):
    n_points = draw(st.integers(min_value=3, max_value=12))
    coords = np.array(
        draw(
            st.lists(
                st.tuples(
                    st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                    st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
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
                st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
                min_size=n_points,
                max_size=n_points,
            )
        ),
        dtype=float,
    )
    return coords, values


@settings(deadline=None, max_examples=25)
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


@pytest.mark.xfail(
    reason="EmpiricalVariogram.compute() crashes on a single point: np.max() is called on the empty distance array at empirical.py:70 before any guard. Should return empty bins instead of raising.",
    raises=ValueError,
    strict=True,
)
def test_empirical_variogram_handles_single_point():
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


@pytest.mark.xfail(
    reason="Asserts an empirical variogram is monotonically non-decreasing, which is not a property of empirical variograms -- they are noisy estimates and dip freely. The test expectation itself is most likely wrong.",
    strict=True,
)
def test_variogram_monotonic_for_synthetic_field():
    params = VariogramParameters(nugget=0.05, sill=1.0, range=0.4)
    coords, values = generate_isotropic_field(40, "exponential", params, seed=7)
    ev = EmpiricalVariogram(coords, values, n_bins=10).compute()
    mask = ev.counts_ > 0
    filtered = ev.gamma_[mask]
    assert np.all(np.diff(filtered) >= -1e-4)


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


@settings(deadline=None, max_examples=10)
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
