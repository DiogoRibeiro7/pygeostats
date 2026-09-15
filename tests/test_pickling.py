"""Fitted variograms and kriging models survive pickling and copying.

ParallelKrigingExecutor sends the fitted model to worker processes by pickling it.
A fitted Variogram holds the FittingResult returned by the Rust core, which could
not be pickled, so process workers failed for every fitted model, and
copy.deepcopy failed too.
"""

import copy
import pickle

import numpy as np
import pytest
from pygeostats._core import fit_variogram_model
from pygeostats.kriging import (
    AnisotropicKriging,
    OrdinaryKriging,
    SimpleKriging,
    UniversalKriging,
)
from pygeostats.variogram import EmpiricalVariogram, Variogram

FITTING_RESULT_FIELDS = (
    "parameters",
    "r_squared",
    "rmse",
    "converged",
    "iterations",
    "message",
    "status",
    "parameter_std",
    "diagnostics",
    "trace",
    "fallback_used",
    "warnings",
)

ROUND_TRIPS = {
    "pickle": lambda obj: pickle.loads(pickle.dumps(obj)),
    "deepcopy": copy.deepcopy,
}


@pytest.fixture(scope="module")
def samples():
    rng = np.random.default_rng(3)
    coords = rng.uniform(0, 10, size=(50, 2))
    values = np.sin(coords[:, 0] / 2) + rng.normal(0, 0.1, len(coords))
    return coords, values


@pytest.fixture(scope="module")
def empirical(samples):
    return EmpiricalVariogram(*samples, n_bins=10).compute()


@pytest.fixture
def variogram(empirical):
    return Variogram(model="exponential").fit(
        empirical.distances_, empirical.gamma_, weights=empirical.counts_
    )


def _anisotropic(_variogram):
    variogram = Variogram(model="exponential")
    variogram.parameters = np.array([0.05, 1.0, 3.0, 1.0, 0.3])
    variogram.is_fitted_ = True
    return AnisotropicKriging(variogram)


KRIGING_MODELS = {
    "ordinary": OrdinaryKriging,
    "simple": lambda variogram: SimpleKriging(variogram, mean=0.0),
    "universal": lambda variogram: UniversalKriging(variogram, trend="linear"),
    "anisotropic": _anisotropic,
}


@pytest.mark.parametrize("round_trip", ROUND_TRIPS.values(), ids=ROUND_TRIPS)
def test_fitting_result_round_trips(empirical, round_trip):
    result = fit_variogram_model(
        empirical.distances_,
        empirical.gamma_,
        "exponential",
        np.array([0.0, 1.0, 2.0]),
        weights=empirical.counts_.astype(float),
    )

    restored = round_trip(result)

    assert type(restored) is type(result)
    for field in FITTING_RESULT_FIELDS:
        # repr, so that NaN diagnostics compare equal
        assert repr(getattr(restored, field)) == repr(getattr(result, field)), field


@pytest.mark.parametrize("round_trip", ROUND_TRIPS.values(), ids=ROUND_TRIPS)
def test_fitted_variogram_round_trips(variogram, round_trip):
    restored = round_trip(variogram)

    lags = np.linspace(0.0, 5.0, 11)
    np.testing.assert_array_equal(restored.predict(lags), variogram.predict(lags))
    assert restored.status_ == variogram.status_
    assert repr(restored.fit_result_) == repr(variogram.fit_result_)


@pytest.mark.parametrize("round_trip", ROUND_TRIPS.values(), ids=ROUND_TRIPS)
@pytest.mark.parametrize("build", KRIGING_MODELS.values(), ids=KRIGING_MODELS)
def test_fitted_kriging_models_round_trip(samples, variogram, build, round_trip):
    coords, values = samples
    model = build(variogram).fit(coords, values)
    targets = np.random.default_rng(4).uniform(0, 10, size=(12, 2))

    restored = round_trip(model)

    np.testing.assert_array_equal(restored.predict(targets), model.predict(targets))
