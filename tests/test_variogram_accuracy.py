"""Numerical accuracy tests comparing against reference solutions."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.metrics import r2_score

from pyspatialstats.variogram.empirical import EmpiricalVariogram
from pyspatialstats.variogram.models import Variogram
from pyspatialstats.kriging.ordinary import OrdinaryKriging

from .gstat_reference import ReferenceDataset, build_reference_datasets

REFERENCE_DATASETS = build_reference_datasets()

@pytest.mark.parametrize("model", ["exponential", "spherical", "gaussian"])
def test_variogram_parameter_accuracy(model: str) -> None:
    dataset: ReferenceDataset = REFERENCE_DATASETS[model]

    ev = EmpiricalVariogram(dataset.coords, dataset.values, n_bins=15).compute()
    variogram = Variogram(model=model)
    variogram.fit(ev.distances_, ev.gamma_, weights=ev.counts_)

    expected = np.array([dataset.params.nugget, dataset.params.sill, dataset.params.range])
    fitted = np.array([variogram.nugget_, variogram.sill_, variogram.range_])
    assert_allclose(fitted, expected, rtol=0.01, atol=1e-6)


@pytest.mark.parametrize("model", ["exponential", "spherical", "gaussian"])
def test_ordinary_kriging_matches_reference(model: str) -> None:
    dataset: ReferenceDataset = REFERENCE_DATASETS[model]

    ev = EmpiricalVariogram(dataset.coords, dataset.values, n_bins=15).compute()
    variogram = Variogram(model=model)
    variogram.fit(ev.distances_, ev.gamma_, weights=ev.counts_)

    ok = OrdinaryKriging(variogram)
    ok.fit(dataset.coords, dataset.values)
    predictions = ok.predict(dataset.prediction_points)

    assert predictions.shape == dataset.expected_predictions.shape
    score = r2_score(dataset.expected_predictions, predictions)
    assert score > 0.99
    assert_allclose(predictions, dataset.expected_predictions, rtol=0.02, atol=1e-6)
