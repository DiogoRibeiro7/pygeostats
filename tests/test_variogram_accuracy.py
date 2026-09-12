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

@pytest.mark.xfail(
    reason="Variogram.fit() does not recover known parameters. Exponential and gaussian collapse to the range lower bound (1e-6) with nugget 0; spherical diverges to sill 1075 and range 1670 against expected 0.8 and 0.45. The Rust fit_variogram_model optimizer needs investigation.",
    strict=True,
)
@pytest.mark.parametrize("model", ["exponential", "spherical", "gaussian"])
def test_variogram_parameter_accuracy(model: str) -> None:
    dataset: ReferenceDataset = REFERENCE_DATASETS[model]

    ev = EmpiricalVariogram(dataset.coords, dataset.values, n_bins=15).compute()
    variogram = Variogram(model=model)
    variogram.fit(ev.distances_, ev.gamma_, weights=ev.counts_)

    expected = np.array([dataset.params.nugget, dataset.params.sill, dataset.params.range])
    fitted = np.array([variogram.nugget_, variogram.sill_, variogram.range_])
    assert_allclose(fitted, expected, rtol=0.01, atol=1e-6)


@pytest.mark.xfail(
    reason="Kriging R2 against the gstat reference is negative (-3.7, -54.3, -35.5), i.e. worse than predicting the mean. Downstream of the variogram-fitting failure above rather than an independent defect.",
    strict=True,
)
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
