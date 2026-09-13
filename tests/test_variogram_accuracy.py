"""Numerical accuracy tests against reference solutions.

The reference datasets are single realisations of 22-28 points on the unit
square. From samples that small no estimator recovers the generating variogram
parameters to a useful tolerance -- the previous version of this module asked
for 1% and could never have passed. These tests check two properties that do
hold exactly:

* fitting is correct: the fitted model explains the empirical variogram at least
  as well as the true parameters do, which a least-squares optimum must;
* kriging is correct: given the true variogram, ordinary kriging reproduces the
  independently computed reference predictions to floating-point precision.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pygeostats.kriging.ordinary import OrdinaryKriging
from pygeostats.variogram.empirical import EmpiricalVariogram
from pygeostats.variogram.models import Variogram

from .gstat_reference import ReferenceDataset, build_reference_datasets

REFERENCE_DATASETS = build_reference_datasets()
MODELS = ["exponential", "spherical", "gaussian"]


def _gamma(h, nugget, sill, range_, model):
    h = np.asarray(h, dtype=float)
    if model == "exponential":
        return nugget + (sill - nugget) * (1.0 - np.exp(-h / range_))
    if model == "gaussian":
        return nugget + (sill - nugget) * (1.0 - np.exp(-((h / range_) ** 2)))
    ratio = h / range_
    return np.where(
        h < range_, nugget + (sill - nugget) * (1.5 * ratio - 0.5 * ratio**3), sill
    )


@pytest.mark.parametrize("model", MODELS)
def test_fitted_variogram_fits_at_least_as_well_as_the_truth(model: str) -> None:
    dataset: ReferenceDataset = REFERENCE_DATASETS[model]
    ev = EmpiricalVariogram(dataset.coords, dataset.values, n_bins=15).compute()
    variogram = Variogram(model=model).fit(ev.distances_, ev.gamma_, weights=ev.counts_)

    # Score over the bins Variogram.fit uses.
    mask = np.isfinite(ev.distances_) & np.isfinite(ev.gamma_) & (ev.counts_ > 0)
    if np.any(ev.counts_ > 1):
        mask &= ev.counts_ > 1
    d, g = ev.distances_[mask], ev.gamma_[mask]
    w = ev.counts_[mask].astype(float)

    def weighted_sse(nugget, sill, range_):
        return float(np.sum(w * (_gamma(d, nugget, sill, range_, model) - g) ** 2))

    fitted = weighted_sse(variogram.nugget_, variogram.sill_, variogram.range_)
    truth = weighted_sse(
        dataset.params.nugget, dataset.params.sill, dataset.params.range
    )
    assert fitted <= truth * (1.0 + 1e-9)


@pytest.mark.parametrize("model", MODELS)
def test_ordinary_kriging_matches_reference(model: str) -> None:
    dataset: ReferenceDataset = REFERENCE_DATASETS[model]

    variogram = Variogram(model=model)
    variogram.nugget_ = dataset.params.nugget
    variogram.sill_ = dataset.params.sill
    variogram.range_ = dataset.params.range
    variogram.is_fitted_ = True

    predictions = (
        OrdinaryKriging(variogram)
        .fit(dataset.coords, dataset.values)
        .predict(dataset.prediction_points)
    )

    assert predictions.shape == dataset.expected_predictions.shape
    assert_allclose(predictions, dataset.expected_predictions, rtol=1e-9, atol=1e-12)
