"""Cross-validation helpers for kriging and variogram models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score

from pygeostats.kriging.ordinary import OrdinaryKriging
from pygeostats.variogram.empirical import EmpiricalVariogram
from pygeostats.variogram.models import Variogram

Coords = np.ndarray
Values = np.ndarray


@dataclass
class CrossValidationResult:
    """Container for cross-validation diagnostics."""

    predictions: np.ndarray
    residuals: np.ndarray
    rmse: float
    r2: float

    def summary(self) -> dict:
        return {"rmse": self.rmse, "r2": self.r2}


VariogramBuilder = Callable[[Coords, Values], Variogram]
KrigingBuilder = Callable[[Variogram], OrdinaryKriging]


def default_variogram_builder(model: str = "exponential") -> VariogramBuilder:
    """Return a builder that fits a :class:Variogram with the requested model."""

    def builder(coords: Coords, values: Values) -> Variogram:
        variogram = Variogram(model=model)
        n_bins = max(6, min(20, max(len(coords) // 2, 2)))
        n_bins = min(n_bins, max(len(coords) - 1, 1))
        empirical = EmpiricalVariogram(coords, values, n_bins=n_bins).compute()
        mask = empirical.counts_ > 0
        variogram.fit(
            empirical.distances_[mask],
            empirical.gamma_[mask],
            weights=empirical.counts_[mask],
        )
        return variogram

    return builder


def default_kriging_builder(kriging_class: type = OrdinaryKriging) -> KrigingBuilder:
    """Return a builder that instantiates a kriging predictor."""

    def builder(variogram: Variogram) -> OrdinaryKriging:
        return kriging_class(variogram)

    return builder


def leave_one_out_cross_validation(
    coords: Coords,
    values: Values,
    build_variogram: VariogramBuilder,
    build_predictor: KrigingBuilder,
) -> CrossValidationResult:
    """Leave-one-out cross-validation emulating geoR-style validation."""

    n = coords.shape[0]
    if n < 3:
        raise ValueError("leave_one_out_cross_validation requires at least 3 observations")

    predictions = np.zeros(n)

    for idx in range(n):
        mask = np.ones(n, dtype=bool)
        mask[idx] = False
        variogram = build_variogram(coords[mask], values[mask])
        predictor = build_predictor(variogram)
        predictor.fit(coords[mask], values[mask])
        predictions[idx] = predictor.predict(coords[[idx]])[0]

    residuals = values - predictions
    rmse = float(np.sqrt(mean_squared_error(values, predictions)))
    r2 = float(r2_score(values, predictions))
    return CrossValidationResult(predictions, residuals, rmse, r2)


def _cluster_labels(coords: Coords, n_clusters: int, random_state: Optional[int]) -> np.ndarray:
    if n_clusters <= 1 or len(coords) <= n_clusters:
        return np.arange(len(coords)) % max(n_clusters, 1)
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    return kmeans.fit_predict(coords)


def spatial_kfold_cross_validation(
    coords: Coords,
    values: Values,
    build_variogram: VariogramBuilder,
    build_predictor: KrigingBuilder,
    n_splits: int = 5,
    random_state: Optional[int] = None,
) -> CrossValidationResult:
    """Cluster-aware spatial k-fold cross-validation."""

    if coords.shape[0] < max(n_splits, 3):
        raise ValueError("spatial_kfold_cross_validation requires more observations than folds")

    labels = _cluster_labels(coords, n_splits, random_state)
    predictions = np.zeros_like(values)

    for fold in range(n_splits):
        test_mask = labels == fold
        train_mask = ~test_mask
        if not np.any(test_mask) or np.sum(train_mask) < 3:
            continue
        variogram = build_variogram(coords[train_mask], values[train_mask])
        predictor = build_predictor(variogram)
        predictor.fit(coords[train_mask], values[train_mask])
        predictions[test_mask] = predictor.predict(coords[test_mask])

    residuals = values - predictions
    rmse = float(np.sqrt(mean_squared_error(values, predictions)))
    r2 = float(r2_score(values, predictions))
    return CrossValidationResult(predictions, residuals, rmse, r2)


def block_cross_validation(
    coords: Coords,
    values: Values,
    build_variogram: VariogramBuilder,
    build_predictor: KrigingBuilder,
    grid_shape: Tuple[int, int] = (2, 2),
) -> CrossValidationResult:
    """Block cross-validation using a regular grid of spatial blocks."""

    if coords.shape[0] < 3:
        raise ValueError("block_cross_validation requires at least 3 observations")

    predictions = np.zeros_like(values)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    edges_x = np.linspace(mins[0], maxs[0], grid_shape[0] + 1)
    edges_y = np.linspace(mins[1], maxs[1], grid_shape[1] + 1)

    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            mask = (
                (coords[:, 0] >= edges_x[i])
                & (coords[:, 0] <= edges_x[i + 1])
                & (coords[:, 1] >= edges_y[j])
                & (coords[:, 1] <= edges_y[j + 1])
            )
            if not np.any(mask) or np.sum(~mask) < 3:
                continue
            variogram = build_variogram(coords[~mask], values[~mask])
            predictor = build_predictor(variogram)
            predictor.fit(coords[~mask], values[~mask])
            predictions[mask] = predictor.predict(coords[mask])

    residuals = values - predictions
    rmse = float(np.sqrt(mean_squared_error(values, predictions)))
    r2 = float(r2_score(values, predictions))
    return CrossValidationResult(predictions, residuals, rmse, r2)
