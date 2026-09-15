# src/python/pygeostats/kriging/universal.py
"""Universal kriging implementation with polynomial trends."""

from __future__ import annotations

from collections.abc import Iterable
from math import log
from typing import Dict, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from ..utils.validation import validate_coordinates, validate_values
from ..variogram.models import Variogram
from ._solver import KrigingSystem, variogram_parameters

_VALID_TRENDS = {"auto", "linear", "quadratic"}


class UniversalKriging(BaseEstimator, RegressorMixin):
    """Universal kriging with automatic polynomial trend selection."""

    def __init__(
        self,
        variogram: Variogram,
        trend: str = "auto",
    ) -> None:
        if variogram is None:
            raise ValueError("variogram must be provided")
        if trend.lower() not in _VALID_TRENDS:
            raise ValueError(f"trend must be one of {_VALID_TRENDS}")

        self.variogram = variogram
        self.trend = trend.lower()

        self.coordinates_: Optional[np.ndarray] = None
        self.values_: Optional[np.ndarray] = None
        self.trend_: Optional[str] = None
        self.trend_aic_: Optional[Dict[str, float]] = None
        self.is_fitted_: bool = False
        self._solution = None

    def fit(
        self,
        coordinates: Union[np.ndarray, gpd.GeoDataFrame, pd.DataFrame],
        values: Union[np.ndarray, pd.Series],
    ) -> UniversalKriging:
        """Fit the universal kriging model and factorise its system.

        Raises ``ValueError`` if the variogram is not fitted, or the system is
        singular, as it is when two samples share a location.
        """
        if not self.variogram.is_fitted_:
            raise ValueError("Variogram must be fitted before kriging")

        coords = validate_coordinates(coordinates)
        vals = validate_values(values)

        if len(coords) != len(vals):
            raise ValueError("Coordinates and values must have same length")

        selected_trend, trend_scores = _select_trend(coords, vals, self.trend)

        self.coordinates_ = coords
        self.values_ = vals
        self.trend_ = selected_trend
        self.trend_aic_ = trend_scores
        self._solution = None
        self._current_solution()
        self.is_fitted_ = True
        return self

    def _current_solution(self):
        """The factorised system and dual weights for the current variogram and trend."""
        parameters = variogram_parameters(self.variogram)
        solution = self._solution
        if (
            solution is None
            or solution[2] != self.trend_
            or not solution[0].matches(parameters, self.variogram.model)
        ):
            drift = _design_matrix(self.coordinates_, self.trend_)
            system = KrigingSystem.build(
                self.coordinates_, parameters, self.variogram.model, drift=drift
            )
            rhs = np.concatenate([self.values_, np.zeros(drift.shape[1])])
            solution = (system, system.solve(rhs), self.trend_)
            self._solution = solution
        return solution

    def predict(
        self,
        coordinates: Union[np.ndarray, gpd.GeoDataFrame, pd.DataFrame],
        return_variance: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Predict values at new locations.

        With ``return_variance=True``, also return the universal kriging variance
        at each location, ``sill - w @ c - mu @ f`` for weights ``w``, covariances
        ``c`` to the samples, trend features ``f`` and Lagrange multipliers ``mu``,
        floored at zero. It used to be the ordinary kriging variance, which is
        smaller, because it does not account for estimating the trend.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
        if self.trend_ is None:
            raise RuntimeError("Trend model not available. Did you call fit?")

        pred_coords = validate_coordinates(coordinates)
        system, weights, trend = self._current_solution()
        n_samples = system.n_samples
        design = _design_matrix(pred_coords, trend)

        # The weights hold one entry per sample, then one per trend coefficient.
        predictions = system.covariance_sum(pred_coords, weights[:n_samples])
        predictions += design @ weights[n_samples:]

        if return_variance:
            return predictions, system.variance(pred_coords, design)

        return predictions

    def score(self, coordinates: np.ndarray, values: np.ndarray) -> float:
        """Coefficient of determination of the prediction."""
        from sklearn.metrics import r2_score

        predictions = self.predict(coordinates)
        return r2_score(values, predictions)


def _select_trend(
    coordinates: np.ndarray,
    values: np.ndarray,
    trend: str,
) -> Tuple[str, Dict[str, float]]:
    candidates: Iterable[str]
    if trend == "auto":
        candidates = ("linear", "quadratic")
    else:
        return trend, {trend: _trend_aic(coordinates, values, trend)}

    scores: Dict[str, float] = {}
    best_trend = None
    best_aic = float("inf")

    for candidate in candidates:
        aic_value = _trend_aic(coordinates, values, candidate)
        scores[candidate] = aic_value
        if aic_value < best_aic:
            best_aic = aic_value
            best_trend = candidate

    assert best_trend is not None
    return best_trend, scores


def _trend_aic(coordinates: np.ndarray, values: np.ndarray, trend: str) -> float:
    design = _design_matrix(coordinates, trend)
    beta, residuals, rank, _ = np.linalg.lstsq(design, values, rcond=None)
    if residuals.size > 0:
        rss = float(residuals[0])
    else:
        fitted = design @ beta
        rss = float(np.sum((values - fitted) ** 2))

    n = len(values)
    k = design.shape[1]
    rss = max(rss, 1e-12)
    return n * log(rss / n) + 2 * k


def _design_matrix(coordinates: np.ndarray, trend: str) -> np.ndarray:
    """Trend features for each row of ``coordinates``.

    The columns are a constant, the coordinates and, for a quadratic trend, their
    squares and then their pairwise products.
    """
    coordinates = np.asarray(coordinates, dtype=float)
    if trend not in ("linear", "quadratic"):
        raise ValueError("trend must be 'linear' or 'quadratic'")

    columns = [np.ones(len(coordinates)), *coordinates.T]
    if trend == "quadratic":
        n_dims = coordinates.shape[1]
        columns.extend(coordinates.T**2)
        columns.extend(
            coordinates[:, i] * coordinates[:, j]
            for i in range(n_dims)
            for j in range(i + 1, n_dims)
        )
    return np.column_stack(columns)
