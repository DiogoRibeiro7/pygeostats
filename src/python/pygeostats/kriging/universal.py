# src/python/pygeostats/kriging/universal.py
"""Universal kriging implementation with polynomial trends."""

from __future__ import annotations

from math import log
from typing import Dict, Iterable, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from .._core import kriging_variance, universal_kriging_predict
from ..utils.validation import validate_coordinates, validate_values
from ..variogram.models import Variogram

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

    def fit(
        self,
        coordinates: Union[np.ndarray, gpd.GeoDataFrame, pd.DataFrame],
        values: Union[np.ndarray, pd.Series],
    ) -> "UniversalKriging":
        """Fit the universal kriging model."""
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
        self.is_fitted_ = True
        return self

    def predict(
        self,
        coordinates: Union[np.ndarray, gpd.GeoDataFrame, pd.DataFrame],
        return_variance: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Predict values at new locations."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
        if self.trend_ is None:
            raise RuntimeError("Trend model not available. Did you call fit?")

        pred_coords = validate_coordinates(coordinates)

        variogram_params = np.array(
            [self.variogram.nugget_, self.variogram.sill_, self.variogram.range_],
            dtype=float,
        )

        predictions = universal_kriging_predict(
            self.coordinates_,
            self.values_,
            pred_coords,
            variogram_params,
            self.variogram.model,
            self.trend_,
        )

        if return_variance:
            variance = kriging_variance(
                self.coordinates_,
                pred_coords,
                variogram_params,
                self.variogram.model,
            )
            return predictions, variance

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
    features = [_trend_features(point, trend) for point in coordinates]
    return np.asarray(features, dtype=float)


def _trend_features(point: np.ndarray, trend: str) -> np.ndarray:
    point = np.asarray(point, dtype=float)
    features = [1.0]

    if trend == "linear":
        features.extend(point.tolist())
    elif trend == "quadratic":
        features.extend(point.tolist())
        features.extend((point * point).tolist())
        for i in range(len(point)):
            for j in range(i + 1, len(point)):
                features.append(point[i] * point[j])
    else:
        raise ValueError("trend must be 'linear' or 'quadratic'")

    return np.asarray(features, dtype=float)
