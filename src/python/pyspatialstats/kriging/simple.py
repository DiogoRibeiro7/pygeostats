# src/python/pyspatialstats/kriging/simple.py
"""Simple kriging implementation."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from .._core import kriging_variance, simple_kriging_predict
from ..utils.validation import validate_coordinates, validate_values
from ..variogram.models import Variogram


class SimpleKriging(BaseEstimator, RegressorMixin):
    """Simple kriging interpolation with known mean."""

    def __init__(
        self,
        variogram: Variogram,
        mean: float,
    ) -> None:
        if variogram is None:
            raise ValueError("variogram must be provided")

        self.variogram = variogram
        self.mean = float(mean)

        self.coordinates_: Optional[np.ndarray] = None
        self.values_: Optional[np.ndarray] = None
        self.is_fitted_: bool = False

    def fit(
        self,
        coordinates: Union[np.ndarray, gpd.GeoDataFrame, pd.DataFrame],
        values: Union[np.ndarray, pd.Series],
    ) -> "SimpleKriging":
        """Store known samples for kriging."""
        if not self.variogram.is_fitted_:
            raise ValueError("Variogram must be fitted before kriging")

        coords = validate_coordinates(coordinates)
        vals = validate_values(values)

        if len(coords) != len(vals):
            raise ValueError("Coordinates and values must have same length")

        self.coordinates_ = coords
        self.values_ = vals
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

        pred_coords = validate_coordinates(coordinates)

        variogram_params = np.array(
            [self.variogram.nugget_, self.variogram.sill_, self.variogram.range_],
            dtype=float,
        )

        predictions = simple_kriging_predict(
            self.coordinates_,
            self.values_,
            pred_coords,
            variogram_params,
            self.variogram.model,
            float(self.mean),
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
