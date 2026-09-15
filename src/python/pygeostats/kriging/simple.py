# src/python/pygeostats/kriging/simple.py
"""Simple kriging implementation."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from ..utils.validation import validate_coordinates, validate_values
from ..variogram.models import Variogram
from ._solver import KrigingSystem, variogram_parameters


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
        self._solution = None

    def fit(
        self,
        coordinates: Union[np.ndarray, gpd.GeoDataFrame, pd.DataFrame],
        values: Union[np.ndarray, pd.Series],
    ) -> SimpleKriging:
        """Store known samples and factorise the kriging system.

        Raises ``ValueError`` if the variogram is not fitted, or the system is
        singular, as it is when two samples share a location.
        """
        if not self.variogram.is_fitted_:
            raise ValueError("Variogram must be fitted before kriging")

        coords = validate_coordinates(coordinates)
        vals = validate_values(values)

        if len(coords) != len(vals):
            raise ValueError("Coordinates and values must have same length")

        self.coordinates_ = coords
        self.values_ = vals
        self._solution = None
        self._current_solution()
        self.is_fitted_ = True
        return self

    def _current_solution(self):
        """The factorised system and dual weights for the current variogram and mean."""
        parameters = variogram_parameters(self.variogram)
        solution = self._solution
        if (
            solution is None
            or solution[2] != self.mean
            or not solution[0].matches(parameters, self.variogram.model)
        ):
            system = KrigingSystem.build(
                self.coordinates_, parameters, self.variogram.model
            )
            weights = system.solve(self.values_ - self.mean)
            solution = (system, weights, self.mean)
            self._solution = solution
        return solution

    def predict(
        self,
        coordinates: Union[np.ndarray, gpd.GeoDataFrame, pd.DataFrame],
        return_variance: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Predict values at new locations.

        With ``return_variance=True``, also return the simple kriging variance at
        each location, ``sill - w @ c`` for weights ``w`` and covariances ``c`` to
        the samples, floored at zero. It used to be the ordinary kriging variance,
        which is larger, because ordinary kriging estimates the mean.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")

        pred_coords = validate_coordinates(coordinates)
        system, weights, mean = self._current_solution()

        predictions = mean + system.covariance_sum(pred_coords, weights)

        if return_variance:
            return predictions, system.variance(pred_coords)

        return predictions

    def score(self, coordinates: np.ndarray, values: np.ndarray) -> float:
        """Coefficient of determination of the prediction."""
        from sklearn.metrics import r2_score

        predictions = self.predict(coordinates)
        return r2_score(values, predictions)
