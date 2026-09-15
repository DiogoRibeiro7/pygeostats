# src/python/pygeostats/kriging/ordinary.py
"""Ordinary kriging implementation."""

from pathlib import Path
from typing import Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from ..utils.validation import validate_coordinates, validate_values
from ..variogram.models import Variogram
from .executor import ParallelKrigingExecutor, spatial_tiles
from ._solver import ordinary_system, ordinary_variance, variogram_parameters
from .neighbor_search import ApproximateNeighborIndex


class OrdinaryKriging(BaseEstimator, RegressorMixin):
    """
    Ordinary kriging interpolation.

    Parameters
    ----------
    variogram : Variogram
        Fitted variogram model.
    """

    def __init__(self, variogram: Variogram):
        self.variogram = variogram

        # Fitted attributes
        self.coordinates_ = None
        self.values_ = None
        self.is_fitted_ = False
        self._solution = None

    def fit(
        self,
        coordinates: Union[np.ndarray, gpd.GeoDataFrame, pd.DataFrame],
        values: Union[np.ndarray, pd.Series],
    ) -> "OrdinaryKriging":
        """
        Fit the kriging model.

        Parameters
        ----------
        coordinates : array-like, shape (n_samples, n_features)
            Known sample coordinates.
        values : array-like, shape (n_samples,)
            Known sample values.

        Returns
        -------
        self : OrdinaryKriging
            Returns self for method chaining.

        Raises
        ------
        ValueError
            If the variogram is not fitted, or the kriging system is singular, as
            it is when two samples share a location.
        """
        if not self.variogram.is_fitted_:
            raise ValueError("Variogram must be fitted before kriging")

        self.coordinates_ = validate_coordinates(coordinates)
        self.values_ = validate_values(values)

        if len(self.coordinates_) != len(self.values_):
            raise ValueError("Coordinates and values must have same length")

        self._solution = None
        self._current_solution()
        self.is_fitted_ = True
        return self

    def _current_solution(self):
        """The factorised system and dual weights for the current variogram.

        They are computed at fit, and again only if the variogram's parameters have
        changed since.
        """
        parameters = variogram_parameters(self.variogram)
        solution = self._solution
        if solution is None or not solution[0].matches(
            parameters, self.variogram.model
        ):
            system = ordinary_system(
                self.coordinates_, parameters, self.variogram.model
            )
            solution = (system, system.solve(np.append(self.values_, 0.0)))
            self._solution = solution
        return solution

    def predict(
        self,
        coordinates: Union[np.ndarray, gpd.GeoDataFrame, pd.DataFrame],
        return_variance: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict values at new locations.

        Parameters
        ----------
        coordinates : array-like, shape (n_points, n_features)
            Coordinates to predict at.
        return_variance : bool, default=False
            If True, also return kriging variance.

        Returns
        -------
        predictions : ndarray, shape (n_points,)
            Predicted values.
        variance : ndarray, shape (n_points,), optional
            Kriging variance at each point. Only returned if return_variance=True.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")

        pred_coords = validate_coordinates(coordinates)
        system, weights = self._current_solution()

        # The weights hold one entry per sample, then the Lagrange multiplier.
        predictions = system.covariance_sum(pred_coords, weights[:-1]) + weights[-1]

        if return_variance:
            return predictions, ordinary_variance(system, pred_coords)

        return predictions

    def predict_parallel(
        self,
        coordinates: Union[np.ndarray, gpd.GeoDataFrame, pd.DataFrame],
        *,
        neighbors: int = 64,
        backend: Optional[str] = None,
        search_k: Optional[int] = None,
        grid_shape: Optional[Tuple[int, int]] = None,
        halo: float = 0.0,
        chunk_size: int = 10_000,
        checkpoint_path: Optional[Union[str, Path]] = None,
        checkpoint_interval: int = 5,
        resume: bool = False,
        progress: bool = True,
    ) -> np.ndarray:
        """Predict values using approximate neighbours and spatial tiling."""

        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")

        pred_coords = validate_coordinates(coordinates)

        variogram_params = np.array(
            [self.variogram.nugget_, self.variogram.sill_, self.variogram.range_],
            dtype=float,
        )

        executor = ParallelKrigingExecutor(
            self.coordinates_, self.values_, variogram_params, self.variogram.model
        )
        neighbor_index = ApproximateNeighborIndex(
            self.coordinates_, backend=backend, metric="euclidean"
        )

        tile_plan = None
        if grid_shape is not None:
            tile_plan = spatial_tiles(pred_coords, grid_shape=grid_shape, halo=halo)

        checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None

        return executor.predict(
            pred_coords,
            neighbor_index=neighbor_index,
            neighbors=int(neighbors),
            search_k=search_k,
            tile_plan=tile_plan,
            chunk_size=int(chunk_size),
            checkpoint_path=checkpoint,
            checkpoint_interval=int(max(checkpoint_interval, 1)),
            progress=progress,
            resume=resume,
        )

    def score(self, coordinates: np.ndarray, values: np.ndarray) -> float:
        """
        Return the coefficient of determination R^2 of the prediction.

        Parameters
        ----------
        coordinates : array-like, shape (n_samples, n_features)
            Test coordinates.
        values : array-like, shape (n_samples,)
            True values.

        Returns
        -------
        score : float
            R^2 score.
        """
        from sklearn.metrics import r2_score

        predictions = self.predict(coordinates)
        return r2_score(values, predictions)
