# src/python/pygeostats/kriging/ordinary.py
"""Ordinary kriging implementation."""

from pathlib import Path
from typing import Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from .._core import kriging_variance, ordinary_kriging_predict
from ..utils.validation import validate_coordinates, validate_values
from ..variogram.models import Variogram
from .executor import ParallelKrigingExecutor, spatial_tiles
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
        """
        if not self.variogram.is_fitted_:
            raise ValueError("Variogram must be fitted before kriging")

        self.coordinates_ = validate_coordinates(coordinates)
        self.values_ = validate_values(values)

        if len(self.coordinates_) != len(self.values_):
            raise ValueError("Coordinates and values must have same length")

        self.is_fitted_ = True
        return self

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

        # Get variogram parameters
        variogram_params = np.array(
            [self.variogram.nugget_, self.variogram.sill_, self.variogram.range_]
        )

        # Call Rust implementation
        predictions = ordinary_kriging_predict(
            self.coordinates_,
            self.values_,
            pred_coords,
            variogram_params,
            self.variogram.model,
        )

        if return_variance:
            variance = kriging_variance(
                self.coordinates_, pred_coords, variogram_params, self.variogram.model
            )
            return predictions, variance

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
