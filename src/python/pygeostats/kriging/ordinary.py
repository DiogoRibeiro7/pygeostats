# src/python/pygeostats/kriging/ordinary.py
"""Ordinary kriging implementation."""

import warnings
from pathlib import Path
from typing import Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from ..utils.validation import validate_coordinates, validate_values
from ..variogram.models import Variogram
from ._solver import ordinary_system, ordinary_variance, variogram_parameters


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
        """Deprecated: use :meth:`predict`, which computes targets in parallel.

        This method never worked. It called ``ParallelKrigingExecutor``,
        ``ApproximateNeighborIndex`` and ``spatial_tiles`` with arguments they do not
        accept, so it raised for any input. It now returns ``predict(coordinates)``
        and emits a ``FutureWarning``, and it will be removed in a future release.

        Neighbour search, tiling, checkpointing and resuming were never implemented.
        Their options are accepted so that existing calls do not fail, but they are
        ignored, and the warning names any that were given.
        """
        defaults = {
            "neighbors": 64,
            "backend": None,
            "search_k": None,
            "grid_shape": None,
            "halo": 0.0,
            "chunk_size": 10_000,
            "checkpoint_path": None,
            "checkpoint_interval": 5,
            "resume": False,
            "progress": True,
        }
        given = {
            "neighbors": neighbors,
            "backend": backend,
            "search_k": search_k,
            "grid_shape": grid_shape,
            "halo": halo,
            "chunk_size": chunk_size,
            "checkpoint_path": checkpoint_path,
            "checkpoint_interval": checkpoint_interval,
            "resume": resume,
            "progress": progress,
        }
        ignored = sorted(
            name for name, value in given.items() if value != defaults[name]
        )

        message = (
            "OrdinaryKriging.predict_parallel is deprecated and will be removed; use "
            "predict, which already computes targets in parallel. It now returns "
            "predict(coordinates)."
        )
        if ignored:
            message += " Ignored: " + ", ".join(ignored) + "."
        warnings.warn(message, FutureWarning, stacklevel=2)

        return self.predict(coordinates)

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
