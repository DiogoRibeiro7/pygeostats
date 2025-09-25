# src/python/pyspatialstats/variogram/models.py
"""Theoretical variogram models."""

import numpy as np
from typing import Dict, Optional, Tuple, Union
from sklearn.base import BaseEstimator
import warnings

from .._core import fit_variogram_model
from ..utils.validation import validate_array


class Variogram(BaseEstimator):
    """
    Theoretical variogram model.

    Parameters
    ----------
    model : str, default='exponential'
        Variogram model type. Options: 'exponential', 'spherical', 'gaussian', 'matern'.
    nugget : float, default=0.0
        Nugget effect (variance at distance 0).
    sill : float, optional
        Sill (total variance). If None, estimated from data.
    range : float, optional
        Range parameter. If None, estimated from data.
    """

    VALID_MODELS = {"exponential", "spherical", "gaussian", "matern"}

    def __init__(
        self,
        model: str = "exponential",
        nugget: float = 0.0,
        sill: Optional[float] = None,
        range: Optional[float] = None,
    ):
        if model not in self.VALID_MODELS:
            raise ValueError(f"Model must be one of {self.VALID_MODELS}")

        self.model = model
        self.nugget = nugget
        self.sill = sill
        self.range = range

        # Fitted parameters
        self.nugget_ = None
        self.sill_ = None
        self.range_ = None
        self.is_fitted_ = False

    def fit(
        self,
        distances: np.ndarray,
        gamma: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> "Variogram":
        """
        Fit variogram model to empirical data.

        Parameters
        ----------
        distances : array-like, shape (n_points,)
            Distance values.
        gamma : array-like, shape (n_points,)
            Semivariance values.
        weights : array-like, shape (n_points,), optional
            Weights for fitting. If None, uses equal weights.

        Returns
        -------
        self : Variogram
            Returns self for method chaining.
        """
        distances = validate_array(distances, name="distances")
        gamma = validate_array(gamma, name="gamma")

        if len(distances) != len(gamma):
            raise ValueError("distances and gamma must have same length")

        # Initial parameter estimates
        initial_params = self._get_initial_params(distances, gamma)

        # Call Rust optimizer
        fitted_params = fit_variogram_model(
            distances, gamma, self.model, initial_params
        )

        self.nugget_ = fitted_params[0]
        self.sill_ = fitted_params[1]
        self.range_ = fitted_params[2]

        self.is_fitted_ = True
        return self

    def predict(self, distances: np.ndarray) -> np.ndarray:
        """
        Predict semivariance at given distances.

        Parameters
        ----------
        distances : array-like, shape (n_points,)
            Distance values to predict at.

        Returns
        -------
        gamma : ndarray, shape (n_points,)
            Predicted semivariance values.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")

        distances = validate_array(distances, name="distances")
        return self._variogram_function(distances)

    def covariance(self, distances: np.ndarray) -> np.ndarray:
        """
        Calculate covariance at given distances.

        Parameters
        ----------
        distances : array-like, shape (n_points,)
            Distance values.

        Returns
        -------
        cov : ndarray, shape (n_points,)
            Covariance values.
        """
        gamma = self.predict(distances)
        return self.sill_ - gamma

    def _get_initial_params(
        self, distances: np.ndarray, gamma: np.ndarray
    ) -> np.ndarray:
        """Get initial parameter estimates."""
        # Simple heuristics for initial values
        nugget_init = self.nugget if self.nugget is not None else np.min(gamma) * 0.1
        sill_init = self.sill if self.sill is not None else np.max(gamma) * 1.1

        if self.range is not None:
            range_init = self.range
        else:
            # Estimate range as distance where gamma reaches ~95% of sill
            target_gamma = sill_init * 0.95
            idx = np.argmin(np.abs(gamma - target_gamma))
            range_init = distances[idx] if idx > 0 else distances[-1] / 3

        return np.array([nugget_init, sill_init, range_init])

    def _variogram_function(self, distances: np.ndarray) -> np.ndarray:
        """Calculate variogram values using the fitted model."""
        h = distances
        nugget, sill, range_param = self.nugget_, self.sill_, self.range_

        if self.model == "exponential":
            return nugget + (sill - nugget) * (1 - np.exp(-h / range_param))

        elif self.model == "spherical":
            gamma = np.full_like(h, sill)
            mask = h < range_param
            h_scaled = h[mask] / range_param
            gamma[mask] = nugget + (sill - nugget) * (
                1.5 * h_scaled - 0.5 * h_scaled**3
            )
            return gamma

        elif self.model == "gaussian":
            return nugget + (sill - nugget) * (1 - np.exp(-((h / range_param) ** 2)))

        elif self.model == "matern":
            # Simplified Matérn with ν=0.5 (exponential)
            warnings.warn("Matérn model using ν=0.5 (equivalent to exponential)")
            return nugget + (sill - nugget) * (1 - np.exp(-h / range_param))

        else:
            raise ValueError(f"Unknown model: {self.model}")
