# src/python/pyspatialstats/variogram/models.py
"""Theoretical variogram models."""

import numpy as np
from typing import Dict, List, Optional
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
        self.converged_ = False
        self.fit_statistics_: Optional[Dict[str, object]] = None
        self.fit_result_ = None
        self.status_: Optional[str] = None
        self.fallback_used_: bool = False
        self.warnings_: Optional[List[str]] = None
        self.parameter_std_: Optional[np.ndarray] = None
        self.trace_: Optional[np.ndarray] = None
        self.diagnostics_: Optional[Dict[str, float]] = None

    def fit(
        self,
        distances: np.ndarray,
        gamma: np.ndarray,
        weights: Optional[np.ndarray] = None,
        fix: Optional[Dict[str, bool]] = None,
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
            Weights for fitting. Typically the number of point pairs in each bin.
        fix : dict, optional
            Dictionary indicating parameters to keep fixed during optimisation.
            Accepted keys: "nugget", "sill", "range". Example: {"nugget": True}.

        Returns
        -------
        self : Variogram
            Returns self for method chaining.
        """
        distances = validate_array(distances, name="distances")
        gamma = validate_array(gamma, name="gamma")

        if len(distances) != len(gamma):
            raise ValueError("distances and gamma must have same length")

        weight_array: Optional[np.ndarray] = None
        if weights is not None:
            weight_array = validate_array(weights, name="weights")
            if len(weight_array) != len(distances):
                raise ValueError("weights must have same length as distances")
        else:
            weight_array = None

        valid_mask = np.isfinite(distances) & np.isfinite(gamma)
        if weight_array is not None:
            valid_mask &= weight_array > 0
            if np.any(weight_array > 1):
                dense_mask = weight_array > 1
                valid_mask &= dense_mask
                if not np.any(valid_mask):
                    valid_mask = weight_array > 0
        if not np.any(valid_mask):
            raise ValueError("No valid empirical variogram bins available for fitting")

        distances = distances[valid_mask]
        gamma = gamma[valid_mask]
        if weight_array is not None:
            weight_array = weight_array[valid_mask]

        fix_mask = None
        if fix:
            valid_keys = {"nugget", "sill", "range"}
            unexpected = set(fix) - valid_keys
            if unexpected:
                raise ValueError(f"Unknown fixed parameter keys: {unexpected}")
            fix_mask = np.array(
                [
                    bool(fix.get("nugget", False)),
                    bool(fix.get("sill", False)),
                    bool(fix.get("range", False)),
                ],
                dtype=bool,
            )

        # Initial parameter estimates
        initial_params = self._get_initial_params(distances, gamma)

        # Call Rust optimizer
        fit_result = fit_variogram_model(
            distances,
            gamma,
            self.model,
            initial_params,
            weights=weight_array,
            fix_mask=fix_mask,
        )

        params = np.asarray(fit_result.parameters, dtype=float)
        if params.shape[0] != 3:
            raise ValueError("Fitting result must contain three parameters")

        self.nugget_ = params[0]
        self.sill_ = params[1]
        self.range_ = params[2]
        self.is_fitted_ = True
        self.converged_ = bool(fit_result.converged)
        self.fit_result_ = fit_result

        self.status_ = str(fit_result.status)
        self.fallback_used_ = bool(fit_result.fallback_used)
        self.warnings_ = [str(msg) for msg in fit_result.warnings]
        self.parameter_std_ = np.asarray(fit_result.parameter_std, dtype=float)
        self.trace_ = (
            np.asarray(fit_result.trace, dtype=float)
            if fit_result.trace
            else np.empty((0, 5), dtype=float)
        )
        diagnostics = {str(key): float(value) for key, value in fit_result.diagnostics}
        self.diagnostics_ = diagnostics

        self.fit_statistics_ = {
            "r2": float(fit_result.r_squared),
            "rmse": float(fit_result.rmse),
            "converged": self.converged_,
            "iterations": int(fit_result.iterations),
            "message": str(fit_result.message),
            "status": self.status_,
            "fallback_used": self.fallback_used_,
            "warnings": list(self.warnings_),
            "parameter_std": self.parameter_std_.copy(),
            "trace": self.trace_,
            "diagnostics": diagnostics,
        }

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

        params = np.array([nugget_init, sill_init, range_init], dtype=float)
        params[0] = max(params[0], 0.0)
        params[2] = max(params[2], 1e-6)
        if params[1] <= params[0]:
            params[1] = params[0] + 1e-6
        return params

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
            # Simplified Matern with nu=0.5 (exponential)
            warnings.warn("Matern model using nu=0.5 (equivalent to exponential)")
            return nugget + (sill - nugget) * (1 - np.exp(-h / range_param))

        else:
            raise ValueError(f"Unknown model: {self.model}")
