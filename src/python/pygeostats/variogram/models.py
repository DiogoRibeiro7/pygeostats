# src/python/pygeostats/variogram/models.py
"""Theoretical variogram models."""

import warnings
from typing import Dict, List, Optional

import numpy as np
from sklearn.base import BaseEstimator

from .._core import fit_variogram_model
from ..utils.validation import validate_array

# Ratio h95/range, where h95 is the lag at which gamma reaches 95% of the sill.
# Used to turn an empirical h95 estimate into an initial range parameter.
_H95_OVER_RANGE = {
    "exponential": 3.0,
    "gaussian": 3.0**0.5,
    "spherical": 0.811,
}


# Identifiability window for a free range parameter, relative to the observed
# lags. Below _RANGE_FLOOR_OVER_MIN_LAG * min_lag every observed lag is many ranges
# out, so gamma is constant across the data and the range has no effect on the
# fit; this is where the optimiser used to settle and report success. Above
# _RANGE_CAP_OVER_MAX_LAG * max_lag the curve has not begun to bend inside the
# data, so range and sill trade off without limit.
_RANGE_FLOOR_OVER_MIN_LAG = 0.05
_RANGE_CAP_OVER_MAX_LAG = 2.0
_SILL_CAP_OVER_MAX_GAMMA = 10.0


def _gamma(
    model: str, h: np.ndarray, nugget: float, sill: float, range_: float
) -> np.ndarray:
    """Evaluate a theoretical variogram. Matern is treated as nu = 0.5 (exponential).

    gamma(0) is 0 by definition: the nugget is a discontinuity at h -> 0+, not the
    value at zero separation. Without that case covariance(0) came out as
    sill - nugget instead of the sill. The Rust kriging core already treats zero
    distance this way, so this also makes the Python layer agree with it.
    """
    h = np.asarray(h, dtype=float)
    if model in ("exponential", "matern"):
        gamma = nugget + (sill - nugget) * (1.0 - np.exp(-h / range_))
    elif model == "gaussian":
        gamma = nugget + (sill - nugget) * (1.0 - np.exp(-((h / range_) ** 2)))
    elif model == "spherical":
        ratio = h / range_
        gamma = np.where(
            h < range_, nugget + (sill - nugget) * (1.5 * ratio - 0.5 * ratio**3), sill
        )
    else:
        raise ValueError(f"Unknown model: {model}")
    return np.where(h == 0, 0.0, gamma)


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

        fit_result, admissible = self._fit_multistart(
            distances, gamma, weight_array, fix_mask
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
        if not admissible:
            self.converged_ = False
            self.warnings_.append(
                "No admissible fit: every starting point converged to a range or "
                "sill outside what the observed lags can identify. The empirical "
                "variogram does not constrain this model; the lowest-cost "
                "parameters are returned but should not be relied on."
            )
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

    def _fit_multistart(self, distances, gamma, weights, fix_mask):
        """Run the optimiser from several starting ranges and keep the best fit.

        Levenberg-Marquardt is local. From a starting range too far out, a step
        can drive the range down to its floor, where the model is constant over
        every observed lag, the range derivative underflows to exactly zero and
        the optimiser reports success at a solution several times worse than the
        least-squares optimum. Starting from several ranges and discarding
        solutions outside the identifiable window reaches that optimum.

        Returns the chosen fit result and whether it lies inside the window.
        """
        base = self._get_initial_params(distances, gamma)
        w = np.ones_like(distances) if weights is None else np.asarray(weights, float)

        positive = distances[distances > 0]
        max_lag = float(np.max(distances))
        range_fixed = fix_mask is not None and bool(fix_mask[2])
        sill_fixed = fix_mask is not None and bool(fix_mask[1])

        if range_fixed or positive.size == 0 or max_lag <= 0.0:
            starts = [float(base[2])]
        else:
            min_lag = float(positive.min())
            starts = sorted(
                {
                    float(base[2]),
                    2.0 * min_lag,
                    max_lag / 8.0,
                    max_lag / 4.0,
                    max_lag / 2.0,
                    max_lag,
                }
            )

        def admissible(params):
            if not np.all(np.isfinite(params)):
                return False
            if not range_fixed and positive.size and max_lag > 0.0:
                floor = _RANGE_FLOOR_OVER_MIN_LAG * float(positive.min())
                cap = _RANGE_CAP_OVER_MAX_LAG * max_lag
                if not floor <= params[2] <= cap:
                    return False
            if not sill_fixed and params[1] > _SILL_CAP_OVER_MAX_GAMMA * float(
                np.max(gamma)
            ):
                return False
            return True

        candidates = []
        for start in starts:
            initial = base.copy()
            initial[2] = max(start, 1e-6)
            result = fit_variogram_model(
                distances,
                gamma,
                self.model,
                initial,
                weights=weights,
                fix_mask=fix_mask,
            )
            params = np.asarray(result.parameters, dtype=float)
            cost = float(
                np.sum(w * (_gamma(self.model, distances, *params) - gamma) ** 2)
            )
            candidates.append((cost, admissible(params), result))

        inside = [c for c in candidates if c[1]]
        pool = inside if inside else candidates
        best = min(pool, key=lambda c: c[0] if np.isfinite(c[0]) else np.inf)
        return best[2], bool(inside)

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
            # Locate the lag at which gamma first reaches ~95% of the sill, then
            # convert that to the model's range parameter. The two are not the
            # same thing: gamma(h) reaches 95% of its sill at h = 3.00*range for
            # the exponential model and 1.73*range (i.e. sqrt(3)) for the
            # gaussian, while the spherical reaches it at 0.81*range. Treating
            # h95 as the range itself overestimated it threefold for the
            # exponential case, which started the optimiser far enough out that
            # it descended into the degenerate range -> 0 solution.
            target_gamma = sill_init * 0.95
            idx = np.argmin(np.abs(gamma - target_gamma))
            h95 = distances[idx] if idx > 0 else distances[-1] / 3
            range_init = h95 / _H95_OVER_RANGE.get(self.model, 3.0)

        params = np.array([nugget_init, sill_init, range_init], dtype=float)
        params[0] = max(params[0], 0.0)
        params[2] = max(params[2], 1e-6)
        if params[1] <= params[0]:
            params[1] = params[0] + 1e-6
        return params

    def _variogram_function(self, distances: np.ndarray) -> np.ndarray:
        """Calculate variogram values using the fitted model."""
        if self.model == "matern":
            warnings.warn(
                "Matern model using nu=0.5 (equivalent to exponential)", stacklevel=2
            )
        return _gamma(self.model, distances, self.nugget_, self.sill_, self.range_)
