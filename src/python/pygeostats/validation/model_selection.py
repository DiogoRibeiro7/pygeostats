"""Model selection helpers mirroring geoR-style workflows."""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np

from pygeostats.variogram.empirical import EmpiricalVariogram
from pygeostats.variogram.models import Variogram
from pygeostats.validation.cross_validation import (
    CrossValidationResult,
    default_kriging_builder,
    default_variogram_builder,
    leave_one_out_cross_validation,
)


def _theoretical_variogram(model: str, params: Tuple[float, float, float], distances: np.ndarray) -> np.ndarray:
    nugget, sill, range_ = params
    if model == "exponential":
        return nugget + (sill - nugget) * (1.0 - np.exp(-distances / range_))
    if model == "spherical":
        ratio = np.clip(distances / range_, 0.0, 1.0)
        values = np.where(
            distances < range_,
            nugget + (sill - nugget) * (1.5 * ratio - 0.5 * ratio**3),
            sill,
        )
        return values
    if model == "gaussian":
        ratio = distances / range_
        return nugget + (sill - nugget) * (1.0 - np.exp(-(ratio**2)))
    raise ValueError(f"Unsupported model '{model}'")


def _residual_sum_squares(
    variogram: Variogram,
    distances: np.ndarray,
    gamma: np.ndarray,
    weights: np.ndarray | None,
) -> float:
    params = (variogram.nugget_, variogram.sill_, variogram.range_)
    predicted = _theoretical_variogram(variogram.model, params, distances)
    residuals = gamma - predicted
    if weights is None:
        return float(np.sum(residuals**2))
    return float(np.sum(weights * residuals**2))


def variogram_aic(
    variogram: Variogram,
    distances: np.ndarray,
    gamma: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    n = len(gamma)
    rss = max(_residual_sum_squares(variogram, distances, gamma, weights), 1e-12)
    k = 3  # nugget, sill, range
    return n * np.log(rss / n) + 2 * k


def variogram_bic(
    variogram: Variogram,
    distances: np.ndarray,
    gamma: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    n = len(gamma)
    rss = max(_residual_sum_squares(variogram, distances, gamma, weights), 1e-12)
    k = 3
    return n * np.log(rss / n) + k * np.log(n)


def select_best_variogram_model(
    coords: np.ndarray,
    values: np.ndarray,
    candidate_models: Iterable[str] = ("exponential", "spherical", "gaussian"),
    criterion: str = "aic",
) -> Tuple[str, Variogram, dict]:
    """Fit candidate models and pick the best one.

    Parameters
    ----------
    coords : ndarray
        Coordinate array of shape (n_samples, 2).
    values : ndarray
        Sample values.
    candidate_models : Iterable[str]
        Variogram model names to compare.
    criterion : {'aic', 'bic', 'loo'}
        Selection criterion following geoR conventions.
    """

    empirical = EmpiricalVariogram(coords, values, n_bins=20).compute()
    mask = empirical.counts_ > 0
    distances = empirical.distances_[mask]
    gamma = empirical.gamma_[mask]
    weights = empirical.counts_[mask]

    results: List[Tuple[str, Variogram, float]] = []

    for model in candidate_models:
        variogram = Variogram(model=model)
        variogram.fit(distances, gamma, weights=weights)
        if criterion.lower() == "aic":
            score = variogram_aic(variogram, distances, gamma, weights)
        elif criterion.lower() == "bic":
            score = variogram_bic(variogram, distances, gamma, weights)
        elif criterion.lower() == "loo":
            build_var = default_variogram_builder(model=model)
            build_krig = default_kriging_builder()
            cv_result: CrossValidationResult = leave_one_out_cross_validation(
                coords, values, build_var, build_krig
            )
            score = cv_result.rmse
        else:
            raise ValueError("criterion must be one of {'aic', 'bic', 'loo'}")
        results.append((model, variogram, score))

    best = min(results, key=lambda item: item[2])
    diagnostics = {model: score for model, _, score in results}
    return best[0], best[1], diagnostics
