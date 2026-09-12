"""Diagnostic utilities inspired by geoR validation workflows."""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy import stats

from pygeostats.kriging.ordinary import OrdinaryKriging
from pygeostats.variogram.empirical import EmpiricalVariogram


def compute_kriging_residuals(
    predictor: OrdinaryKriging,
    coords: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Return residuals for fitted kriging model."""

    predictions = predictor.predict(coords)
    return values - predictions


def variogram_cloud(coords: np.ndarray, values: np.ndarray) -> Dict[str, np.ndarray]:
    """Return variogram cloud values for diagnostic plotting."""

    distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    semivariances = 0.5 * (values[:, None] - values[None, :]) ** 2
    triu_idx = np.triu_indices(len(coords), k=1)

    return {
        "distance": distances[triu_idx],
        "gamma": semivariances[triu_idx],
    }


def standardized_residuals(residuals: np.ndarray) -> np.ndarray:
    """Scale residuals by their standard deviation."""

    std = np.std(residuals, ddof=1)
    if std == 0.0:
        return np.zeros_like(residuals)
    return residuals / std


def normality_test(residuals: np.ndarray) -> Dict[str, float]:
    """Run a Shapiro-Wilk normality test on residuals."""

    if len(residuals) < 3:
        return {"statistic": np.nan, "pvalue": np.nan}
    stat, pvalue = stats.shapiro(residuals)
    return {"statistic": float(stat), "pvalue": float(pvalue)}
