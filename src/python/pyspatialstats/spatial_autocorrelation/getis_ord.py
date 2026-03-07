"""Getis-Ord spatial autocorrelation statistics."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.stats import norm

from pyspatialstats.utils.validation import validate_array


def local_getis_ord_g(
    values: np.ndarray,
    weights: np.ndarray,
    include_self: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Compute local Getis-Ord G_i* z-scores.

    Parameters
    ----------
    values : ndarray, shape (n,)
        Observed attribute values.
    weights : ndarray, shape (n, n)
        Spatial weights matrix.
    include_self : bool
        Whether to include self-weight in each local neighborhood.
    """

    x = validate_array(values, name="values").ravel()
    w = validate_array(weights, name="weights")
    if w.ndim != 2 or w.shape[0] != w.shape[1]:
        raise ValueError("weights must be a square 2D array")
    if len(x) != w.shape[0]:
        raise ValueError("values length must match weights dimensions")
    if len(x) < 3:
        raise ValueError("At least 3 observations are required")

    w_eff = w.copy()
    if not include_self:
        np.fill_diagonal(w_eff, 0.0)

    n = len(x)
    x_bar = np.mean(x)
    s = np.sqrt(np.mean(x**2) - x_bar**2)
    if s == 0.0:
        return {"G_local": np.zeros_like(x), "z_score": np.zeros_like(x)}

    w_sum = np.sum(w_eff, axis=1)
    w_sq_sum = np.sum(w_eff**2, axis=1)

    numerator = w_eff @ x - x_bar * w_sum
    denominator = s * np.sqrt((n * w_sq_sum - w_sum**2) / (n - 1))
    z = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator > 0.0,
    )
    return {"G_local": z.astype(np.float64), "z_score": z.astype(np.float64)}


def global_getis_ord_g(
    values: np.ndarray,
    weights: np.ndarray,
    permutations: int = 0,
    random_state: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute global Getis-Ord General G statistic.

    This implementation returns the observed G and an optional permutation
    p-value around the permutation null distribution.
    """

    x = validate_array(values, name="values").ravel()
    w = validate_array(weights, name="weights")
    if w.ndim != 2 or w.shape[0] != w.shape[1]:
        raise ValueError("weights must be a square 2D array")
    if len(x) != w.shape[0]:
        raise ValueError("values length must match weights dimensions")
    if permutations < 0:
        raise ValueError("permutations must be >= 0")

    n = len(x)
    if n < 3:
        raise ValueError("At least 3 observations are required")

    w_eff = w.copy()
    np.fill_diagonal(w_eff, 0.0)

    def _general_g(vals: np.ndarray) -> float:
        xx = vals[:, None] * vals[None, :]
        numerator = float(np.sum(w_eff * xx))
        denominator = float(np.sum(xx) - np.sum(vals**2))
        if denominator == 0.0:
            return 0.0
        return numerator / denominator

    g_obs = _general_g(x)
    result = {"G": float(g_obs)}

    if permutations > 0:
        rng = np.random.default_rng(random_state)
        permuted = np.array([_general_g(rng.permutation(x)) for _ in range(permutations)])
        p_two_sided = (
            np.sum(np.abs(permuted - np.mean(permuted)) >= abs(g_obs - np.mean(permuted)))
            + 1.0
        ) / (permutations + 1.0)
        result["p_value"] = float(p_two_sided)
        result["expected_G"] = float(np.mean(permuted))
    else:
        rng = np.random.default_rng(0)
        approx = np.array([_general_g(rng.permutation(x)) for _ in range(200)])
        mu = float(np.mean(approx))
        sigma = float(np.std(approx, ddof=1))
        if sigma > 0.0:
            z = (g_obs - mu) / sigma
            p = 2.0 * (1.0 - norm.cdf(abs(z)))
        else:
            z = 0.0
            p = 1.0
        result["expected_G"] = mu
        result["z_score"] = float(z)
        result["p_value"] = float(p)

    return result
