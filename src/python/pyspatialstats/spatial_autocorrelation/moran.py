"""Moran's I spatial autocorrelation statistics."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.stats import norm

from pyspatialstats.utils.validation import validate_array


def _morans_i_stat(values: np.ndarray, weights: np.ndarray) -> float:
    n = len(values)
    z = values - np.mean(values)
    s0 = np.sum(weights)
    if s0 == 0.0:
        return 0.0
    denom = np.sum(z**2)
    if denom == 0.0:
        return 0.0
    num = np.sum(weights * (z[:, None] * z[None, :]))
    return float((n / s0) * (num / denom))


def morans_i(
    values: np.ndarray,
    weights: np.ndarray,
    permutations: int = 0,
    random_state: Optional[int] = None,
) -> Dict[str, float]:
    """Compute global Moran's I with optional permutation p-value."""

    vals = validate_array(values, name="values").ravel()
    w = validate_array(weights, name="weights")
    if w.ndim != 2 or w.shape[0] != w.shape[1]:
        raise ValueError("weights must be a square 2D array")
    if len(vals) != w.shape[0]:
        raise ValueError("values length must match weights dimensions")
    if permutations < 0:
        raise ValueError("permutations must be >= 0")

    i_obs = _morans_i_stat(vals, w)
    expected_i = -1.0 / (len(vals) - 1) if len(vals) > 1 else 0.0

    result = {"I": i_obs, "expected_I": float(expected_i)}

    if permutations > 0:
        rng = np.random.default_rng(random_state)
        permuted = np.array([_morans_i_stat(rng.permutation(vals), w) for _ in range(permutations)])
        p_two_sided = (np.sum(np.abs(permuted) >= abs(i_obs)) + 1.0) / (permutations + 1.0)
        result["p_value"] = float(p_two_sided)
    else:
        # Simple normal approximation from permutation moments.
        rng = np.random.default_rng(0)
        approx = np.array([_morans_i_stat(rng.permutation(vals), w) for _ in range(200)])
        std = float(np.std(approx, ddof=1))
        if std > 0.0:
            z_score = (i_obs - expected_i) / std
            p_two_sided = 2.0 * (1.0 - norm.cdf(abs(z_score)))
        else:
            z_score = 0.0
            p_two_sided = 1.0
        result["z_score"] = float(z_score)
        result["p_value"] = float(p_two_sided)

    return result


def local_morans_i(
    values: np.ndarray,
    weights: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Compute local Moran's I for each observation."""

    vals = validate_array(values, name="values").ravel()
    w = validate_array(weights, name="weights")
    if w.ndim != 2 or w.shape[0] != w.shape[1]:
        raise ValueError("weights must be a square 2D array")
    if len(vals) != w.shape[0]:
        raise ValueError("values length must match weights dimensions")

    z = vals - np.mean(vals)
    m2 = np.sum(z**2) / len(vals)
    if m2 == 0.0:
        return {"I_local": np.zeros_like(vals), "z": z}

    lag = w @ z
    i_local = (z / m2) * lag
    return {"I_local": i_local.astype(np.float64), "z": z.astype(np.float64)}


def _gearys_c_stat(values: np.ndarray, weights: np.ndarray) -> float:
    n = len(values)
    z = values - np.mean(values)
    s0 = np.sum(weights)
    if s0 == 0.0:
        return 1.0
    denom = np.sum(z**2)
    if denom == 0.0:
        return 0.0
    diffs = values[:, None] - values[None, :]
    num = np.sum(weights * (diffs**2))
    return float(((n - 1.0) / (2.0 * s0)) * (num / denom))


def gearys_c(
    values: np.ndarray,
    weights: np.ndarray,
    permutations: int = 0,
    random_state: Optional[int] = None,
) -> Dict[str, float]:
    """Compute global Geary's C with optional permutation p-value."""

    vals = validate_array(values, name="values").ravel()
    w = validate_array(weights, name="weights")
    if w.ndim != 2 or w.shape[0] != w.shape[1]:
        raise ValueError("weights must be a square 2D array")
    if len(vals) != w.shape[0]:
        raise ValueError("values length must match weights dimensions")
    if permutations < 0:
        raise ValueError("permutations must be >= 0")

    c_obs = _gearys_c_stat(vals, w)
    result = {"C": c_obs, "expected_C": 1.0}

    if permutations > 0:
        rng = np.random.default_rng(random_state)
        permuted = np.array(
            [_gearys_c_stat(rng.permutation(vals), w) for _ in range(permutations)]
        )
        # Use distance from the null expectation C=1.
        p_two_sided = (
            np.sum(np.abs(permuted - 1.0) >= abs(c_obs - 1.0)) + 1.0
        ) / (permutations + 1.0)
        result["p_value"] = float(p_two_sided)
    else:
        rng = np.random.default_rng(0)
        approx = np.array([_gearys_c_stat(rng.permutation(vals), w) for _ in range(200)])
        std = float(np.std(approx, ddof=1))
        if std > 0.0:
            z_score = (c_obs - 1.0) / std
            p_two_sided = 2.0 * (1.0 - norm.cdf(abs(z_score)))
        else:
            z_score = 0.0
            p_two_sided = 1.0
        result["z_score"] = float(z_score)
        result["p_value"] = float(p_two_sided)

    return result


def local_gearys_c(
    values: np.ndarray,
    weights: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Compute local Geary components for each observation."""

    vals = validate_array(values, name="values").ravel()
    w = validate_array(weights, name="weights")
    if w.ndim != 2 or w.shape[0] != w.shape[1]:
        raise ValueError("weights must be a square 2D array")
    if len(vals) != w.shape[0]:
        raise ValueError("values length must match weights dimensions")

    z = vals - np.mean(vals)
    m2 = np.sum(z**2) / len(vals)
    if m2 == 0.0:
        return {"C_local": np.zeros_like(vals), "z": z}

    diffs_sq = (vals[:, None] - vals[None, :]) ** 2
    c_local = np.sum(weights * diffs_sq, axis=1) / m2
    return {"C_local": c_local.astype(np.float64), "z": z.astype(np.float64)}
