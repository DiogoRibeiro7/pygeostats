"""Synthetic spatial data generation utilities for testing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np


@dataclass(frozen=True)
class VariogramParameters:
    nugget: float
    sill: float
    range: float


def _euclidean_distance_matrix(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def _covariance_from_variogram(
    distances: np.ndarray,
    model: str,
    params: VariogramParameters,
) -> np.ndarray:
    nugget, sill, range_ = params.nugget, params.sill, params.range
    if model == "exponential":
        gamma = nugget + (sill - nugget) * (1.0 - np.exp(-distances / range_))
    elif model == "spherical":
        ratio = np.clip(distances / range_, 0.0, 1.0)
        gamma = np.where(
            distances < range_,
            nugget + (sill - nugget) * (1.5 * ratio - 0.5 * ratio**3),
            sill,
        )
    elif model == "gaussian":
        ratio = distances / range_
        gamma = nugget + (sill - nugget) * (1.0 - np.exp(-(ratio**2)))
    else:
        raise ValueError(f"Unsupported model '{model}'")

    covariance = sill - gamma
    np.fill_diagonal(covariance, sill)
    return covariance


def generate_isotropic_field(
    n_points: int,
    model: str,
    params: VariogramParameters,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0.0, 1.0, size=(n_points, 2))
    distances = _euclidean_distance_matrix(coords)
    cov = _covariance_from_variogram(distances, model, params)
    jitter = 1e-10 * np.eye(n_points)
    values = rng.multivariate_normal(np.zeros(n_points), cov + jitter)
    return coords, values


def generate_anisotropic_field(
    n_points: int,
    model: str,
    params: VariogramParameters,
    stretch: float = 0.4,
    rotation: float = np.pi / 6,
    seed: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    coords, values = generate_isotropic_field(n_points, model, params, seed=seed)
    rot = np.array(
        [[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]]
    )
    scaled = coords @ rot.T
    scaled[:, 0] *= stretch
    return scaled, values


def generate_nested_structure(
    n_points: int,
    components: Tuple[Tuple[str, VariogramParameters], ...],
    seed: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0.0, 1.0, size=(n_points, 2))
    distances = _euclidean_distance_matrix(coords)
    cov = np.zeros((n_points, n_points))
    for model, params in components:
        cov += _covariance_from_variogram(distances, model, params)
    cov /= len(components)
    jitter = 1e-10 * np.eye(n_points)
    values = rng.multivariate_normal(np.zeros(n_points), cov + jitter)
    return coords, values


def inject_trend(
    coords: np.ndarray,
    values: np.ndarray,
    trend_fn: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    return values + trend_fn(coords)


def inject_noise(
    values: np.ndarray,
    noise_level: float,
    outlier_fraction: float = 0.0,
    outlier_scale: float = 3.0,
    seed: int = 3,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = values + rng.normal(scale=noise_level, size=values.shape)
    if outlier_fraction > 0.0:
        n_outliers = max(1, int(len(values) * outlier_fraction))
        idx = rng.choice(len(values), n_outliers, replace=False)
        noisy[idx] += rng.normal(scale=noise_level * outlier_scale, size=n_outliers)
    return noisy
