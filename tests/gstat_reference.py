"""Reference datasets emulating R gstat outputs for validation tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .data_generation import VariogramParameters, _covariance_from_variogram, _euclidean_distance_matrix


@dataclass(frozen=True)
class ReferenceDataset:
    coords: np.ndarray
    values: np.ndarray
    params: VariogramParameters
    model: str
    prediction_points: np.ndarray
    expected_predictions: np.ndarray


def _ordinary_kriging_reference(
    coords: np.ndarray,
    values: np.ndarray,
    targets: np.ndarray,
    model: str,
    params: VariogramParameters,
) -> np.ndarray:
    n_known = coords.shape[0]
    distances = _euclidean_distance_matrix(coords)
    cov = _covariance_from_variogram(distances, model, params)
    system = np.zeros((n_known + 1, n_known + 1))
    system[:n_known, :n_known] = cov
    system[-1, :-1] = 1.0
    system[:-1, -1] = 1.0

    predictions = []
    for target in targets:
        dists = np.linalg.norm(coords - target, axis=1)
        cross_cov = _covariance_from_variogram(
            dists.reshape(-1, 1), model, params
        ).ravel()
        rhs = np.concatenate([cross_cov, np.array([1.0])])
        weights = np.linalg.solve(system, rhs)
        predictions.append(float(weights[:-1] @ values))
    return np.asarray(predictions)


def build_reference_datasets() -> Dict[str, ReferenceDataset]:
    rng = np.random.default_rng(42)

    datasets: Dict[str, ReferenceDataset] = {}

    expo_params = VariogramParameters(nugget=0.05, sill=1.0, range=0.35)
    coords = rng.uniform(0.0, 1.0, size=(24, 2))
    values = rng.standard_normal(24)
    values = values @ np.linalg.cholesky(
        _covariance_from_variogram(
            _euclidean_distance_matrix(coords), "exponential", expo_params
        )
    )
    targets = np.array([[0.2, 0.3], [0.7, 0.6], [0.5, 0.9]])
    predictions = _ordinary_kriging_reference(coords, values, targets, "exponential", expo_params)
    datasets["exponential"] = ReferenceDataset(
        coords=coords,
        values=values,
        params=expo_params,
        model="exponential",
        prediction_points=targets,
        expected_predictions=predictions,
    )

    sph_params = VariogramParameters(nugget=0.02, sill=0.8, range=0.45)
    coords = rng.uniform(0.0, 1.0, size=(28, 2))
    distances = _euclidean_distance_matrix(coords)
    chol = np.linalg.cholesky(
        _covariance_from_variogram(distances, "spherical", sph_params)
    )
    values = rng.standard_normal(28) @ chol
    targets = np.array([[0.1, 0.8], [0.4, 0.4]])
    predictions = _ordinary_kriging_reference(coords, values, targets, "spherical", sph_params)
    datasets["spherical"] = ReferenceDataset(
        coords=coords,
        values=values,
        params=sph_params,
        model="spherical",
        prediction_points=targets,
        expected_predictions=predictions,
    )

    gau_params = VariogramParameters(nugget=0.01, sill=1.2, range=0.55)
    coords = rng.uniform(0.0, 1.0, size=(22, 2))
    cov = _covariance_from_variogram(
        _euclidean_distance_matrix(coords), "gaussian", gau_params
    )
    chol = np.linalg.cholesky(cov)
    values = rng.standard_normal(22) @ chol
    targets = np.array([[0.3, 0.2], [0.6, 0.1], [0.8, 0.8]])
    predictions = _ordinary_kriging_reference(coords, values, targets, "gaussian", gau_params)
    datasets["gaussian"] = ReferenceDataset(
        coords=coords,
        values=values,
        params=gau_params,
        model="gaussian",
        prediction_points=targets,
        expected_predictions=predictions,
    )

    return datasets



