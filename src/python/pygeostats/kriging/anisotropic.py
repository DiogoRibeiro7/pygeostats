# src/python/pygeostats/kriging/anisotropic.py
"""Anisotropic kriging implementation with elliptical distance calculations."""

from __future__ import annotations

from typing import Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from ..utils.validation import validate_coordinates, validate_values
from ..variogram.models import Variogram
from ._solver import ordinary_system, ordinary_variance

_MODELS = {"exponential", "spherical", "gaussian"}


class AnisotropicKriging(BaseEstimator, RegressorMixin):
    """
    Anisotropic kriging with elliptical distance calculations.

    Parameters
    ----------
    variogram : Variogram
        Fitted anisotropic variogram model with parameters:
        [nugget, sill, range_major, range_minor, rotation_angle]
    """

    def __init__(self, variogram: Variogram):
        if not variogram.is_fitted_:
            raise ValueError("Variogram must be fitted before kriging")

        # Check if variogram has anisotropic parameters
        if len(variogram.parameters) != 5:
            raise ValueError(
                "Anisotropic kriging requires 5 variogram parameters: "
                "[nugget, sill, range_major, range_minor, rotation_angle]"
            )

        self.variogram = variogram

        # Extract anisotropic parameters
        self.nugget = variogram.parameters[0]
        self.sill = variogram.parameters[1]
        self.range_major = variogram.parameters[2]
        self.range_minor = variogram.parameters[3]
        self.rotation_angle = variogram.parameters[4]  # radians

        # Precompute rotation matrix
        cos_theta = np.cos(self.rotation_angle)
        sin_theta = np.sin(self.rotation_angle)
        self.rotation_matrix = np.array(
            [[cos_theta, -sin_theta], [sin_theta, cos_theta]]
        )

        # Fitted attributes
        self.coordinates_ = None
        self.values_ = None
        self.is_fitted_ = False
        self._solution = None

    def fit(
        self,
        coordinates: Union[np.ndarray, gpd.GeoDataFrame, pd.DataFrame],
        values: Union[np.ndarray, pd.Series],
    ) -> AnisotropicKriging:
        """
        Fit the anisotropic kriging model.

        Parameters
        ----------
        coordinates : array-like, shape (n_samples, 2)
            Known sample coordinates (must be 2D for anisotropic kriging).
        values : array-like, shape (n_samples,)
            Known sample values.

        Returns
        -------
        self : AnisotropicKriging
            Returns self for method chaining.

        Raises
        ------
        ValueError
            If the coordinates are not 2D, the variogram model is not exponential,
            spherical or Gaussian, or the kriging system is singular, as it is when
            two samples share a location.
        """
        self.coordinates_ = validate_coordinates(coordinates)
        self.values_ = validate_values(values)

        if self.coordinates_.shape[1] != 2:
            raise ValueError("Anisotropic kriging requires 2D coordinates")

        if len(self.coordinates_) != len(self.values_):
            raise ValueError("Coordinates and values must have same length")

        if self.variogram.model not in _MODELS:
            raise ValueError(f"Unknown variogram model: {self.variogram.model}")

        self._solution = None
        self._current_solution()
        self.is_fitted_ = True
        return self

    def predict(
        self,
        coordinates: Union[np.ndarray, gpd.GeoDataFrame, pd.DataFrame],
        return_variance: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict values at new locations using anisotropic kriging.

        Parameters
        ----------
        coordinates : array-like, shape (n_points, 2)
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

        if pred_coords.shape[1] != 2:
            raise ValueError(
                "Prediction coordinates must be 2D for anisotropic kriging"
            )

        system, weights, _ = self._current_solution()
        targets = self._to_isotropic(pred_coords)

        # The weights hold one entry per sample, then the Lagrange multiplier.
        predictions = system.covariance_sum(targets, weights[:-1]) + weights[-1]

        if return_variance:
            return predictions, ordinary_variance(system, targets)
        return predictions

    def _current_solution(self):
        """The factorised system and dual weights for the current parameters.

        Rotating and scaling the coordinates turns anisotropic distances into
        Euclidean ones with a range of 1, so ordinary kriging in those coordinates
        is anisotropic kriging in the original ones.
        """
        key = (
            self.nugget,
            self.sill,
            self.range_major,
            self.range_minor,
            self.rotation_angle,
            self.variogram.model,
        )
        solution = self._solution
        if solution is None or solution[2] != key:
            parameters = np.array([self.nugget, self.sill, 1.0], dtype=float)
            system = ordinary_system(
                self._to_isotropic(self.coordinates_), parameters, self.variogram.model
            )
            weights = system.solve(np.append(self.values_, 0.0))
            solution = (system, weights, key)
            self._solution = solution
        return solution

    def _to_isotropic(self, coordinates: np.ndarray) -> np.ndarray:
        """Coordinates in which anisotropic distance is Euclidean distance."""
        rotated = np.asarray(coordinates, dtype=float) @ self.rotation_matrix.T
        return rotated / np.array([self.range_major, self.range_minor], dtype=float)

    def _anisotropic_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """Compute anisotropic distance between two points."""
        # Vector difference
        delta = point2 - point1

        # Rotate to align with anisotropy axes
        rotated_delta = self.rotation_matrix @ delta

        # Scale by anisotropic ranges
        scaled_delta = np.array(
            [rotated_delta[0] / self.range_major, rotated_delta[1] / self.range_minor]
        )

        # Euclidean distance in scaled space
        return np.linalg.norm(scaled_delta)

    def score(self, coordinates: np.ndarray, values: np.ndarray) -> float:
        """
        Return the coefficient of determination R^2 of the prediction.

        Parameters
        ----------
        coordinates : array-like, shape (n_samples, 2)
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

    def get_anisotropy_info(self) -> dict:
        """Return anisotropy parameters and derived statistics."""
        ratio = self.range_major / self.range_minor
        angle_degrees = np.degrees(self.rotation_angle)

        return {
            "range_major": self.range_major,
            "range_minor": self.range_minor,
            "anisotropy_ratio": ratio,
            "rotation_angle_radians": self.rotation_angle,
            "rotation_angle_degrees": angle_degrees,
            "nugget": self.nugget,
            "sill": self.sill,
            "is_anisotropic": ratio > 1.2,  # threshold for significant anisotropy
        }


def create_anisotropic_variogram_from_directional(
    directional_variogram, model: str = "exponential"
) -> Variogram:
    """
    Create an anisotropic variogram from directional variogram analysis.

    Parameters
    ----------
    directional_variogram : DirectionalVariogram
        Fitted directional variogram with anisotropy detection.
    model : str, default="exponential"
        Variogram model type.

    Returns
    -------
    variogram : Variogram
        Anisotropic variogram model ready for kriging.
    """
    if not directional_variogram.is_fitted:
        raise ValueError("Directional variogram must be computed first")

    # Get anisotropy initialization parameters
    init_summary = directional_variogram.estimate_initial_parameters()
    ensemble_result = init_summary  # Assuming this returns the ensemble result

    # Create anisotropic variogram with 5 parameters
    variogram = Variogram(model=model)

    # Set initial parameters
    initial_params = np.array(
        [
            ensemble_result.nugget,
            ensemble_result.sill,
            ensemble_result.range_major,
            ensemble_result.range_minor,
            np.radians(ensemble_result.angle_deg),  # Convert to radians
        ]
    )

    # Mock fitting for demonstration - in practice this would call the Rust implementation
    variogram.parameters = initial_params
    variogram.is_fitted_ = True
    variogram.nugget_ = initial_params[0]
    variogram.sill_ = initial_params[1]
    variogram.range_ = initial_params[2]  # Use major range as primary range

    return variogram


def plot_anisotropy_ellipse(kriging_model, center=(0, 0), scale=1.0, ax=None):
    """
    Plot anisotropy ellipse showing spatial correlation structure.

    Parameters
    ----------
    kriging_model : AnisotropicKriging
        Fitted anisotropic kriging model.
    center : tuple, default=(0, 0)
        Center point for the ellipse.
    scale : float, default=1.0
        Scale factor for ellipse size.
    ax : matplotlib axes, optional
        Axes to plot on.

    Returns
    -------
    ax : matplotlib axes
        The axes object with the ellipse plot.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Create ellipse
    ellipse = Ellipse(
        center,
        width=2 * kriging_model.range_major * scale,
        height=2 * kriging_model.range_minor * scale,
        angle=np.degrees(kriging_model.rotation_angle),
        fill=False,
        color="red",
        linewidth=2,
        label=f"Anisotropy (ratio={kriging_model.range_major / kriging_model.range_minor:.2f})",
    )

    ax.add_patch(ellipse)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title("Anisotropy Ellipse")

    return ax
