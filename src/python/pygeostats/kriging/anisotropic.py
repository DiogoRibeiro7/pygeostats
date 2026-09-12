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
        """
        self.coordinates_ = validate_coordinates(coordinates)
        self.values_ = validate_values(values)

        if self.coordinates_.shape[1] != 2:
            raise ValueError("Anisotropic kriging requires 2D coordinates")

        if len(self.coordinates_) != len(self.values_):
            raise ValueError("Coordinates and values must have same length")

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

        n_known = len(self.coordinates_)
        n_pred = len(pred_coords)

        # Build anisotropic covariance matrix for known points
        C = self._build_covariance_matrix(self.coordinates_, self.coordinates_)

        # Add unbiasedness constraint
        system_matrix = np.zeros((n_known + 1, n_known + 1))
        system_matrix[:n_known, :n_known] = C
        system_matrix[n_known, :n_known] = 1.0
        system_matrix[:n_known, n_known] = 1.0

        # Solve for weights for each prediction point
        predictions = np.zeros(n_pred)
        variances = np.zeros(n_pred) if return_variance else None

        try:
            # Pre-factorize the system matrix
            from scipy.linalg import solve

            for i in range(n_pred):
                # Build RHS vector
                rhs = np.zeros(n_known + 1)

                # Compute covariances between prediction point and known points
                pred_point = pred_coords[i : i + 1]
                c0 = self._build_covariance_matrix(self.coordinates_, pred_point)
                rhs[:n_known] = c0.flatten()
                rhs[n_known] = 1.0  # unbiasedness constraint

                # Solve for weights
                weights = solve(system_matrix, rhs)

                # Compute prediction
                predictions[i] = np.dot(weights[:n_known], self.values_)

                # Compute variance if requested
                if return_variance:
                    # Kriging variance: C(0,0) - w^T * c0 - lambda
                    c00 = self._anisotropic_covariance(
                        0.0
                    )  # variance at prediction point
                    variances[i] = (
                        c00 - np.dot(weights[:n_known], c0.flatten()) - weights[n_known]
                    )
                    variances[i] = max(variances[i], 0.0)  # ensure non-negative

        except np.linalg.LinAlgError as err:
            raise ValueError(
                "Singular covariance matrix - check for duplicate points or poor conditioning"
            ) from err

        if return_variance:
            return predictions, variances
        return predictions

    def _build_covariance_matrix(
        self, coords1: np.ndarray, coords2: np.ndarray
    ) -> np.ndarray:
        """Build anisotropic covariance matrix between two sets of coordinates."""
        n1, n2 = len(coords1), len(coords2)
        C = np.zeros((n1, n2))

        for i in range(n1):
            for j in range(n2):
                aniso_distance = self._anisotropic_distance(coords1[i], coords2[j])
                C[i, j] = self._anisotropic_covariance(aniso_distance)

        return C

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

    def _anisotropic_covariance(self, aniso_distance: float) -> float:
        """Compute covariance from anisotropic distance using variogram model."""
        if aniso_distance == 0.0:
            return self.sill

        # Convert covariance to semivariance then back to covariance
        gamma = self._variogram_function(aniso_distance)
        return self.sill - gamma

    def _variogram_function(self, h: float) -> float:
        """Calculate variogram values using the fitted model."""
        if self.variogram.model == "exponential":
            return self.nugget + (self.sill - self.nugget) * (1 - np.exp(-h))

        elif self.variogram.model == "spherical":
            if h >= 1.0:
                return self.sill
            else:
                return self.nugget + (self.sill - self.nugget) * (1.5 * h - 0.5 * h**3)

        elif self.variogram.model == "gaussian":
            return self.nugget + (self.sill - self.nugget) * (1 - np.exp(-(h**2)))

        else:
            raise ValueError(f"Unknown variogram model: {self.variogram.model}")

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
