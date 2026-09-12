# src/python/pygeostats/utils/validation.py
"""Input validation utilities."""

from typing import Union

import numpy as np
import pandas as pd

try:
    import geopandas as gpd
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    gpd = None


def validate_coordinates(
    coordinates: Union[np.ndarray, pd.DataFrame, "gpd.GeoDataFrame"],
) -> np.ndarray:
    """
    Validate and convert coordinates to numpy array.

    Parameters
    ----------
    coordinates : array-like or GeoDataFrame
        Input coordinates.

    Returns
    -------
    coords : ndarray, shape (n_samples, n_features)
        Validated coordinate array.
    """
    if gpd is not None and isinstance(coordinates, gpd.GeoDataFrame):
        # Extract coordinates from geometry column
        if coordinates.geometry.isna().any():
            raise ValueError("GeoDataFrame contains null geometries")

        geom_type = coordinates.geometry.iloc[0].geom_type
        if geom_type != "Point":
            raise ValueError(f"Only Point geometries supported, got {geom_type}")

        coords = np.column_stack(
            [coordinates.geometry.x.values, coordinates.geometry.y.values]
        )

    elif isinstance(coordinates, pd.DataFrame):
        # Assume first two columns are x, y coordinates
        if coordinates.shape[1] < 2:
            raise ValueError("DataFrame must have at least 2 columns for coordinates")
        coords = coordinates.iloc[:, :2].values

    else:
        coords = np.asarray(coordinates)

    if coords.ndim != 2:
        raise ValueError(f"Coordinates must be 2D, got {coords.ndim}D")

    if coords.shape[1] < 2:
        raise ValueError("Coordinates must have at least 2 dimensions")

    if not np.isfinite(coords).all():
        raise ValueError("Coordinates must not contain NaN or infinite values")

    return coords.astype(np.float64)


def validate_values(values: Union[np.ndarray, pd.Series]) -> np.ndarray:
    """
    Validate and convert values to numpy array.

    Parameters
    ----------
    values : array-like
        Input values.

    Returns
    -------
    vals : ndarray, shape (n_samples,)
        Validated values array.
    """
    vals = np.asarray(values).flatten()

    if not np.isfinite(vals).all():
        raise ValueError("Values must not contain NaN or infinite values")

    return vals.astype(np.float64)


def validate_array(array: np.ndarray, name: str = "array") -> np.ndarray:
    """
    Generic array validation.

    Parameters
    ----------
    array : array-like
        Input array.
    name : str
        Name of the array for error messages.

    Returns
    -------
    arr : ndarray
        Validated array.
    """
    arr = np.asarray(array)

    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must not contain NaN or infinite values")

    return arr.astype(np.float64)
