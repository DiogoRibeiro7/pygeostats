# src/python/pygeostats/variogram/empirical.py
"""Empirical variogram computation."""

from typing import Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd

from .._core import empirical_variogram
from ..utils.validation import validate_coordinates, validate_values


class EmpiricalVariogram:
    """
    Compute and analyze empirical variograms.

    Parameters
    ----------
    coordinates : array-like, shape (n_samples, n_features)
        Sample coordinates.
    values : array-like, shape (n_samples,)
        Sample values.
    max_distance : float, optional
        Maximum distance for variogram computation. If None, uses half the
        maximum distance in the dataset.
    n_bins : int, default=15
        Number of distance bins.
    bin_edges : array-like, optional
        Custom bin edges. If provided, overrides n_bins.
    """

    def __init__(
        self,
        coordinates: Union[np.ndarray, gpd.GeoDataFrame, pd.DataFrame],
        values: Union[np.ndarray, pd.Series],
        max_distance: Optional[float] = None,
        n_bins: int = 15,
        bin_edges: Optional[np.ndarray] = None,
    ):
        self.coordinates = validate_coordinates(coordinates)
        self.values = validate_values(values)

        if len(self.coordinates) != len(self.values):
            raise ValueError("Coordinates and values must have the same length")

        self.max_distance = max_distance
        self.n_bins = n_bins
        self.bin_edges = bin_edges

        # Computed attributes
        self.distances_ = None
        self.gamma_ = None
        self.counts_ = None
        self.is_fitted_ = False

    def compute(self) -> "EmpiricalVariogram":
        """
        Compute the empirical variogram.

        Returns
        -------
        self : EmpiricalVariogram
            Returns self for method chaining.
        """
        if self.bin_edges is None:
            if self.max_distance is None:
                # Calculate max distance if not provided
                from scipy.spatial.distance import pdist

                distances = pdist(self.coordinates)
                # A single point has no pairs, so its variogram is empty -- which is
                # what the core returns for it when bin edges are given. Taking the
                # maximum of no distances raised instead.
                self.max_distance = (
                    float(np.max(distances)) / 2.0 if distances.size else 0.0
                )

            self.bin_edges = np.linspace(0, self.max_distance, self.n_bins + 1)

        # Call Rust implementation
        self.distances_, self.gamma_, self.counts_ = empirical_variogram(
            self.coordinates, self.values, self.bin_edges
        )

        self.is_fitted_ = True
        return self

    def plot(self, ax=None, **kwargs):
        """
        Plot the empirical variogram.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
        **kwargs
            Additional arguments passed to matplotlib.pyplot.scatter.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes object.
        """
        if not self.is_fitted_:
            raise ValueError("Must call compute() before plotting")

        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        # Filter out bins with no data
        valid_mask = self.counts_ > 0
        distances = self.distances_[valid_mask]
        gamma = self.gamma_[valid_mask]
        counts = self.counts_[valid_mask]

        # Size points by number of pairs
        sizes = kwargs.pop("s", 50 * counts / np.max(counts) + 10)

        ax.scatter(distances, gamma, s=sizes, alpha=0.7, **kwargs)

        ax.set_xlabel("Distance")
        ax.set_ylabel("Semivariance (γ)")
        ax.set_title("Empirical Variogram")
        ax.grid(True, alpha=0.3)

        return ax
