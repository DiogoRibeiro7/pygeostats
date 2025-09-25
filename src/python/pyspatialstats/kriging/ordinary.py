# src/python/pyspatialstats/kriging/ordinary.py
"""Ordinary kriging implementation."""

import numpy as np
from typing import Optional, Union, Tuple
import geopandas as gpd
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from .._core import ordinary_kriging_predict, kriging_variance
from ..utils.validation import validate_coordinates, validate_values
from ..variogram.models import Variogram


class OrdinaryKriging(BaseEstimator, RegressorMixin):
    """
    Ordinary kriging interpolation.
    
    Parameters
    ----------
    variogram : Variogram
        Fitted variogram model.
    """
    
    def __init__(self, variogram: Variogram):
        self.variogram = variogram
        
        # Fitted attributes
        self.coordinates_ = None
        self.values_ = None
        self.is_fitted_ = False
        
    def fit(
        self, 
        coordinates: Union[np.ndarray, gpd.GeoDataFrame, pd.DataFrame],
        values: Union[np.ndarray, pd.Series]
    ) -> "OrdinaryKriging":
        """
        Fit the kriging model.
        
        Parameters
        ----------
        coordinates : array-like, shape (n_samples, n_features)
            Known sample coordinates.
        values : array-like, shape (n_samples,)
            Known sample values.
            
        Returns
        -------
        self : OrdinaryKriging
            Returns self for method chaining.
        """
        if not self.variogram.is_fitted_:
            raise ValueError("Variogram must be fitted before kriging")
            
        self.coordinates_ = validate_coordinates(coordinates)
        self.values_ = validate_values(values)
        
        if len(self.coordinates_) != len(self.values_):
            raise ValueError("Coordinates and values must have same length")
            
        self.is_fitted_ = True
        return self
        
    def predict(
        self, 
        coordinates: Union[np.ndarray, gpd.GeoDataFrame, pd.DataFrame],
        return_variance: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict values at new locations.
        
        Parameters
        ---------- 
        coordinates : array-like, shape (n_points, n_features)
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
        
        # Get variogram parameters
        variogram_params = np.array([
            self.variogram.nugget_,
            self.variogram.sill_, 
            self.variogram.range_
        ])
        
        # Call Rust implementation
        predictions = ordinary_kriging_predict(
            self.coordinates_,
            self.values_,
            pred_coords,
            variogram_params,
            self.variogram.model
        )
        
        if return_variance:
            variance = kriging_variance(
                self.coordinates_,
                pred_coords,
                variogram_params,
                self.variogram.model
            )
            return predictions, variance
        
        return predictions
        
    def score(self, coordinates: np.ndarray, values: np.ndarray) -> float:
        """
        Return the coefficient of determination R^2 of the prediction.
        
        Parameters
        ----------
        coordinates : array-like, shape (n_samples, n_features)
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
