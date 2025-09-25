"""Tests for kriging functionality."""

import numpy as np
import pytest
from pyspatialstats.kriging import OrdinaryKriging
from pyspatialstats.variogram import Variogram


class TestOrdinaryKriging:
    """Test ordinary kriging."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        n_known = 50
        
        # Known points  
        self.known_coords = np.random.uniform(0, 10, size=(n_known, 2))
        
        # Generate spatially correlated values
        distances = np.linalg.norm(
            self.known_coords[:, None, :] - self.known_coords[None, :, :], 
            axis=2
        )
        correlation = np.exp(-distances / 2.0)
        self.known_values = np.random.multivariate_normal(
            mean=np.zeros(n_known), cov=correlation
        )
        
        # Prediction points
        n_pred = 20
        self.pred_coords = np.random.uniform(0, 10, size=(n_pred, 2))
        
        # Fitted variogram
        self.variogram = Variogram(model='exponential')
        # Mock fitting (normally would fit to empirical variogram)
        self.variogram.nugget_ = 0.1
        self.variogram.sill_ = 1.0
        self.variogram.range_ = 2.0
        self.variogram.is_fitted_ = True
        
    def test_unfitted_variogram(self):
        """Test that unfitted variogram raises error."""
        unfitted_vario = Variogram(model='exponential')
        kriging = OrdinaryKriging(unfitted_vario)
        
        with pytest.raises(ValueError, match="Variogram must be fitted"):
            kriging.fit(self.known_coords, self.known_values)
    
    def test_fit_predict(self):
        """Test basic fitting and prediction."""
        kriging = OrdinaryKriging(self.variogram)
        kriging.fit(self.known_coords, self.known_values)
        
        assert kriging.is_fitted_
        
        # Test prediction
        predictions = kriging.predict(self.pred_coords)
        assert len(predictions) == len(self.pred_coords)
        assert np.all(np.isfinite(predictions))
    
    def test_predict_with_variance(self):
        """Test prediction with variance."""
        kriging = OrdinaryKriging(self.variogram)
        kriging.fit(self.known_coords, self.known_values)
        
        predictions, variance = kriging.predict(
            self.pred_coords, return_variance=True
        )
        
        assert len(predictions) == len(self.pred_coords)
        assert len(variance) == len(self.pred_coords)
        assert np.all(variance >= 0)  # Variance should be non-negative
    
    def test_score(self):
        """Test R² scoring."""
        kriging = OrdinaryKriging(self.variogram)
        kriging.fit(self.known_coords, self.known_values)
        
        # Use subset of known points for testing
        test_coords = self.known_coords[:10]
        test_values = self.known_values[:10]
        
        score = kriging.score(test_coords, test_values)
        assert 0 <= score <= 1  # R² should be between 0 and 1 for good fits
    
    def test_mismatched_lengths(self):
        """Test mismatched coordinate and value lengths."""
        kriging = OrdinaryKriging(self.variogram)
        
        with pytest.raises(ValueError, match="same length"):
            kriging.fit(self.known_coords, self.known_values[:-1])
    
    def test_predict_unfitted(self):
        """Test prediction without fitting."""
        kriging = OrdinaryKriging(self.variogram)
        
        with pytest.raises(ValueError, match="must be fitted"):
            kriging.predict(self.pred_coords)
