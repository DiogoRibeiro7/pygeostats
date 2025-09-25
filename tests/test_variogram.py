"""Tests for variogram functionality."""

import numpy as np
import pytest
from pyspatialstats.variogram import EmpiricalVariogram, Variogram


class TestEmpiricalVariogram:
    """Test empirical variogram computation."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        n = 100
        self.coords = np.random.uniform(0, 10, size=(n, 2))
        
        # Create spatially correlated data
        distances = np.linalg.norm(
            self.coords[:, None, :] - self.coords[None, :, :], axis=2
        )
        correlation = np.exp(-distances / 2.0)
        self.values = np.random.multivariate_normal(
            mean=np.zeros(n), cov=correlation
        )
    
    def test_init_validation(self):
        """Test initialization validation."""
        # Mismatched lengths
        with pytest.raises(ValueError, match="same length"):
            EmpiricalVariogram(self.coords, self.values[:-1])
    
    def test_compute_basic(self):
        """Test basic variogram computation."""
        emp_vario = EmpiricalVariogram(self.coords, self.values)
        emp_vario.compute()
        
        assert emp_vario.is_fitted_
        assert emp_vario.distances_ is not None
        assert emp_vario.gamma_ is not None
        assert emp_vario.counts_ is not None
        assert len(emp_vario.distances_) == emp_vario.n_bins
    
    def test_custom_bins(self):
        """Test with custom bin edges."""
        bin_edges = np.linspace(0, 5, 11)
        emp_vario = EmpiricalVariogram(
            self.coords, self.values, bin_edges=bin_edges
        )
        emp_vario.compute()
        
        assert len(emp_vario.distances_) == len(bin_edges) - 1
        
    def test_plot_unfitted(self):
        """Test plotting without fitting raises error."""
        emp_vario = EmpiricalVariogram(self.coords, self.values)
        
        with pytest.raises(ValueError, match="Must call compute"):
            emp_vario.plot()


class TestVariogram:
    """Test theoretical variogram models."""
    
    def setup_method(self):
        """Set up test data."""
        self.distances = np.linspace(0, 10, 50)
        # Generate synthetic variogram data
        true_nugget, true_sill, true_range = 0.1, 1.0, 3.0
        self.gamma = true_nugget + (true_sill - true_nugget) * (
            1 - np.exp(-self.distances / true_range)
        )
        # Add some noise
        np.random.seed(42)
        self.gamma += np.random.normal(0, 0.05, len(self.gamma))
    
    def test_invalid_model(self):
        """Test invalid model raises error."""
        with pytest.raises(ValueError, match="Model must be one of"):
            Variogram(model='invalid')
    
    def test_fit_predict(self):
        """Test fitting and prediction."""
        vario = Variogram(model='exponential')
        vario.fit(self.distances, self.gamma)
        
        assert vario.is_fitted_
        assert vario.nugget_ is not None
        assert vario.sill_ is not None
        assert vario.range_ is not None
        
        # Test prediction
        pred_gamma = vario.predict(self.distances)
        assert len(pred_gamma) == len(self.distances)
        assert np.all(pred_gamma >= 0)  # Variogram values should be non-negative
    
    def test_covariance(self):
        """Test covariance calculation."""
        vario = Variogram(model='exponential')
        vario.fit(self.distances, self.gamma)
        
        cov = vario.covariance(self.distances)
        
        # Covariance at distance 0 should equal sill
        assert np.isclose(cov[0], vario.sill_, rtol=1e-3)
        
        # Covariance should be non-increasing with distance
        assert np.all(np.diff(cov) <= 1e-10)  # Allow for numerical errors
    
    def test_spherical_model(self):
        """Test spherical model specifics."""
        vario = Variogram(model='spherical')
        vario.fit(self.distances, self.gamma)
        
        pred_gamma = vario.predict(self.distances)
        
        # For spherical model, gamma should reach sill at range
        range_idx = np.argmin(np.abs(self.distances - vario.range_))
        if range_idx < len(pred_gamma) - 1:
            assert np.isclose(pred_gamma[range_idx], vario.sill_, rtol=1e-2)
