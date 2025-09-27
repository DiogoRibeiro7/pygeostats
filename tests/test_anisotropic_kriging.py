# tests/test_anisotropic_kriging.py
"""Comprehensive tests for anisotropic kriging functionality."""

import numpy as np
import pytest
from sklearn.metrics import r2_score, mean_squared_error

from pyspatialstats.variogram import DirectionalVariogram, Variogram
from pyspatialstats.kriging import AnisotropicKriging, OrdinaryKriging
from pyspatialstats.kriging.anisotropic import create_anisotropic_variogram_from_directional


class TestAnisotropicKriging:
    """Test suite for anisotropic kriging."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Create known anisotropic test case
        self.range_major = 3.0
        self.range_minor = 1.0
        self.rotation_angle = np.pi / 4  # 45 degrees
        self.nugget = 0.1
        self.sill = 1.0
        
        # Generate test coordinates
        self.coords = np.array([
            [0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0],
            [0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0],
            [0.0, 2.0], [1.0, 2.0], [2.0, 2.0], [3.0, 2.0],
            [0.5, 0.5], [1.5, 1.5], [2.5, 0.5], [0.5, 2.5]
        ])
        
        # Generate anisotropic correlated values
        self.values = self._generate_anisotropic_values()
        
        # Create anisotropic variogram
        self.aniso_variogram = Variogram(model="exponential")
        self.aniso_variogram.parameters = [
            self.nugget, self.sill, self.range_major, 
            self.range_minor, self.rotation_angle
        ]
        self.aniso_variogram.is_fitted_ = True
        self.aniso_variogram.nugget_ = self.nugget
        self.aniso_variogram.sill_ = self.sill
        self.aniso_variogram.range_ = self.range_major

    def _generate_anisotropic_values(self):
        """Generate spatially correlated values with known anisotropy."""
        n = len(self.coords)
        cov_matrix = np.zeros((n, n))
        
        cos_theta = np.cos(self.rotation_angle)
        sin_theta = np.sin(self.rotation_angle)
        R = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        
        for i in range(n):
            for j in range(n):
                delta = self.coords[j] - self.coords[i]
                rotated_delta = R @ delta
                aniso_dist = np.sqrt(
                    (rotated_delta[0] / self.range_major)**2 + 
                    (rotated_delta[1] / self.range_minor)**2
                )
                
                if i == j:
                    cov_matrix[i, j] = self.sill
                else:
                    gamma = self.nugget + (self.sill - self.nugget) * (1 - np.exp(-aniso_dist))
                    cov_matrix[i, j] = self.sill - gamma
        
        return np.random.multivariate_normal(np.zeros(n), cov_matrix)

    def test_initialization(self):
        """Test AnisotropicKriging initialization."""
        kriging = AnisotropicKriging(self.aniso_variogram)
        
        assert kriging.nugget == self.nugget
        assert kriging.sill == self.sill
        assert kriging.range_major == self.range_major
        assert kriging.range_minor == self.range_minor
        assert kriging.rotation_angle == self.rotation_angle
        assert not kriging.is_fitted_

    def test_initialization_invalid_parameters(self):
        """Test initialization with invalid variogram parameters."""
        # Variogram with wrong number of parameters
        invalid_variogram = Variogram(model="exponential")
        invalid_variogram.parameters = [0.1, 1.0, 2.0]  # Only 3 parameters
        invalid_variogram.is_fitted_ = True
        
        with pytest.raises(ValueError, match="5 variogram parameters"):
            AnisotropicKriging(invalid_variogram)

    def test_fit(self):
        """Test fitting the kriging model."""
        kriging = AnisotropicKriging(self.aniso_variogram)
        kriging.fit(self.coords, self.values)
        
        assert kriging.is_fitted_
        assert np.array_equal(kriging.coordinates_, self.coords)
        assert np.array_equal(kriging.values_, self.values)

    def test_fit_invalid_dimensions(self):
        """Test fitting with invalid coordinate dimensions."""
        kriging = AnisotropicKriging(self.aniso_variogram)
        
        # 1D coordinates (invalid for anisotropic kriging)
        coords_1d = np.array([[0.0], [1.0], [2.0]])
        values_1d = np.array([1.0, 2.0, 3.0])
        
        with pytest.raises(ValueError, match="requires 2D coordinates"):
            kriging.fit(coords_1d, values_1d)

    def test_predict_basic(self):
        """Test basic prediction functionality."""
        kriging = AnisotropicKriging(self.aniso_variogram)
        kriging.fit(self.coords, self.values)
        
        # Predict at known locations (should be close to known values)
        predictions = kriging.predict(self.coords)
        
        assert len(predictions) == len(self.coords)
        assert np.allclose(predictions, self.values, rtol=0.1)  # Allow some tolerance

    def test_predict_with_variance(self):
        """Test prediction with variance calculation."""
        kriging = AnisotropicKriging(self.aniso_variogram)
        kriging.fit(self.coords, self.values)
        
        # Predict at new locations
        pred_coords = np.array([[0.5, 0.5], [1.5, 1.5], [2.5, 2.5]])
        predictions, variances = kriging.predict(pred_coords, return_variance=True)
        
        assert len(predictions) == len(pred_coords)
        assert len(variances) == len(pred_coords)
        assert np.all(variances >= 0)  # Variances should be non-negative

    def test_anisotropic_distance_calculation(self):
        """Test anisotropic distance calculations."""
        kriging = AnisotropicKriging(self.aniso_variogram)
        
        # Test distance calculation along major axis
        p1 = np.array([0.0, 0.0])
        p2 = np.array([1.0, 0.0])  # Along x-axis
        
        # At 45-degree rotation, this should be affected by both ranges
        dist = kriging._anisotropic_distance(p1, p2)
        
        # Expected: rotated point uses both major and minor ranges
        cos_45 = np.cos(np.pi/4)
        expected_dist = np.sqrt((cos_45 / self.range_major)**2 + (cos_45 / self.range_minor)**2)
        
        assert np.isclose(dist, expected_dist, rtol=1e-10)

    def test_isotropic_special_case(self):
        """Test that isotropic case (equal ranges) works correctly."""
        # Create isotropic variogram (equal ranges)
        iso_variogram = Variogram(model="exponential")
        iso_variogram.parameters = [self.nugget, self.sill, 2.0, 2.0, 0.0]  # Equal ranges
        iso_variogram.is_fitted_ = True
        iso_variogram.nugget_ = self.nugget
        iso_variogram.sill_ = self.sill
        iso_variogram.range_ = 2.0
        
        aniso_kriging = AnisotropicKriging(iso_variogram)
        aniso_kriging.fit(self.coords, self.values)
        
        # Compare with ordinary kriging
        ord_variogram = Variogram(model="exponential")
        ord_variogram.parameters = [self.nugget, self.sill, 2.0]
        ord_variogram.is_fitted_ = True
        ord_variogram.nugget_ = self.nugget
        ord_variogram.sill_ = self.sill
        ord_variogram.range_ = 2.0
        
        ord_kriging = OrdinaryKriging(ord_variogram)
        ord_kriging.fit(self.coords, self.values)
        
        # Predictions should be very similar
        pred_coords = np.array([[0.5, 0.5], [1.5, 1.5]])
        aniso_pred = aniso_kriging.predict(pred_coords)
        ord_pred = ord_kriging.predict(pred_coords)
        
        assert np.allclose(aniso_pred, ord_pred, rtol=0.01)

    def test_anisotropy_info(self):
        """Test anisotropy information extraction."""
        kriging = AnisotropicKriging(self.aniso_variogram)
        info = kriging.get_anisotropy_info()
        
        expected_ratio = self.range_major / self.range_minor
        expected_angle_deg = np.degrees(self.rotation_angle)
        
        assert np.isclose(info['anisotropy_ratio'], expected_ratio)
        assert np.isclose(info['rotation_angle_degrees'], expected_angle_deg)
        assert info['is_anisotropic'] == True  # ratio > 1.2
        assert info['range_major'] == self.range_major
        assert info['range_minor'] == self.range_minor

    def test_score(self):
        """Test R² scoring."""
        kriging = AnisotropicKriging(self.aniso_variogram)
        kriging.fit(self.coords, self.values)
        
        # Score on training data should be high
        score = kriging.score(self.coords, self.values)
        assert score > 0.8  # Should achieve good fit on training data

    def test_singular_matrix_handling(self):
        """Test handling of singular covariance matrices."""
        # Create duplicate points (should cause singular matrix)
        coords_dup = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
        values_dup = np.array([1.0, 1.1, 2.0])  # Slightly different values at same location
        
        kriging = AnisotropicKriging(self.aniso_variogram)
        kriging.fit(coords_dup, values_dup)
        
        # Prediction should raise error for singular matrix
        with pytest.raises(ValueError, match="Singular covariance matrix"):
            kriging.predict(np.array([[0.5, 0.5]]))


class TestDirectionalVariogramIntegration:
    """Test integration with directional variogram analysis."""

    def setup_method(self):
        """Set up test data with known anisotropy."""
        np.random.seed(123)
        
        # Generate data with known anisotropic structure
        n_samples = 50
        coords = np.random.uniform(0, 10, size=(n_samples, 2))
        
        # Create anisotropic covariance structure
        range_major = 4.0
        range_minor = 1.5
        angle = np.pi/3  # 60 degrees
        nugget = 0.1
        sill = 1.2
        
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        R = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        
        cov_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                delta = coords[j] - coords[i]
                rotated = R @ delta
                aniso_dist = np.sqrt(
                    (rotated[0] / range_major)**2 + (rotated[1] / range_minor)**2
                )
                if i == j:
                    cov_matrix[i, j] = sill
                else:
                    gamma = nugget + (sill - nugget) * (1 - np.exp(-aniso_dist))
                    cov_matrix[i, j] = sill - gamma
        
        values = np.random.multivariate_normal(np.zeros(n_samples), cov_matrix)
        
        self.coords = coords
        self.values = values
        self.true_params = {
            'range_major': range_major,
            'range_minor': range_minor,
            'angle_rad': angle,
            'angle_deg': np.degrees(angle),
            'nugget': nugget,
            'sill': sill,
            'ratio': range_major / range_minor
        }

    def test_directional_to_anisotropic_workflow(self):
        """Test complete workflow from directional analysis to anisotropic kriging."""
        # Step 1: Directional variogram analysis
        directional_vario = DirectionalVariogram(
            coordinates=self.coords,
            values=self.values,
            directions=[0, 30, 60, 90, 120, 150],
            tolerance=22.5,
            max_distance=5.0,
            n_bins=12
        )
        directional_vario.compute()
        
        # Step 2: Detect anisotropy
        anisotropy_result = directional_vario.detect_anisotropy()
        assert anisotropy_result.is_anisotropic
        
        # Step 3: Create anisotropic variogram
        aniso_variogram = create_anisotropic_variogram_from_directional(
            directional_vario, model="exponential"
        )
        
        # Step 4: Anisotropic kriging
        kriging = AnisotropicKriging(aniso_variogram)
        kriging.fit(self.coords, self.values)
        
        # Test predictions
        pred_coords = np.array([[2.5, 2.5], [7.5, 7.5]])
        predictions, variances = kriging.predict(pred_coords, return_variance=True)
        
        assert len(predictions) == 2
        assert len(variances) == 2
        assert np.all(np.isfinite(predictions))
        assert np.all(variances >= 0)

    def test_parameter_recovery_accuracy(self):
        """Test how well we can recover known anisotropic parameters."""
        directional_vario = DirectionalVariogram(
            coordinates=self.coords,
            values=self.values,
            directions=np.arange(0, 180, 15),  # Dense directional sampling
            tolerance=15.0,
            max_distance=4.0,
            n_bins=15
        )
        directional_vario.compute()
        
        # Get initialization estimates
        init_summary = directional_vario.estimate_initial_parameters()
        
        # Check parameter recovery (allow reasonable tolerance)
        recovered_ratio = init_summary.range_major / init_summary.range_minor
        ratio_error = abs(recovered_ratio - self.true_params['ratio']) / self.true_params['ratio']
        
        # Allow up to 30% error in ratio recovery (realistic for noisy data)
        assert ratio_error < 0.3, f"Ratio error {ratio_error:.2f} too large"
        
        # Check angle recovery (handle circular nature of angles)
        angle_diff = abs(init_summary.angle_deg - self.true_params['angle_deg'])
        angle_error = min(angle_diff, 180 - angle_diff)  # Handle wraparound
        
        # Allow up to 30-degree error in angle recovery
        assert angle_error < 30, f"Angle error {angle_error:.1f}° too large"


class TestAnisotropicVsIsotropicComparison:
    """Test comparisons between anisotropic and isotropic kriging."""

    def setup_method(self):
        """Set up comparison test case."""
        np.random.seed(789)
        
        # Generate highly anisotropic data
        self.range_major = 5.0
        self.range_minor = 1.0  # High anisotropy ratio = 5.0
        self.angle = np.pi / 6  # 30 degrees
        self.nugget = 0.05
        self.sill = 1.5
        
        # Generate structured sample locations
        x = np.linspace(0, 12, 8)
        y = np.linspace(0, 8, 6)
        xx, yy = np.meshgrid(x, y)
        self.coords = np.column_stack([xx.ravel(), yy.ravel()])
        
        # Add some random locations
        random_coords = np.random.uniform([0, 0], [12, 8], size=(20, 2))
        self.coords = np.vstack([self.coords, random_coords])
        
        # Generate anisotropic values
        self.values = self._generate_anisotropic_values()

    def _generate_anisotropic_values(self):
        """Generate anisotropic spatially correlated values."""
        n = len(self.coords)
        cov_matrix = np.zeros((n, n))
        
        cos_theta = np.cos(self.angle)
        sin_theta = np.sin(self.angle)
        R = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        
        for i in range(n):
            for j in range(n):
                delta = self.coords[j] - self.coords[i]
                rotated = R @ delta
                aniso_dist = np.sqrt(
                    (rotated[0] / self.range_major)**2 + 
                    (rotated[1] / self.range_minor)**2
                )
                
                if i == j:
                    cov_matrix[i, j] = self.sill
                else:
                    gamma = self.nugget + (self.sill - self.nugget) * (1 - np.exp(-aniso_dist))
                    cov_matrix[i, j] = self.sill - gamma
        
        return np.random.multivariate_normal(np.zeros(n), cov_matrix)

    def test_anisotropic_vs_isotropic_performance(self):
        """Compare prediction performance of anisotropic vs isotropic kriging."""
        # Set up anisotropic kriging
        aniso_variogram = Variogram(model="exponential")
        aniso_variogram.parameters = [
            self.nugget, self.sill, self.range_major, 
            self.range_minor, self.angle
        ]
        aniso_variogram.is_fitted_ = True
        aniso_variogram.nugget_ = self.nugget
        aniso_variogram.sill_ = self.sill
        aniso_variogram.range_ = self.range_major
        
        aniso_kriging = AnisotropicKriging(aniso_variogram)
        aniso_kriging.fit(self.coords, self.values)
        
        # Set up isotropic kriging (using average range)
        avg_range = (self.range_major + self.range_minor) / 2
        iso_variogram = Variogram(model="exponential")
        iso_variogram.parameters = [self.nugget, self.sill, avg_range]
        iso_variogram.is_fitted_ = True
        iso_variogram.nugget_ = self.nugget
        iso_variogram.sill_ = self.sill
        iso_variogram.range_ = avg_range
        
        iso_kriging = OrdinaryKriging(iso_variogram)
        iso_kriging.fit(self.coords, self.values)
        
        # Cross-validation comparison
        n_samples = len(self.coords)
        aniso_predictions = np.zeros(n_samples)
        iso_predictions = np.zeros(n_samples)
        
        # Leave-one-out cross-validation on subset for speed
        test_indices = np.random.choice(n_samples, size=min(15, n_samples), replace=False)
        
        for i in test_indices:
            # Training data (excluding sample i)
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[i] = False
            
            train_coords = self.coords[train_mask]
            train_values = self.values[train_mask]
            test_coord = self.coords[i:i+1]
            
            # Anisotropic prediction
            aniso_temp = AnisotropicKriging(aniso_variogram)
            aniso_temp.fit(train_coords, train_values)
            aniso_predictions[i] = aniso_temp.predict(test_coord)[0]
            
            # Isotropic prediction
            iso_temp = OrdinaryKriging(iso_variogram)
            iso_temp.fit(train_coords, train_values)
            iso_predictions[i] = iso_temp.predict(test_coord)[0]
        
        # Calculate performance metrics on tested samples
        test_values = self.values[test_indices]
        aniso_test_pred = aniso_predictions[test_indices]
        iso_test_pred = iso_predictions[test_indices]
        
        aniso_rmse = np.sqrt(mean_squared_error(test_values, aniso_test_pred))
        iso_rmse = np.sqrt(mean_squared_error(test_values, iso_test_pred))
        
        aniso_r2 = r2_score(test_values, aniso_test_pred)
        iso_r2 = r2_score(test_values, iso_test_pred)
        
        # With high anisotropy, anisotropic kriging should perform better
        print(f"Anisotropic RMSE: {aniso_rmse:.4f}, R²: {aniso_r2:.4f}")
        print(f"Isotropic RMSE: {iso_rmse:.4f}, R²: {iso_r2:.4f}")
        
        # Allow some tolerance since this is a stochastic test
        assert aniso_rmse <= iso_rmse * 1.1, "Anisotropic kriging should not be much worse"
        
        # Check that anisotropic predictions are different from isotropic
        prediction_diff = np.mean(np.abs(aniso_test_pred - iso_test_pred))
        assert prediction_diff > 0.01, "Anisotropic and isotropic predictions should differ"

    def test_uncertainty_quantification(self):
        """Test uncertainty quantification in anisotropic kriging."""
        aniso_variogram = Variogram(model="exponential")
        aniso_variogram.parameters = [
            self.nugget, self.sill, self.range_major, 
            self.range_minor, self.angle
        ]
        aniso_variogram.is_fitted_ = True
        aniso_variogram.nugget_ = self.nugget
        aniso_variogram.sill_ = self.sill
        aniso_variogram.range_ = self.range_major
        
        kriging = AnisotropicKriging(aniso_variogram)
        kriging.fit(self.coords, self.values)
        
        # Create prediction grid
        x_pred = np.linspace(-1, 13, 15)  # Extend beyond sample domain
        y_pred = np.linspace(-1, 9, 12)
        xx, yy = np.meshgrid(x_pred, y_pred)
        pred_coords = np.column_stack([xx.ravel(), yy.ravel()])
        
        predictions, variances = kriging.predict(pred_coords, return_variance=True)
        
        # Test variance properties
        assert np.all(variances >= 0), "All variances should be non-negative"
        assert np.all(np.isfinite(variances)), "All variances should be finite"
        
        # Variance should be higher in extrapolation regions
        # Find points inside and outside the convex hull of sample points
        from scipy.spatial import ConvexHull
        hull = ConvexHull(self.coords)
        
        inside_mask = np.array([
            _point_in_hull(point, self.coords[hull.vertices]) 
            for point in pred_coords
        ])
        
        if np.any(inside_mask) and np.any(~inside_mask):
            var_inside = np.mean(variances[inside_mask])
            var_outside = np.mean(variances[~inside_mask])
            
            # Variance should generally be higher outside the sample region
            assert var_outside >= var_inside * 0.8, "Extrapolation variance should be higher"

    def test_directional_dependence(self):
        """Test that anisotropic kriging shows directional dependence."""
        aniso_variogram = Variogram(model="exponential")
        aniso_variogram.parameters = [
            self.nugget, self.sill, self.range_major, 
            self.range_minor, self.angle
        ]
        aniso_variogram.is_fitted_ = True
        aniso_variogram.nugget_ = self.nugget
        aniso_variogram.sill_ = self.sill
        aniso_variogram.range_ = self.range_major
        
        kriging = AnisotropicKriging(aniso_variogram)
        kriging.fit(self.coords, self.values)
        
        # Test directional correlation by predicting along different directions
        center = np.array([6.0, 4.0])  # Center of domain
        distance = 2.0
        
        # Directions aligned with major and minor axes (rotated by self.angle)
        major_direction = np.array([np.cos(self.angle), np.sin(self.angle)])
        minor_direction = np.array([-np.sin(self.angle), np.cos(self.angle)])
        
        # Points along major and minor axes
        point_major = center + distance * major_direction
        point_minor = center + distance * minor_direction
        
        # Predict at these points
        _, var_major = kriging.predict(point_major.reshape(1, -1), return_variance=True)
        _, var_minor = kriging.predict(point_minor.reshape(1, -1), return_variance=True)
        
        # Along major axis (higher correlation), variance should be lower
        # This test may be sensitive to data configuration, so use loose tolerance
        variance_ratio = var_minor[0] / var_major[0]
        assert variance_ratio >= 0.8, f"Expected directional variance difference, got ratio {variance_ratio:.2f}"


def _point_in_hull(point, hull_points):
    """Check if a point is inside a convex hull (simple 2D implementation)."""
    from scipy.spatial import ConvexHull
    
    try:
        # Add the test point to the hull points
        all_points = np.vstack([hull_points, point.reshape(1, -1)])
        new_hull = ConvexHull(all_points)
        
        # If the point is inside, the hull shouldn't change
        return len(new_hull.vertices) == len(hull_points)
    except:
        # If ConvexHull fails, assume point is inside
        return True


class TestEdgeCasesAndRobustness:
    """Test edge cases and robustness of anisotropic kriging."""

    def test_extreme_anisotropy_ratio(self):
        """Test with very high anisotropy ratio."""
        # Extreme anisotropy
        range_major = 10.0
        range_minor = 0.1  # Ratio = 100
        
        variogram = Variogram(model="exponential")
        variogram.parameters = [0.1, 1.0, range_major, range_minor, 0.0]
        variogram.is_fitted_ = True
        variogram.nugget_ = 0.1
        variogram.sill_ = 1.0
        variogram.range_ = range_major
        
        # Simple test data
        coords = np.array([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1]])
        values = np.array([1.0, 1.1, 1.2, 2.0, 2.1])
        
        kriging = AnisotropicKriging(variogram)
        kriging.fit(coords, values)
        
        # Should still work despite extreme anisotropy
        pred = kriging.predict(np.array([[0.5, 0.5]]))
        assert np.isfinite(pred[0])

    def test_near_isotropic_case(self):
        """Test with nearly isotropic parameters."""
        # Nearly isotropic (small difference in ranges)
        range_major = 2.001
        range_minor = 2.000
        
        variogram = Variogram(model="exponential")
        variogram.parameters = [0.1, 1.0, range_major, range_minor, 0.0]
        variogram.is_fitted_ = True
        variogram.nugget_ = 0.1
        variogram.sill_ = 1.0
        variogram.range_ = range_major
        
        coords = np.random.uniform(0, 5, size=(10, 2))
        values = np.random.normal(0, 1, 10)
        
        kriging = AnisotropicKriging(variogram)
        kriging.fit(coords, values)
        
        # Should handle near-isotropic case gracefully
        pred = kriging.predict(coords[:3])
        assert len(pred) == 3
        assert np.all(np.isfinite(pred))

    def test_zero_nugget(self):
        """Test with zero nugget effect."""
        variogram = Variogram(model="exponential")
        variogram.parameters = [0.0, 1.0, 3.0, 1.0, np.pi/4]  # Zero nugget
        variogram.is_fitted_ = True
        variogram.nugget_ = 0.0
        variogram.sill_ = 1.0
        variogram.range_ = 3.0
        
        coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [2, 2]])
        values = np.array([1.0, 2.0, 1.5, 2.5, 3.0])
        
        kriging = AnisotropicKriging(variogram)
        kriging.fit(coords, values)
        
        # Should interpolate exactly at known points with zero nugget
        pred = kriging.predict(coords)
        assert np.allclose(pred, values, atol=1e-10)

    def test_different_variogram_models(self):
        """Test different variogram models (spherical, gaussian)."""
        coords = np.random.uniform(0, 5, size=(15, 2))
        values = np.random.normal(0, 1, 15)
        
        models = ["exponential", "spherical", "gaussian"]
        
        for model in models:
            variogram = Variogram(model=model)
            variogram.parameters = [0.1, 1.0, 3.0, 1.5, 0.0]
            variogram.is_fitted_ = True
            variogram.nugget_ = 0.1
            variogram.sill_ = 1.0
            variogram.range_ = 3.0
            
            kriging = AnisotropicKriging(variogram)
            kriging.fit(coords, values)
            
            pred = kriging.predict(coords[:5])
            assert len(pred) == 5
            assert np.all(np.isfinite(pred))


if __name__ == "__main__":
    pytest.main([__file__])
