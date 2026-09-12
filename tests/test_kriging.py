"""Tests for kriging functionality."""

import numpy as np
import pytest
from pygeostats.kriging import (
    OrdinaryKriging,
    SimpleKriging,
    UniversalKriging,
)
from pygeostats.variogram import Variogram


def _covariance(
    distance: float, nugget: float, sill: float, range_: float, model: str
) -> float:
    if distance == 0.0:
        return sill
    if model == "exponential":
        gamma = nugget + (sill - nugget) * (1.0 - np.exp(-distance / range_))
    elif model == "spherical":
        if distance < range_:
            ratio = distance / range_
            gamma = nugget + (sill - nugget) * (1.5 * ratio - 0.5 * ratio**3)
        else:
            gamma = sill
    elif model == "gaussian":
        ratio = distance / range_
        gamma = nugget + (sill - nugget) * (1.0 - np.exp(-(ratio**2)))
    else:
        raise ValueError("Unknown model")
    return sill - gamma


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
            axis=2,
        )
        correlation = np.exp(-distances / 2.0)
        self.known_values = np.random.multivariate_normal(
            mean=np.zeros(n_known), cov=correlation
        )

        # Prediction points
        n_pred = 20
        self.pred_coords = np.random.uniform(0, 10, size=(n_pred, 2))

        # Fitted variogram
        self.variogram = Variogram(model="exponential")
        # Mock fitting (normally would fit to empirical variogram)
        self.variogram.nugget_ = 0.1
        self.variogram.sill_ = 1.0
        self.variogram.range_ = 2.0
        self.variogram.is_fitted_ = True

    def test_unfitted_variogram(self):
        """Test that unfitted variogram raises error."""
        unfitted_vario = Variogram(model="exponential")
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

        predictions, variance = kriging.predict(self.pred_coords, return_variance=True)

        assert len(predictions) == len(self.pred_coords)
        assert len(variance) == len(self.pred_coords)
        assert np.all(variance >= 0)  # Variance should be non-negative

    def test_variance_near_zero_at_observed_points_without_nugget(self):
        """Kriging variance should be ~0 at observed points when nugget is zero."""
        variogram = Variogram(model="exponential")
        variogram.nugget_ = 0.0
        variogram.sill_ = 1.0
        variogram.range_ = 2.0
        variogram.is_fitted_ = True

        kriging = OrdinaryKriging(variogram)
        kriging.fit(self.known_coords, self.known_values)
        _, variance = kriging.predict(self.known_coords, return_variance=True)

        assert np.all(variance >= 0.0)
        assert np.max(variance) < 1e-6

    def test_score(self):
        """Test R^2 scoring."""
        kriging = OrdinaryKriging(self.variogram)
        kriging.fit(self.known_coords, self.known_values)

        # Use subset of known points for testing
        test_coords = self.known_coords[:10]
        test_values = self.known_values[:10]

        score = kriging.score(test_coords, test_values)
        assert 0 <= score <= 1  # R^2 should be between 0 and 1 for good fits

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


class TestSimpleKriging:
    """Tests for the simple kriging workflow."""

    def setup_method(self):
        self.coords = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
        self.values = np.array([1.0, 2.0, 2.5, 3.0])
        self.pred = np.array([[0.25, 0.75]])
        self.mean = float(np.mean(self.values))

        self.variogram = Variogram(model="exponential")
        self.variogram.nugget_ = 0.05
        self.variogram.sill_ = 1.5
        self.variogram.range_ = 3.0
        self.variogram.is_fitted_ = True

    def test_unfitted_variogram(self):
        sk = SimpleKriging(Variogram(model="exponential"), mean=self.mean)
        with pytest.raises(ValueError, match="Variogram must be fitted"):
            sk.fit(self.coords, self.values)

    def test_matches_manual_solution(self):
        sk = SimpleKriging(self.variogram, mean=self.mean)
        sk.fit(self.coords, self.values)

        prediction = sk.predict(self.pred)

        params = (self.variogram.nugget_, self.variogram.sill_, self.variogram.range_)
        model = self.variogram.model

        n = len(self.coords)
        cov = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist = np.linalg.norm(self.coords[i] - self.coords[j])
                cov[i, j] = _covariance(dist, *params, model)

        rhs = np.zeros(n)
        for i in range(n):
            rhs[i] = _covariance(
                np.linalg.norm(self.coords[i] - self.pred[0]), *params, model
            )

        weights = np.linalg.solve(cov, rhs)
        manual = self.mean + weights.dot(self.values - self.mean)

        assert np.allclose(prediction, manual, atol=1e-6)


class TestUniversalKriging:
    """Tests for universal kriging with automatic trend detection."""

    def setup_method(self):
        self.coords = np.array(
            [
                [-1.0, -1.0],
                [-1.0, 0.5],
                [0.0, -0.5],
                [0.5, 1.0],
                [1.0, -0.5],
                [1.0, 1.0],
                [0.0, 0.75],
            ]
        )

        def quadratic(x, y):
            return 1.0 + 0.6 * x - 0.2 * y + 0.5 * x * x + 0.3 * y * y - 0.4 * x * y

        self.values = np.array([quadratic(x, y) for x, y in self.coords])
        self.pred = np.array([[0.3, -0.2], [-0.4, 0.6]])

        self.variogram = Variogram(model="exponential")
        self.variogram.nugget_ = 0.0
        self.variogram.sill_ = 2.0
        self.variogram.range_ = 5.0
        self.variogram.is_fitted_ = True

        self.quadratic_fn = quadratic

    def test_auto_selects_quadratic(self):
        uk = UniversalKriging(self.variogram, trend="auto")
        uk.fit(self.coords, self.values)

        assert uk.trend_ == "quadratic"
        assert "quadratic" in uk.trend_aic_

        predictions = uk.predict(self.pred)
        expected = np.array([self.quadratic_fn(*pt) for pt in self.pred])
        assert np.allclose(predictions, expected, atol=1e-6)

    def test_linear_trend(self):
        uk = UniversalKriging(self.variogram, trend="linear")
        uk.fit(self.coords, self.values)
        predictions = uk.predict(self.pred)
        assert predictions.shape == (len(self.pred),)

    def test_requires_fitted_variogram(self):
        unfitted = Variogram(model="exponential")
        uk = UniversalKriging(unfitted)
        with pytest.raises(ValueError, match="Variogram must be fitted"):
            uk.fit(self.coords, self.values)
