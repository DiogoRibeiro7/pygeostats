"""Kriging from a system factorised once matches solving it for every target.

The estimators used to build and factorise their kriging system on every call to
predict, then solve it once per target. They now factorise it at fit and predict
from dual weights. The previous implementations, still exported by the Rust core,
are the reference for ordinary, simple and universal kriging predictions and for the
ordinary kriging variance. Direct NumPy solves are the reference for the simple and
universal kriging variances, which used to be the ordinary kriging variance, and
for anisotropic kriging.
"""

import numpy as np
import pytest
from pygeostats import _core
from pygeostats.kriging import (
    AnisotropicKriging,
    OrdinaryKriging,
    SimpleKriging,
    UniversalKriging,
)
from pygeostats.variogram import Variogram

MODELS = ["exponential", "spherical", "gaussian"]
NUGGET, SILL, RANGE = 0.1, 1.3, 3.0


def _variogram(model):
    variogram = Variogram(model=model)
    variogram.nugget_, variogram.sill_, variogram.range_ = NUGGET, SILL, RANGE
    variogram.is_fitted_ = True
    return variogram


def _anisotropic_variogram(model, parameters=(NUGGET, SILL, 4.0, 1.5, 0.6)):
    variogram = Variogram(model=model)
    variogram.parameters = np.array(parameters)
    variogram.is_fitted_ = True
    return variogram


def _covariance(scaled_distance, model):
    """Covariance at distances already divided by the range."""
    h = np.asarray(scaled_distance, dtype=float)
    if model == "exponential":
        gamma = NUGGET + (SILL - NUGGET) * (1.0 - np.exp(-h))
    elif model == "spherical":
        gamma = np.where(
            h >= 1.0, SILL, NUGGET + (SILL - NUGGET) * (1.5 * h - 0.5 * h**3)
        )
    else:
        gamma = NUGGET + (SILL - NUGGET) * (1.0 - np.exp(-(h**2)))
    return np.where(h == 0.0, SILL, SILL - gamma)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(11)
    coords = rng.uniform(0, 10, size=(80, 2))
    values = np.sin(coords[:, 0] / 2) + 0.3 * coords[:, 1]
    values += rng.normal(0, 0.1, len(coords))
    # Targets outside the samples' extent and at sample locations too.
    targets = np.vstack([rng.uniform(-1, 11, size=(150, 2)), coords[:5]])
    return coords, values, targets


def _parameters():
    return np.array([NUGGET, SILL, RANGE])


def _distances(first, second):
    return np.linalg.norm(first[:, None, :] - second[None, :, :], axis=-1)


def _trend(points, trend):
    x, y = points[:, 0], points[:, 1]
    columns = [np.ones(len(points)), x, y]
    if trend == "quadratic":
        columns += [x * x, y * y, x * y]
    return np.column_stack(columns)


def _direct_variance(coords, targets, model, trend=None):
    """Kriging variance solved directly: sill minus r @ solve(K, r) per target.

    With no trend this is simple kriging; with a trend, universal kriging.
    """
    n_samples = len(coords)
    system = _covariance(_distances(coords, coords) / RANGE, model)
    rhs = _covariance(_distances(targets, coords) / RANGE, model)
    if trend is not None:
        drift = _trend(coords, trend)
        n_drift = drift.shape[1]
        bordered = np.zeros((n_samples + n_drift, n_samples + n_drift))
        bordered[:n_samples, :n_samples] = system
        bordered[:n_samples, n_samples:] = drift
        bordered[n_samples:, :n_samples] = drift.T
        system = bordered
        rhs = np.hstack([rhs, _trend(targets, trend)])
    solution = np.linalg.solve(system, rhs.T)
    return np.maximum(SILL - np.einsum("ij,ji->i", rhs, solution), 0.0)


@pytest.mark.parametrize("model", MODELS)
def test_ordinary_kriging_matches_per_target_solution(data, model):
    coords, values, targets = data

    predictions, variance = (
        OrdinaryKriging(_variogram(model))
        .fit(coords, values)
        .predict(targets, return_variance=True)
    )

    expected = _core.ordinary_kriging_predict(
        coords, values, targets, _parameters(), model
    )
    expected_variance = _core.kriging_variance(coords, targets, _parameters(), model)
    np.testing.assert_allclose(predictions, expected, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(variance, expected_variance, rtol=1e-7, atol=1e-10)


@pytest.mark.parametrize("model", MODELS)
def test_simple_kriging_matches_per_target_solution(data, model):
    coords, values, targets = data
    mean = float(values.mean())

    predictions, variance = (
        SimpleKriging(_variogram(model), mean=mean)
        .fit(coords, values)
        .predict(targets, return_variance=True)
    )

    expected = _core.simple_kriging_predict(
        coords, values, targets, _parameters(), model, mean
    )
    expected_variance = _direct_variance(coords, targets, model)
    np.testing.assert_allclose(predictions, expected, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(variance, expected_variance, rtol=1e-7, atol=1e-10)


@pytest.mark.parametrize("trend", ["linear", "quadratic"])
@pytest.mark.parametrize("model", MODELS)
def test_universal_kriging_matches_per_target_solution(data, model, trend):
    coords, values, targets = data

    predictions, variance = (
        UniversalKriging(_variogram(model), trend=trend)
        .fit(coords, values)
        .predict(targets, return_variance=True)
    )

    expected = _core.universal_kriging_predict(
        coords, values, targets, _parameters(), model, trend
    )
    expected_variance = _direct_variance(coords, targets, model, trend=trend)
    np.testing.assert_allclose(predictions, expected, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(variance, expected_variance, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("model", MODELS)
def test_variance_orders_simple_below_ordinary_below_universal(data, model):
    # Knowing the mean removes uncertainty and estimating a trend adds it, so at
    # the same locations simple kriging's variance is the smallest and universal
    # kriging's the largest. Both used to report the ordinary kriging variance.
    coords, values, targets = data
    variogram = _variogram(model)
    away_from_samples = targets[:150]

    def variance(estimator):
        fitted = estimator.fit(coords, values)
        return fitted.predict(away_from_samples, return_variance=True)[1]

    simple = variance(SimpleKriging(variogram, mean=0.0))
    ordinary = variance(OrdinaryKriging(variogram))
    universal = variance(UniversalKriging(variogram, trend="linear"))

    assert np.all(simple <= ordinary + 1e-10)
    assert np.all(ordinary <= universal + 1e-10)
    assert simple.sum() < ordinary.sum() < universal.sum()


@pytest.mark.parametrize("model", MODELS)
def test_anisotropic_kriging_matches_direct_solution(data, model):
    coords, values, targets = data
    coords, values, targets = coords[:40], values[:40], targets[:30]
    kriging = AnisotropicKriging(_anisotropic_variogram(model)).fit(coords, values)

    predictions, variance = kriging.predict(targets, return_variance=True)

    def distances(first, second):
        return np.array(
            [[kriging._anisotropic_distance(p, q) for q in second] for p in first]
        )

    n_samples = len(coords)
    system = np.ones((n_samples + 1, n_samples + 1))
    system[n_samples, n_samples] = 0.0
    system[:n_samples, :n_samples] = _covariance(distances(coords, coords), model)
    rhs = np.ones((len(targets), n_samples + 1))
    rhs[:, :n_samples] = _covariance(distances(targets, coords), model)
    solution = np.linalg.solve(system, rhs.T)

    expected = values @ solution[:n_samples]
    expected_variance = np.maximum(SILL - np.einsum("ij,ji->i", rhs, solution), 0.0)
    np.testing.assert_allclose(predictions, expected, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(variance, expected_variance, rtol=1e-7, atol=1e-10)


def test_predictions_follow_parameters_changed_after_fit(data):
    coords, values, targets = data
    variogram = _variogram("exponential")
    kriging = OrdinaryKriging(variogram).fit(coords, values)
    kriging.predict(targets)

    variogram.range_ = 6.0

    expected = _core.ordinary_kriging_predict(
        coords, values, targets, np.array([NUGGET, SILL, 6.0]), "exponential"
    )
    np.testing.assert_allclose(
        kriging.predict(targets), expected, rtol=1e-8, atol=1e-10
    )


@pytest.mark.parametrize(
    "build",
    [
        lambda: OrdinaryKriging(_variogram("exponential")),
        lambda: SimpleKriging(_variogram("exponential"), mean=1.0),
        lambda: UniversalKriging(_variogram("exponential"), trend="linear"),
        lambda: AnisotropicKriging(_anisotropic_variogram("exponential")),
    ],
    ids=["ordinary", "simple", "universal", "anisotropic"],
)
def test_duplicate_locations_are_reported_at_fit(build):
    coords = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [2.0, 0.5]])
    values = np.array([1.0, 1.1, 2.0, 1.5])

    with pytest.raises(ValueError, match="Singular covariance matrix"):
        build().fit(coords, values)


def test_results_do_not_depend_on_how_targets_are_split(data):
    # Each target is computed independently. Variance from an inverted system
    # moved by up to 4e-5 with the number of targets computed together.
    coords, values, targets = data
    kriging = OrdinaryKriging(_variogram("gaussian")).fit(coords, values)
    whole, whole_variance = kriging.predict(targets, return_variance=True)

    parts = [
        kriging.predict(part, return_variance=True)
        for part in np.array_split(targets, 7)
    ]

    np.testing.assert_array_equal(np.concatenate([p for p, _ in parts]), whole)
    np.testing.assert_array_equal(np.concatenate([v for _, v in parts]), whole_variance)


def test_lu_factors_solve_like_numpy():
    rng = np.random.default_rng(8)
    matrix = rng.normal(size=(9, 9))
    # A zero on the leading diagonal forces a row swap.
    matrix[0, 0] = 0.0
    rhs = rng.normal(size=9)

    lu, permutation = _core.lu_factorize(matrix)

    np.testing.assert_allclose(
        _core.lu_solve(lu, permutation, rhs), np.linalg.solve(matrix, rhs), rtol=1e-10
    )
    lower = np.tril(lu, -1) + np.eye(9)
    upper = np.triu(lu)
    np.testing.assert_allclose(lower @ upper, matrix[permutation], atol=1e-12)


def test_lu_factorize_reports_a_singular_matrix():
    matrix = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [0.0, 1.0, 5.0]])

    with pytest.raises(ValueError, match="Singular covariance matrix"):
        _core.lu_factorize(matrix)


@pytest.mark.parametrize("model", MODELS)
def test_covariance_between_matches_the_model(model):
    rng = np.random.default_rng(5)
    first = rng.uniform(0, 5, size=(12, 2))
    # Two rows repeat points of the first set, so some distances are exactly zero.
    second = np.vstack([rng.uniform(0, 5, size=(9, 2)), first[:2]])

    result = _core.covariance_between(first, second, _parameters(), model)

    distances = np.linalg.norm(first[:, None, :] - second[None, :, :], axis=-1)
    # Near the range, covariances are close to zero, and NumPy and Rust round
    # them differently in the last bit.
    np.testing.assert_allclose(
        result, _covariance(distances / RANGE, model), rtol=1e-12, atol=1e-14
    )


def test_dual_kriging_predict_rejects_weights_of_the_wrong_length():
    with pytest.raises(ValueError, match="one entry per known coordinate"):
        _core.dual_kriging_predict(
            np.zeros((3, 2)),
            np.zeros((2, 2)),
            _parameters(),
            "exponential",
            np.zeros(2),
        )
