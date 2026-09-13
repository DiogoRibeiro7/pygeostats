"""Regression tests for variogram model fitting.

The Levenberg-Marquardt optimiser in the Rust core is local. From a starting
range too far out it could drive the range down to its lower bound, where the
model is constant across every observed lag and the range derivative underflows
to exactly zero, then report success at a solution several times worse than the
least-squares optimum. It could also run the range off to hundreds of units.
Variogram.fit now starts from several ranges and discards solutions outside the
window the observed lags can identify.

The oracle here is an independent bounded scipy fit over the same bins, not the
generating parameters: a single realisation does not determine those, but the
least-squares optimum of its empirical variogram is well defined.
"""

import numpy as np
import pytest
from pygeostats._core import fit_variogram_model
from pygeostats.variogram.empirical import EmpiricalVariogram
from pygeostats.variogram.models import Variogram
from scipy.optimize import least_squares

from .data_generation import (
    VariogramParameters,
    _covariance_from_variogram,
    _euclidean_distance_matrix,
)
from .gstat_reference import build_reference_datasets

ADMISSIBILITY_WARNING = "No admissible fit"
# The optimiser caps at 250 iterations. Every synthetic fit here converges in
# under 50, so a fit that reaches this budget has regressed to running to the
# cap without converging.
ITERATION_BUDGET = 100


def _gamma(h, nugget, sill, range_, model):
    h = np.asarray(h, dtype=float)
    if model == "exponential":
        return nugget + (sill - nugget) * (1.0 - np.exp(-h / range_))
    if model == "gaussian":
        return nugget + (sill - nugget) * (1.0 - np.exp(-((h / range_) ** 2)))
    ratio = h / range_
    return np.where(
        h < range_, nugget + (sill - nugget) * (1.5 * ratio - 0.5 * ratio**3), sill
    )


def _synthetic_variogram(model, params, seed, n_points=500, extent=6.0):
    """Empirical variogram of one Gaussian-field realisation on a wide domain."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0.0, extent, size=(n_points, 2))
    cov = _covariance_from_variogram(_euclidean_distance_matrix(coords), model, params)
    chol = np.linalg.cholesky(cov + 1e-10 * np.eye(n_points))
    values = chol @ rng.standard_normal(n_points)
    return EmpiricalVariogram(coords, values, n_bins=20).compute()


def _fitting_bins(ev):
    """The bins Variogram.fit uses: finite, occupied, and dense where possible."""
    d, g, w = ev.distances_, ev.gamma_, ev.counts_.astype(float)
    mask = np.isfinite(d) & np.isfinite(g) & (w > 0)
    if np.any(w > 1):
        mask &= w > 1
    return d[mask], g[mask], w[mask]


def _weighted_sse(d, g, w, params, model):
    return float(np.sum(w * (_gamma(d, *params, model) - g) ** 2))


def _least_squares_optimum(d, g, w, model):
    """Bounded multi-start scipy fit over the same identifiable window."""
    floor, cap = 0.05 * d[d > 0].min(), 2.0 * d.max()
    gamma_max = float(g.max())
    best = None
    for nugget0 in (0.0, 0.2 * gamma_max):
        for range0 in (0.05, 0.15, 0.35, 0.7, 1.5, 3.0):
            result = least_squares(
                lambda q: np.sqrt(w) * (_gamma(d, *q, model) - g),
                x0=[nugget0, gamma_max, min(max(range0, floor), cap)],
                bounds=([0.0, 1e-9, floor], [np.inf, 10.0 * gamma_max, cap]),
            )
            if best is None or result.cost < best.cost:
                best = result
    return best.x


@pytest.mark.parametrize(
    "model, params",
    [
        # Seed 2 is where single-start fitting failed: the exponential collapsed
        # to range 1e-6 at 6.7x the optimal cost, the gaussian ran to range 340.
        ("exponential", VariogramParameters(nugget=0.05, sill=1.0, range=0.35)),
        ("gaussian", VariogramParameters(nugget=0.01, sill=1.2, range=0.55)),
    ],
)
def test_fit_reaches_least_squares_optimum_where_single_start_failed(model, params):
    ev = _synthetic_variogram(model, params, seed=2)
    d, g, w = _fitting_bins(ev)

    variogram = Variogram(model=model).fit(ev.distances_, ev.gamma_, weights=ev.counts_)
    fitted = (variogram.nugget_, variogram.sill_, variogram.range_)
    optimum = _least_squares_optimum(d, g, w, model)

    assert 0.05 * d[d > 0].min() <= variogram.range_ <= 2.0 * d.max()
    assert _weighted_sse(d, g, w, fitted, model) <= 1.001 * _weighted_sse(
        d, g, w, optimum, model
    )
    assert not any(ADMISSIBILITY_WARNING in msg for msg in variogram.warnings_)
    assert variogram.converged_
    assert variogram.fit_statistics_["iterations"] < ITERATION_BUDGET


def test_optimum_with_nugget_on_its_bound_converges_quickly():
    # This realisation's least-squares optimum has the nugget exactly at zero.
    # The optimizer used to solve with the pinned nugget included, clip it, and
    # reject every resulting step -- running all 250 iterations to report
    # max_iterations at the correct answer. Holding pinned parameters out of the
    # solve converges in a handful of iterations.
    ev = _synthetic_variogram(
        "exponential", VariogramParameters(nugget=0.05, sill=1.0, range=0.35), seed=0
    )
    variogram = Variogram(model="exponential").fit(
        ev.distances_, ev.gamma_, weights=ev.counts_
    )

    assert variogram.nugget_ == 0.0
    assert variogram.status_ == "succeeded"
    assert variogram.converged_
    assert variogram.fit_statistics_["iterations"] < ITERATION_BUDGET


def test_range_collapsed_to_its_floor_is_not_reported_as_success():
    # Called on the Rust core directly: Variogram.fit's multi-start never returns
    # this solution, so there is no public path to a single start. From range 2.0
    # this realisation descends to the 1e-6 range floor, where the model is
    # constant over every observed lag and the optimality test passes vacuously.
    # That used to be labelled succeeded.
    ev = _synthetic_variogram(
        "exponential", VariogramParameters(nugget=0.05, sill=1.0, range=0.35), seed=0
    )
    d, g, w = _fitting_bins(ev)
    result = fit_variogram_model(
        d, g, "exponential", np.array([0.0, 1.1 * float(g.max()), 2.0]), weights=w
    )

    assert result.parameters[2] <= 1e-6 * (1 + 1e-9)
    assert result.status == "invalid_parameters"
    assert result.converged is False


def test_fixed_range_is_left_exactly_where_it_was_fixed():
    ev = _synthetic_variogram(
        "exponential", VariogramParameters(nugget=0.05, sill=1.0, range=0.35), seed=0
    )
    # 0.9 is well away from this realisation's optimum (~0.23), so a fix that
    # was ignored would move it.
    variogram = Variogram(model="exponential", range=0.9).fit(
        ev.distances_, ev.gamma_, weights=ev.counts_, fix={"range": True}
    )
    assert variogram.range_ == 0.9


def test_unidentifiable_variogram_is_flagged_rather_than_trusted():
    # 28 points whose empirical variogram is still rising at the largest lag
    # (~0.52): every start converges to a range an order of magnitude past the
    # data, so no fitted range should be presented as reliable.
    dataset = build_reference_datasets()["spherical"]
    ev = EmpiricalVariogram(dataset.coords, dataset.values, n_bins=15).compute()

    variogram = Variogram(model="spherical").fit(
        ev.distances_, ev.gamma_, weights=ev.counts_
    )

    assert variogram.converged_ is False
    assert any(ADMISSIBILITY_WARNING in msg for msg in variogram.warnings_)


def test_identifiable_variogram_is_not_flagged():
    # Guards the test above against passing because every fit is flagged.
    ev = _synthetic_variogram(
        "spherical", VariogramParameters(nugget=0.02, sill=0.8, range=0.45), seed=0
    )
    variogram = Variogram(model="spherical").fit(
        ev.distances_, ev.gamma_, weights=ev.counts_
    )
    assert not any(ADMISSIBILITY_WARNING in msg for msg in variogram.warnings_)


def test_covariance_at_zero_separation_is_the_sill():
    variogram = Variogram(model="exponential")
    variogram.nugget_, variogram.sill_, variogram.range_ = 0.124, 0.736, 0.3
    variogram.is_fitted_ = True

    assert variogram.predict(np.array([0.0]))[0] == 0.0
    assert variogram.covariance(np.array([0.0]))[0] == pytest.approx(0.736)
    # Just past zero the nugget discontinuity applies.
    assert variogram.predict(np.array([1e-12]))[0] == pytest.approx(0.124)
