# Validation and model selection

`pygeostats.validation` measures how well a variogram and kriging workflow predicts,
compares variogram models, and checks residuals.

The examples use 80 samples of a spatially correlated field:

```python
import numpy as np
from pygeostats.variogram import EmpiricalVariogram

rng = np.random.default_rng(42)
coords = rng.uniform(0, 10, size=(80, 2))
distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
values = rng.multivariate_normal(
    np.zeros(len(coords)), np.exp(-distances / 2.0) + 0.05 * np.eye(len(coords))
)

empirical = EmpiricalVariogram(coords, values, n_bins=10).compute()
```

## Cross-validation

Cross-validation holds samples out, predicts them from the rest, and compares. Each
function takes two builders, so that the whole workflow is repeated for every fold:

- `build_variogram(coords, values)` returns a fitted `Variogram` for the training
  samples.
- `build_predictor(variogram)` returns an unfitted kriging estimator.

`default_variogram_builder(model)` computes an empirical variogram with between 6
and 20 bins, depending on the number of training samples, drops empty bins, and fits
`model` weighted by pair counts. `default_kriging_builder()` builds
`OrdinaryKriging`; pass another estimator class to use that instead.

### Leave-one-out

```python
from pygeostats.validation import (
    default_kriging_builder,
    default_variogram_builder,
    leave_one_out_cross_validation,
)

loo = leave_one_out_cross_validation(
    coords, values, default_variogram_builder("exponential"), default_kriging_builder()
)
print(loo.summary())  # {"rmse": ..., "r2": ...}
```

The result also holds `predictions` and `residuals` for every sample. Leave-one-out
fits a variogram and a kriging system per sample, so it slows down quickly as the
number of samples grows.

### Spatial folds

Nearby samples are correlated, so predicting a sample while its neighbours stay in
the training set mostly measures interpolation. Holding out whole regions measures
prediction further from the data, and gives lower, more cautious scores:

```python
from pygeostats.validation import block_cross_validation, spatial_kfold_cross_validation

kfold = spatial_kfold_cross_validation(
    coords,
    values,
    default_variogram_builder(),
    default_kriging_builder(),
    n_splits=5,
    random_state=0,
)
blocks = block_cross_validation(
    coords, values, default_variogram_builder(), default_kriging_builder(), grid_shape=(2, 2)
)
print(kfold.summary(), blocks.summary())
```

`spatial_kfold_cross_validation` groups samples into spatial clusters with k-means
and holds out one cluster at a time. `block_cross_validation` divides the extent of
the samples into a regular grid of blocks and holds out one block at a time.

### Custom builders

Any callables with the same signatures work, for example to compare a spherical
model with a linear trend against the defaults:

```python
from pygeostats.kriging import UniversalKriging
from pygeostats.variogram import Variogram


def spherical_variogram(train_coords, train_values):
    train_empirical = EmpiricalVariogram(train_coords, train_values, n_bins=8).compute()
    populated = train_empirical.counts_ > 0
    return Variogram(model="spherical").fit(
        train_empirical.distances_[populated],
        train_empirical.gamma_[populated],
        weights=train_empirical.counts_[populated],
    )


def linear_trend(variogram):
    return UniversalKriging(variogram, trend="linear")


custom = spatial_kfold_cross_validation(
    coords, values, spherical_variogram, linear_trend, n_splits=5, random_state=0
)
print(custom.summary())
```

## Choosing a variogram model

`select_best_variogram_model` fits each candidate model and returns the best one's
name, the fitted model, and every candidate's score. Lower scores are better:

```python
from pygeostats.validation import select_best_variogram_model

name, best_model, scores = select_best_variogram_model(
    coords, values, candidate_models=("exponential", "spherical", "gaussian"), criterion="aic"
)
print(name, scores)
```

`criterion` is `"aic"`, `"bic"` or `"loo"`. `"loo"` scores each candidate by
leave-one-out cross-validation, which is much slower.

To score a model fitted separately, compute its AIC or BIC against the empirical
variogram it was fitted to:

```python
from pygeostats.validation import variogram_aic, variogram_bic

candidate = Variogram(model="gaussian").fit(
    empirical.distances_, empirical.gamma_, weights=empirical.counts_
)
print(variogram_aic(candidate, empirical.distances_, empirical.gamma_, weights=empirical.counts_))
print(variogram_bic(candidate, empirical.distances_, empirical.gamma_, weights=empirical.counts_))
```

Compare these scores only between models fitted to the same empirical variogram.

## Residual diagnostics

```python
from pygeostats.kriging import OrdinaryKriging
from pygeostats.validation import (
    compute_kriging_residuals,
    normality_test,
    standardized_residuals,
    variogram_cloud,
)

order = rng.permutation(len(coords))
train, test = order[:60], order[60:]

train_model = default_variogram_builder(name)(coords[train], values[train])
kriging = OrdinaryKriging(train_model).fit(coords[train], values[train])
residuals = compute_kriging_residuals(kriging, coords[test], values[test])

print(normality_test(loo.residuals))  # Shapiro-Wilk: {"statistic": ..., "pvalue": ...}
scaled = standardized_residuals(loo.residuals)

cloud = variogram_cloud(coords, values)  # {"distance": ..., "gamma": ...}
```

- `compute_kriging_residuals` returns observed minus predicted values. At the
  samples a model was fitted to, kriging reproduces the data, so compute residuals
  on held-out samples.
- `standardized_residuals` divides residuals by their standard deviation.
- `normality_test` runs a Shapiro-Wilk test.
- `variogram_cloud` gives the distance and semivariance of every pair of samples,
  before binning, which shows outliers that an empirical variogram averages away.
  It builds the full distance matrix, so keep it to a few thousand samples.

[Plotting](plotting.md#diagnostics) shows these as figures.
