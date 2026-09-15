# Quickstart

This page walks through a complete geostatistical workflow on synthetic data:
estimate how values vary with distance, fit a model to that, interpolate with
kriging, and check the result.

## 1. Data

pygeostats works with coordinates as an array of shape `(n_samples, 2)` and values
as an array of shape `(n_samples,)`. Here the values come from a Gaussian random
field with an exponential covariance of range 2, so there is real spatial
structure to find:

```python
import numpy as np

rng = np.random.default_rng(42)
coords = rng.uniform(0, 10, size=(80, 2))

distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
covariance = np.exp(-distances / 2.0) + 0.05 * np.eye(len(coords))
values = rng.multivariate_normal(np.zeros(len(coords)), covariance)
```

Coordinates can also be a pandas DataFrame, whose first two columns are used, or a
GeoPandas GeoDataFrame of points.

## 2. Empirical variogram

The empirical variogram averages half the squared difference between values, over
pairs of samples grouped by the distance between them:

```python
from pygeostats.variogram import EmpiricalVariogram

empirical = EmpiricalVariogram(coords, values, n_bins=15)
empirical.compute()

print(empirical.distances_)  # bin centres
print(empirical.gamma_)      # semivariance in each bin
print(empirical.counts_)     # number of pairs in each bin
```

By default the bins span half the largest distance between samples. Pass
`max_distance` or explicit `bin_edges` to change that.

## 3. Fit a model

A theoretical model turns the binned estimate into a function that can be evaluated
at any distance, which kriging needs:

```python
from pygeostats.variogram import Variogram

model = Variogram(model="exponential")
model.fit(empirical.distances_, empirical.gamma_, weights=empirical.counts_)

print(f"nugget {model.nugget_:.3f}, sill {model.sill_:.3f}, range {model.range_:.3f}")
print(model.converged_, model.status_)
```

Weighting by the pair counts gives well-populated bins more say. Always check
`converged_`: when the data does not constrain the model, typically because the
variogram is still rising at the largest lag, it is `False` and `warnings_` says
why. [Variograms](../guide/variograms.md) covers the models and the fit report.

## 4. Kriging

Ordinary kriging predicts at new locations from the fitted model and the samples,
and reports a variance for each prediction:

```python
from pygeostats.kriging import OrdinaryKriging

kriging = OrdinaryKriging(model)
kriging.fit(coords, values)

xs = np.linspace(0, 10, 30)
grid_x, grid_y = np.meshgrid(xs, xs)
targets = np.column_stack([grid_x.ravel(), grid_y.ravel()])

predictions, variance = kriging.predict(targets, return_variance=True)
prediction_grid = predictions.reshape(grid_x.shape)
variance_grid = variance.reshape(grid_x.shape)
```

The variance is smallest near samples and grows with distance from them.

## 5. Check the result

Cross-validation predicts each sample from the others, which measures how well the
whole workflow generalises:

```python
from pygeostats.validation import (
    default_kriging_builder,
    default_variogram_builder,
    leave_one_out_cross_validation,
)

result = leave_one_out_cross_validation(
    coords,
    values,
    default_variogram_builder("exponential"),
    default_kriging_builder(),
)
print(result.summary())
```

[Validation and model selection](../guide/validation.md) covers spatial k-fold
cross-validation, choosing between models, and residual diagnostics.

## 6. Plot

```python
from pygeostats.utils import plot_kriging_results, plot_variogram

plot_variogram(
    empirical.distances_,
    empirical.gamma_,
    counts=empirical.counts_,
    model_curves=[("exponential", xs, model.predict(xs))],
    show=False,
)
plot_kriging_results(
    xs, xs, prediction_grid, sample_coords=coords, sample_values=values, show=False
)
```

Each plotting function returns the figure; pass `save_path` to write it to a file.
See [Plotting](../guide/plotting.md).

## Next steps

- [Kriging](../guide/kriging.md) for simple, universal and anisotropic kriging
- [Large datasets](../guide/large-data.md) when the sample count grows into the
  tens of thousands
- [Point patterns](../guide/point-patterns.md) and
  [spatial autocorrelation](../guide/autocorrelation.md) for other kinds of spatial
  analysis
