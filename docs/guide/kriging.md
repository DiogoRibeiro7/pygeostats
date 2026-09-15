# Kriging

Kriging predicts values at unsampled locations as weighted combinations of the
samples, with weights derived from a variogram model, and gives a variance for each
prediction. Every estimator in `pygeostats.kriging` works the same way: construct it
with a fitted `Variogram`, call `fit` with the sample coordinates and values, then
call `predict`.

The examples on this page share a dataset, a fitted model and a grid of target
locations:

```python
import numpy as np
from pygeostats.variogram import EmpiricalVariogram, Variogram

rng = np.random.default_rng(42)
coords = rng.uniform(0, 10, size=(80, 2))
distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
values = rng.multivariate_normal(
    np.zeros(len(coords)), np.exp(-distances / 2.0) + 0.05 * np.eye(len(coords))
)

empirical = EmpiricalVariogram(coords, values, n_bins=10).compute()
model = Variogram(model="exponential").fit(
    empirical.distances_, empirical.gamma_, weights=empirical.counts_
)

xs = np.linspace(0, 10, 25)
grid_x, grid_y = np.meshgrid(xs, xs)
targets = np.column_stack([grid_x.ravel(), grid_y.ravel()])
```

## Ordinary kriging

Ordinary kriging assumes the mean is constant but unknown. It is the usual default.

```python
from pygeostats.kriging import OrdinaryKriging

kriging = OrdinaryKriging(model).fit(coords, values)

predictions = kriging.predict(targets)
predictions, variance = kriging.predict(targets, return_variance=True)
prediction_grid = predictions.reshape(grid_x.shape)
```

The estimators raise `ValueError` for a variogram that has not been fitted.

`score()` returns the coefficient of determination, $R^2$, of the predictions at the
given locations. On the samples the model was fitted to, it says little, so score on
held-out samples:

```python
held_out = rng.uniform(0, 10, size=(10, 2))
held_out_values = kriging.predict(held_out) + rng.normal(0, 0.1, size=10)

print(kriging.score(held_out, held_out_values))
```

For a proper estimate of prediction error, see
[Validation and model selection](validation.md).

## Simple kriging

Simple kriging takes the mean as known, which suits data that has been standardised
or detrended:

```python
from pygeostats.kriging import SimpleKriging

simple = SimpleKriging(model, mean=0.0).fit(coords, values)
simple_predictions, simple_variance = simple.predict(targets, return_variance=True)
```

## Universal kriging

Universal kriging models the mean as a polynomial trend in the coordinates. `trend`
is `"linear"`, `"quadratic"` or `"auto"`, the default, which fits both and keeps the
one with the lower AIC:

```python
from pygeostats.kriging import UniversalKriging

universal = UniversalKriging(model, trend="auto").fit(coords, values)

print(universal.trend_)      # the trend that was used
print(universal.trend_aic_)  # AIC of each candidate trend
```

## Anisotropic kriging

`AnisotropicKriging` stretches distances so that correlation reaches further along
one axis than across it. Its variogram carries five parameters: nugget, sill, the
range along the major axis, the range across it, and a rotation angle in radians.

!!! warning "Parameters are set by hand"

    Fitting these five parameters from directional variograms is not implemented yet,
    so they are assigned directly and the variogram is marked as fitted.
    [Directional variograms and anisotropy](anisotropy.md) covers where values for
    them can come from.

```python
from pygeostats.kriging import AnisotropicKriging

major_direction = 30.0  # degrees counter-clockwise from the x-axis

anisotropic_model = Variogram(model="exponential")
anisotropic_model.parameters = np.array(
    [0.05, 1.0, 3.0, 1.0, -np.radians(major_direction)]
)
anisotropic_model.is_fitted_ = True

anisotropic = AnisotropicKriging(anisotropic_model).fit(coords, values)
aniso_predictions, aniso_variance = anisotropic.predict(targets, return_variance=True)
print(anisotropic.get_anisotropy_info())
```

!!! note "The rotation angle is negated"

    `AnisotropicKriging` puts the major axis at minus the rotation angle, so it
    measures clockwise from the x-axis. `DirectionalVariogram` measures directions
    counter-clockwise. To put the major axis at 30 degrees counter-clockwise, as
    above, pass `-np.radians(30)`.

The ranges are read the same way as for isotropic models: for the exponential and
Gaussian models they are scale parameters, not the distances at which the sill is
reached. With equal major and minor ranges, `AnisotropicKriging` gives the same
predictions as `OrdinaryKriging`.

## Coordinates and values

Coordinates can be a NumPy array of shape `(n_samples, n_dimensions)`, a pandas
DataFrame or a GeoPandas GeoDataFrame of points. Values can be an array or a pandas
Series.

```python
import geopandas as gpd
import pandas as pd

frame = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "value": values})
from_frame = OrdinaryKriging(model).fit(frame[["x", "y"]], frame["value"])

points = gpd.GeoDataFrame(
    {"value": values}, geometry=gpd.points_from_xy(coords[:, 0], coords[:, 1])
)
from_points = OrdinaryKriging(model).fit(points, points["value"])
```

!!! warning "DataFrame columns are taken by position"

    A DataFrame's first two columns are used as the coordinates, whatever their
    names. Select the coordinate columns explicitly, as above.

Predicting on large grids is covered in [Large datasets](large-data.md).
