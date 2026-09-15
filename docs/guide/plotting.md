# Plotting

The plotting functions in `pygeostats.utils` each draw one figure from arrays and
return it: a matplotlib `Figure` by default, or a plotly `Figure` with
`backend="plotly"`. They share three more arguments:

- `title`, the figure title
- `save_path`, a file to write the figure to; for matplotlib, the extension picks
  the format
- `show`, which displays the figure and defaults to `True`; pass `show=False` to
  only return it

The examples share a dataset, a fitted model and kriging predictions on a grid:

```python
import numpy as np
from pygeostats.kriging import OrdinaryKriging
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

xs = np.linspace(0, 10, 20)
grid_x, grid_y = np.meshgrid(xs, xs)
kriging = OrdinaryKriging(model).fit(coords, values)
predictions, variance = kriging.predict(
    np.column_stack([grid_x.ravel(), grid_y.ravel()]), return_variance=True
)
prediction_grid = predictions.reshape(grid_x.shape)
variance_grid = variance.reshape(grid_x.shape)
```

## Variograms

```python
from pygeostats.utils import plot_variogram, plot_variogram_cloud
from pygeostats.validation import variogram_cloud

lags = np.linspace(0, empirical.max_distance, 50)
plot_variogram(
    empirical.distances_,
    empirical.gamma_,
    counts=empirical.counts_,
    model_curves=[("exponential", lags, model.predict(lags))],
    title="Empirical variogram",
    show=False,
)

cloud = variogram_cloud(coords, values)
plot_variogram_cloud(cloud["distance"], cloud["gamma"], show=False)
```

`model_curves` takes any number of `(label, distances, semivariance)` tuples, and
`confidence_interval` a `(lower, upper)` pair of arrays to shade.

`EmpiricalVariogram.plot()` draws onto a matplotlib `Axes`, a new one or one passed
as `ax`, and returns it:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
empirical.plot(ax=ax)
```

## Directional variograms

```python
from pygeostats.utils import plot_anisotropy_rose, plot_directional_variograms
from pygeostats.variogram import DirectionalVariogram

directional = DirectionalVariogram(coords, values, n_bins=8).compute()

plot_directional_variograms(directional.directional_summary(), show=False)
plot_anisotropy_rose(*directional.anisotropy_rose_data(), show=False)
```

## Kriging results

```python
from pygeostats.utils import (
    plot_kriging_cross_section,
    plot_kriging_results,
    plot_kriging_uncertainty,
)

plot_kriging_results(
    xs, xs, prediction_grid, sample_coords=coords, sample_values=values, show=False
)
plot_kriging_uncertainty(xs, xs, variance_grid, show=False)
plot_kriging_cross_section(xs, prediction_grid[10], show=False)
```

The grid functions take the x and y coordinates of the grid's columns and rows,
and a prediction array of shape `(len(y), len(x))`, which is what reshaping
predictions made on a `np.meshgrid` grid gives. The cross-section plots one row of
it against distance along the row, with `observations` optionally overlaid.

## Diagnostics

```python
from pygeostats.utils import (
    plot_prediction_comparison,
    plot_residuals_qq,
    plot_spatial_correlation,
)
from pygeostats.validation import (
    default_kriging_builder,
    default_variogram_builder,
    spatial_kfold_cross_validation,
)

cv = spatial_kfold_cross_validation(
    coords,
    values,
    default_variogram_builder(),
    default_kriging_builder(),
    n_splits=5,
    random_state=0,
)
plot_prediction_comparison(cv.predictions, values, show=False)
plot_residuals_qq(cv.residuals, show=False)
plot_spatial_correlation(lags, model.covariance(lags) / model.sill_, show=False)
```

## Saving

```python
plot_variogram(empirical.distances_, empirical.gamma_, save_path="variogram.png", show=False)
```

## Interactive plots with plotly

The plotly backend needs the `plotting` extra:

```bash
pip install "pygeostats[plotting]"
```

```py
figure = plot_variogram(empirical.distances_, empirical.gamma_, backend="plotly", show=False)
figure.show()
```
