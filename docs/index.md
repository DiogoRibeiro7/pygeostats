# pygeostats

Geostatistics for Python with a Rust-accelerated core: variograms, kriging,
point-pattern analysis and spatial autocorrelation.

!!! warning "Alpha release"

    pygeostats is at version 0.1.0a1, an alpha pre-release. The API can still
    change, and one workflow is unfinished. Read
    [Known limitations](known-limitations.md) before relying on directional or
    anisotropy analysis.

## Install

```bash
pip install pygeostats
```

Wheels are published for Linux, macOS and Windows, for Python 3.11 and newer. See
[Installation](getting-started/installation.md) for the details.

## A first example

Estimate a variogram from scattered samples, fit a model to it, and interpolate
with ordinary kriging:

```python
import numpy as np
from pygeostats.kriging import OrdinaryKriging
from pygeostats.variogram import EmpiricalVariogram, Variogram

rng = np.random.default_rng(0)
coords = rng.uniform(0, 10, size=(100, 2))
values = np.sin(coords[:, 0]) + np.cos(coords[:, 1]) + rng.normal(0, 0.1, 100)

empirical = EmpiricalVariogram(coords, values, n_bins=12).compute()
model = Variogram(model="exponential")
model.fit(empirical.distances_, empirical.gamma_, weights=empirical.counts_)

kriging = OrdinaryKriging(model)
kriging.fit(coords, values)
predictions, variance = kriging.predict([[5.0, 5.0], [2.5, 7.5]], return_variance=True)
```

The [Quickstart](getting-started/quickstart.md) walks through the same workflow in
more detail.

## What is in the package

| Area | What it covers | Guide |
|------|----------------|-------|
| Variograms | Empirical variograms, exponential, spherical and Gaussian models, fitting with a convergence report | [Variograms](guide/variograms.md) |
| Anisotropy | Directional variograms, anisotropy detection, starting values for anisotropic models | [Directional variograms and anisotropy](guide/anisotropy.md) |
| Kriging | Ordinary, simple, universal and anisotropic kriging, with prediction variance | [Kriging](guide/kriging.md) |
| Large datasets | Streaming and memory-mapped variograms, kriging many locations | [Large datasets](guide/large-data.md) |
| Point patterns | Ripley's K and L, G, F and pair correlation functions, clustering, kernel density, point process simulation | [Point patterns](guide/point-patterns.md) |
| Spatial autocorrelation | Moran's I, Geary's C, Getis-Ord statistics and spatial weights | [Spatial autocorrelation](guide/autocorrelation.md) |
| Validation | Leave-one-out, spatial k-fold and block cross-validation, model selection, residual diagnostics | [Validation and model selection](guide/validation.md) |
| Plotting | Variogram, kriging and diagnostic plots with matplotlib or plotly | [Plotting](guide/plotting.md) |

The distance, variogram and kriging kernels are compiled Rust extensions built
with PyO3. Coordinates can be given as NumPy arrays, pandas DataFrames or
GeoPandas GeoDataFrames of points.

The source is on [GitHub](https://github.com/DiogoRibeiro7/pygeostats), released
under the MIT License.
