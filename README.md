# pygeostats

**Geostatistics for Python with a Rust-accelerated core.** Variograms, kriging,
point-pattern analysis and spatial autocorrelation, behind a familiar `fit` /
`predict` API that accepts NumPy arrays, pandas DataFrames and GeoPandas
GeoDataFrames.

[![PyPI](https://img.shields.io/pypi/v/pygeostats)](https://pypi.org/project/pygeostats/)
[![Python versions](https://img.shields.io/pypi/pyversions/pygeostats)](https://pypi.org/project/pygeostats/)
[![Development status](https://img.shields.io/pypi/status/pygeostats)](https://pypi.org/project/pygeostats/)
[![License: MIT](https://img.shields.io/pypi/l/pygeostats)](https://github.com/DiogoRibeiro7/pygeostats/blob/main/LICENSE)
[![CI](https://github.com/DiogoRibeiro7/pygeostats/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/DiogoRibeiro7/pygeostats/actions/workflows/ci.yml)
[![Docs](https://github.com/DiogoRibeiro7/pygeostats/actions/workflows/docs.yml/badge.svg?branch=main)](https://diogoribeiro7.github.io/pygeostats/)
[![Rust core: PyO3](https://img.shields.io/badge/core-Rust%20%2B%20PyO3-dea584?logo=rust)](https://pyo3.rs)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**[Documentation](https://diogoribeiro7.github.io/pygeostats/)** ·
[Quickstart](https://diogoribeiro7.github.io/pygeostats/getting-started/quickstart/) ·
[API reference](https://diogoribeiro7.github.io/pygeostats/reference/variogram/) ·
[Known limitations](https://diogoribeiro7.github.io/pygeostats/known-limitations/) ·
[Changelog](https://github.com/DiogoRibeiro7/pygeostats/blob/main/CHANGELOG.md) ·
[Issues](https://github.com/DiogoRibeiro7/pygeostats/issues)

> **Alpha release.** `0.1.0a1` is on PyPI as a pre-release. The API may still
> change, and some features are unfinished or unreliable: read the
> [known limitations](https://diogoribeiro7.github.io/pygeostats/known-limitations/)
> before relying on anisotropy analysis.

## Highlights

- **A compiled core.** Distances, empirical variograms, model fitting and kriging
  run in Rust, built with PyO3 and maturin.
- **The full workflow.** Estimate a variogram, fit a model, krige with prediction
  variance, and cross-validate the result, all in one package.
- **Honest fits.** When the data does not constrain a variogram model, `fit()` says
  so through `converged_` and `warnings_`, instead of returning a range as though
  it were reliable.
- **Beyond kriging.** Point-pattern statistics, spatial clustering and spatial
  autocorrelation share the same inputs.
- **Wheels for every major platform.** One stable-ABI wheel per platform covers
  Python 3.11 to 3.14, with no Rust toolchain needed to install.
- **Tested documentation.** Every example in the documentation and in this README
  runs as part of the test suite.

## Installation

```bash
pip install pygeostats
```

pygeostats needs Python 3.11 or newer. While no stable release exists, pip installs
the pre-release; after one does, use `pip install --pre pygeostats` to get
pre-releases.

Optional extras:

| Extra | Adds | For |
|-------|------|-----|
| `plotting` | plotly | interactive plots, with `backend="plotly"` |
| `progress` | tqdm | progress bars in the parallel executor |
| `approx` | annoy | approximate neighbour search |

```bash
pip install "pygeostats[plotting]"
```

Wheels are published for:

| Platform | Architectures |
|----------|---------------|
| Linux (manylinux2014) | x86_64, aarch64 |
| macOS | x86_64 (10.12+), arm64 (11+) |
| Windows | x86_64 |

Elsewhere, pip builds from the source distribution, which needs a
[Rust toolchain](https://rustup.rs).

## Quick start

Estimate a variogram from scattered samples, fit a model, and krige onto a grid:

```python
import numpy as np
from pygeostats.kriging import OrdinaryKriging
from pygeostats.variogram import EmpiricalVariogram, Variogram

rng = np.random.default_rng(0)
coords = rng.uniform(0, 10, size=(100, 2))
values = np.sin(coords[:, 0]) + np.cos(coords[:, 1]) + rng.normal(0, 0.1, 100)

# 1. Empirical variogram: semivariance by distance bin
empirical = EmpiricalVariogram(coords, values, n_bins=12).compute()

# 2. Fit a model, and check that the fit can be trusted
model = Variogram(model="exponential")
model.fit(empirical.distances_, empirical.gamma_, weights=empirical.counts_)
print(model.converged_, model.nugget_, model.sill_, model.range_)

# 3. Krige onto a grid, with the prediction variance
xs = np.linspace(0, 10, 40)
grid_x, grid_y = np.meshgrid(xs, xs)
targets = np.column_stack([grid_x.ravel(), grid_y.ravel()])

kriging = OrdinaryKriging(model).fit(coords, values)
predictions, variance = kriging.predict(targets, return_variance=True)
```

Check how well the workflow predicts, holding out whole regions at a time:

```python
from pygeostats.validation import (
    default_kriging_builder,
    default_variogram_builder,
    spatial_kfold_cross_validation,
)

cv = spatial_kfold_cross_validation(
    coords,
    values,
    default_variogram_builder("exponential"),
    default_kriging_builder(),
    n_splits=5,
    random_state=0,
)
print(cv.summary())  # {"rmse": ..., "r2": ...}
```

Point patterns and spatial autocorrelation work directly on coordinates and values:

```python
from pygeostats import (
    morans_i,
    ripley_l_function,
    simulate_poisson_process,
    spatial_weights_knn,
)

# Ripley's L for a random pattern: close to r at every radius
points = simulate_poisson_process(200, bounds=(0.0, 1.0, 0.0, 1.0), random_state=0)
radii = np.linspace(0.01, 0.15, 15)
l_values = ripley_l_function(points, radii, area=1.0)

# Moran's I for the samples above, with a permutation test
weights = spatial_weights_knn(coords, k=8)
print(morans_i(values, weights, permutations=999, random_state=0))
```

The [Quickstart](https://diogoribeiro7.github.io/pygeostats/getting-started/quickstart/)
walks through a complete workflow, including plots, and
[`examples/basic_kriging.py`](https://github.com/DiogoRibeiro7/pygeostats/blob/main/examples/basic_kriging.py)
and
[`examples/variogram_fitting.py`](https://github.com/DiogoRibeiro7/pygeostats/blob/main/examples/variogram_fitting.py)
are complete scripts.

## Features

| Module | What it covers | Guide |
|--------|----------------|-------|
| `pygeostats.variogram` | Empirical and directional variograms; exponential, spherical and Gaussian models with a fit report; anisotropy detection; streaming and memory-mapped variograms | [Variograms](https://diogoribeiro7.github.io/pygeostats/guide/variograms/), [Anisotropy](https://diogoribeiro7.github.io/pygeostats/guide/anisotropy/) |
| `pygeostats.kriging` | Ordinary, simple, universal and anisotropic kriging, with prediction variance; neighbour search | [Kriging](https://diogoribeiro7.github.io/pygeostats/guide/kriging/), [Large datasets](https://diogoribeiro7.github.io/pygeostats/guide/large-data/) |
| `pygeostats.point_patterns` | Nearest neighbours; Ripley's K and L; G, F and pair correlation functions; DBSCAN; kernel density; Gi* hot spots; Poisson, Cox and marked process simulation; segregation indices | [Point patterns](https://diogoribeiro7.github.io/pygeostats/guide/point-patterns/) |
| `pygeostats.spatial_autocorrelation` | Moran's I and Geary's C, global and local; Getis-Ord G and Gi*; k-nearest-neighbour, distance-band and inverse-distance weights | [Spatial autocorrelation](https://diogoribeiro7.github.io/pygeostats/guide/autocorrelation/) |
| `pygeostats.validation` | Leave-one-out, spatial k-fold and block cross-validation; model selection by AIC, BIC or leave-one-out; residual diagnostics | [Validation](https://diogoribeiro7.github.io/pygeostats/guide/validation/) |
| `pygeostats.utils` | matplotlib and plotly plots of variograms, kriging surfaces and diagnostics | [Plotting](https://diogoribeiro7.github.io/pygeostats/guide/plotting/) |

## Status and known limitations

pygeostats is alpha software. The variogram, kriging, point-pattern,
autocorrelation and validation workflows are tested and documented, but:

- **Fitting an anisotropic model from directional variograms is not implemented.**
  `AnisotropicKriging` works with parameters set by hand, and measures its rotation
  angle clockwise, unlike `DirectionalVariogram`.
- **Anisotropy estimates are rough.** The ratio from `detect_anisotropy()` runs low,
  and on a few hundred samples the estimated axis can be tens of degrees off.
- **Matérn models cannot be fitted**, only exponential, spherical and Gaussian ones.
- **`StreamingVariogramBuilder.add_pairs()` needs `weights`**, although it is
  documented as optional.
- **Type annotations are incomplete**, and mypy runs as an advisory CI step.

The [known limitations](https://diogoribeiro7.github.io/pygeostats/known-limitations/)
page has the details and workarounds.

## Development

A Rust toolchain is needed to build from source. The compiler version is pinned in
`rust-toolchain.toml`, and rustup fetches it automatically.

```bash
git clone https://github.com/DiogoRibeiro7/pygeostats.git
cd pygeostats
pip install -e ".[dev,test]"

pytest tests/
black --check src/python/ tests/
ruff check src/python/ tests/
cargo test --no-default-features --features parallel
```

The [development guide](https://diogoribeiro7.github.io/pygeostats/development/)
explains the checks and how to build the documentation.

## Contributing

Bug reports, questions and pull requests are welcome in the
[issue tracker](https://github.com/DiogoRibeiro7/pygeostats/issues). See
[CONTRIBUTING.md](https://github.com/DiogoRibeiro7/pygeostats/blob/main/CONTRIBUTING.md)
for the workflow, and the
[Code of Conduct](https://github.com/DiogoRibeiro7/pygeostats/blob/main/CODE_OF_CONDUCT.md).

## Name

This project was previously called `pyspatialstats`. It was renamed because that name
belongs to [an existing, actively maintained package](https://github.com/jasperroebroek/pyspatialstats)
on PyPI by Jasper Roebroek. The two are unrelated.

## License

MIT. See [LICENSE](https://github.com/DiogoRibeiro7/pygeostats/blob/main/LICENSE).
