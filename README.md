# pygeostats

[![CI](https://github.com/DiogoRibeiro7/pygeostats/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/DiogoRibeiro7/pygeostats/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Geostatistics for Python with a Rust-accelerated core: variograms, kriging,
point-pattern analysis and spatial autocorrelation.

## Status

**Pre-release. Not yet published to PyPI, and not ready for production use.**

The package builds and imports, the test suite runs green, and wheels build for
Linux, macOS and Windows on x86_64 and arm64. Several numerical defects are
known and tracked as strict `xfail` tests, the most serious being in variogram
model fitting — see [Known limitations](#known-limitations). Read that section
before relying on any fitted parameter.

This project was previously called `pyspatialstats`, and was renamed because
that name belongs to [an existing, actively maintained package](https://github.com/jasperroebroek/pyspatialstats)
on PyPI by Jasper Roebroek. The two are unrelated.

## Features

* **Variograms** — empirical estimation, theoretical models (exponential,
  spherical, gaussian, Matérn), directional variograms and anisotropy detection
* **Kriging** — ordinary, simple and universal, with prediction variance
* **Large data** — streaming variogram accumulators, memory-mapped input and
  sparse bin summaries; kriging with approximate neighbours, spatial tiling,
  checkpointing and resume via `predict_parallel()`
* **Point patterns** — nearest-neighbour distances, Ripley's K and L, G and F
  functions, pair correlation, DBSCAN, kernel density, Poisson and Cox process
  simulation, spatial segregation indices
* **Spatial autocorrelation** — Moran's I and Geary's C (global and local),
  Getis-Ord G and Gi*, and weight-matrix builders (kNN, distance band, inverse
  distance)
* **Validation** — leave-one-out and spatial k-fold cross-validation, AIC/BIC
  model selection, residual diagnostics
* **Rust core** — the distance, variogram and kriging kernels are compiled
  extensions built with PyO3 and maturin
* **Familiar API** — estimators expose `.fit()`, `.predict()` and `.score()`,
  and coordinate arguments accept NumPy arrays or GeoPandas GeoDataFrames

## Installation

Not on PyPI yet, so install from source. A Rust toolchain is required, since
the core extension is compiled:

```bash
git clone https://github.com/DiogoRibeiro7/pygeostats.git
cd pygeostats
pip install .
```

For development, including the test dependencies:

```bash
pip install -e ".[dev,test]"
```

Requires Python 3.9 or newer. Released wheels will target the stable ABI
(`cp39-abi3`), so one wheel per platform covers every supported Python version.

## Quick start

```python
import numpy as np
from pygeostats.variogram import EmpiricalVariogram, Variogram
from pygeostats.kriging import OrdinaryKriging

rng = np.random.default_rng(0)
coords = rng.uniform(0, 10, size=(100, 2))
values = rng.standard_normal(100)

# Empirical variogram
ev = EmpiricalVariogram(coords, values)
ev.compute()

# Fit a theoretical model
model = Variogram(model="exponential")
model.fit(ev.distances_, ev.gamma_)

# Interpolate
kriging = OrdinaryKriging(model)
kriging.fit(coords, values)
predictions = kriging.predict(coords)
```

Point patterns and spatial autocorrelation are used directly:

```python
from pygeostats import ripley_k_function, morans_i, spatial_weights_knn

radii = np.linspace(0.01, 0.25, 25)
k = ripley_k_function(coords, radii, area=100.0)

weights = spatial_weights_knn(coords, k=8)
result = morans_i(values, weights)
```

## Known limitations

These are real defects, each covered by a strict `xfail` test so that the build
fails if one is silently fixed. They are not cosmetic:

* **Variogram model fitting does not recover known parameters.** Exponential and
  gaussian fits collapse to the range lower bound; spherical diverges by three
  orders of magnitude. The Levenberg–Marquardt implementation in the Rust core
  needs investigation.
* **Kriging accuracy against reference data is poor**, a consequence of the
  above: R² against the gstat reference datasets is negative.
* `Variogram.covariance()` does not equal the sill at lag zero.
* `detect_anisotropy()` reports a major direction 45° away from the truth on a
  synthetic anisotropic field.
* Directional bandwidth filtering discards every pair rather than a subset.
* `EmpiricalVariogram.compute()` raises on a single input point instead of
  returning empty bins.
* `DirectionalVariogram.estimate_initial_parameters()` and
  `create_anisotropic_variogram_from_directional()` raise `NotImplementedError`.
  They call a helper that was never written. Use `InitializationEnsemble` or
  `RangeInitializer` from `pygeostats.variogram.initialization` instead.

The point-pattern, spatial-autocorrelation and clustering modules are not
affected by any of the above and pass their tests.

Type annotations are incomplete: mypy reports findings in first-party code and
runs as an advisory CI step rather than a gate.

## Development

```bash
pip install -e ".[dev,test]"

pytest tests/                  # 84 passed, 15 xfailed
black --check src/python/ tests/
ruff check src/python/ tests/
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow and
[ROADMAP.md](ROADMAP.md) for planned work.

## License

MIT. See [LICENSE](LICENSE).
