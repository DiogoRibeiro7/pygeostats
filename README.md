# pygeostats

[![CI](https://github.com/DiogoRibeiro7/pygeostats/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/DiogoRibeiro7/pygeostats/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Geostatistics for Python with a Rust-accelerated core: variograms, kriging,
point-pattern analysis and spatial autocorrelation.

**Documentation:** <https://diogoribeiro7.github.io/pygeostats/>

## Status

**Alpha. `0.1.0a1` is published to PyPI as a pre-release, and the package is not
ready for production use.**

The package builds and imports, and the test suite runs green. Wheels are
published for Linux and macOS on x86_64 and arm64, and for Windows on x86_64.
Known defects and unfinished features are listed under
[Known limitations](#known-limitations); read that section before relying on
directional or anisotropy analysis.

This project was previously called `pyspatialstats`, and was renamed because
that name belongs to [an existing, actively maintained package](https://github.com/jasperroebroek/pyspatialstats)
on PyPI by Jasper Roebroek. The two are unrelated.

## Features

* **Variograms** — empirical estimation, theoretical models (exponential,
  spherical and Gaussian), directional variograms and anisotropy detection
* **Kriging** — ordinary, simple and universal, with prediction variance
* **Large data** — variograms computed in streaming chunks, from memory-mapped
  input if needed, with sparse bin summaries; neighbour search for local
  kriging neighbourhoods
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

```bash
pip install pygeostats
```

Requires Python 3.11 or newer. Wheels are built against the stable ABI
(`cp311-abi3`), so one wheel per platform covers every supported Python version.
On platforms without a wheel, pip builds from the source distribution, which
needs a Rust toolchain.

`0.1.0a1` is a pre-release. pip installs it while no stable release exists; once
one does, `pip install --pre pygeostats` is needed to get pre-releases.

To build from source, a Rust toolchain is required, since the core extension is
compiled:

```bash
git clone https://github.com/DiogoRibeiro7/pygeostats.git
cd pygeostats
pip install .
```

For development, including the test dependencies:

```bash
pip install -e ".[dev,test]"
```

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

* The workflow from directional variograms to anisotropic kriging is not
  implemented. `DirectionalVariogram.estimate_initial_parameters()` and
  `create_anisotropic_variogram_from_directional()` raise
  `NotImplementedError`, and both are covered by strict `xfail` tests.
  `InitializationEnsemble` and `RangeInitializer` from
  `pygeostats.variogram.initialization` provide starting values in the meantime.
* `detect_anisotropy()` estimates the anisotropy axis, but its ratio runs low:
  about 1.4 for a 2.9:1 field, against about 1.06 for an isotropic one. Treat it
  as a detection statistic rather than an estimate of the true ratio. At the
  default `ratio_threshold` of 1.2, some isotropic fields are flagged as
  anisotropic.
* `RangeInitializer` returns starting values for a fit, not estimates of a
  model's range parameter.
* `InitializationEnsemble` blends the principal axis of the sampling locations
  into its angle, which can pull it off the field's axis: in one test it reported
  74 degrees for an axis at 60.
* `OrdinaryKriging.predict_parallel()` fails before making any prediction: it
  calls `ParallelKrigingExecutor`, `ApproximateNeighborIndex` and
  `spatial_tiles` with arguments they do not accept.
* `ParallelKrigingExecutor` can return predictions in the wrong order in thread
  mode, fails for fitted models in process mode, and can return NaN with its
  spatial strategy. Predict large grids in batches with `predict()` instead.
* `StreamingVariogramBuilder.add_pairs()` raises `TypeError` unless `weights`
  is passed, although it is documented as optional.

The point-pattern, spatial-autocorrelation and clustering modules are not
affected by any of the above and pass their tests.

Variogram fitting reports when it cannot be trusted. If the empirical variogram
does not constrain the chosen model — typically because it is still rising at
the largest observed lag — `Variogram.fit` sets `converged_` to `False` and adds
a warning, rather than returning a range as though it were reliable.

Type annotations are incomplete: mypy reports findings in first-party code and
runs as an advisory CI step rather than a gate.

## Development

```bash
pip install -e ".[dev,test]"

pytest tests/                  # 195 passed, 2 xfailed
black --check src/python/ tests/
ruff check src/python/ tests/
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings

# note the flags: extension-module tells the linker not to link libpython,
# which is correct for the cdylib but breaks a plain `cargo test` on Linux
# and macOS with undefined Python symbols
cargo test --no-default-features --features parallel
```

The Rust toolchain is pinned in `rust-toolchain.toml`, so rustup will fetch the
matching compiler automatically.

See [CHANGELOG.md](CHANGELOG.md) for release notes,
[CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow and
[ROADMAP.md](ROADMAP.md) for planned work.

## License

MIT. See [LICENSE](LICENSE).
