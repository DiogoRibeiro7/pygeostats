# Roadmap

This is the plan for pygeostats from its current alpha releases to 1.0.0. The
milestones are ordered, not dated: each is released when its work is done, and
items can move between milestones as the work shows what is needed. Suggestions
and pull requests are welcome in the
[issue tracker](https://github.com/DiogoRibeiro7/pygeostats/issues).

## How versions work until 1.0

- **Releases before 1.0 can change the API.** From 0.2.0, anything removed or
  renamed is deprecated first, with a warning, for at least one minor release.
- **Releases can be published as pre-releases first**, such as `0.2.0a1`. pip
  installs pre-releases only while no stable release exists.
- **1.0.0 starts Semantic Versioning:** no breaking changes without a new major
  version.

## Where things stand: 0.1.0a2

What works today is described in the
[documentation](https://diogoribeiro7.github.io/pygeostats/):

- **Variograms.** Empirical, directional and streaming variograms, and
  exponential, spherical and Gaussian models fitted with a convergence report.
- **Kriging.** Ordinary, simple, universal and anisotropic kriging. Each
  estimator factorises its system once at `fit` and predicts in parallel in the
  Rust core.
- **Other analysis.** Point patterns, spatial autocorrelation, cross-validation,
  model selection and plotting.
- **Packaging.** Wheels for Linux, macOS and Windows covering Python 3.11 to
  3.14, and documentation examples that run as tests.

The [known limitations](https://diogoribeiro7.github.io/pygeostats/known-limitations/)
page lists what does not work yet. Each limitation on it is assigned to a
milestone below, except the note that fits which cannot be trusted are reported,
which is intended behaviour.

## Released

### 0.1.0a1 (2026-09-15)

- First release on PyPI, as an alpha pre-release, with wheels for five platforms
  built against the stable ABI.
- Variogram fitting that reaches the least-squares optimum and reports fits it
  cannot trust.
- Anisotropy detection that fits an ellipse through the directional ranges.

### 0.1.0a2 (2026-09-15)

- A documentation site with user guides, an API reference and tested examples.
- Kriging factorised once at `fit`: ordinary kriging of 4,000 targets from 2,000
  samples fell from 14 s to 0.01 s.
- Correct results from `ParallelKrigingExecutor`, and each kriging estimator's own
  variance.
- `OrdinaryKriging.predict_parallel()` deprecated.

The [changelog](https://github.com/DiogoRibeiro7/pygeostats/blob/main/CHANGELOG.md)
has the details.

## Planned

### 0.2.0: anisotropy end to end

**Goal:** go from directional variograms to anisotropic kriging without setting
parameters by hand, with estimates whose accuracy is measured.

- **Fit anisotropic models from directional variograms.** Implement
  `DirectionalVariogram.estimate_initial_parameters()` and
  `create_anisotropic_variogram_from_directional()`, which raise
  `NotImplementedError`. Their two strict `xfail` tests become ordinary tests.
- **Measure and improve the anisotropy estimates.** On samples of a few hundred
  points:
  - the axis from `detect_anisotropy()` is often tens of degrees off, and its
    ratio runs low;
  - `InitializationEnsemble` blends the layout of the samples into its angle;
  - `RangeInitializer` returns starting values rather than model ranges.

  Benchmark them on simulated fields with known anisotropy, publish the errors,
  and set accuracy thresholds in the tests.
- **One angle convention.** `AnisotropicKriging` measures its rotation clockwise
  and `DirectionalVariogram` counter-clockwise. Move to counter-clockwise
  everywhere, deprecating the old convention first.
- **Matérn models.** Fit them, including their smoothness, or stop accepting
  `model="matern"`.

**Done when** the anisotropy guide and example run without hand-set parameters,
and the accuracy results are in the documentation.

### 0.3.0: large datasets

**Goal:** krige tens of thousands of samples with a documented, benchmarked
workflow.

- **Neighbourhood kriging.** Predict each target from its nearest samples, rather
  than building local neighbourhoods by hand as the large-datasets guide does
  now. The Rust core already has neighbour-based ordinary kriging,
  `ordinary_kriging_predict_neighbors`, which nothing calls.
- **Remove `OrdinaryKriging.predict_parallel()`**, deprecated in 0.1.0a2.
- **Fork safety.** Kriging can hang in a process forked after pygeostats has
  predicted, because the Rust core's thread pool does not survive `fork`. Make
  the core safe to use after a fork, or detect the fork and fall back.
- **Faster variance.** The variance needs a solve for every target: 33 s for
  16,000 targets from 4,000 samples.
- **The parallel executor.** `ParallelKrigingExecutor` prints its progress to
  standard output, and rarely helps now that `predict` runs in parallel. Report
  progress through logging, and decide whether the executor stays.
- **Variograms from subsamples.** Streaming variograms bound memory but still
  visit every pair. Add a sampled option for very large datasets.

**Done when** the large-datasets guide shows neighbourhood kriging of a large
dataset, with published timings.

### 0.4.0: more kriging methods

From the original roadmap, not yet started:

- Block kriging.
- Indicator kriging.
- Co-kriging basics.

### 0.5.0: spatial regression

From the original roadmap, not yet started:

- Spatial lag and spatial error models.
- Geographically weighted regression.
- Diagnostics and tests for these models.

### 0.6.0: validation against reference implementations

**Goal:** results that agree with established tools, checked by the tests.

- **R gstat.** `tests/gstat_reference.py` emulates gstat in Python instead of
  using gstat's own output. Generate reference results with gstat, store them as
  test fixtures, and test against them. The original targets stand: fitted
  variograms matching gstat's with an R² above 0.99, and ordinary kriging
  predictions within 1% of gstat's.
- **Other references.** Where an established implementation exists, such as
  PySAL for spatial autocorrelation, compare against it the same way.
- **Reproducible benchmarks.** Publish benchmarks together with the environment
  they ran in, and measure the original goal of a tenfold speed-up over pure-Python
  implementations. CONTRIBUTING asks for results from
  `benchmarks/benchmark_variogram.py` and `benchmarks/compare_with_gstat.py`,
  which do not exist yet.

### 0.7.0: code health

- **Type annotations.** mypy reports 143 errors in 14 files and runs as an
  advisory CI step. Clear them, make mypy a required check, and ship a `py.typed`
  marker.
- **Untested modules.** `pygeostats.acceleration` (GPU) and
  `pygeostats.optimization` (memory-efficient kriging) have no tests, and
  `pygeostats.kriging.neighbor_search` runs only in a documentation example. Test
  them, or remove what should not ship. GPU support stays only if it is tested and
  measurably faster.
- **Coverage.** CI measures test coverage but only stores the report. Publish it,
  and keep it from falling.
- **Clean-up.** Remove the Rust prediction functions that factorising at `fit`
  replaced, once the tests no longer use them as references, and lint
  `benchmarks/` and `examples/` in CI.

### 0.8.0: API review

**Goal:** a public API consistent enough to freeze.

- **Bounds.** Point-pattern functions take `(xmin, xmax, ymin, ymax)`, and
  `spatial_tiles` takes `(xmin, ymin, xmax, ymax)`. Use one order.
- **DataFrame coordinates.** A DataFrame's first two columns are taken as the
  coordinates, whatever their names. Let callers name the columns.
- **scikit-learn conventions.** The estimators set fitted attributes in
  `__init__`. Run scikit-learn's estimator checks, and fix or document what fails.
- **Deprecations.** Complete every removal announced in earlier releases.
- **Documentation.** Version the documentation site, and write a migration guide
  from 0.x.

### 0.9.0: release candidate

- The API is frozen: only fixes and documentation changes.
- The Python versions and platforms supported by 1.0 are confirmed.
- Every known limitation is resolved, or documented as out of scope.

### 1.0.0: stable

- A stable public API under Semantic Versioning, with deprecations lasting at least
  one minor release.
- Validation and benchmark results published in the documentation.
- A citation file, `CITATION.cff`, so that the package can be cited.

## After 1.0

Ideas, not scheduled:

- Tutorials and worked examples on real datasets.
- Closer integration with the wider geospatial ecosystem, such as GeoPandas and
  PySAL workflows.

## Contributing

To work on an item, open an issue first for anything larger than a fix, so the
approach can be agreed before the code. The workflow is in
[CONTRIBUTING.md](https://github.com/DiogoRibeiro7/pygeostats/blob/main/CONTRIBUTING.md).
