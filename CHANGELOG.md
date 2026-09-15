# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and version numbers
follow [PEP 440](https://peps.python.org/pep-0440/).

## [0.1.0a1] - 2026-09-15

First release, published to PyPI as an alpha pre-release. The project was
developed as `pyspatialstats` until that name turned out to belong to an
unrelated package on PyPI.

### Added

- Wheels for Linux (x86_64 and aarch64), macOS (Intel and Apple silicon) and
  Windows (x86_64), built against the stable ABI so that one wheel per platform
  covers Python 3.11 and newer (#3, #7, #11).
- Publishing to PyPI through trusted publishing, with no API tokens (#3).

### Changed

- Python 3.11 is the minimum supported version. Dependency minimums are the
  earliest releases that work on it (#11).

### Fixed

- `Variogram.fit` reaches the least-squares optimum and reports when a fit
  cannot be trusted (#5).
- Anisotropic variogram fitting reports the longer axis as `range_major`, tries
  several starts for its isotropic fallback, and no longer reports a fit that is
  flat across the data as converged (#9, #12).
- `DirectionalVariogram.detect_anisotropy` fits an ellipse through the
  directional ranges instead of taking the longest one, which put the axis 22 to
  45 degrees off in the median (#13).
- `estimate_rotation_angle` reports the major axis, with a ratio of at least 1
  (#13).
- Anisotropy angles are averaged as axes, so directions either side of 0 degrees
  no longer average to 90 (#16).
- `InitializationEnsemble.run` keeps each directional range with its angle, and
  no longer raises when a direction has no range or fewer than three do (#17).
- `EmpiricalVariogram.compute` returns empty bins for a single point instead of
  raising (#14).
- The sampling-pattern diagnostic no longer depends on the order of the points
  (#15).

### Known limitations

- The workflow from directional variograms to anisotropic kriging is not
  implemented: `DirectionalVariogram.estimate_initial_parameters()` and
  `create_anisotropic_variogram_from_directional()` raise
  `NotImplementedError`.
- The anisotropy ratio from `detect_anisotropy` runs low, and `RangeInitializer`
  returns starting values rather than model ranges. The README has details.

[0.1.0a1]: https://github.com/DiogoRibeiro7/pygeostats/releases/tag/v0.1.0a1
