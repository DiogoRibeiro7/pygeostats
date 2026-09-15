# Directional variograms and anisotropy

A field is anisotropic when its correlation reaches further in some directions than
in others. `DirectionalVariogram` estimates a variogram for each of several
directions and tests for geometric anisotropy, and
`pygeostats.variogram.initialization` derives starting values for an anisotropic
model.

!!! warning "Part of this workflow is unfinished"

    Fitting an anisotropic model from directional variograms is not implemented:
    `DirectionalVariogram.estimate_initial_parameters()` and
    `create_anisotropic_variogram_from_directional()` raise `NotImplementedError`.
    The estimates on this page are also rougher than their names suggest. Read
    [Known limitations](../known-limitations.md) before relying on them.

The examples use 300 samples of a field whose correlation reaches three times
further along 30 degrees than across it:

```python
import numpy as np

rng = np.random.default_rng(0)
coords = rng.uniform(0, 10, size=(300, 2))

axis = np.radians(30.0)
delta = coords[:, None, :] - coords[None, :, :]
along = delta[..., 0] * np.cos(axis) + delta[..., 1] * np.sin(axis)
across = -delta[..., 0] * np.sin(axis) + delta[..., 1] * np.cos(axis)
scaled_distance = np.sqrt((along / 1.5) ** 2 + (across / 0.5) ** 2)
values = rng.multivariate_normal(
    np.zeros(len(coords)), np.exp(-scaled_distance) + 0.05 * np.eye(len(coords))
)
```

## Directional variograms

```python
from pygeostats.variogram import DirectionalVariogram

directional = DirectionalVariogram(
    coords,
    values,
    directions=[0, 30, 60, 90, 120, 150],
    tolerance=15.0,
    max_distance=5.0,
    n_bins=12,
).compute()

summary = directional.directional_summary()
print(summary[30.0]["bin_centers"], summary[30.0]["gamma"], summary[30.0]["counts"])
```

Directions are in degrees, counter-clockwise from the x-axis, and default to 0, 45,
90 and 135. A direction and its opposite select the same pairs, so directions from 0
to 180 degrees cover them all. A pair counts towards a direction when its orientation
is within `tolerance` degrees of it, and, if `bandwidth` is given, within that
perpendicular distance of the direction's axis. Narrower windows separate directions
better but leave fewer pairs in each bin. Only two-dimensional coordinates are
supported.

`directional_summary()` maps each direction to its `bin_centers`, `gamma`, `counts`,
and a confidence band, `ci_lower` and `ci_upper`. The same results are available as
`DirectionalResult` objects in `directional.directional_results_`.

`DirectionalVariogram.automatic_direction_set(coords, n_directions=4)` returns evenly
spaced directions starting from the principal axis of the sampling locations.

## Detecting anisotropy

```python
result = directional.detect_anisotropy()

print(result.is_anisotropic)
print(result.major_direction, result.minor_direction)  # degrees, counter-clockwise
print(result.anisotropy_ratio)
print(result.ranges)  # direction -> range
```

`detect_anisotropy()` works in three steps:

1. Each direction's range is the first lag at which its variogram reaches
   `sill_fraction`, 0.95 by default, of the largest semivariance in any direction,
   or its largest lag if it never does. Ranges are therefore bin centres.
2. With three or more directions, an ellipse is fitted through the ranges.
   `major_direction` is the angle of its long axis, in [0, 180), `minor_direction` is
   perpendicular to it, and `anisotropy_ratio` is the long axis over the short one.
   With two directions, the longer range gives the major direction.
3. `is_anisotropic` is `True` when the ratio reaches `ratio_threshold`, 1.2 by
   default, and the major range exceeds the minor one by at least
   `range_difference`, 0 by default.

!!! warning "Treat the results as rough"

    - **The ratio runs low.** Ranges are capped at the largest lag, and a noisy
      maximum sets the threshold, so the ratio understates the true one. Treat it as
      a detection statistic.
    - **The axis varies between samples.** On samples of a few hundred points from
      fields like the one above, the estimated axis was often tens of degrees from
      the true one.
    - **Look at the directional variograms.** A direction whose variogram has not
      levelled off by `max_distance` reports the largest lag as its range, and
      several such directions make the ranges tie.

The ellipse fit is available on its own, for ranges from any source:

```python
from pygeostats.variogram.initialization import estimate_rotation_angle

angle = estimate_rotation_angle(list(result.ranges), list(result.ranges.values()))
print(angle.angle_deg, angle.angle_confidence, angle.ratio, angle.significant)
```

## Starting values for an anisotropic model

```python
from pygeostats.variogram.initialization import InitializationEnsemble, RangeInitializer

starting = RangeInitializer(directional.directional_results_).estimate()
print(starting.range_major, starting.range_minor, starting.sill, starting.nugget)

ensemble = InitializationEnsemble(coords, directional.directional_results_).run()
print(ensemble.angle_deg, ensemble.ratio, ensemble.confidence)
```

- **`RangeInitializer`** returns starting values for a fit: ranges along and across
  the axis, with standard errors and bounds, and a sill and nugget. Each direction's
  range is the first lag at which its variogram reaches 90% of its own largest value,
  so these are not estimates of a model's range parameter.
- **`InitializationEnsemble`** combines those ranges with an angle and a confidence
  label. Its angle blends in the principal axis of the sampling locations, which
  describes where samples were taken rather than the field, and can pull the angle
  well away from the field's axis. Prefer `detect_anisotropy()` for the axis.

## Kriging with anisotropic parameters

Until the fitting step exists, `AnisotropicKriging` takes parameters you set:
nugget, sill, major range, minor range and a rotation angle. The estimates above are
one source of values. Since they are rough, compare candidates on held-out samples,
here against isotropic ordinary kriging:

```python
from pygeostats.kriging import AnisotropicKriging, OrdinaryKriging
from pygeostats.variogram import EmpiricalVariogram, Variogram

order = rng.permutation(len(coords))
train, test = order[:240], order[240:]

candidate = Variogram(model="exponential")
candidate.parameters = np.array(
    [
        starting.nugget,
        starting.sill,
        starting.range_major,
        starting.range_minor,
        -np.radians(result.major_direction),
    ]
)
candidate.is_fitted_ = True
anisotropic_score = (
    AnisotropicKriging(candidate)
    .fit(coords[train], values[train])
    .score(coords[test], values[test])
)

train_empirical = EmpiricalVariogram(coords[train], values[train], n_bins=12).compute()
isotropic = Variogram(model="exponential").fit(
    train_empirical.distances_, train_empirical.gamma_, weights=train_empirical.counts_
)
isotropic_score = (
    OrdinaryKriging(isotropic).fit(coords[train], values[train]).score(coords[test], values[test])
)
print(anisotropic_score, isotropic_score)
```

Two details matter here:

- **The rotation angle is negated.** `AnisotropicKriging` places the major axis at
  minus its rotation angle, measuring clockwise, while `detect_anisotropy()` measures
  counter-clockwise.
- **Estimate from training samples only for a fair comparison.** This example reuses
  the estimates from all 300 samples, for brevity.

[Kriging](kriging.md#anisotropic-kriging) describes `AnisotropicKriging` itself.

## Plots

```python
from pygeostats.utils import plot_anisotropy_rose, plot_directional_variograms

plot_directional_variograms(summary, show=False)
plot_anisotropy_rose(*directional.anisotropy_rose_data(), show=False)
```

`anisotropy_rose_data()` returns the directions and their ranges.
