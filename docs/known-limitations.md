# Known limitations

pygeostats 0.1.0a1 is an alpha release. These are the limitations known at the time
of release. Unfinished features are covered by strict `xfail` tests, so the test
suite fails if one is completed without the tests being updated.

## Directional to anisotropic kriging is not implemented

`DirectionalVariogram.estimate_initial_parameters()` and
`create_anisotropic_variogram_from_directional()` raise `NotImplementedError`. They
depend on a helper that was never written.

`InitializationEnsemble` and `RangeInitializer` from
`pygeostats.variogram.initialization` provide starting values in the meantime, and
`AnisotropicKriging` works when its variogram's five parameters are set directly and
the variogram is marked as fitted. See
[Directional variograms and anisotropy](guide/anisotropy.md).

## `OrdinaryKriging.predict_parallel` fails

`OrdinaryKriging.predict_parallel()` fails before making any prediction, whatever
arguments it is given: it constructs `ParallelKrigingExecutor`,
`ApproximateNeighborIndex` and `spatial_tiles` with arguments those APIs do not
accept. It is also the only route to the neighbour-based kriging in the Rust core.
[Large datasets](guide/large-data.md#kriging-many-locations) shows how to predict in
batches, and how to krige from local neighbourhoods by hand.
`benchmarks/kriging_parallel.py` fails for the same reason.

## `AnisotropicKriging` measures its angle clockwise

`AnisotropicKriging` rotates each separation counter-clockwise by the variogram's
fifth parameter, `rotation_angle`, before scaling it, so the major axis lies at
`-rotation_angle`: clockwise from the x-axis. `DirectionalVariogram` and
`detect_anisotropy()` measure directions counter-clockwise. Negate a direction taken
from them: `rotation_angle = -np.radians(major_direction)`.
`get_anisotropy_info()` reports the parameter as given.

## `StreamingVariogramBuilder.add_pairs` needs weights

`weights` is documented as optional, but `add_pairs()` raises `TypeError` without
it. Pass `weights=np.ones(len(distances))` for unweighted pairs.
`benchmarks/variogram_streaming.py --mode pairs` fails for this reason; its memmap
mode works.

## Matérn models cannot be fitted

`Variogram` accepts `model="matern"`, but `fit()` raises `ValueError`: the Rust core
fits exponential, spherical and Gaussian models only. Where a Matérn model is
evaluated, it is treated as exponential, which is the Matérn model with smoothness
0.5.

## The anisotropy ratio runs low

`DirectionalVariogram.detect_anisotropy()` estimates the anisotropy axis by fitting
an ellipse through the directional ranges. Its ratio runs low: about 1.4 for a 2.9:1
field, against about 1.06 for an isotropic one. Treat it as a detection statistic,
not an estimate of the true ratio. At the default `ratio_threshold` of 1.2, some
isotropic fields are flagged as anisotropic.

## Starting values are not model ranges

`RangeInitializer` returns starting values for a fit, not estimates of a model's
range parameter. Each direction's range is the first lag at which its variogram
reaches 90% of its own largest value.

## The ensemble angle mixes in the sampling layout

`InitializationEnsemble` blends the principal axis of the sampling locations into
its angle. That axis describes where samples were taken, not the field, so it can
pull the angle away from the field's axis: in one test it reported 74 degrees for an
axis at 60.

## Fits that cannot be trusted are reported, not hidden

This is behaviour rather than a defect, but it surprises people: when the empirical
variogram does not constrain the chosen model, typically because it is still rising
at the largest observed lag, `Variogram.fit` sets `converged_` to `False` and adds a
warning, instead of returning a range as though it were reliable.

## Type annotations are incomplete

mypy reports findings in first-party code. It runs as an advisory step in CI rather
than a gate.
