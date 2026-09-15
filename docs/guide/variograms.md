# Variograms

A variogram describes how the difference between values grows with the distance
between their locations. pygeostats estimates it from samples with
`EmpiricalVariogram` and fits a model to the estimate with `Variogram`. Kriging needs
the fitted model.

The examples on this page share one dataset: 80 samples of a Gaussian random field
with an exponential covariance.

```python
import numpy as np

rng = np.random.default_rng(42)
coords = rng.uniform(0, 10, size=(80, 2))

distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
covariance = np.exp(-distances / 2.0) + 0.05 * np.eye(len(coords))
values = rng.multivariate_normal(np.zeros(len(coords)), covariance)
```

## Empirical variogram

For each distance bin, the empirical variogram is half the mean squared difference
over the pairs of samples whose separation falls in that bin:

$$
\hat\gamma(h) = \frac{1}{2\,|N(h)|} \sum_{(i, j) \in N(h)} \left(z_i - z_j\right)^2
$$

```python
from pygeostats.variogram import EmpiricalVariogram

empirical = EmpiricalVariogram(coords, values, n_bins=10).compute()

print(empirical.max_distance)  # upper edge of the last bin
print(empirical.distances_)    # bin centres
print(empirical.gamma_)        # semivariance in each bin
print(empirical.counts_)       # number of pairs in each bin
```

`compute()` returns the estimator, so construction and computation can be chained.

The bins are equally wide and run from 0 to `max_distance`, which defaults to half the
largest distance between two samples. Pairs further apart are left out: at those
distances few pairs remain, and the estimate becomes unreliable. Pass `max_distance`
to change the cut-off, or `bin_edges` for bins of different widths:

```python
edges = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.5, 6.0])
custom = EmpiricalVariogram(coords, values, bin_edges=edges).compute()
```

A bin that no pair falls into has a count of 0 and a semivariance of 0, not NaN, so
drop empty bins before fitting:

```python
populated = custom.counts_ > 0
lags, semivariance = custom.distances_[populated], custom.gamma_[populated]
```

## Models

Three models can be fitted. With nugget $c_0$, sill $c$ and range $a$, each gives the
semivariance at a distance $h > 0$ as:

**Exponential**

$$
\gamma(h) = c_0 + (c - c_0)\left(1 - e^{-h/a}\right)
$$

**Spherical**

$$
\gamma(h) =
\begin{cases}
c_0 + (c - c_0)\left(\dfrac{3h}{2a} - \dfrac{h^3}{2a^3}\right) & h < a \\
c & h \ge a
\end{cases}
$$

**Gaussian**

$$
\gamma(h) = c_0 + (c - c_0)\left(1 - e^{-(h/a)^2}\right)
$$

At $h = 0$ every model is 0; the nugget is the jump just above it.

Two conventions matter when reading fitted parameters:

- **The sill includes the nugget.** It is the level the variogram approaches at long
  distances, not the height above the nugget.
- **The range is a scale parameter for the exponential and Gaussian models.** Only the
  spherical model reaches its sill at $h = a$. The exponential model reaches 95% of
  the way from nugget to sill at about $3a$, and the Gaussian model at about $1.73a$.

`Variogram` also accepts `model="matern"`, but cannot fit it; see
[Known limitations](../known-limitations.md#matern-models-cannot-be-fitted).

## Fitting a model

```python
from pygeostats.variogram import Variogram

model = Variogram(model="exponential")
model.fit(empirical.distances_, empirical.gamma_, weights=empirical.counts_)

print(model.nugget_, model.sill_, model.range_)
```

`fit()` returns the model. Weighting by the pair counts gives bins with more pairs
more influence, which is usually what is wanted.

Once fitted, the model evaluates the semivariance and the covariance, which is the
sill minus the semivariance, at any distance:

```python
lags = np.linspace(0, 6, 25)
semivariance = model.predict(lags)
covariance = model.covariance(lags)
```

Both raise `ValueError` on a model that has not been fitted.

### Holding parameters fixed

`nugget`, `sill` and `range` given to the constructor are starting values. Name any
of them in `fix` to keep it at that value instead:

```python
with_nugget = Variogram(model="exponential", nugget=0.05)
with_nugget.fit(
    empirical.distances_,
    empirical.gamma_,
    weights=empirical.counts_,
    fix={"nugget": True},
)
```

### Reading the fit report

A fit that finishes is not necessarily a fit to trust, so `Variogram` reports on
each one:

```python
print(model.converged_)       # the flag to check
print(model.status_)          # how the optimiser stopped
print(model.warnings_)        # why a fit is not to be trusted, if it is not
print(model.parameter_std_)   # approximate standard errors of the parameters
print(model.fit_statistics_["r2"], model.fit_statistics_["rmse"])
```

When the empirical variogram does not constrain the model, typically because it is
still rising at the largest lag, `converged_` is `False` and `warnings_` says why.
Extending `max_distance`, or fixing the sill or range from other knowledge, are the
usual remedies.

`status_` is one of `succeeded`, `max_iterations`, `singular_matrix`, `diverged`,
`timeout`, `invalid_parameters`, `fallback_to_isotropic` or `failed`.
`fit_statistics_` also holds the iteration count, the optimiser's message and
further diagnostics.

## Choosing between models

`select_best_variogram_model` fits several models and ranks them by AIC, BIC or
leave-one-out cross-validation. See
[Validation and model selection](validation.md#choosing-a-variogram-model).
