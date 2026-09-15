# Spatial autocorrelation

Spatial autocorrelation statistics measure whether observations close to each other
have similar values. Each needs a spatial weights matrix, which says how strongly
each pair of observations counts as neighbours. The functions can be imported from
`pygeostats` or `pygeostats.spatial_autocorrelation`.

The examples use 80 samples of a spatially correlated field:

```python
import numpy as np

rng = np.random.default_rng(42)
coords = rng.uniform(0, 10, size=(80, 2))
distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
values = rng.multivariate_normal(
    np.zeros(len(coords)), np.exp(-distances / 2.0) + 0.05 * np.eye(len(coords))
)
```

## Spatial weights

```python
from pygeostats import (
    spatial_weights_distance_band,
    spatial_weights_inverse_distance,
    spatial_weights_knn,
)

knn = spatial_weights_knn(coords, k=6)
band = spatial_weights_distance_band(coords, threshold=2.0)
inverse = spatial_weights_inverse_distance(coords, power=2.0, max_distance=3.0)
```

- `spatial_weights_knn` gives each observation's `k` nearest neighbours a weight of 1.
- `spatial_weights_distance_band` gives a weight of 1 to every observation within
  `threshold`, or, with `binary=False`, inverse-distance weights within it. Pick a
  threshold at which every observation has at least one neighbour.
- `spatial_weights_inverse_distance` weights every pair by distance to the power
  `-power`, optionally only within `max_distance`.

Each returns a dense array of shape `(n, n)`, with row `i` holding the weights of
observation `i`'s neighbours. Rows are standardised to sum to 1 unless
`row_standardize=False` is passed, and `row_standardize_weights()` standardises a
matrix built another way.

!!! note "Weights are dense"

    Memory grows with the square of the number of observations: 10,000 observations
    take about 800 MB per matrix.

## Moran's I

```python
from pygeostats import morans_i

result = morans_i(values, knn)
print(result["I"], result["expected_I"], result["z_score"], result["p_value"])

permuted = morans_i(values, knn, permutations=999, random_state=0)
print(permuted["p_value"])
```

Moran's I above its expectation under no autocorrelation, $-1/(n-1)$, indicates
that neighbours have similar values; below it, dissimilar values.

The p-value is two-sided either way, and is computed in one of two ways:

- **Without permutations**, from `z_score` under a normal approximation.
- **With `permutations`**, from the share of random relabellings of the values that
  give a statistic at least as extreme. It can be no smaller than
  `1 / (permutations + 1)`, and no `z_score` is returned.

## Geary's C

```python
from pygeostats import gearys_c

geary = gearys_c(values, knn, permutations=999, random_state=0)
print(geary["C"], geary["expected_C"], geary["p_value"])
```

Geary's C is more sensitive to differences between close neighbours than Moran's I.
Its expectation is 1; below 1 indicates similar neighbours. P-values work as for
Moran's I.

## Local statistics

Local statistics give one value per observation, to locate clusters rather than
test for them overall:

```python
from pygeostats import local_gearys_c, local_morans_i

local_i = local_morans_i(values, knn)  # {"I_local": ..., "z": ...}
local_c = local_gearys_c(values, knn)  # {"C_local": ..., "z": ...}
```

They come without p-values, and with many observations some stand out by chance.

## Getis-Ord statistics

Getis-Ord statistics measure whether high or low values concentrate in space. They
are meant for positive values and binary weights that are not row-standardised:

```python
from pygeostats import global_getis_ord_g, local_getis_ord_g

binary_band = spatial_weights_distance_band(coords, threshold=2.0, row_standardize=False)
positive = values - values.min() + 1.0

general_g = global_getis_ord_g(positive, binary_band, permutations=999, random_state=0)
print(general_g["G"], general_g["expected_G"], general_g["p_value"])

local_g = local_getis_ord_g(positive, binary_band)
hot_spots = local_g["z_score"] > 1.96
```

`local_getis_ord_g` returns z-scores, under both `z_score` and `G_local`, which hold
the same array. With `include_self=True`, the default, each observation counts in its
own neighbourhood, which gives the Gi* statistic.

For hot spots among events given as coordinates, without building a weights matrix,
see `getis_ord_gi_star` in [Point patterns](point-patterns.md#hot-spots).
