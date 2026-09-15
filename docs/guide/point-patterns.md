# Point patterns

Point-pattern analysis asks whether the locations of events are clustered, regularly
spaced, or consistent with complete spatial randomness (CSR). The functions take
planar coordinates as an array of shape `(n_points, 2)`, and can be imported from
`pygeostats` or `pygeostats.point_patterns`.

!!! note "Bounds are `(xmin, xmax, ymin, ymax)`"

    Every function on this page that takes a window expects
    `(xmin, xmax, ymin, ymax)`. `spatial_tiles` in `pygeostats.kriging` orders its
    bounds differently, as `(xmin, ymin, xmax, ymax)`.

## Simulating patterns

```python
import numpy as np
from pygeostats import simulate_cox_process, simulate_poisson_process

window = (0.0, 1.0, 0.0, 1.0)  # xmin, xmax, ymin, ymax
area = 1.0

random_points = simulate_poisson_process(200, bounds=window, random_state=0)

cox = simulate_cox_process(
    200, bounds=window, field_sigma=1.5, random_state=1, return_intensity=True
)
clustered = cox["points"]
```

`simulate_poisson_process` places points independently and uniformly, with the given
intensity per unit area, so the number of points varies between runs.
`simulate_cox_process` draws a log-Gaussian Cox process: the intensity varies over a
random, smoothed surface, which clusters the points. `field_sigma` sets how strongly
the intensity varies and `field_smoothness` how smoothly. With
`return_intensity=True`, it returns a dict with the `points` and the `intensity`
surface on the grid `x`, `y`.

## Nearest neighbours

```python
from pygeostats import nearest_neighbor_distances

nearest = nearest_neighbor_distances(clustered)
second, second_index = nearest_neighbor_distances(clustered, k=2, return_indices=True)

intensity = len(clustered) / area
clark_evans = nearest.mean() / (0.5 / np.sqrt(intensity))
```

`nearest_neighbor_distances` gives each point's distance to its `k`-th nearest
neighbour and, with `return_indices=True`, that neighbour's index. The Clark-Evans
ratio compares the mean nearest-neighbour distance with its expectation under CSR:
below 1 suggests clustering and above 1 regularity. It ignores edge effects.

## Ripley's K and L

Ripley's $K(r)$ is the expected number of further points within distance $r$ of a
typical point, divided by the intensity. Under CSR, $K(r) = \pi r^2$. The L function,
$L(r) = \sqrt{K(r)/\pi}$, turns that into $L(r) = r$, which is easier to read.

```python
from pygeostats import ripley_k_function, ripley_l_function

radii = np.linspace(0.01, 0.15, 15)

k = ripley_k_function(clustered, radii, area=area)
l = ripley_l_function(clustered, radii, area=area)
```

Where $L(r) > r$, points have more neighbours within $r$ than CSR would give, which
indicates clustering at that scale. Without `area`, the area of the points' bounding
box is used.

!!! warning "No edge correction"

    Points near the edge of the window have part of their neighbourhood outside it,
    so K and L are biased low, increasingly so at larger radii. Comparing against
    simulations in the same window, which share the bias, avoids reading it as
    regularity.

A simulation envelope does that comparison:

```python
simulated = np.array(
    [
        ripley_l_function(
            simulate_poisson_process(intensity, bounds=window, random_state=seed),
            radii,
            area=area,
        )
        for seed in range(19)
    ]
)
lower, upper = simulated.min(axis=0), simulated.max(axis=0)
above_envelope = l > upper
```

## Pair correlation, G and F

```python
from pygeostats import f_function, g_function, pair_correlation_function

pcf = pair_correlation_function(clustered, radii, area=area)
g = g_function(clustered, radii)
f = f_function(clustered, radii, bounds=window, random_state=0)

csr = 1 - np.exp(-intensity * np.pi * radii**2)
```

- **Pair correlation** `pcf["g"]` at distances `pcf["r"]` is 1 under CSR, and above 1
  at distances where pairs are more common than under CSR. It is estimated from
  finite differences of K, so it is noisy, especially at the smallest radii.
- **G** is the distribution function of nearest-neighbour distances.
- **F**, the empty-space function, is the distribution function of the distance from
  a random location in the window to the nearest point. It is estimated from
  `n_random` random locations.

Under CSR, G and F both follow `csr` above. Clustering pushes G above it, because
points have close neighbours, and F below it, because clusters leave empty space.

## Clustering

`spatial_dbscan` runs scikit-learn's DBSCAN on the coordinates. Noise points get the
label `-1`.

```python
from pygeostats import cluster_validation_metrics, spatial_dbscan

labels = spatial_dbscan(clustered, eps=0.05, min_samples=5)
metrics = cluster_validation_metrics(clustered, labels)
```

`cluster_validation_metrics` returns `n_clusters` and the silhouette,
Calinski-Harabasz and Davies-Bouldin scores of the clusters, leaving noise out. The
scores are NaN when there are fewer than two clusters.

## Kernel density

```python
from pygeostats import kernel_density_estimate

density = kernel_density_estimate(clustered, grid_size=50, bounds=window)
```

The result holds `x` and `y` grids and `density` evaluated on them, each of shape
`(grid_size, grid_size)`. The density integrates to 1, so multiply by the number of
points for an intensity.

!!! note "`bandwidth` is not a distance"

    `bandwidth` is passed to `scipy.stats.gaussian_kde` as `bw_method`. `None` uses
    Scott's rule, and a number is a factor that scales the kernel relative to the
    spread of the points, not a distance in coordinate units.

## Hot spots

`getis_ord_gi_star` scores each point by how its value compares with the values of
points within `distance_threshold` of it, as a z-score:

```python
from pygeostats import getis_ord_gi_star

marks = np.exp(-((clustered[:, 0] - 0.5) ** 2 + (clustered[:, 1] - 0.5) ** 2) / 0.05)
z = getis_ord_gi_star(clustered, marks, distance_threshold=0.1)
hot_spots = z > 1.96
```

No correction is made for testing many points at once, so some points pass any
fixed threshold by chance. For values attached to fixed locations rather than to
events, see [Spatial autocorrelation](autocorrelation.md).

## Marked patterns and segregation

```python
from pygeostats import compute_spatial_segregation_indices, simulate_marked_poisson_process

marked = simulate_marked_poisson_process(
    300,
    marks=("A", "B"),
    mark_probabilities=np.array([0.7, 0.3]),
    bounds=window,
    random_state=0,
)
segregation = compute_spatial_segregation_indices(
    marked["points"], marked["marks"], n_cells=5, bounds=window
)
```

`compute_spatial_segregation_indices` divides the window into `n_cells` by `n_cells`
cells and compares the mix of marks in each cell with the overall mix:

- `entropy_segregation`, Theil's H, from 0, when every cell has the overall mix, to 1,
  when each cell holds a single mark
- `dissimilarity_index`, from 0 to 1, for two groups only; NaN with more
- `mean_cell_entropy` and `global_entropy`, the entropies it is built from
- `n_groups`, the number of distinct marks

With few points per cell, chance variation alone moves both indices above 0.
