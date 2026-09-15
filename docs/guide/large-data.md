# Large datasets

Two costs grow quickly with the number of samples. An empirical variogram visits
every pair of samples, and ordinary kriging uses every sample for every prediction.
This page covers what pygeostats offers for each, and what that does not solve.

!!! warning "Parallel kriging is unreliable in 0.1.0a1"

    `OrdinaryKriging.predict_parallel()` fails, and `ParallelKrigingExecutor` can
    return predictions in the wrong order or as NaN. Neither is used on this page.
    See [Known limitations](../known-limitations.md#parallelkrigingexecutor-returns-wrong-or-missing-predictions).

The examples use 1,000 samples:

```python
import numpy as np

rng = np.random.default_rng(0)
coords = rng.uniform(0, 100, size=(1000, 2))
values = (
    np.sin(coords[:, 0] / 15) + np.cos(coords[:, 1] / 20) + rng.normal(0, 0.2, len(coords))
)
```

## Streaming variograms

`EmpiricalVariogram` holds the distances of all pairs in memory at once, which for
100,000 samples is about 40 GB. `streaming_variogram` processes the samples in
chunks instead, pairing each chunk with itself and with every other chunk, so memory
depends on `chunk_size` rather than on the number of samples:

```python
from pygeostats.variogram import streaming_variogram

bin_edges = np.linspace(0, 40, 21)
result = streaming_variogram(coords, values, bin_edges, chunk_size=250)

print(result.centers)  # bin centres
print(result.gamma)    # semivariance in each bin
print(result.weights)  # number of pairs in each bin
```

The result is the same as `EmpiricalVariogram` with the same bin edges, whatever the
chunk size. Bin edges are required; pairs further apart than the last edge are left
out, and empty bins have a weight and semivariance of 0. Coordinates can have any
number of dimensions.

!!! note "Memory, not time"

    Every pair is still computed, so the run time grows with the square of the number
    of samples. For very large datasets, a variogram estimated from a random subsample
    is often good enough and much faster.

The result feeds a model fit as usual:

```python
from pygeostats.variogram import Variogram

populated = result.weights > 0
model = Variogram(model="exponential").fit(
    result.centers[populated], result.gamma[populated], weights=result.weights[populated]
)
```

`result.as_array()` stacks centres, semivariances and weights as three columns.

### Sparse results

With `sparse=True`, only the bins that contain pairs are kept, which helps when
there are many narrow bins:

```python
sparse = streaming_variogram(coords, values, bin_edges, chunk_size=250, sparse=True)

print(sparse.indices)       # which bins are populated
print(sparse.gamma, sparse.weights)
print(sparse.total_weight)  # total number of pairs
```

`sparse.bin_edges` and `sparse.bin_centers` still describe every bin, and
`sparse.to_coo()` returns a SciPy sparse matrix with one row per bin. A dense result
converts with `to_sparse()`.

### Building a variogram from blocks

`StreamingVariogramBuilder` accumulates pairs from blocks you supply, for data that
arrives in pieces or does not fit in memory:

```python
from pygeostats.variogram import StreamingVariogramBuilder

first, second = slice(0, 500), slice(500, None)

builder = StreamingVariogramBuilder(bin_edges)
builder.add_chunk(coords[first], values[first])
builder.add_chunk(coords[second], values[second])
builder.add_cross(coords[first], values[first], coords[second], values[second])
combined = builder.finalize()
```

- `add_chunk` adds the pairs within one block.
- `add_cross` adds the pairs between two blocks. Every pair of blocks needs one call,
  or those pairs are missing from the result.
- `merge` adds everything another builder has accumulated, so blocks can be
  processed separately and combined.
- `reset` clears the builder, and `finalize(sparse=...)` returns the result.

With pairs whose distances and semivariances are already known, `add_pairs` adds
them directly. Each semivariance is half the squared difference of the pair's
values:

```python
pairs = StreamingVariogramBuilder(bin_edges)
pairs.add_pairs(
    np.array([1.2, 3.4, 7.9]),
    np.array([0.05, 0.30, 0.12]),
    weights=np.ones(3),
)
```

`weights` is documented as optional, but pass it: `add_pairs` currently raises
`TypeError` without it.

### Data on disk

`streaming_variogram_memmap` reads coordinates and values from raw binary files
through memory maps, so they never have to fit in memory:

```python
from pygeostats.variogram import streaming_variogram_memmap

coords.tofile("coords.bin")
values.tofile("values.bin")

from_disk = streaming_variogram_memmap(
    "coords.bin", "values.bin", coords.shape, bin_edges, chunk_size=250
)
```

`shape` is the shape of the coordinates, `(n_samples, n_dimensions)`, and the values
file must hold `n_samples` values of the same `dtype`, float64 by default.
`memory_map_array(path, shape, mode="w+")` creates such a file to write into.

!!! warning "Not for `.npy` files"

    `.npy` files start with a header, which a raw memory map reads as data, giving
    wrong results without an error. Open them with `np.load` instead:

```python
np.save("coords.npy", coords)
np.save("values.npy", values)

from_npy = streaming_variogram(
    np.load("coords.npy", mmap_mode="r"),
    np.load("values.npy", mmap_mode="r"),
    bin_edges,
    chunk_size=250,
)
```

## Kriging many locations

`OrdinaryKriging.predict` uses every sample for every target, and its cost grows
roughly with the square of the number of samples, times the number of targets.
Predicting a large grid in batches keeps the memory of each call bounded, but takes
as long overall:

```python
from pygeostats.kriging import OrdinaryKriging

xs = np.linspace(0, 100, 20)
grid_x, grid_y = np.meshgrid(xs, xs)
targets = np.column_stack([grid_x.ravel(), grid_y.ravel()])

kriging = OrdinaryKriging(model).fit(coords[:300], values[:300])
batched = np.concatenate(
    [kriging.predict(targets[start : start + 100]) for start in range(0, len(targets), 100)]
)
```

### Local neighbourhoods

With thousands of samples, a common approximation is to krige each part of the grid
from the samples near it. pygeostats does not do this for you yet, but
`ApproximateNeighborIndex` finds the neighbours:

```python
from pygeostats.kriging import ApproximateNeighborIndex

index = ApproximateNeighborIndex(method="kdtree", max_neighbors=32).fit(coords)

tile = 10
local_grid = np.empty(grid_x.shape)
for row in range(0, grid_x.shape[0], tile):
    for col in range(0, grid_x.shape[1], tile):
        window = (slice(row, row + tile), slice(col, col + tile))
        block = np.column_stack([grid_x[window].ravel(), grid_y[window].ravel()])

        nearby = np.unique(index.query(block, k=32).indices)
        local = OrdinaryKriging(model).fit(coords[nearby], values[nearby])
        local_grid[window] = local.predict(block).reshape(grid_x[window].shape)
```

Each tile of the grid is kriged from the union of its targets' 32 nearest samples.
This is an approximation: predictions can change abruptly at tile boundaries, where
neighbouring tiles use different samples, and more neighbours bring the result
closer to kriging with every sample, at a higher cost.

`method` is `"kdtree"` or `"sklearn"`, both exact, `"annoy"`, which is approximate
and needs the `approx` extra, or `"auto"`. `query` returns `indices` and
`distances`, each of shape `(n_queries, k)`. Building an index prints a message.
