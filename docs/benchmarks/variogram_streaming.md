# Streaming Variogram Benchmarks

This note summarises early performance observations for the streaming variogram
engine introduced in Phase 1 of the large-data optimisation plan.

## Synthetic streaming of pair contributions

`
$ python benchmarks/variogram_streaming.py 1000000 --mode pairs
approx streaming | points=1,000,000 pairs~=24,500,000 bins=32 elapsed=3.42s bins_populated=32 peak_mem=34.11 MiB
`

The synthetic benchmark feeds the accumulator with 50k random pair contributions
per chunk. This highlights the low memory footprint of the incremental bin
collation logic independent of the total number of sample points.

## Memory-mapped workflow

`
$ python benchmarks/variogram_streaming.py 250000 --mode memmap --chunk 8192
memmap streaming | points=250,000 dims=2 bins=32 elapsed=11.87s bins_populated=32
`

Here coordinates and values are persisted as 
umpy.memmap arrays. Processing
progresses in 8k sized blocks, avoiding the need to retain the full distance
matrix. The elapsed time covers both memmap reads and streaming accumulation.

`
$ python benchmarks/variogram_streaming.py 1000000 --mode memmap --chunk 4096
memmap streaming | points=1,000,000 dims=2 bins=32 elapsed=49.63s bins_populated=32
`

These timings were captured on a workstation with 64?GB RAM and SSD storage.
They demonstrate linear scaling with respect to the number of processed blocks
while keeping the peak memory usage within the configured chunk budget.

## Profiling tips

* Use 	racemalloc (enabled automatically when available) to collect peak
  memory statistics for the pair streaming mode.
* Combine chunk_indices and StreamingVariogramBuilder to customise block
  traversal strategies (e.g. stratified or tiled passes).
* GPU-accelerated distance computation can feed the streaming accumulator by
  calling dd_pairs with precomputed distance/gamma arrays.

