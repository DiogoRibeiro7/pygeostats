# Parallel Kriging Benchmarks

The parallel kriging executor combines approximate neighbour search, spatial
tiling, and resumable checkpoints. The following commands illustrate throughput
on synthetic datasets (Intel i9 workstation, 16 cores).

`
$ python benchmarks/kriging_parallel.py throughput --known 75000 --pred 250000 --neighbors 64 --chunk 12000
throughput | known=75,000 pred=250,000 neighbors=64 chunk=12000 elapsed=18.42s throughput=13,569 preds/s
accuracy | rmse (approx vs full)=0.0317
`

Checkpoint resilience can be demonstrated by interrupting and resuming the same
job. The executor stores intermediate results inside a .npz container.

`
$ python benchmarks/kriging_parallel.py resilience --known 60000 --pred 150000 --checkpoint kriging_ckpt.npz
resilience | completed predictions=150,000
resilience | final predictions=150,000
`

The restored run reuses the saved state and finalises the output without
recomputing previously processed tiles.
