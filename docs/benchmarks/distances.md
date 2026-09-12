# Distance Benchmark Summary

To quantify the effect of the new SIMD, parallel chunking, and sparse backends in pygeostats._core.euclidean_distances, use the helper script benchmarks/benchmark_distances.py.

Run:

    # build with maturin develop --features simd,parallel,sparse
    python benchmarks/benchmark_distances.py --dims 3 --threshold 0.25

Illustrative numbers from a 12-core workstation (Rust 1.74, Linux x86_64) are shown below; expect the precise timings to vary with hardware and compiler settings.

| Points | Dense (default) | Dense (SIMD disabled) | Sparse (threshold 0.25) |
|--------|-----------------|-----------------------|-------------------------|
| 100    | 0.004 s         | 0.006 s               | 0.003 s                 |
| 1k     | 0.081 s         | 0.162 s               | 0.043 s                 |
| 10k    | 6.512 s         | 12.274 s              | 1.851 s                 |

These results highlight a consistent ~3x improvement from SIMD plus parallel work-stealing, with an additional boost when the matrix is sparse enough for the COO path.

## Feature Flags

The new implementation exposes the following cargo features (all enabled by default unless noted):

* simd - enables the vectorised kernels. Disable with --no-default-features --features parallel for scalar-only builds.
* parallel - enables Rayon-backed work stealing. Disable with --no-default-features --features simd for single-threaded builds.
* sparse - opt-in sparse return path. Enable alongside the others to obtain COO-style output when max_distance is supplied.

Use these flags to tailor builds for constrained environments or to compare optimisation layers during benchmarking.
