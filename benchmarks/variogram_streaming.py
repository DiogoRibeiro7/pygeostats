"""Benchmark utilities for streaming variogram computation."""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np
from pygeostats.variogram.streaming import (
    StreamingVariogramBuilder,
)
from pygeostats.variogram.streaming import (
    streaming_variogram_memmap as compute_streaming_variogram_memmap,
)


def _write_memmap(path: Path, data: np.ndarray) -> None:
    mm = np.memmap(path, dtype=data.dtype, mode="w+", shape=data.shape)
    mm[:] = data
    mm.flush()


def benchmark_streaming_pairs(
    num_points: int,
    *,
    bins: int = 32,
    chunk_size: int = 2048,
    samples_per_chunk: int = 50_000,
) -> None:
    rng = np.random.default_rng(42)
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    builder = StreamingVariogramBuilder(bin_edges)
    try:
        import tracemalloc

        tracemalloc.start()
        tracing = True
    except Exception:
        tracing = False

    total_pairs = 0
    start = time.perf_counter()
    for _ in range(max(1, num_points // chunk_size)):
        distances = rng.random(samples_per_chunk)
        semivariances = rng.random(samples_per_chunk)
        builder.add_pairs(distances, semivariances)
        total_pairs += samples_per_chunk
    elapsed = time.perf_counter() - start

    peak_mem = None
    if tracing:
        import tracemalloc

        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mem = peak / (1024**2)

    result = builder.finalize(sparse=True)
    message = (
        f"approx streaming | points={num_points:,} pairs~={total_pairs:,} bins={bins} "
        f"elapsed={elapsed:.2f}s bins_populated={result.indices.size}"
    )
    if peak_mem is not None:
        message += f" peak_mem={peak_mem:.2f} MiB"
    print(message)


def benchmark_memmap(
    num_points: int,
    *,
    dims: int = 2,
    bins: int = 32,
    chunk_size: int = 4096,
) -> None:
    rng = np.random.default_rng(0)
    coords = rng.random((num_points, dims))
    values = rng.random(num_points)
    bin_edges = np.linspace(0.0, 1.0, bins + 1)

    with tempfile.TemporaryDirectory() as tmp:
        coords_path = Path(tmp) / "coords.dat"
        values_path = Path(tmp) / "values.dat"
        _write_memmap(coords_path, coords)
        _write_memmap(values_path, values)

        start = time.perf_counter()
        result = compute_streaming_variogram_memmap(
            coords_path,
            values_path,
            (num_points, dims),
            bin_edges,
            chunk_size=chunk_size,
            sparse=True,
        )
        elapsed = time.perf_counter() - start
        print(
            f"memmap streaming | points={num_points:,} dims={dims} bins={bins} "
            f"elapsed={elapsed:.2f}s bins_populated={result.indices.size}"
        )


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="Streaming variogram benchmarks")
    parser.add_argument("points", type=int, nargs="?", default=1_000_000)
    parser.add_argument("--chunk", type=int, default=4096)
    parser.add_argument("--bins", type=int, default=32)
    parser.add_argument("--mode", choices=["pairs", "memmap"], default="pairs")
    args = parser.parse_args()

    if args.mode == "pairs":
        benchmark_streaming_pairs(args.points, bins=args.bins, chunk_size=args.chunk)
    else:
        benchmark_memmap(args.points, bins=args.bins, chunk_size=args.chunk)


if __name__ == "__main__":
    run_cli()
