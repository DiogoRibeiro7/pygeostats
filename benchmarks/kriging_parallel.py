"""Benchmarks for parallel kriging execution."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from pygeostats.kriging import (
    ApproximateNeighborIndex,
    OrdinaryKriging,
    ParallelKrigingExecutor,
)
from pygeostats.variogram import Variogram


def synthetic_data(n_known: int, n_pred: int, dims: int = 2):
    rng = np.random.default_rng(13)
    known_coords = rng.random((n_known, dims)) * 1000.0
    pred_coords = rng.random((n_pred, dims)) * 1000.0
    kernel = np.exp(-np.linalg.norm(known_coords[:, None, :] - known_coords[None, :, :], axis=-1) / 200.0)
    values = rng.normal(size=n_known)
    return known_coords, values, pred_coords


def fit_variogram(known_coords: np.ndarray, values: np.ndarray) -> Variogram:
    variogram = Variogram(model="exponential")
    distances = np.linalg.norm(known_coords[:, None, :] - known_coords[None, :, :], axis=-1)
    gamma = 0.5 * (values[:, None] - values[None, :]) ** 2
    mask = np.triu(np.ones_like(distances), k=1).astype(bool)
    variogram.fit(distances[mask], gamma[mask])
    return variogram


def throughput_benchmark(n_known: int, n_pred: int, neighbors: int, chunk_size: int) -> None:
    known_coords, values, pred_coords = synthetic_data(n_known, n_pred)
    variogram = fit_variogram(known_coords, values)
    ok = OrdinaryKriging(variogram).fit(known_coords, values)

    params = np.array([variogram.nugget_, variogram.sill_, variogram.range_], dtype=float)
    executor = ParallelKrigingExecutor(known_coords, values, params, variogram.model)
    index = ApproximateNeighborIndex(known_coords)

    start = time.perf_counter()
    preds = executor.predict(
        pred_coords,
        neighbor_index=index,
        neighbors=neighbors,
        chunk_size=chunk_size,
        progress=False,
    )
    elapsed = time.perf_counter() - start
    print(
        f"throughput | known={n_known:,} pred={n_pred:,} neighbors={neighbors} chunk={chunk_size} "
        f"elapsed={elapsed:.2f}s throughput={n_pred/elapsed:,.0f} preds/s"
    )

    ref = ok.predict(pred_coords[: min(n_pred, 1000)])
    approx = preds[: ref.shape[0]]
    rmse = float(np.sqrt(np.mean((ref - approx) ** 2)))
    print(f"accuracy | rmse (approx vs full)={rmse:.4f}")


def resilience_benchmark(n_known: int, n_pred: int, checkpoint: Path) -> None:
    known_coords, values, pred_coords = synthetic_data(n_known, n_pred)
    variogram = fit_variogram(known_coords, values)

    params = np.array([variogram.nugget_, variogram.sill_, variogram.range_], dtype=float)
    executor = ParallelKrigingExecutor(known_coords, values, params, variogram.model)
    index = ApproximateNeighborIndex(known_coords)

    try:
        checkpoint.unlink()
    except FileNotFoundError:
        pass

    # Run first half and simulate interruption
    preds_half = executor.predict(
        pred_coords,
        neighbor_index=index,
        neighbors=48,
        chunk_size=20_000,
        checkpoint_path=checkpoint,
        checkpoint_interval=1,
        progress=False,
    )
    print(f"resilience | completed predictions={np.count_nonzero(~np.isnan(preds_half)):,}")

    # Resume from checkpoint
    preds_full = executor.predict(
        pred_coords,
        neighbor_index=index,
        neighbors=48,
        chunk_size=20_000,
        checkpoint_path=checkpoint,
        resume=True,
        progress=False,
    )
    print(f"resilience | final predictions={np.count_nonzero(~np.isnan(preds_full)):,}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel kriging benchmarks")
    parser.add_argument("mode", choices=["throughput", "resilience"], default="throughput")
    parser.add_argument("--known", type=int, default=50_000)
    parser.add_argument("--pred", type=int, default=200_000)
    parser.add_argument("--neighbors", type=int, default=64)
    parser.add_argument("--chunk", type=int, default=10_000)
    parser.add_argument("--checkpoint", type=Path, default=Path("kriging_checkpoint.npz"))
    args = parser.parse_args()

    if args.mode == "throughput":
        throughput_benchmark(args.known, args.pred, args.neighbors, args.chunk)
    else:
        resilience_benchmark(args.known, args.pred, args.checkpoint)


if __name__ == "__main__":
    main()

