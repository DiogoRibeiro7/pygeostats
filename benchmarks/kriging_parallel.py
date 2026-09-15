"""Time ordinary kriging prediction: one call, batches and ParallelKrigingExecutor.

``predict`` already computes targets in parallel in the Rust core, so this measures
what batching and the executor's thread and process workers add on top of it. It
also checks that every route returns the same predictions.

    python benchmarks/kriging_parallel.py --known 2000 --pred 100000
    python benchmarks/kriging_parallel.py --known 1000 --pred 20000 --variance
"""

from __future__ import annotations

import argparse
import contextlib
import io
import time

import numpy as np
from pygeostats.kriging import OrdinaryKriging, ParallelKrigingExecutor
from pygeostats.variogram import Variogram


def synthetic_data(n_known: int, n_pred: int, seed: int = 13):
    rng = np.random.default_rng(seed)
    known = rng.uniform(0, 1000, size=(n_known, 2))
    values = np.sin(known[:, 0] / 150) + np.cos(known[:, 1] / 200)
    values += rng.normal(0, 0.1, n_known)
    targets = rng.uniform(0, 1000, size=(n_pred, 2))
    return known, values, targets


def exponential_variogram(range_: float = 100.0) -> Variogram:
    """A fitted exponential variogram, fitted to lags rather than to every pair."""
    lags = np.linspace(10, 500, 25)
    gamma = 0.05 + 0.95 * (1 - np.exp(-lags / range_))
    return Variogram(model="exponential").fit(lags, gamma)


def timed(label: str, call):
    start = time.perf_counter()
    # The executor prints progress; keep the benchmark output to the timings.
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        result = call()
    print(f"{label:<36s} {time.perf_counter() - start:8.2f}s")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Time ordinary kriging prediction routes"
    )
    parser.add_argument("--known", type=int, default=2_000, help="samples")
    parser.add_argument("--pred", type=int, default=100_000, help="targets")
    parser.add_argument(
        "--batch", type=int, default=10_000, help="targets per batch or chunk"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--variance", action="store_true", help="also compute the kriging variance"
    )
    args = parser.parse_args()

    known, values, targets = synthetic_data(args.known, args.pred)
    kriging = timed(
        f"fit ({args.known:,} samples)",
        lambda: OrdinaryKriging(exponential_variogram()).fit(known, values),
    )

    def predictions_of(result):
        return result[0] if args.variance else result

    whole = timed(
        f"predict ({args.pred:,} targets)",
        lambda: kriging.predict(targets, return_variance=args.variance),
    )
    batches = [targets[i : i + args.batch] for i in range(0, args.pred, args.batch)]
    batched = timed(
        f"predict in {len(batches)} batches",
        lambda: [kriging.predict(b, return_variance=args.variance) for b in batches],
    )
    if not np.array_equal(
        np.concatenate([predictions_of(part) for part in batched]),
        predictions_of(whole),
    ):
        raise SystemExit("batched predictions differ from one call")

    for method in ("thread", "process"):
        executor = ParallelKrigingExecutor(
            n_workers=args.workers, execution_method=method, chunk_size=args.batch
        )
        result = timed(
            f"executor, {args.workers} {method} workers",
            lambda executor=executor: executor.predict_parallel(
                kriging, targets, return_variance=args.variance
            ),
        )
        if not np.array_equal(predictions_of(result), predictions_of(whole)):
            raise SystemExit(f"{method} workers returned different predictions")


if __name__ == "__main__":
    main()
