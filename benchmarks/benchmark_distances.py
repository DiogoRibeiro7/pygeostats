"""Benchmark utilities for distance computations."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Callable, Iterable, List, Tuple

import numpy as np

from pygeostats import _core


@dataclass
class Scenario:
    label: str
    size: int


def time_call(fn: Callable[[], None], repeats: int = 3) -> float:
    durations: List[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        durations.append(time.perf_counter() - start)
    return min(durations)


def generate_points(size: int, dims: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.random((size, dims))


def run_benchmarks(scenarios: Iterable[Scenario], dims: int, threshold: float | None) -> List[Tuple[str, float, float, float]]:
    results: List[Tuple[str, float, float, float]] = []

    for scenario in scenarios:
        coords = generate_points(scenario.size, dims)

        dense_time = time_call(lambda: _core.euclidean_distances(coords), repeats=3)

        no_simd_time = time_call(
            lambda: _core.euclidean_distances(coords, use_simd=False),
            repeats=3,
        )

        if threshold is not None:
            sparse_time = time_call(
                lambda: _core.euclidean_distances(coords, max_distance=threshold),
                repeats=3,
            )
        else:
            sparse_time = float("nan")

        results.append((scenario.label, dense_time, no_simd_time, sparse_time))

    return results


def format_table(rows: Iterable[Tuple[str, float, float, float]]) -> str:
    header = "scenario | dense (s) | dense no-simd (s) | sparse threshold (s)"
    underline = "-" * len(header)
    body = [header, underline]
    for label, dense, no_simd, sparse in rows:
        body.append(
            f"{label:<8} | {dense:>10.4f} | {no_simd:>18.4f} | {sparse:>20.4f}"
        )
    return "\n".join(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark distance backends")
    parser.add_argument(
        "--dims",
        type=int,
        default=3,
        help="Number of spatial dimensions",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional distance threshold for sparse mode",
    )
    args = parser.parse_args()

    scenarios = [
        Scenario("100", 100),
        Scenario("1k", 1_000),
        Scenario("10k", 10_000),
    ]

    rows = run_benchmarks(scenarios, dims=args.dims, threshold=args.threshold)
    print(format_table(rows))


if __name__ == "__main__":
    main()

