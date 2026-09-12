# src/python/pygeostats/variogram/streaming.py
"""Streaming variogram utilities for large datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.sparse as sp

from .._core import (
    StreamingVariogramAccumulator as _RustStreamingVariogramAccumulator,
    streaming_variogram as _streaming_variogram,
)

NumericArray = Union[np.ndarray, Sequence[float]]


@dataclass(frozen=True)
class StreamingVariogramDenseResult:
    """Dense representation of a streaming variogram outcome."""

    centers: np.ndarray
    gamma: np.ndarray
    weights: np.ndarray

    def as_array(self) -> np.ndarray:
        """Return columns suitable for dataframe construction."""

        return np.column_stack([self.centers, self.gamma, self.weights])

    def to_sparse(self) -> "StreamingVariogramSparseResult":
        """Convert to sparse representation by filtering non-zero bins."""

        nonzero = self.weights > 0
        return StreamingVariogramSparseResult(
            indices=np.nonzero(nonzero)[0],
            centers=self.centers[nonzero],
            gamma=self.gamma[nonzero],
            weights=self.weights[nonzero],
            bin_edges=None,
            bin_centers=self.centers,
            total_weight=float(self.weights.sum()),
        )


@dataclass(frozen=True)
class StreamingVariogramSparseResult:
    """Sparse representation storing only populated bins."""

    indices: np.ndarray
    centers: np.ndarray
    gamma: np.ndarray
    weights: np.ndarray
    bin_edges: Optional[np.ndarray]
    bin_centers: np.ndarray
    total_weight: float

    def to_coo(self) -> sp.coo_matrix:
        """Return a COO matrix with semivariances in a single column."""

        return sp.coo_matrix(
            (self.gamma, (self.indices, np.zeros_like(self.indices))),
            shape=(self.bin_centers.shape[0], 1),
        )

    def as_array(self) -> np.ndarray:
        """Return stacked columns for dataframe construction."""

        return np.column_stack([
            self.indices,
            self.centers,
            self.gamma,
            self.weights,
        ])


class StreamingVariogramBuilder:
    """Incremental variogram computation helper."""

    def __init__(self, bin_edges: NumericArray):
        self._acc = _RustStreamingVariogramAccumulator(np.asarray(bin_edges, dtype=float))

    @property
    def bin_edges(self) -> np.ndarray:
        return np.asarray(self._acc.bin_edges, dtype=float)

    @property
    def bin_centers(self) -> np.ndarray:
        return np.asarray(self._acc.bin_centers, dtype=float)

    @property
    def total_weight(self) -> float:
        return float(self._acc.total_weight())

    def add_pairs(
        self,
        distances: NumericArray,
        semivariances: NumericArray,
        weights: Optional[NumericArray] = None,
    ) -> None:
        distances = np.asarray(distances, dtype=float)
        semivariances = np.asarray(semivariances, dtype=float)
        if weights is None:
            self._acc.update_pairs(distances, semivariances)
        else:
            self._acc.update_pairs(distances, semivariances, np.asarray(weights, dtype=float))

    def add_chunk(self, coords: np.ndarray, values: np.ndarray) -> None:
        self._acc.update_chunk(np.asarray(coords, dtype=float), np.asarray(values, dtype=float))

    def add_cross(
        self,
        coords_left: np.ndarray,
        values_left: np.ndarray,
        coords_right: np.ndarray,
        values_right: np.ndarray,
    ) -> None:
        self._acc.update_cross(
            np.asarray(coords_left, dtype=float),
            np.asarray(values_left, dtype=float),
            np.asarray(coords_right, dtype=float),
            np.asarray(values_right, dtype=float),
        )

    def merge(self, other: "StreamingVariogramBuilder") -> None:
        self._acc.merge(other._acc)

    def reset(self) -> None:
        self._acc.reset()

    def finalize(self, sparse: bool = False) -> Union[StreamingVariogramDenseResult, StreamingVariogramSparseResult]:
        if sparse:
            indices, centers, gamma, weights = self._acc.finalize_sparse()
            return StreamingVariogramSparseResult(
                indices=np.asarray(indices, dtype=np.int64),
                centers=np.asarray(centers, dtype=float),
                gamma=np.asarray(gamma, dtype=float),
                weights=np.asarray(weights, dtype=float),
                bin_edges=self.bin_edges,
                bin_centers=self.bin_centers,
                total_weight=self.total_weight,
            )
        centers, gamma, weights = self._acc.finalize_dense()
        return StreamingVariogramDenseResult(
            centers=np.asarray(centers, dtype=float),
            gamma=np.asarray(gamma, dtype=float),
            weights=np.asarray(weights, dtype=float),
        )


def streaming_variogram(
    coords: np.ndarray,
    values: np.ndarray,
    bin_edges: NumericArray,
    *,
    chunk_size: int = 2048,
    sparse: bool = False,
) -> Union[StreamingVariogramDenseResult, StreamingVariogramSparseResult]:
    """Compute a variogram using chunked streaming blocks.

    Parameters
    ----------
    coords : ndarray
        Coordinate matrix `(n_samples, n_dims)`.
    values : ndarray
        Observation vector `(n_samples,)`.
    bin_edges : array-like
        Monotonic increasing distance bin edges.
    chunk_size : int, default=2048
        Number of points loaded into memory for each streaming block.
    sparse : bool, default=False
        If True, return only populated bins.
    """

    payload = _streaming_variogram(
        np.asarray(coords, dtype=float),
        np.asarray(values, dtype=float),
        np.asarray(bin_edges, dtype=float),
        int(chunk_size),
        bool(sparse),
    )
    if sparse:
        return StreamingVariogramSparseResult(
            indices=np.asarray(payload["indices"], dtype=np.int64),
            centers=np.asarray(payload["centers"], dtype=float),
            gamma=np.asarray(payload["gamma"], dtype=float),
            weights=np.asarray(payload["weights"], dtype=float),
            bin_edges=np.asarray(payload["bin_edges"], dtype=float),
            bin_centers=np.asarray(payload["bin_centers"], dtype=float),
            total_weight=float(payload["total_weight"]),
        )
    centers, gamma, weights = payload
    return StreamingVariogramDenseResult(
        centers=np.asarray(centers, dtype=float),
        gamma=np.asarray(gamma, dtype=float),
        weights=np.asarray(weights, dtype=float),
    )


def memory_map_array(
    path: Union[str, Path],
    shape: Sequence[int],
    *,
    dtype: np.dtype = np.float64,
    mode: str = "r",
) -> np.memmap:
    """Return a numpy.memmap for the provided path and shape."""

    return np.memmap(Path(path), dtype=dtype, mode=mode, shape=tuple(shape), order="C")


def streaming_variogram_memmap(
    coords_path: Union[str, Path],
    values_path: Union[str, Path],
    shape: Sequence[int],
    bin_edges: NumericArray,
    *,
    chunk_size: int = 2048,
    dtype: np.dtype = np.float64,
    mode: str = "r",
    sparse: bool = False,
) -> Union[StreamingVariogramDenseResult, StreamingVariogramSparseResult]:
    """Compute a streaming variogram backed by memory-mapped arrays."""

    coords = memory_map_array(coords_path, shape, dtype=dtype, mode=mode)
    values = memory_map_array(values_path, (shape[0],), dtype=dtype, mode=mode)
    return streaming_variogram(coords, values, bin_edges, chunk_size=chunk_size, sparse=sparse)


def chunk_indices(n_rows: int, chunk_size: int) -> Iterator[Tuple[int, int]]:
    """Iterate over `(start, end)` pairs for chunked processing."""

    chunk = max(int(chunk_size), 1)
    start = 0
    while start < n_rows:
        end = min(n_rows, start + chunk)
        yield start, end
        start = end


__all__ = [
    "StreamingVariogramBuilder",
    "StreamingVariogramDenseResult",
    "StreamingVariogramSparseResult",
    "streaming_variogram",
    "streaming_variogram_memmap",
    "memory_map_array",
    "chunk_indices",
]


