# src/python/pyspatialstats/kriging/neighbor_search.py
"""Approximate nearest-neighbour utilities for kriging support."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no-cover - optional dependency
    from annoy import AnnoyIndex

    _HAS_ANNOY = True
except Exception:  # pragma: no-cover - optional dependency
    _HAS_ANNOY = False

try:  # pragma: no-cover - optional dependency
    from sklearn.neighbors import KDTree

    _HAS_KDTREE = True
except Exception:  # pragma: no-cover - optional dependency
    _HAS_KDTREE = False

Backend = Literal["annoy", "kdtree"]


def _ensure_2d(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=float)
    if array.ndim != 2:
        raise ValueError("coordinate array must be 2-dimensional: (n_samples, n_dims)")
    return array


@dataclass
class NeighborQueryResult:
    """Container for neighbour indices and distances."""

    indices: np.ndarray
    distances: Optional[np.ndarray]


class ApproximateNeighborIndex:
    """Approximate nearest-neighbour index with graceful fallbacks."""

    def __init__(
        self,
        coords: np.ndarray,
        *,
        backend: Optional[Backend] = None,
        metric: str = "euclidean",
        n_trees: int = 20,
    ) -> None:
        self.coords = _ensure_2d(coords)
        self.metric = metric
        self._backend: Backend

        if backend is not None:
            backend = backend.lower()  # type: ignore[assignment]
            if backend not in {"annoy", "kdtree"}:
                raise ValueError("backend must be 'annoy' or 'kdtree'")

        if backend == "annoy" or (backend is None and _HAS_ANNOY):
            if not _HAS_ANNOY:
                raise ImportError("annoy package not available; install nnoy to use this backend")
            self._backend = "annoy"
            self._annoy_index = AnnoyIndex(self.coords.shape[1], metric)
            for idx, vector in enumerate(self.coords):
                self._annoy_index.add_item(idx, vector.tolist())
            self._annoy_index.build(n_trees)
        else:
            if not _HAS_KDTREE:
                raise ImportError(
                    "No suitable neighbour backend available. Install nnoy or scikit-learn for KDTree."
                )
            self._backend = "kdtree"
            self._kdtree = KDTree(self.coords)

    def query(
        self,
        targets: np.ndarray,
        k: int,
        *,
        search_k: Optional[int] = None,
        include_distances: bool = False,
        chunk_size: int = 10_000,
    ) -> NeighborQueryResult:
        """Return indices (and optionally distances) to the *k* nearest neighbours."""

        targets = _ensure_2d(targets)
        if k <= 0:
            raise ValueError("k must be positive")

        if self._backend == "annoy":
            return self._query_annoy(targets, k, search_k, include_distances)
        return self._query_kdtree(targets, k, include_distances, chunk_size)

    def _query_annoy(
        self,
        targets: np.ndarray,
        k: int,
        search_k: Optional[int],
        include_distances: bool,
    ) -> NeighborQueryResult:
        search_k = search_k or (k * 2)
        indices = np.full((targets.shape[0], k), -1, dtype=np.int64)
        distances: Optional[np.ndarray]
        distances = np.full((targets.shape[0], k), np.nan, dtype=float) if include_distances else None

        for row, vector in enumerate(targets):
            result = self._annoy_index.get_nns_by_vector(vector.tolist(), k, search_k, include_distances)
            if include_distances:
                nn_indices, nn_distances = result
            else:
                nn_indices = result
                nn_distances = None
            count = min(len(nn_indices), k)
            indices[row, :count] = nn_indices[:count]
            if include_distances and nn_distances is not None:
                distances[row, :count] = nn_distances[:count]

        return NeighborQueryResult(indices=indices, distances=distances)

    def _query_kdtree(
        self,
        targets: np.ndarray,
        k: int,
        include_distances: bool,
        chunk_size: int,
    ) -> NeighborQueryResult:
        chunk = max(int(chunk_size), 1)
        indices = np.full((targets.shape[0], k), -1, dtype=np.int64)
        distances = (
            np.full((targets.shape[0], k), np.nan, dtype=float)
            if include_distances
            else None
        )
        for start in range(0, targets.shape[0], chunk):
            end = min(targets.shape[0], start + chunk)
            dists, inds = self._kdtree.query(targets[start:end], k=k, return_distance=True)
            indices[start:end] = inds
            if distances is not None:
                distances[start:end] = dists
        return NeighborQueryResult(indices=indices, distances=distances)


__all__ = ["ApproximateNeighborIndex", "NeighborQueryResult"]
