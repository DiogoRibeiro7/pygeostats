# src/python/pyspatialstats/kriging/executor.py
"""High-level utilities for parallel kriging execution."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no-cover - optional dependency
    from tqdm import tqdm

    _HAS_TQDM = True
except Exception:  # pragma: no-cover
    _HAS_TQDM = False

from .._core import ordinary_kriging_predict_neighbors
from .neighbor_search import ApproximateNeighborIndex, NeighborQueryResult


@dataclass
class TilePlan:
    """Description of a spatial tile and its point indices."""

    bounds: Tuple[np.ndarray, np.ndarray]
    primary_indices: np.ndarray
    halo_indices: np.ndarray


def spatial_tiles(
    coords: np.ndarray,
    grid_shape: Tuple[int, int],
    halo: float = 0.0,
) -> List[TilePlan]:
    """Create spatial tiles for prediction coordinates.

    Parameters
    ----------
    coords : ndarray
        Prediction coordinates (n, d). Only the first two dimensions are
        considered.
    grid_shape : tuple of int
        Number of tiles in each spatial dimension (rows, cols).
    halo : float, default=0.0
        Additional halo radius to extend tile bounds.
    """

    coords = np.asarray(coords, dtype=float)
    if coords.shape[1] < 2:
        raise ValueError("spatial tiling requires at least two coordinate dimensions")

    min_corner = coords[:, :2].min(axis=0)
    max_corner = coords[:, :2].max(axis=0)
    rows, cols = grid_shape
    row_edges = np.linspace(min_corner[0], max_corner[0], rows + 1)
    col_edges = np.linspace(min_corner[1], max_corner[1], cols + 1)

    plans: List[TilePlan] = []
    for i in range(rows):
        for j in range(cols):
            lower = np.array([row_edges[i], col_edges[j]])
            upper = np.array([row_edges[i + 1], col_edges[j + 1]])
            mask = np.all((coords[:, :2] >= lower) & (coords[:, :2] < upper), axis=1)
            primary = np.nonzero(mask)[0]
            if primary.size == 0:
                continue
            lower_halo = lower - halo
            upper_halo = upper + halo
            halo_mask = np.all((coords[:, :2] >= lower_halo) & (coords[:, :2] < upper_halo), axis=1)
            halo_indices = np.nonzero(halo_mask)[0]
            plans.append(
                TilePlan(
                    bounds=(lower_halo, upper_halo),
                    primary_indices=primary,
                    halo_indices=halo_indices,
                )
            )
    return plans


def chunk_indices(n: int, chunk_size: int) -> Iterator[np.ndarray]:
    chunk = max(int(chunk_size), 1)
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        yield np.arange(start, end)


class ParallelKrigingExecutor:
    """Coordinate orchestrator for large kriging jobs."""

    def __init__(
        self,
        known_coords: np.ndarray,
        known_values: np.ndarray,
        variogram_params: Sequence[float],
        model: str,
    ) -> None:
        self.known_coords = np.asarray(known_coords, dtype=float)
        self.known_values = np.asarray(known_values, dtype=float)
        self.params = np.asarray(variogram_params, dtype=float)
        if self.params.size != 3:
            raise ValueError("variogram_params must contain [nugget, sill, range]")
        self.model = str(model)

    def predict(
        self,
        pred_coords: np.ndarray,
        *,
        neighbor_index: ApproximateNeighborIndex,
        neighbors: int = 64,
        search_k: Optional[int] = None,
        tile_plan: Optional[List[TilePlan]] = None,
        chunk_size: int = 10_000,
        checkpoint_path: Optional[Path] = None,
        checkpoint_interval: int = 5,
        progress: bool = True,
        resume: bool = False,
    ) -> np.ndarray:
        pred_coords = np.asarray(pred_coords, dtype=float)
        n_pred = pred_coords.shape[0]
        predictions = np.full(n_pred, np.nan, dtype=float)
        completed = np.zeros(n_pred, dtype=bool)

        if checkpoint_path and resume and checkpoint_path.exists():
            data = np.load(checkpoint_path, allow_pickle=False)
            predictions = np.asarray(data["predictions"], dtype=float)
            completed = np.asarray(data["completed"], dtype=bool)

        batches = (
            tile_plan
            if tile_plan is not None
            else [
                TilePlan(bounds=(np.array([0.0, 0.0]), np.array([0.0, 0.0])), primary_indices=idx, halo_indices=idx)
                for idx in chunk_indices(n_pred, chunk_size)
            ]
        )

        iterator: Iterable[TilePlan] = batches
        if progress and _HAS_TQDM:
            iterator = tqdm(batches, desc="Kriging tiles")

        for i, plan in enumerate(iterator):
            work_indices = plan.halo_indices if plan.halo_indices.size else plan.primary_indices
            pending = work_indices[~completed[work_indices]]
            if pending.size == 0:
                continue
            targets = pred_coords[pending]
            query = neighbor_index.query(targets, neighbors, search_k=search_k, include_distances=False)
            chunk_predictions = ordinary_kriging_predict_neighbors(
                self.known_coords,
                self.known_values,
                targets,
                self.params,
                query.indices,
                self.model,
            )
            predictions[pending] = chunk_predictions
            completed[pending] = True

            if checkpoint_path and ((i + 1) % max(checkpoint_interval, 1) == 0):
                self._write_checkpoint(checkpoint_path, predictions, completed)

        if checkpoint_path:
            self._write_checkpoint(checkpoint_path, predictions, completed)

        if not np.all(completed):
            raise RuntimeError("Kriging executor completed with pending predictions. Check neighbor coverage or tile plan.")

        return predictions

    @staticmethod
    def _write_checkpoint(path: Path, predictions: np.ndarray, completed: np.ndarray) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, predictions=predictions, completed=completed)


__all__ = [
    "ApproximateNeighborIndex",
    "NeighborQueryResult",
    "ParallelKrigingExecutor",
    "TilePlan",
    "spatial_tiles",
    "chunk_indices",
]

