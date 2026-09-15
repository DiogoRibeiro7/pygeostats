# src/python/pygeostats/kriging/executor.py
"""Parallel execution framework for large-scale kriging computations."""

from __future__ import annotations

import multiprocessing as mp
import warnings
from collections.abc import Iterator
from concurrent.futures import (
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from ..optimization.memory import MemoryManager
from ..utils.validation import validate_coordinates


def spatial_tiles(
    bounds: Tuple[float, float, float, float],
    tile_size: Union[float, Tuple[float, float]],
    overlap: float = 0.1,
) -> List[Tuple[float, float, float, float]]:
    """
    Generate spatial tiles for parallel processing.

    Parameters
    ----------
    bounds : tuple
        Spatial bounds (xmin, ymin, xmax, ymax).
    tile_size : float or tuple
        Size of each tile. If float, assumes square tiles. Must be positive.
    overlap : float, default=0.1
        Overlap fraction between adjacent tiles.

    Returns
    -------
    tiles : list of tuples
        List of tile bounds (xmin, ymin, xmax, ymax). Bounds with no width or
        height still give one row or column of tiles.

    Raises
    ------
    ValueError
        If a tile size is not positive.
    """
    xmin, ymin, xmax, ymax = bounds

    if isinstance(tile_size, (int, float)):
        tile_width = tile_height = float(tile_size)
    else:
        tile_width, tile_height = (float(size) for size in tile_size)
    # A size that is not positive never advances past the first tile, and the
    # list of tiles grew until memory ran out.
    if tile_width <= 0 or tile_height <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")

    # Calculate overlap in absolute units
    overlap_width = tile_width * overlap
    overlap_height = tile_height * overlap

    tiles = []
    for x_start, x_end in _tile_edges(xmin, xmax, tile_width):
        for y_start, y_end in _tile_edges(ymin, ymax, tile_height):
            # Extend bounds by overlap (but don't exceed domain bounds)
            tile_xmin = max(x_start - overlap_width, xmin)
            tile_ymin = max(y_start - overlap_height, ymin)
            tile_xmax = min(x_end + overlap_width, xmax)
            tile_ymax = min(y_end + overlap_height, ymax)

            tiles.append((tile_xmin, tile_ymin, tile_xmax, tile_ymax))

    return tiles


def _tile_edges(low: float, high: float, size: float) -> List[Tuple[float, float]]:
    """Split ``[low, high]`` into consecutive intervals no longer than ``size``.

    An extent with no width still gets one interval; it used to get none, so
    bounds with no width produced no tiles.
    """
    edges = []
    start = low
    while True:
        end = min(start + size, high)
        if end <= start:
            # Either the extent has no width, or size is too small to move start
            # at this magnitude; finish rather than loop.
            end = high
        edges.append((start, end))
        if end >= high:
            return edges
        start = end


def chunk_indices(n_items: int, chunk_size: int) -> Iterator[Tuple[int, int]]:
    """
    Generate chunk index ranges.

    Parameters
    ----------
    n_items : int
        Total number of items.
    chunk_size : int
        Size of each chunk.

    Yields
    ------
    start, end : tuple of int
        Start and end indices for each chunk.
    """
    for start in range(0, n_items, chunk_size):
        end = min(start + chunk_size, n_items)
        yield start, end


def _tile_size_for(coordinates: np.ndarray, points_per_tile: int) -> float:
    """Side of square tiles holding about ``points_per_tile`` targets each.

    Targets are taken as evenly spread over their bounding box. When the box has
    no area, because the targets lie along a line parallel to an axis or at a
    single location, the spread is measured along whatever extent remains.
    """
    extent = coordinates.max(axis=0) - coordinates.min(axis=0)
    fraction = min(1.0, points_per_tile / len(coordinates))
    spanned = extent[extent > 0]
    if spanned.size == 2:
        return float(np.sqrt(np.prod(spanned) * fraction))
    if spanned.size == 1:
        return float(spanned[0] * fraction)
    return 1.0


def _group_into_tiles(coordinates: np.ndarray, tile_size: float) -> List[np.ndarray]:
    """Group target indices by square tile, putting each target in exactly one.

    Targets used to be matched against tile bounds instead. Bounds with no width
    produced no tiles, so targets along a line parallel to an axis were never
    predicted and came back as NaN.
    """
    offsets = coordinates - coordinates.min(axis=0)
    cells = np.floor(offsets / tile_size).astype(np.int64)
    n_rows = int(cells[:, 1].max()) + 1
    keys = cells[:, 0] * n_rows + cells[:, 1]
    order = np.argsort(keys, kind="stable")
    boundaries = np.flatnonzero(np.diff(keys[order])) + 1
    return np.split(order, boundaries)


class ParallelKrigingExecutor:
    """
    Parallel executor for large-scale kriging computations.

    Provides various parallelization strategies for kriging predictions
    on large datasets.
    """

    def __init__(
        self,
        n_workers: Optional[int] = None,
        execution_method: str = "process",
        chunk_size: int = 1000,
        memory_limit_gb: float = 8.0,
        progress_callback: Optional[Callable] = None,
    ):
        """
        Initialize parallel kriging executor.

        Parameters
        ----------
        n_workers : int, optional
            Number of worker processes/threads. Uses CPU count if None.
        execution_method : str, default="process"
            Execution method: "process", "thread", or "sequential".
        chunk_size : int, default=1000
            Size of prediction chunks.
        memory_limit_gb : float, default=8.0
            Memory limit per worker process.
        progress_callback : callable, optional
            Callback function for progress updates.
        """
        self.n_workers = n_workers or mp.cpu_count()
        self.execution_method = execution_method
        self.chunk_size = chunk_size
        self.memory_limit_gb = memory_limit_gb
        self.progress_callback = progress_callback

        self.memory_manager = MemoryManager()

        # Validate execution method
        valid_methods = {"process", "thread", "sequential"}
        if execution_method not in valid_methods:
            raise ValueError(f"execution_method must be one of {valid_methods}")

    def predict_parallel(
        self,
        kriging_model,
        prediction_coordinates: np.ndarray,
        return_variance: bool = False,
        strategy: str = "chunk",
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Perform parallel kriging predictions.

        Parameters
        ----------
        kriging_model : object
            Fitted kriging model with predict method.
        prediction_coordinates : ndarray, shape (n_pred, d)
            Coordinates for prediction.
        return_variance : bool, default=False
            Whether to return prediction variance.
        strategy : str, default="chunk"
            Parallelization strategy: "chunk", "spatial", or "adaptive".

        Returns
        -------
        predictions : ndarray, shape (n_pred,)
            Predicted values.
        variances : ndarray, shape (n_pred,), optional
            Prediction variances if return_variance=True.

        Raises
        ------
        Exception
            Whatever ``kriging_model.predict`` raises for any part of the targets.
            No partial result is returned.

        Notes
        -----
        With ``execution_method="process"``, the model is pickled and sent to
        worker processes, which are always spawned rather than forked. Call this
        from code guarded by ``if __name__ == "__main__":``.
        """
        prediction_coordinates = validate_coordinates(prediction_coordinates)
        n_pred = len(prediction_coordinates)

        print(
            f"Parallel kriging: {n_pred:,} predictions using {self.n_workers} workers"
        )

        if strategy == "chunk":
            return self._predict_chunked(
                kriging_model, prediction_coordinates, return_variance
            )
        elif strategy == "spatial":
            return self._predict_spatial(
                kriging_model, prediction_coordinates, return_variance
            )
        elif strategy == "adaptive":
            return self._predict_adaptive(
                kriging_model, prediction_coordinates, return_variance
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _predict_chunked(
        self, kriging_model, prediction_coordinates: np.ndarray, return_variance: bool
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Parallel prediction using simple chunking."""
        n_pred = len(prediction_coordinates)
        chunks = list(chunk_indices(n_pred, self.chunk_size))

        print(f"Processing {len(chunks)} chunks of size ~{self.chunk_size}")

        worker_func = partial(
            _predict_chunk_worker,
            kriging_model=kriging_model,
            return_variance=return_variance,
        )
        results = self._run_tasks(
            worker_func, [prediction_coordinates[start:end] for start, end in chunks]
        )
        return self._combine_chunk_results(results, return_variance)

    def _predict_spatial(
        self, kriging_model, prediction_coordinates: np.ndarray, return_variance: bool
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Parallel prediction using spatial tiling."""
        if prediction_coordinates.shape[1] != 2:
            warnings.warn(
                "Spatial tiling only supports 2D coordinates, falling back to chunking",
                stacklevel=2,
            )
            return self._predict_chunked(
                kriging_model, prediction_coordinates, return_variance
            )

        n_pred = len(prediction_coordinates)
        target_points_per_tile = max(self.chunk_size, n_pred // (self.n_workers * 2))
        tile_size = _tile_size_for(prediction_coordinates, target_points_per_tile)
        tiles = _group_into_tiles(prediction_coordinates, tile_size)

        print(f"Processing {len(tiles)} spatial tiles")

        worker_func = partial(
            _predict_chunk_worker,
            kriging_model=kriging_model,
            return_variance=return_variance,
        )
        results = self._run_tasks(
            worker_func, [prediction_coordinates[indices] for indices in tiles]
        )
        return self._combine_spatial_results(results, tiles, n_pred, return_variance)

    def _predict_adaptive(
        self, kriging_model, prediction_coordinates: np.ndarray, return_variance: bool
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Adaptive parallel prediction based on data characteristics."""
        n_pred = len(prediction_coordinates)
        n_dims = prediction_coordinates.shape[1]

        # Choose strategy based on data characteristics
        if n_dims == 2 and n_pred > 10000:
            # Use spatial tiling for large 2D datasets
            print("Using spatial tiling strategy for large 2D dataset")
            return self._predict_spatial(
                kriging_model, prediction_coordinates, return_variance
            )
        else:
            # Use chunking for other cases
            print("Using chunking strategy")
            return self._predict_chunked(
                kriging_model, prediction_coordinates, return_variance
            )

    def _run_tasks(self, worker_func: Callable, tasks: List[Any]) -> List[Any]:
        """Run ``worker_func`` on every task and return the results in task order.

        An exception raised by a task is raised here, rather than turned into NaN
        or dropped, so a result is never returned with parts missing.
        """
        if self.execution_method == "sequential":
            results = []
            for completed, task in enumerate(tasks, start=1):
                results.append(worker_func(task))
                if self.progress_callback:
                    self.progress_callback(completed, len(tasks))
            return results

        if self.execution_method == "thread":
            pool = ThreadPoolExecutor(max_workers=self.n_workers)
        else:
            # Always spawned, never forked. The Rust core predicts on a pool of
            # threads that does not survive a fork, so a forked worker predicting
            # after the parent had waited on that pool forever: the tests hung on
            # Linux with Python 3.11 to 3.13, where fork is the default.
            pool = ProcessPoolExecutor(
                max_workers=self.n_workers, mp_context=mp.get_context("spawn")
            )
        with pool:
            futures = [pool.submit(worker_func, task) for task in tasks]
            return self._collect_results_with_progress(futures)

    def _collect_results_with_progress(self, futures: List[Future]) -> List[Any]:
        """Collect results in submission order, reporting progress as tasks finish.

        Results used to be appended in the order tasks finished, so chunks that
        finished early were joined first and their predictions landed at other
        targets' positions.
        """
        results: List[Any] = [None] * len(futures)
        positions = {future: index for index, future in enumerate(futures)}
        total_tasks = len(futures)

        try:
            from tqdm import tqdm

            progress_bar = tqdm(total=total_tasks, desc="Processing tasks")
        except ImportError:
            progress_bar = None

        try:
            for completed, future in enumerate(as_completed(futures), start=1):
                try:
                    results[positions[future]] = future.result()
                except BaseException:
                    for pending in futures:
                        pending.cancel()
                    raise

                if progress_bar is not None:
                    progress_bar.update(1)
                # The callback used to be skipped whenever tqdm was installed.
                if self.progress_callback:
                    self.progress_callback(completed, total_tasks)
                elif (
                    progress_bar is None and completed % max(1, total_tasks // 10) == 0
                ):
                    print(f"Completed {completed}/{total_tasks} tasks")
        finally:
            if progress_bar is not None:
                progress_bar.close()

        return results

    def _combine_chunk_results(
        self, results: List, return_variance: bool
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Join per-chunk results, which are in chunk order."""
        if return_variance:
            predictions = np.concatenate([result[0] for result in results])
            variances = np.concatenate([result[1] for result in results])
            return predictions, variances
        return np.concatenate(results)

    def _combine_spatial_results(
        self,
        results: List,
        tiles: List[np.ndarray],
        n_total: int,
        return_variance: bool,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Put each tile's results back at the positions of its targets."""
        predictions = np.empty(n_total)
        variances = np.empty(n_total)

        for indices, result in zip(tiles, results, strict=True):
            if return_variance:
                predictions[indices] = result[0]
                variances[indices] = result[1]
            else:
                predictions[indices] = result

        if return_variance:
            return predictions, variances
        return predictions

    def estimate_computation_time(
        self, n_predictions: int, sample_size: int = 100, kriging_model=None
    ) -> Dict[str, float]:
        """
        Estimate computation time for different parallelization strategies.

        Parameters
        ----------
        n_predictions : int
            Number of predictions to make.
        sample_size : int, default=100
            Sample size for timing estimation.
        kriging_model : object, optional
            Kriging model for accurate timing.

        Returns
        -------
        estimates : dict
            Time estimates for different strategies in seconds.
        """
        if kriging_model is None:
            # Use synthetic timing model
            base_time_per_prediction = 0.001  # 1ms per prediction
            parallelization_efficiency = 0.7  # 70% efficiency
        else:
            # Benchmark with actual model
            sample_coords = np.random.uniform(0, 10, size=(sample_size, 2))

            import time

            start_time = time.time()
            kriging_model.predict(sample_coords)
            elapsed = time.time() - start_time

            base_time_per_prediction = elapsed / sample_size
            parallelization_efficiency = 0.8  # Assume good efficiency for real models

        # Estimate times
        sequential_time = n_predictions * base_time_per_prediction

        parallel_time = sequential_time / (self.n_workers * parallelization_efficiency)

        # Add overhead for different strategies
        chunk_overhead = 0.1 * parallel_time  # 10% overhead for chunking
        spatial_overhead = 0.2 * parallel_time  # 20% overhead for spatial processing

        return {
            "sequential": sequential_time,
            "chunk_parallel": parallel_time + chunk_overhead,
            "spatial_parallel": parallel_time + spatial_overhead,
            "workers": self.n_workers,
            "base_time_per_prediction": base_time_per_prediction,
        }


def _predict_chunk_worker(
    chunk_coordinates: np.ndarray, kriging_model, return_variance: bool
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Predict one chunk or tile of targets.

    Exceptions propagate. They used to be caught and replaced with NaN, so a
    failing model returned NaN predictions without an error or a warning.
    """
    if return_variance:
        return kriging_model.predict(chunk_coordinates, return_variance=True)
    return kriging_model.predict(chunk_coordinates)


class ProgressiveKrigingComputation:
    """
    Progressive kriging computation with checkpointing and early stopping.

    Enables resumable computations for very large prediction grids.
    """

    def __init__(
        self,
        checkpoint_dir: str = "kriging_checkpoints",
        checkpoint_interval: int = 1000,
        convergence_tolerance: float = 1e-6,
        max_iterations: Optional[int] = None,
    ):
        """
        Initialize progressive computation.

        Parameters
        ----------
        checkpoint_dir : str, default="kriging_checkpoints"
            Directory for checkpoint files.
        checkpoint_interval : int, default=1000
            Number of predictions between checkpoints.
        convergence_tolerance : float, default=1e-6
            Tolerance for convergence detection.
        max_iterations : int, optional
            Maximum number of iterations before stopping.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_interval = checkpoint_interval
        self.convergence_tolerance = convergence_tolerance
        self.max_iterations = max_iterations

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def predict_progressive(
        self,
        kriging_model,
        prediction_coordinates: np.ndarray,
        return_variance: bool = False,
        resume_from_checkpoint: bool = True,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Perform progressive kriging with checkpointing.

        Parameters
        ----------
        kriging_model : object
            Fitted kriging model.
        prediction_coordinates : ndarray
            Prediction coordinates.
        return_variance : bool, default=False
            Whether to return variance estimates.
        resume_from_checkpoint : bool, default=True
            Whether to resume from existing checkpoint.

        Returns
        -------
        predictions : ndarray
            Final predictions.
        variances : ndarray, optional
            Final variances if return_variance=True.
        """
        n_pred = len(prediction_coordinates)

        # Check for existing checkpoint
        checkpoint_file = self.checkpoint_dir / "latest_checkpoint.npz"

        if resume_from_checkpoint and checkpoint_file.exists():
            print("Resuming from checkpoint...")
            checkpoint = np.load(checkpoint_file)

            start_idx = int(checkpoint["completed_predictions"])
            predictions = checkpoint["predictions"]
            variances = checkpoint.get("variances", None) if return_variance else None

            print(f"Resuming from prediction {start_idx:,}/{n_pred:,}")
        else:
            print("Starting new progressive computation...")
            start_idx = 0
            predictions = np.full(n_pred, np.nan)
            variances = np.full(n_pred, np.nan) if return_variance else None

        # Progressive computation
        try:
            from tqdm import tqdm

            progress_bar = tqdm(
                total=n_pred, initial=start_idx, desc="Progressive kriging"
            )
        except ImportError:
            progress_bar = None

        for i in range(start_idx, n_pred):
            # Predict single point
            coord = prediction_coordinates[i : i + 1]

            if return_variance:
                pred, var = kriging_model.predict(coord, return_variance=True)
                predictions[i] = pred[0]
                variances[i] = var[0]
            else:
                pred = kriging_model.predict(coord)
                predictions[i] = pred[0]

            if progress_bar:
                progress_bar.update(1)

            # Checkpoint periodically
            if (i + 1) % self.checkpoint_interval == 0:
                self._save_checkpoint(i + 1, predictions, variances, checkpoint_file)

                if progress_bar:
                    progress_bar.set_postfix({"checkpointed": i + 1})

            # Check for early stopping (if applicable)
            if self.max_iterations and i >= self.max_iterations:
                print(f"Stopping at maximum iterations: {self.max_iterations}")
                break

        if progress_bar:
            progress_bar.close()

        # Final checkpoint
        self._save_checkpoint(n_pred, predictions, variances, checkpoint_file)

        print("Progressive computation completed!")

        if return_variance:
            return predictions, variances
        else:
            return predictions

    def _save_checkpoint(
        self,
        completed_predictions: int,
        predictions: np.ndarray,
        variances: Optional[np.ndarray],
        checkpoint_file: Path,
    ) -> None:
        """Save computation checkpoint."""
        checkpoint_data = {
            "completed_predictions": completed_predictions,
            "predictions": predictions,
            "timestamp": np.datetime64("now"),
        }

        if variances is not None:
            checkpoint_data["variances"] = variances

        # Save to temporary file first, then rename (atomic operation)
        temp_file = checkpoint_file.with_suffix(".tmp")
        np.savez_compressed(temp_file, **checkpoint_data)
        temp_file.rename(checkpoint_file)

    def clear_checkpoints(self) -> None:
        """Clear all checkpoint files."""
        for checkpoint_file in self.checkpoint_dir.glob("*.npz"):
            checkpoint_file.unlink()

        print("All checkpoints cleared.")

    def get_checkpoint_info(self) -> Optional[Dict[str, Any]]:
        """Get information about existing checkpoint."""
        checkpoint_file = self.checkpoint_dir / "latest_checkpoint.npz"

        if not checkpoint_file.exists():
            return None

        checkpoint = np.load(checkpoint_file)

        return {
            "completed_predictions": int(checkpoint["completed_predictions"]),
            "timestamp": str(checkpoint["timestamp"]),
            "has_variances": "variances" in checkpoint,
            "checkpoint_file": str(checkpoint_file),
        }
