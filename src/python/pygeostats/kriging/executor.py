# src/python/pygeostats/kriging/executor.py
"""Parallel execution framework for large-scale kriging computations."""

from __future__ import annotations

import multiprocessing as mp
import warnings
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
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
        Size of each tile. If float, assumes square tiles.
    overlap : float, default=0.1
        Overlap fraction between adjacent tiles.

    Returns
    -------
    tiles : list of tuples
        List of tile bounds (xmin, ymin, xmax, ymax).
    """
    xmin, ymin, xmax, ymax = bounds

    if isinstance(tile_size, (int, float)):
        tile_width = tile_height = tile_size
    else:
        tile_width, tile_height = tile_size

    # Calculate overlap in absolute units
    overlap_width = tile_width * overlap
    overlap_height = tile_height * overlap

    # Generate tiles with overlap
    tiles = []

    x_start = xmin
    while x_start < xmax:
        x_end = min(x_start + tile_width, xmax)

        y_start = ymin
        while y_start < ymax:
            y_end = min(y_start + tile_height, ymax)

            # Extend bounds by overlap (but don't exceed domain bounds)
            tile_xmin = max(x_start - overlap_width, xmin)
            tile_ymin = max(y_start - overlap_height, ymin)
            tile_xmax = min(x_end + overlap_width, xmax)
            tile_ymax = min(y_end + overlap_height, ymax)

            tiles.append((tile_xmin, tile_ymin, tile_xmax, tile_ymax))

            y_start = y_end

        x_start = x_end

    return tiles


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

        # Create chunks
        chunks = list(chunk_indices(n_pred, self.chunk_size))
        n_chunks = len(chunks)

        print(f"Processing {n_chunks} chunks of size ~{self.chunk_size}")

        # Prepare worker function
        worker_func = partial(
            _predict_chunk_worker,
            kriging_model=kriging_model,
            return_variance=return_variance,
        )

        # Execute in parallel
        if self.execution_method == "sequential":
            results = []
            for i, (start, end) in enumerate(chunks):
                chunk_coords = prediction_coordinates[start:end]
                result = worker_func(chunk_coords)
                results.append(result)

                if self.progress_callback:
                    self.progress_callback(i + 1, n_chunks)

        elif self.execution_method == "thread":
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                futures = []
                for start, end in chunks:
                    chunk_coords = prediction_coordinates[start:end]
                    future = executor.submit(worker_func, chunk_coords)
                    futures.append(future)

                results = self._collect_results_with_progress(futures, n_chunks)

        elif self.execution_method == "process":
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                futures = []
                for start, end in chunks:
                    chunk_coords = prediction_coordinates[start:end]
                    future = executor.submit(worker_func, chunk_coords)
                    futures.append(future)

                results = self._collect_results_with_progress(futures, n_chunks)

        # Combine results
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

        # Calculate spatial bounds and tile size
        bounds = (
            prediction_coordinates[:, 0].min(),
            prediction_coordinates[:, 1].min(),
            prediction_coordinates[:, 0].max(),
            prediction_coordinates[:, 1].max(),
        )

        # Estimate tile size based on number of workers and points
        n_pred = len(prediction_coordinates)
        target_points_per_tile = max(self.chunk_size, n_pred // (self.n_workers * 2))

        # Estimate tile size assuming uniform distribution
        domain_area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
        density = n_pred / domain_area if domain_area > 0 else 1
        tile_area = target_points_per_tile / density
        tile_size = np.sqrt(tile_area)

        # Generate tiles
        tiles = spatial_tiles(bounds, tile_size, overlap=0.1)

        print(f"Processing {len(tiles)} spatial tiles")

        # Assign points to tiles
        tile_assignments = self._assign_points_to_tiles(prediction_coordinates, tiles)

        # Prepare worker function
        worker_func = partial(
            _predict_spatial_tile_worker,
            kriging_model=kriging_model,
            return_variance=return_variance,
        )

        # Execute in parallel
        if self.execution_method == "sequential":
            results = []
            for i, (tile_coords, tile_indices) in enumerate(tile_assignments):
                if len(tile_coords) > 0:
                    result = worker_func((tile_coords, tile_indices))
                    results.append(result)

                if self.progress_callback:
                    self.progress_callback(i + 1, len(tile_assignments))

        elif self.execution_method == "thread":
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                futures = []
                for tile_coords, tile_indices in tile_assignments:
                    if len(tile_coords) > 0:
                        future = executor.submit(
                            worker_func, (tile_coords, tile_indices)
                        )
                        futures.append(future)

                results = self._collect_results_with_progress(futures, len(futures))

        elif self.execution_method == "process":
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                futures = []
                for tile_coords, tile_indices in tile_assignments:
                    if len(tile_coords) > 0:
                        future = executor.submit(
                            worker_func, (tile_coords, tile_indices)
                        )
                        futures.append(future)

                results = self._collect_results_with_progress(futures, len(futures))

        # Combine spatial results
        return self._combine_spatial_results(
            results, len(prediction_coordinates), return_variance
        )

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

    def _assign_points_to_tiles(
        self, coordinates: np.ndarray, tiles: List[Tuple[float, float, float, float]]
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Assign prediction points to spatial tiles."""
        assignments = []

        for tile_bounds in tiles:
            xmin, ymin, xmax, ymax = tile_bounds

            # Find points within tile
            mask = (
                (coordinates[:, 0] >= xmin)
                & (coordinates[:, 0] <= xmax)
                & (coordinates[:, 1] >= ymin)
                & (coordinates[:, 1] <= ymax)
            )

            tile_indices = np.where(mask)[0]
            tile_coords = coordinates[mask]

            assignments.append((tile_coords, tile_indices))

        return assignments

    def _collect_results_with_progress(self, futures, total_tasks: int):
        """Collect results from futures with progress tracking."""
        results = []
        completed = 0

        try:
            from tqdm import tqdm

            progress_bar = tqdm(total=total_tasks, desc="Processing tasks")
        except ImportError:
            progress_bar = None

        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                completed += 1

                if progress_bar:
                    progress_bar.update(1)
                elif self.progress_callback:
                    self.progress_callback(completed, total_tasks)
                elif completed % max(1, total_tasks // 10) == 0:
                    print(f"Completed {completed}/{total_tasks} tasks")

            except Exception as e:
                warnings.warn(f"Task failed: {e!s}", stacklevel=2)
                results.append(None)

        if progress_bar:
            progress_bar.close()

        return results

    def _combine_chunk_results(
        self, results: List, return_variance: bool
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Combine results from chunked processing."""
        # Filter out failed results
        valid_results = [r for r in results if r is not None]

        if not valid_results:
            raise RuntimeError("All prediction tasks failed")

        if return_variance:
            predictions_list = [r[0] for r in valid_results]
            variances_list = [r[1] for r in valid_results]

            predictions = np.concatenate(predictions_list)
            variances = np.concatenate(variances_list)

            return predictions, variances
        else:
            predictions = np.concatenate(valid_results)
            return predictions

    def _combine_spatial_results(
        self, results: List, n_total: int, return_variance: bool
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Combine results from spatial tiling."""
        # Initialize output arrays
        predictions = np.full(n_total, np.nan)
        variances = np.full(n_total, np.nan) if return_variance else None

        # Assign results to correct positions
        for result in results:
            if result is None:
                continue

            if return_variance:
                tile_predictions, tile_variances, tile_indices = result
                predictions[tile_indices] = tile_predictions
                variances[tile_indices] = tile_variances
            else:
                tile_predictions, tile_indices = result
                predictions[tile_indices] = tile_predictions

        # Check for missing predictions
        missing_mask = np.isnan(predictions)
        if np.any(missing_mask):
            warnings.warn(
                f"{np.sum(missing_mask)} predictions failed and are set to NaN",
                stacklevel=2,
            )

        if return_variance:
            return predictions, variances
        else:
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
    """Worker function for chunk-based parallel prediction."""
    try:
        if return_variance:
            predictions, variances = kriging_model.predict(
                chunk_coordinates, return_variance=True
            )
            return predictions, variances
        else:
            predictions = kriging_model.predict(chunk_coordinates)
            return predictions
    except Exception:
        # Return NaN array on failure
        n_pred = len(chunk_coordinates)
        if return_variance:
            return (np.full(n_pred, np.nan), np.full(n_pred, np.nan))
        else:
            return np.full(n_pred, np.nan)


def _predict_spatial_tile_worker(
    tile_data: Tuple[np.ndarray, np.ndarray], kriging_model, return_variance: bool
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Worker function for spatial tile-based parallel prediction."""
    tile_coords, tile_indices = tile_data

    if len(tile_coords) == 0:
        if return_variance:
            return np.array([]), np.array([]), np.array([])
        else:
            return np.array([]), np.array([])

    try:
        if return_variance:
            predictions, variances = kriging_model.predict(
                tile_coords, return_variance=True
            )
            return predictions, variances, tile_indices
        else:
            predictions = kriging_model.predict(tile_coords)
            return predictions, tile_indices
    except Exception:
        # Return NaN arrays on failure
        n_pred = len(tile_coords)
        if return_variance:
            return (np.full(n_pred, np.nan), np.full(n_pred, np.nan), tile_indices)
        else:
            return np.full(n_pred, np.nan), tile_indices


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
