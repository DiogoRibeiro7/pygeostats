# src/python/pygeostats/optimization/memory.py
"""Memory optimization utilities for large-scale spatial computations."""

from __future__ import annotations

import gc
import os
import psutil
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, lil_matrix

from ..utils.validation import validate_coordinates


class MemoryManager:
    """Memory management utilities for large spatial datasets."""
    
    def __init__(self, warning_threshold_gb: float = 0.8, error_threshold_gb: float = 0.95):
        """
        Initialize memory manager.
        
        Parameters
        ----------
        warning_threshold_gb : float, default=0.8
            Memory usage fraction to trigger warnings.
        error_threshold_gb : float, default=0.95
            Memory usage fraction to trigger errors.
        """
        self.warning_threshold = warning_threshold_gb
        self.error_threshold = error_threshold_gb
        self.process = psutil.Process()
        
    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory usage information."""
        memory_info = self.process.memory_info()
        virtual_memory = psutil.virtual_memory()
        
        return {
            'process_gb': memory_info.rss / (1024**3),
            'process_percent': memory_info.rss / virtual_memory.total * 100,
            'system_available_gb': virtual_memory.available / (1024**3),
            'system_used_percent': virtual_memory.percent,
            'system_total_gb': virtual_memory.total / (1024**3)
        }
    
    def check_memory_usage(self, operation: str = "operation") -> None:
        """Check memory usage and warn/error if thresholds exceeded."""
        info = self.get_memory_info()
        usage_fraction = info['system_used_percent'] / 100
        
        if usage_fraction > self.error_threshold:
            raise MemoryError(
                f"Memory usage too high for {operation}: "
                f"{usage_fraction:.1%} (threshold: {self.error_threshold:.1%})"
            )
        elif usage_fraction > self.warning_threshold:
            warnings.warn(
                f"High memory usage for {operation}: "
                f"{usage_fraction:.1%} (threshold: {self.warning_threshold:.1%})",
                UserWarning
            )
    
    def estimate_array_memory(self, shape: Tuple[int, ...], dtype: np.dtype = np.float64) -> float:
        """Estimate memory requirements for an array in GB."""
        elements = np.prod(shape)
        bytes_per_element = np.dtype(dtype).itemsize
        return elements * bytes_per_element / (1024**3)
    
    def can_allocate(self, shape: Tuple[int, ...], dtype: np.dtype = np.float64) -> bool:
        """Check if an array can be allocated without exceeding memory limits."""
        required_gb = self.estimate_array_memory(shape, dtype)
        available_gb = self.get_memory_info()['system_available_gb']
        return required_gb < available_gb * self.warning_threshold
    
    @contextmanager
    def memory_monitor(self, operation: str = "operation"):
        """Context manager to monitor memory usage during operations."""
        initial_info = self.get_memory_info()
        
        try:
            yield self
            
        finally:
            final_info = self.get_memory_info()
            memory_increase = final_info['process_gb'] - initial_info['process_gb']
            
            if memory_increase > 1.0:  # More than 1GB increase
                print(f"Memory increase for {operation}: {memory_increase:.2f} GB")
    
    def cleanup_memory(self) -> float:
        """Force garbage collection and return memory freed."""
        initial_memory = self.get_memory_info()['process_gb']
        gc.collect()
        final_memory = self.get_memory_info()['process_gb']
        return initial_memory - final_memory


class SparseDistanceMatrix:
    """Sparse distance matrix with efficient storage and computation."""
    
    def __init__(
        self, 
        coords: np.ndarray, 
        max_distance: Optional[float] = None,
        max_neighbors: Optional[int] = None,
        chunk_size: int = 10000
    ):
        """
        Initialize sparse distance matrix.
        
        Parameters
        ----------
        coords : ndarray, shape (n, d)
            Coordinate array.
        max_distance : float, optional
            Maximum distance to store. Distances beyond this are set to 0.
        max_neighbors : int, optional
            Maximum number of neighbors to keep per point.
        chunk_size : int, default=10000
            Chunk size for computation.
        """
        self.coords = validate_coordinates(coords)
        self.n_points = len(coords)
        self.max_distance = max_distance
        self.max_neighbors = max_neighbors
        self.chunk_size = chunk_size
        
        self._distance_matrix = None
        self._is_computed = False
        
    def compute(self, show_progress: bool = True) -> 'SparseDistanceMatrix':
        """Compute the sparse distance matrix."""
        if self._is_computed:
            return self
        
        memory_manager = MemoryManager()
        
        # Estimate memory requirements
        if self.max_distance is None and self.max_neighbors is None:
            # Dense case - check if we can fit it
            dense_memory = memory_manager.estimate_array_memory(
                (self.n_points, self.n_points), np.float32
            )
            if dense_memory > 10:  # More than 10GB
                warnings.warn(
                    f"Dense distance matrix would require {dense_memory:.1f} GB. "
                    "Consider setting max_distance or max_neighbors for sparse computation.",
                    UserWarning
                )
        
        if self.max_neighbors is not None:
            self._compute_knn_sparse(show_progress)
        elif self.max_distance is not None:
            self._compute_distance_sparse(show_progress)
        else:
            self._compute_dense()
        
        self._is_computed = True
        return self
    
    def _compute_knn_sparse(self, show_progress: bool) -> None:
        """Compute sparse matrix with k nearest neighbors."""
        from sklearn.neighbors import NearestNeighbors
        
        if show_progress:
            print(f"Computing {self.max_neighbors}-NN sparse distance matrix...")
        
        nbrs = NearestNeighbors(
            n_neighbors=min(self.max_neighbors + 1, self.n_points),
            algorithm='auto'
        ).fit(self.coords)
        
        distances, indices = nbrs.kneighbors(self.coords)
        
        # Remove self-distances
        distances = distances[:, 1:]
        indices = indices[:, 1:]
        
        # Build sparse matrix
        row_indices = np.repeat(np.arange(self.n_points), distances.shape[1])
        col_indices = indices.ravel()
        distance_values = distances.ravel()
        
        self._distance_matrix = csr_matrix(
            (distance_values, (row_indices, col_indices)),
            shape=(self.n_points, self.n_points)
        )
        
        # Make symmetric
        self._distance_matrix = self._distance_matrix.maximum(self._distance_matrix.T)
    
    def _compute_distance_sparse(self, show_progress: bool) -> None:
        """Compute sparse matrix with distance threshold."""
        if show_progress:
            print(f"Computing sparse distance matrix (max_distance={self.max_distance})...")
        
        # Use chunked computation to manage memory
        row_indices = []
        col_indices = []
        distances = []
        
        total_chunks = (self.n_points + self.chunk_size - 1) // self.chunk_size
        
        for i, chunk_start in enumerate(range(0, self.n_points, self.chunk_size)):
            if show_progress and i % 10 == 0:
                print(f"  Processing chunk {i+1}/{total_chunks}")
            
            chunk_end = min(chunk_start + self.chunk_size, self.n_points)
            chunk_coords = self.coords[chunk_start:chunk_end]
            
            # Compute distances from chunk to all points
            from scipy.spatial.distance import cdist
            chunk_distances = cdist(chunk_coords, self.coords)
            
            # Find points within threshold
            valid_mask = (chunk_distances <= self.max_distance) & (chunk_distances > 0)
            chunk_rows, chunk_cols = np.where(valid_mask)
            
            # Adjust row indices for global indexing
            global_rows = chunk_rows + chunk_start
            
            row_indices.extend(global_rows)
            col_indices.extend(chunk_cols)
            distances.extend(chunk_distances[valid_mask])
        
        self._distance_matrix = csr_matrix(
            (distances, (row_indices, col_indices)),
            shape=(self.n_points, self.n_points)
        )
    
    def _compute_dense(self) -> None:
        """Compute dense distance matrix."""
        from scipy.spatial.distance import pdist, squareform
        
        print("Computing dense distance matrix...")
        distances = pdist(self.coords)
        self._distance_matrix = squareform(distances)
    
    @property
    def matrix(self) -> Union[np.ndarray, sp.csr_matrix]:
        """Get the distance matrix."""
        if not self._is_computed:
            self.compute()
        return self._distance_matrix
    
    @property
    def is_sparse(self) -> bool:
        """Check if matrix is sparse."""
        return sp.issparse(self._distance_matrix) if self._is_computed else True
    
    def get_neighbors(self, point_idx: int, k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get k nearest neighbors for a point.
        
        Parameters
        ----------
        point_idx : int
            Index of the query point.
        k : int, optional
            Number of neighbors. If None, returns all non-zero distances.
            
        Returns
        -------
        distances : ndarray
            Distances to neighbors.
        indices : ndarray
            Indices of neighbors.
        """
        if not self._is_computed:
            self.compute()
        
        if self.is_sparse:
            row = self._distance_matrix.getrow(point_idx)
            nonzero_indices = row.nonzero()[1]
            nonzero_distances = row.data
            
            if k is not None and len(nonzero_indices) > k:
                # Sort and take k closest
                sort_indices = np.argsort(nonzero_distances)[:k]
                nonzero_indices = nonzero_indices[sort_indices]
                nonzero_distances = nonzero_distances[sort_indices]
            
            return nonzero_distances, nonzero_indices
        else:
            distances = self._distance_matrix[point_idx]
            if k is not None:
                # Get k+1 smallest (including self-distance)
                indices = np.argpartition(distances, k+1)[:k+1]
                # Remove self-distance
                mask = indices != point_idx
                indices = indices[mask][:k]
                return distances[indices], indices
            else:
                # Return all non-self distances
                indices = np.arange(len(distances))
                mask = indices != point_idx
                return distances[mask], indices[mask]
    
    def memory_usage_gb(self) -> float:
        """Get memory usage of the distance matrix in GB."""
        if not self._is_computed:
            return 0.0
        
        if self.is_sparse:
            # Sparse matrix memory usage
            data_bytes = self._distance_matrix.data.nbytes
            indices_bytes = self._distance_matrix.indices.nbytes
            indptr_bytes = self._distance_matrix.indptr.nbytes
            return (data_bytes + indices_bytes + indptr_bytes) / (1024**3)
        else:
            return self._distance_matrix.nbytes / (1024**3)
    
    def sparsity_ratio(self) -> float:
        """Get sparsity ratio (fraction of zero elements)."""
        if not self._is_computed:
            return 0.0
        
        if self.is_sparse:
            total_elements = self.n_points * self.n_points
            nonzero_elements = self._distance_matrix.nnz
            return 1.0 - (nonzero_elements / total_elements)
        else:
            nonzero_elements = np.count_nonzero(self._distance_matrix)
            total_elements = self._distance_matrix.size
            return 1.0 - (nonzero_elements / total_elements)


class ChunkedArrayProcessor:
    """Process large arrays in memory-efficient chunks."""
    
    def __init__(self, max_memory_gb: float = 4.0):
        """
        Initialize chunked processor.
        
        Parameters
        ----------
        max_memory_gb : float, default=4.0
            Maximum memory to use for chunk processing.
        """
        self.max_memory_gb = max_memory_gb
        self.memory_manager = MemoryManager()
    
    def optimal_chunk_size(
        self, 
        array_shape: Tuple[int, ...], 
        dtype: np.dtype = np.float64,
        operations_factor: float = 3.0
    ) -> int:
        """
        Calculate optimal chunk size based on memory constraints.
        
        Parameters
        ----------
        array_shape : tuple
            Shape of the array to process.
        dtype : np.dtype, default=np.float64
            Data type of the array.
        operations_factor : float, default=3.0
            Multiplier for temporary arrays created during operations.
            
        Returns
        -------
        chunk_size : int
            Optimal chunk size for the first dimension.
        """
        bytes_per_element = np.dtype(dtype).itemsize
        elements_per_row = np.prod(array_shape[1:]) if len(array_shape) > 1 else 1
        bytes_per_row = elements_per_row * bytes_per_element
        
        # Account for temporary arrays
        effective_bytes_per_row = bytes_per_row * operations_factor
        
        # Calculate chunk size that fits in memory budget
        max_bytes = self.max_memory_gb * (1024**3)
        chunk_size = int(max_bytes // effective_bytes_per_row)
        
        # Ensure reasonable bounds
        chunk_size = max(1, min(chunk_size, array_shape[0]))
        
        return chunk_size
    
    def process_chunks(
        self, 
        array: np.ndarray, 
        process_func, 
        chunk_size: Optional[int] = None,
        show_progress: bool = True,
        **kwargs
    ) -> Iterator:
        """
        Process array in chunks.
        
        Parameters
        ----------
        array : ndarray
            Array to process.
        process_func : callable
            Function to apply to each chunk.
        chunk_size : int, optional
            Chunk size. Auto-calculated if None.
        show_progress : bool, default=True
            Whether to show progress.
        **kwargs
            Additional arguments for process_func.
            
        Yields
        ------
        result
            Result from processing each chunk.
        """
        if chunk_size is None:
            chunk_size = self.optimal_chunk_size(array.shape, array.dtype)
        
        n_chunks = (len(array) + chunk_size - 1) // chunk_size
        
        if show_progress:
            try:
                from tqdm import tqdm
                chunk_iter = tqdm(range(n_chunks), desc="Processing chunks")
            except ImportError:
                chunk_iter = range(n_chunks)
                print(f"Processing {n_chunks} chunks...")
        else:
            chunk_iter = range(n_chunks)
        
        for i in chunk_iter:
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, len(array))
            chunk = array[start_idx:end_idx]
            
            yield process_func(chunk, **kwargs)
    
    def concatenate_results(self, results: Iterator, output_shape: Optional[Tuple[int, ...]] = None) -> np.ndarray:
        """Concatenate chunked results into a single array."""
        result_list = list(results)
        
        if not result_list:
            return np.array([])
        
        # Try to concatenate
        try:
            return np.concatenate(result_list, axis=0)
        except ValueError as e:
            # Handle inconsistent shapes
            print(f"Warning: Could not concatenate results: {e}")
            return result_list


def create_memory_mapped_array(
    filepath: Union[str, Path],
    shape: Tuple[int, ...],
    dtype: np.dtype = np.float64,
    mode: str = 'w+'
) -> np.memmap:
    """
    Create a memory-mapped array for large dataset storage.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the memory-mapped file.
    shape : tuple
        Shape of the array.
    dtype : np.dtype, default=np.float64
        Data type.
    mode : str, default='w+'
        File access mode.
        
    Returns
    -------
    memmap : np.memmap
        Memory-mapped array.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    return np.memmap(filepath, dtype=dtype, mode=mode, shape=shape)


def estimate_kriging_memory(
    n_known: int,
    n_pred: int,
    use_sparse: bool = False,
    dtype: np.dtype = np.float64
) -> Dict[str, float]:
    """
    Estimate memory requirements for kriging computation.
    
    Parameters
    ----------
    n_known : int
        Number of known points.
    n_pred : int
        Number of prediction points.
    use_sparse : bool, default=False
        Whether sparse matrices will be used.
    dtype : np.dtype, default=np.float64
        Data type for computations.
        
    Returns
    -------
    memory_estimates : dict
        Dictionary with memory estimates in GB.
    """
    bytes_per_element = np.dtype(dtype).itemsize
    
    # Covariance matrix
    if use_sparse:
        # Assume 10% sparsity for sparse matrices
        cov_matrix_gb = (n_known * n_known * 0.1 * bytes_per_element) / (1024**3)
    else:
        cov_matrix_gb = (n_known * n_known * bytes_per_element) / (1024**3)
    
    # Kriging system (augmented matrix)
    system_matrix_gb = ((n_known + 1) * (n_known + 1) * bytes_per_element) / (1024**3)
    
    # Prediction arrays
    predictions_gb = (n_pred * bytes_per_element) / (1024**3)
    
    # Working memory (temporary arrays)
    working_memory_gb = max(cov_matrix_gb, system_matrix_gb) * 2
    
    total_gb = cov_matrix_gb + system_matrix_gb + predictions_gb + working_memory_gb
    
    return {
        'covariance_matrix_gb': cov_matrix_gb,
        'system_matrix_gb': system_matrix_gb,
        'predictions_gb': predictions_gb,
        'working_memory_gb': working_memory_gb,
        'total_gb': total_gb,
        'sparse_used': use_sparse
    }


class MemoryEfficientKriging:
    """Memory-efficient kriging for large datasets."""
    
    def __init__(
        self,
        max_memory_gb: float = 8.0,
        use_sparse: bool = True,
        chunk_size: Optional[int] = None
    ):
        """
        Initialize memory-efficient kriging.
        
        Parameters
        ----------
        max_memory_gb : float, default=8.0
            Maximum memory to use.
        use_sparse : bool, default=True
            Whether to use sparse matrices when beneficial.
        chunk_size : int, optional
            Chunk size for processing. Auto-determined if None.
        """
        self.max_memory_gb = max_memory_gb
        self.use_sparse = use_sparse
        self.chunk_size = chunk_size
        self.memory_manager = MemoryManager()
        self.processor = ChunkedArrayProcessor(max_memory_gb)
    
    def should_use_chunks(self, n_known: int, n_pred: int) -> bool:
        """Determine if chunked processing is needed."""
        memory_estimate = estimate_kriging_memory(
            n_known, n_pred, self.use_sparse
        )
        return memory_estimate['total_gb'] > self.max_memory_gb
    
    def predict_chunked(
        self,
        kriging_model,
        pred_coords: np.ndarray,
        return_variance: bool = False,
        show_progress: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict using chunked processing for memory efficiency.
        
        Parameters
        ----------
        kriging_model
            Fitted kriging model.
        pred_coords : ndarray
            Prediction coordinates.
        return_variance : bool, default=False
            Whether to return variance estimates.
        show_progress : bool, default=True
            Whether to show progress.
            
        Returns
        -------
        predictions : ndarray
            Predicted values.
        variances : ndarray, optional
            Prediction variances.
        """
        n_pred = len(pred_coords)
        
        if self.chunk_size is None:
            # Estimate chunk size based on memory constraints
            memory_per_pred = self.memory_manager.estimate_array_memory((1000,), np.float64)
            chunk_size = int(self.max_memory_gb / memory_per_pred)
            chunk_size = max(100, min(chunk_size, n_pred))
        else:
            chunk_size = self.chunk_size
        
        predictions = np.zeros(n_pred)
        variances = np.zeros(n_pred) if return_variance else None
        
        def process_chunk(chunk_coords):
            if return_variance:
                return kriging_model.predict(chunk_coords, return_variance=True)
            else:
                return kriging_model.predict(chunk_coords), None
        
        chunk_results = self.processor.process_chunks(
            pred_coords,
            process_chunk,
            chunk_size=chunk_size,
            show_progress=show_progress
        )
        
        # Collect results
        start_idx = 0
        for chunk_pred, chunk_var in chunk_results:
            end_idx = start_idx + len(chunk_pred)
            predictions[start_idx:end_idx] = chunk_pred
            
            if return_variance and chunk_var is not None:
                variances[start_idx:end_idx] = chunk_var
            
            start_idx = end_idx
        
        if return_variance:
            return predictions, variances
        else:
            return predictions


def optimize_dtype_for_precision(
    values: np.ndarray,
    target_precision: float = 1e-6
) -> np.dtype:
    """
    Optimize data type based on required precision.
    
    Parameters
    ----------
    values : ndarray
        Array values to analyze.
    target_precision : float, default=1e-6
        Required precision.
        
    Returns
    -------
    optimal_dtype : np.dtype
        Optimal data type.
    """
    value_range = np.ptp(values)  # peak-to-peak (max - min)
    
    # Check if float32 provides sufficient precision
    float32_precision = value_range * np.finfo(np.float32).eps
    
    if float32_precision <= target_precision:
        return np.float32
    else:
        return np.float64


class ProgressiveComputation:
    """Handle progressive computation with early stopping and checkpointing."""
    
    def __init__(
        self,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        patience: int = 10,
        checkpoint_interval: int = 100
    ):
        """
        Initialize progressive computation handler.
        
        Parameters
        ----------
        max_iterations : int, default=1000
            Maximum number of iterations.
        tolerance : float, default=1e-6
            Convergence tolerance.
        patience : int, default=10
            Number of iterations without improvement before stopping.
        checkpoint_interval : int, default=100
            Iterations between checkpoints.
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.patience = patience
        self.checkpoint_interval = checkpoint_interval
        
        self.iteration = 0
        self.best_score = float('inf')
        self.patience_counter = 0
        self.converged = False
        
    def should_continue(self, current_score: float) -> bool:
        """Check if computation should continue."""
        self.iteration += 1
        
        # Check convergence
        improvement = self.best_score - current_score
        
        if improvement > self.tolerance:
            self.best_score = current_score
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        # Check stopping criteria
        if self.patience_counter >= self.patience:
            self.converged = True
            return False
        
        if self.iteration >= self.max_iterations:
            return False
        
        return True
    
    def should_checkpoint(self) -> bool:
        """Check if a checkpoint should be created."""
        return self.iteration % self.checkpoint_interval == 0


# Global memory manager instance
_memory_manager = MemoryManager()

def get_memory_manager() -> MemoryManager:
    """Get the global memory manager."""
    return _memory_manager

def memory_profile(func):
    """Decorator to profile memory usage of a function."""
    def wrapper(*args, **kwargs):
        manager = get_memory_manager()
        with manager.memory_monitor(func.__name__):
            return func(*args, **kwargs)
    return wrapper
