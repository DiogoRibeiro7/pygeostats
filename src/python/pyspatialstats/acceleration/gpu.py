# src/python/pyspatialstats/acceleration/gpu.py
"""GPU acceleration utilities using CuPy for large-scale spatial computations."""

from __future__ import annotations

import warnings
from typing import Optional, Tuple, Union

import numpy as np

try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    cp = None
    _HAS_CUPY = False

from ..utils.validation import validate_coordinates


class GPUAccelerator:
    """GPU acceleration manager for spatial computations."""
    
    def __init__(self, enable_gpu: bool = True, device_id: Optional[int] = None):
        """
        Initialize GPU accelerator.
        
        Parameters
        ----------
        enable_gpu : bool, default=True
            Whether to enable GPU acceleration if available.
        device_id : int, optional
            Specific GPU device ID to use.
        """
        self.gpu_available = _HAS_CUPY and enable_gpu
        self.device_id = device_id
        
        if enable_gpu and not _HAS_CUPY:
            warnings.warn(
                "GPU acceleration requested but CuPy not available. "
                "Install CuPy to enable GPU acceleration: pip install cupy",
                UserWarning
            )
        
        if self.gpu_available:
            if device_id is not None:
                cp.cuda.Device(device_id).use()
            self._check_gpu_memory()
    
    def _check_gpu_memory(self) -> None:
        """Check available GPU memory."""
        if not self.gpu_available:
            return
            
        try:
            mempool = cp.get_default_memory_pool()
            free_bytes = mempool.free_bytes()
            total_bytes = mempool.total_bytes()
            
            if total_bytes == 0:  # No memory allocated yet
                # Try to allocate a small array to get memory info
                test_array = cp.zeros(1000)
                del test_array
                free_bytes = mempool.free_bytes()
                total_bytes = mempool.total_bytes()
            
            self.gpu_memory_gb = total_bytes / (1024**3)
            self.gpu_free_gb = free_bytes / (1024**3)
            
        except Exception:
            self.gpu_memory_gb = 0.0
            self.gpu_free_gb = 0.0
    
    @property
    def is_available(self) -> bool:
        """Check if GPU acceleration is available."""
        return self.gpu_available
    
    def get_memory_info(self) -> dict:
        """Get GPU memory information."""
        if not self.gpu_available:
            return {'available': False}
        
        try:
            mempool = cp.get_default_memory_pool()
            return {
                'available': True,
                'total_gb': mempool.total_bytes() / (1024**3),
                'used_gb': mempool.used_bytes() / (1024**3),
                'free_gb': mempool.free_bytes() / (1024**3)
            }
        except Exception:
            return {'available': False, 'error': 'Could not get memory info'}
    
    def to_gpu(self, array: np.ndarray) -> Union[np.ndarray, 'cp.ndarray']:
        """Transfer array to GPU if available."""
        if self.gpu_available:
            return cp.asarray(array)
        return array
    
    def to_cpu(self, array: Union[np.ndarray, 'cp.ndarray']) -> np.ndarray:
        """Transfer array to CPU."""
        if self.gpu_available and hasattr(array, 'get'):
            return array.get()
        return np.asarray(array)
    
    def euclidean_distances_gpu(
        self, 
        coords1: np.ndarray, 
        coords2: Optional[np.ndarray] = None,
        chunk_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute euclidean distances on GPU with automatic chunking.
        
        Parameters
        ----------
        coords1 : ndarray, shape (n1, d)
            First set of coordinates.
        coords2 : ndarray, shape (n2, d), optional
            Second set of coordinates. If None, compute pairwise distances in coords1.
        chunk_size : int, optional
            Chunk size for memory management. Auto-determined if None.
            
        Returns
        -------
        distances : ndarray, shape (n1, n2) or (n1, n1)
            Distance matrix.
        """
        if not self.gpu_available:
            return self._euclidean_distances_cpu(coords1, coords2)
        
        coords1 = validate_coordinates(coords1)
        if coords2 is None:
            coords2 = coords1
            symmetric = True
        else:
            coords2 = validate_coordinates(coords2)
            symmetric = False
        
        n1, n2 = len(coords1), len(coords2)
        
        # Estimate memory requirements and determine chunk size
        if chunk_size is None:
            chunk_size = self._estimate_chunk_size(n1, n2, coords1.shape[1])
        
        # Transfer to GPU in chunks
        if n1 * n2 > chunk_size * chunk_size:
            return self._chunked_distances_gpu(coords1, coords2, chunk_size, symmetric)
        else:
            return self._direct_distances_gpu(coords1, coords2)
    
    def _estimate_chunk_size(self, n1: int, n2: int, dims: int) -> int:
        """Estimate optimal chunk size based on available GPU memory."""
        if not self.gpu_available:
            return min(n1, n2, 10000)
        
        try:
            # Estimate memory requirements
            coord_memory = (n1 + n2) * dims * 8  # float64
            available_bytes = cp.get_default_memory_pool().free_bytes()
            
            # Reserve 20% for other operations
            usable_bytes = int(available_bytes * 0.8)
            
            # Calculate chunk size that fits in memory
            # Distance matrix needs chunk_size^2 * 8 bytes
            max_chunk_squared = (usable_bytes - coord_memory) // 8
            max_chunk = int(np.sqrt(max_chunk_squared))
            
            # Clamp to reasonable bounds
            chunk_size = max(1000, min(max_chunk, min(n1, n2), 50000))
            
        except Exception:
            chunk_size = min(n1, n2, 10000)
        
        return chunk_size
    
    def _direct_distances_gpu(self, coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
        """Compute distances directly on GPU."""
        try:
            coords1_gpu = cp.asarray(coords1, dtype=cp.float32)
            coords2_gpu = cp.asarray(coords2, dtype=cp.float32)
            
            # Compute squared distances using broadcasting
            diff = coords1_gpu[:, None, :] - coords2_gpu[None, :, :]
            squared_distances = cp.sum(diff * diff, axis=2)
            distances = cp.sqrt(squared_distances)
            
            return distances.get()
            
        except cp.cuda.memory.OutOfMemoryError:
            warnings.warn("GPU out of memory, falling back to CPU", UserWarning)
            return self._euclidean_distances_cpu(coords1, coords2)
    
    def _chunked_distances_gpu(
        self, 
        coords1: np.ndarray, 
        coords2: np.ndarray, 
        chunk_size: int,
        symmetric: bool
    ) -> np.ndarray:
        """Compute distances in chunks to manage GPU memory."""
        n1, n2 = len(coords1), len(coords2)
        distances = np.zeros((n1, n2), dtype=np.float32)
        
        coords2_gpu = cp.asarray(coords2, dtype=cp.float32)
        
        try:
            for i in range(0, n1, chunk_size):
                i_end = min(i + chunk_size, n1)
                coords1_chunk = cp.asarray(coords1[i:i_end], dtype=cp.float32)
                
                if symmetric:
                    # For symmetric case, only compute upper triangle
                    j_start = i
                else:
                    j_start = 0
                
                for j in range(j_start, n2, chunk_size):
                    j_end = min(j + chunk_size, n2)
                    coords2_chunk = coords2_gpu[j:j_end]
                    
                    # Compute distances for this chunk
                    diff = coords1_chunk[:, None, :] - coords2_chunk[None, :, :]
                    chunk_distances = cp.sqrt(cp.sum(diff * diff, axis=2))
                    distances[i:i_end, j:j_end] = chunk_distances.get()
                    
                    # For symmetric case, fill lower triangle
                    if symmetric and i != j:
                        distances[j:j_end, i:i_end] = chunk_distances.T.get()
            
            return distances
            
        except cp.cuda.memory.OutOfMemoryError:
            warnings.warn("GPU out of memory during chunked computation, falling back to CPU", UserWarning)
            return self._euclidean_distances_cpu(coords1, coords2)
    
    def _euclidean_distances_cpu(self, coords1: np.ndarray, coords2: Optional[np.ndarray]) -> np.ndarray:
        """Fallback CPU implementation."""
        from scipy.spatial.distance import cdist
        
        if coords2 is None:
            from scipy.spatial.distance import pdist, squareform
            return squareform(pdist(coords1))
        else:
            return cdist(coords1, coords2)
    
    def variogram_distances_gpu(
        self,
        coords: np.ndarray,
        max_pairs: int = 1_000_000,
        batch_size: int = 50_000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute variogram pair distances on GPU with memory management.
        
        Parameters
        ----------
        coords : ndarray, shape (n, d)
            Coordinate array.
        max_pairs : int, default=1_000_000
            Maximum number of pairs to process.
        batch_size : int, default=50_000
            Batch size for processing.
            
        Returns
        -------
        distances : ndarray
            Pairwise distances.
        indices : ndarray, shape (n_pairs, 2)
            Indices of point pairs.
        """
        if not self.gpu_available:
            return self._variogram_distances_cpu(coords, max_pairs, batch_size)
        
        n = len(coords)
        total_pairs = (n * (n - 1)) // 2
        
        if total_pairs <= max_pairs:
            # Compute all pairs
            distances = self.euclidean_distances_gpu(coords)
            i, j = np.triu_indices(n, k=1)
            return distances[i, j], np.column_stack([i, j])
        else:
            # Subsample pairs
            return self._subsample_pairs_gpu(coords, max_pairs, batch_size)
    
    def _subsample_pairs_gpu(
        self, 
        coords: np.ndarray, 
        max_pairs: int, 
        batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Subsample coordinate pairs for variogram computation."""
        n = len(coords)
        
        # Generate random pairs
        rng = np.random.default_rng()
        i_indices = rng.integers(0, n, size=max_pairs)
        j_indices = rng.integers(0, n, size=max_pairs)
        
        # Ensure i < j and i != j
        mask = i_indices < j_indices
        i_indices = i_indices[mask]
        j_indices = j_indices[mask]
        
        if len(i_indices) < max_pairs // 2:
            # Not enough valid pairs, generate more systematically
            i_indices, j_indices = np.triu_indices(n, k=1)
            if len(i_indices) > max_pairs:
                selection = rng.choice(len(i_indices), size=max_pairs, replace=False)
                i_indices = i_indices[selection]
                j_indices = j_indices[selection]
        
        # Compute distances in batches
        distances = np.zeros(len(i_indices), dtype=np.float32)
        
        if self.gpu_available:
            coords_gpu = cp.asarray(coords, dtype=cp.float32)
            
            for start in range(0, len(i_indices), batch_size):
                end = min(start + batch_size, len(i_indices))
                i_batch = i_indices[start:end]
                j_batch = j_indices[start:end]
                
                coords_i = coords_gpu[i_batch]
                coords_j = coords_gpu[j_batch]
                
                batch_distances = cp.sqrt(cp.sum((coords_i - coords_j)**2, axis=1))
                distances[start:end] = batch_distances.get()
        else:
            # CPU fallback
            for start in range(0, len(i_indices), batch_size):
                end = min(start + batch_size, len(i_indices))
                i_batch = i_indices[start:end]
                j_batch = j_indices[start:end]
                
                coords_i = coords[i_batch]
                coords_j = coords[j_batch]
                
                batch_distances = np.sqrt(np.sum((coords_i - coords_j)**2, axis=1))
                distances[start:end] = batch_distances
        
        return distances, np.column_stack([i_indices, j_indices])
    
    def _variogram_distances_cpu(
        self, 
        coords: np.ndarray, 
        max_pairs: int, 
        batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """CPU fallback for variogram distance computation."""
        from scipy.spatial.distance import pdist
        
        n = len(coords)
        total_pairs = (n * (n - 1)) // 2
        
        if total_pairs <= max_pairs:
            distances = pdist(coords)
            i, j = np.triu_indices(n, k=1)
            return distances, np.column_stack([i, j])
        else:
            return self._subsample_pairs_gpu(coords, max_pairs, batch_size)


# Global GPU accelerator instance
_gpu_accelerator = None

def get_gpu_accelerator(enable_gpu: bool = True, device_id: Optional[int] = None) -> GPUAccelerator:
    """Get or create the global GPU accelerator instance."""
    global _gpu_accelerator
    if _gpu_accelerator is None or _gpu_accelerator.device_id != device_id:
        _gpu_accelerator = GPUAccelerator(enable_gpu=enable_gpu, device_id=device_id)
    return _gpu_accelerator

def is_gpu_available() -> bool:
    """Check if GPU acceleration is available."""
    return get_gpu_accelerator().is_available

def gpu_memory_info() -> dict:
    """Get GPU memory information."""
    return get_gpu_accelerator().get_memory_info()

def set_gpu_device(device_id: int) -> None:
    """Set the GPU device to use."""
    global _gpu_accelerator
    _gpu_accelerator = GPUAccelerator(enable_gpu=True, device_id=device_id)
