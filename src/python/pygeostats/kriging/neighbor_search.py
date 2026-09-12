# src/python/pygeostats/kriging/neighbor_search.py
"""Neighbor search and approximation for large-scale kriging."""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors

from ..optimization.memory import MemoryManager
from ..utils.validation import validate_coordinates


class NeighborQueryResult:
    """Result of a neighbor query operation."""

    def __init__(
        self,
        indices: np.ndarray,
        distances: np.ndarray,
        query_indices: Optional[np.ndarray] = None,
    ):
        """
        Initialize neighbor query result.

        Parameters
        ----------
        indices : ndarray, shape (n_queries, k)
            Indices of k nearest neighbors for each query point.
        distances : ndarray, shape (n_queries, k)
            Distances to k nearest neighbors for each query point.
        query_indices : ndarray, optional
            Original indices of query points.
        """
        self.indices = indices
        self.distances = distances
        self.query_indices = query_indices
        self.n_queries, self.k = indices.shape

    def filter_by_distance(self, max_distance: float) -> NeighborQueryResult:
        """Filter neighbors by maximum distance."""
        mask = self.distances <= max_distance

        # Handle ragged arrays by padding with -1
        max_valid = np.max(np.sum(mask, axis=1))

        new_indices = np.full((self.n_queries, max_valid), -1, dtype=self.indices.dtype)
        new_distances = np.full((self.n_queries, max_valid), np.inf)

        for i in range(self.n_queries):
            valid = mask[i]
            n_valid = np.sum(valid)
            if n_valid > 0:
                new_indices[i, :n_valid] = self.indices[i, valid]
                new_distances[i, :n_valid] = self.distances[i, valid]

        return NeighborQueryResult(new_indices, new_distances, self.query_indices)

    def get_valid_neighbors(self, query_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get valid (non-negative) neighbors for a query point."""
        valid_mask = self.indices[query_idx] >= 0
        return (
            self.indices[query_idx][valid_mask],
            self.distances[query_idx][valid_mask],
        )


class ApproximateNeighborIndex:
    """
    Approximate neighbor search index for large-scale kriging.

    Provides efficient neighbor queries for datasets with millions of points
    using various spatial indexing strategies.
    """

    def __init__(
        self,
        method: str = "auto",
        max_neighbors: int = 64,
        leaf_size: int = 30,
        algorithm: str = "auto",
    ):
        """
        Initialize approximate neighbor index.

        Parameters
        ----------
        method : str, default="auto"
            Indexing method: "auto", "kdtree", "sklearn", "annoy".
        max_neighbors : int, default=64
            Maximum number of neighbors to retrieve.
        leaf_size : int, default=30
            Leaf size for tree-based methods.
        algorithm : str, default="auto"
            Algorithm for sklearn methods.
        """
        self.method = method
        self.max_neighbors = max_neighbors
        self.leaf_size = leaf_size
        self.algorithm = algorithm

        self._index = None
        self._coordinates = None
        self._is_fitted = False

        self.memory_manager = MemoryManager()

    def fit(self, coordinates: np.ndarray) -> ApproximateNeighborIndex:
        """
        Build the neighbor index.

        Parameters
        ----------
        coordinates : ndarray, shape (n_points, d)
            Coordinates to index.

        Returns
        -------
        self : ApproximateNeighborIndex
            Returns self for method chaining.
        """
        coordinates = validate_coordinates(coordinates)
        self._coordinates = coordinates

        # Choose method automatically if needed
        if self.method == "auto":
            n_points, n_dims = coordinates.shape
            if n_points > 100_000 and n_dims <= 10:
                method = "kdtree"
            else:
                method = "sklearn"
        else:
            method = self.method

        print(f"Building {method} neighbor index for {len(coordinates):,} points...")

        # Build index based on method
        if method == "kdtree":
            self._index = cKDTree(
                coordinates,
                leafsize=self.leaf_size,
                balanced_tree=True,
                compact_nodes=True,
            )

        elif method == "sklearn":
            self._index = NearestNeighbors(
                n_neighbors=min(self.max_neighbors, len(coordinates)),
                algorithm=self.algorithm,
                leaf_size=self.leaf_size,
                n_jobs=-1,  # Use all available cores
            )
            self._index.fit(coordinates)

        elif method == "annoy":
            self._build_annoy_index(coordinates)

        else:
            raise ValueError(f"Unknown neighbor search method: {method}")

        self._method_used = method
        self._is_fitted = True
        return self

    def _build_annoy_index(self, coordinates: np.ndarray) -> None:
        """Build Annoy approximate index."""
        try:
            from annoy import AnnoyIndex
        except ImportError as err:
            raise ImportError("annoy package required for annoy method") from err

        n_points, n_dims = coordinates.shape

        # Build Annoy index
        index = AnnoyIndex(n_dims, "euclidean")

        for i, coord in enumerate(coordinates):
            index.add_item(i, coord.tolist())

        # Build with reasonable number of trees
        n_trees = min(100, max(10, int(np.log2(n_points))))
        index.build(n_trees)

        self._index = index

    def query(
        self,
        query_points: np.ndarray,
        k: Optional[int] = None,
        max_distance: Optional[float] = None,
        return_indices: bool = True,
    ) -> NeighborQueryResult:
        """
        Query neighbors for given points.

        Parameters
        ----------
        query_points : ndarray, shape (n_queries, d)
            Points to query neighbors for.
        k : int, optional
            Number of neighbors to return. Uses max_neighbors if None.
        max_distance : float, optional
            Maximum distance for neighbors.
        return_indices : bool, default=True
            Whether to return neighbor indices.

        Returns
        -------
        result : NeighborQueryResult
            Query results with neighbor indices and distances.
        """
        if not self._is_fitted:
            raise ValueError("Index must be fitted before querying")

        query_points = validate_coordinates(query_points)

        if k is None:
            k = self.max_neighbors

        k = min(k, len(self._coordinates))

        # Query based on method
        if self._method_used == "kdtree":
            distances, indices = self._index.query(
                query_points,
                k=k,
                distance_upper_bound=max_distance if max_distance else np.inf,
            )

            # Handle single query case
            if query_points.ndim == 1 or len(query_points) == 1:
                distances = distances.reshape(1, -1)
                indices = indices.reshape(1, -1)

        elif self._method_used == "sklearn":
            if max_distance is not None:
                distances, indices = self._index.radius_neighbors(
                    query_points, radius=max_distance
                )
                # Convert to fixed-size arrays
                distances, indices = self._pad_ragged_arrays(distances, indices, k)
            else:
                distances, indices = self._index.kneighbors(query_points, n_neighbors=k)

        elif self._method_used == "annoy":
            distances, indices = self._query_annoy(query_points, k, max_distance)

        else:
            raise RuntimeError(f"Unknown method: {self._method_used}")

        result = NeighborQueryResult(indices, distances)

        # Filter by distance if specified
        if max_distance is not None and self._method_used != "sklearn":
            result = result.filter_by_distance(max_distance)

        return result

    def _query_annoy(
        self, query_points: np.ndarray, k: int, max_distance: Optional[float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Query Annoy index."""
        # Annoy returns approximate results, so request more neighbors
        search_k = min(k * 10, len(self._coordinates))

        indices_list = []
        distances_list = []

        for query_point in query_points:
            neighbor_indices = self._index.get_nns_by_vector(
                query_point.tolist(), search_k, include_distances=False
            )

            # Compute exact distances
            neighbor_coords = self._coordinates[neighbor_indices]
            distances = np.linalg.norm(neighbor_coords - query_point, axis=1)

            # Sort by distance and take top k
            sort_idx = np.argsort(distances)[:k]
            indices_list.append(np.array(neighbor_indices)[sort_idx])
            distances_list.append(distances[sort_idx])

        # Convert to arrays
        indices = np.array(indices_list)
        distances = np.array(distances_list)

        return distances, indices

    def _pad_ragged_arrays(
        self,
        distances_list: List[np.ndarray],
        indices_list: List[np.ndarray],
        max_k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert ragged arrays to padded fixed-size arrays."""
        n_queries = len(distances_list)

        # Find actual max neighbors
        actual_k = min(
            max_k, max(len(arr) for arr in indices_list) if indices_list else 0
        )

        if actual_k == 0:
            return (
                np.full((n_queries, 1), np.inf),
                np.full((n_queries, 1), -1, dtype=int),
            )

        # Pad arrays
        padded_distances = np.full((n_queries, actual_k), np.inf)
        padded_indices = np.full((n_queries, actual_k), -1, dtype=int)

        for i, (dist, idx) in enumerate(zip(distances_list, indices_list)):
            n_neighbors = min(len(dist), actual_k)
            padded_distances[i, :n_neighbors] = dist[:n_neighbors]
            padded_indices[i, :n_neighbors] = idx[:n_neighbors]

        return padded_distances, padded_indices

    def memory_usage_gb(self) -> float:
        """Estimate memory usage of the index."""
        if not self._is_fitted:
            return 0.0

        # Estimate based on coordinate storage and index overhead
        coord_memory = self._coordinates.nbytes / (1024**3)

        # Index overhead varies by method
        if self._method_used == "kdtree":
            # KDTree overhead is typically 2-3x coordinates
            overhead = coord_memory * 2.5
        elif self._method_used == "sklearn":
            # Sklearn overhead varies by algorithm
            overhead = coord_memory * 1.5
        elif self._method_used == "annoy":
            # Annoy creates disk-based index
            overhead = coord_memory * 0.5
        else:
            overhead = coord_memory

        return coord_memory + overhead

    def get_statistics(self) -> Dict[str, Union[int, float, str]]:
        """Get index statistics."""
        if not self._is_fitted:
            return {"fitted": False}

        return {
            "fitted": True,
            "method": self._method_used,
            "n_points": len(self._coordinates),
            "n_dimensions": self._coordinates.shape[1],
            "max_neighbors": self.max_neighbors,
            "memory_usage_gb": self.memory_usage_gb(),
            "leaf_size": self.leaf_size,
        }


class LocalKrigingPredictor:
    """
    Local kriging predictor using neighbor-based approximation.

    Enables kriging on large datasets by using only local neighborhoods
    around each prediction point.
    """

    def __init__(
        self,
        variogram,
        neighbor_index: ApproximateNeighborIndex,
        max_neighbors: int = 64,
        min_neighbors: int = 4,
        max_distance: Optional[float] = None,
        cache_covariance: bool = True,
    ):
        """
        Initialize local kriging predictor.

        Parameters
        ----------
        variogram : Variogram
            Fitted variogram model.
        neighbor_index : ApproximateNeighborIndex
            Fitted neighbor search index.
        max_neighbors : int, default=64
            Maximum neighbors for local kriging.
        min_neighbors : int, default=4
            Minimum neighbors required for prediction.
        max_distance : float, optional
            Maximum distance for neighbors.
        cache_covariance : bool, default=True
            Whether to cache covariance computations.
        """
        self.variogram = variogram
        self.neighbor_index = neighbor_index
        self.max_neighbors = max_neighbors
        self.min_neighbors = min_neighbors
        self.max_distance = max_distance
        self.cache_covariance = cache_covariance

        # Cache for covariance matrices
        self._covariance_cache = {} if cache_covariance else None

        # Fitted data
        self.coordinates_ = None
        self.values_ = None
        self.is_fitted_ = False

    def fit(self, coordinates: np.ndarray, values: np.ndarray) -> LocalKrigingPredictor:
        """
        Fit the local kriging predictor.

        Parameters
        ----------
        coordinates : ndarray, shape (n_samples, d)
            Known sample coordinates.
        values : ndarray, shape (n_samples,)
            Known sample values.

        Returns
        -------
        self : LocalKrigingPredictor
            Returns self for method chaining.
        """
        self.coordinates_ = validate_coordinates(coordinates)
        self.values_ = np.asarray(values)

        if len(self.coordinates_) != len(self.values_):
            raise ValueError("Coordinates and values must have same length")

        # Ensure neighbor index is fitted on the same data
        if not self.neighbor_index._is_fitted:
            self.neighbor_index.fit(self.coordinates_)
        elif not np.array_equal(self.neighbor_index._coordinates, self.coordinates_):
            warnings.warn(
                "Neighbor index coordinates differ from training coordinates",
                stacklevel=2,
            )

        self.is_fitted_ = True
        return self

    def predict(
        self,
        coordinates: np.ndarray,
        return_variance: bool = False,
        show_progress: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict at new locations using local kriging.

        Parameters
        ----------
        coordinates : ndarray, shape (n_pred, d)
            Prediction coordinates.
        return_variance : bool, default=False
            Whether to return prediction variance.
        show_progress : bool, default=False
            Whether to show prediction progress.

        Returns
        -------
        predictions : ndarray, shape (n_pred,)
            Predicted values.
        variances : ndarray, shape (n_pred,), optional
            Prediction variances if return_variance=True.
        """
        if not self.is_fitted_:
            raise ValueError("Predictor must be fitted before prediction")

        pred_coords = validate_coordinates(coordinates)
        n_pred = len(pred_coords)

        # Query neighbors for all prediction points
        neighbor_result = self.neighbor_index.query(
            pred_coords, k=self.max_neighbors, max_distance=self.max_distance
        )

        # Initialize outputs
        predictions = np.full(n_pred, np.nan)
        variances = np.full(n_pred, np.nan) if return_variance else None

        if show_progress:
            try:
                from tqdm import tqdm

                pred_iter = tqdm(range(n_pred), desc="Local kriging")
            except ImportError:
                pred_iter = range(n_pred)
                if n_pred > 1000:
                    print(f"Predicting at {n_pred:,} locations...")
        else:
            pred_iter = range(n_pred)

        # Predict at each location
        for i in pred_iter:
            neighbor_indices, neighbor_distances = neighbor_result.get_valid_neighbors(
                i
            )

            if len(neighbor_indices) < self.min_neighbors:
                # Not enough neighbors - use global mean or skip
                if len(neighbor_indices) > 0:
                    predictions[i] = np.mean(self.values_[neighbor_indices])
                    if return_variance:
                        variances[i] = np.var(self.values_)
                continue

            try:
                # Perform local kriging
                pred_point = pred_coords[i]
                local_coords = self.coordinates_[neighbor_indices]
                local_values = self.values_[neighbor_indices]

                pred_val, pred_var = self._local_kriging_single(
                    pred_point, local_coords, local_values, return_variance
                )

                predictions[i] = pred_val
                if return_variance:
                    variances[i] = pred_var

            except Exception as e:
                # Handle singular matrices or other numerical issues
                warnings.warn(f"Local kriging failed at point {i}: {e!s}", stacklevel=2)
                # Fallback to inverse distance weighting
                if len(neighbor_distances) > 0 and np.any(neighbor_distances > 0):
                    weights = 1.0 / (neighbor_distances + 1e-10)
                    weights /= np.sum(weights)
                    predictions[i] = np.sum(weights * self.values_[neighbor_indices])
                    if return_variance:
                        variances[i] = np.var(self.values_[neighbor_indices])

        if return_variance:
            return predictions, variances
        return predictions

    def _local_kriging_single(
        self,
        pred_point: np.ndarray,
        local_coords: np.ndarray,
        local_values: np.ndarray,
        return_variance: bool,
    ) -> Tuple[float, Optional[float]]:
        """Perform kriging for a single prediction point using local neighbors."""
        n_neighbors = len(local_coords)

        # Create cache key for covariance matrix
        if self.cache_covariance:
            coord_hash = hash(local_coords.tobytes())
            cache_key = (coord_hash, n_neighbors)
        else:
            cache_key = None

        # Build or retrieve covariance matrix
        if cache_key and cache_key in self._covariance_cache:
            cov_matrix, cov_decomp = self._covariance_cache[cache_key]
        else:
            cov_matrix = self._build_local_covariance_matrix(local_coords)

            # Pre-decompose for efficiency
            from scipy.linalg import solve

            try:
                cov_decomp = np.linalg.cholesky(cov_matrix)
            except np.linalg.LinAlgError:
                # Fallback to LU decomposition for ill-conditioned matrices
                from scipy.linalg import lu_factor

                cov_decomp = lu_factor(cov_matrix)

            if cache_key and len(self._covariance_cache) < 1000:  # Limit cache size
                self._covariance_cache[cache_key] = (cov_matrix, cov_decomp)

        # Set up kriging system: [C | 1] [w]   [c0]
        #                        [1 | 0] [μ] = [1 ]
        system_size = n_neighbors + 1
        system_matrix = np.zeros((system_size, system_size))
        system_matrix[:n_neighbors, :n_neighbors] = cov_matrix
        system_matrix[n_neighbors, :n_neighbors] = 1.0
        system_matrix[:n_neighbors, n_neighbors] = 1.0

        # Right-hand side
        rhs = np.zeros(system_size)

        # Compute covariances between prediction point and neighbors
        for i, neighbor_coord in enumerate(local_coords):
            distance = np.linalg.norm(pred_point - neighbor_coord)
            rhs[i] = self._variogram_to_covariance(distance)

        rhs[n_neighbors] = 1.0  # Unbiasedness constraint

        # Solve kriging system
        try:
            from scipy.linalg import solve

            weights = solve(system_matrix, rhs)
        except np.linalg.LinAlgError:
            # Regularize and retry
            system_matrix[:n_neighbors, :n_neighbors] += np.eye(n_neighbors) * 1e-6
            weights = solve(system_matrix, rhs)

        # Compute prediction
        prediction = np.dot(weights[:n_neighbors], local_values)

        # Compute variance if requested
        variance = None
        if return_variance:
            # Kriging variance: C(0,0) - w^T * c0 - μ
            c00 = self._variogram_to_covariance(0.0)  # Variance at prediction point
            variance = (
                c00
                - np.dot(weights[:n_neighbors], rhs[:n_neighbors])
                - weights[n_neighbors]
            )
            variance = max(variance, 0.0)  # Ensure non-negative

        return prediction, variance

    def _build_local_covariance_matrix(self, coords: np.ndarray) -> np.ndarray:
        """Build covariance matrix for local coordinates."""
        n = len(coords)
        cov_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                distance = np.linalg.norm(coords[i] - coords[j])
                cov_matrix[i, j] = self._variogram_to_covariance(distance)

        return cov_matrix

    def _variogram_to_covariance(self, distance: float) -> float:
        """Convert distance to covariance using variogram model."""
        if distance == 0.0:
            return self.variogram.sill_

        # Compute semivariance
        gamma = self.variogram.predict(np.array([distance]))[0]

        # Convert to covariance
        return self.variogram.sill_ - gamma

    def clear_cache(self) -> None:
        """Clear the covariance matrix cache."""
        if self._covariance_cache:
            self._covariance_cache.clear()

    def get_cache_statistics(self) -> Dict[str, int]:
        """Get cache usage statistics."""
        if not self.cache_covariance:
            return {"caching_enabled": False}

        return {
            "caching_enabled": True,
            "cache_size": len(self._covariance_cache),
            "cache_limit": 1000,
        }
