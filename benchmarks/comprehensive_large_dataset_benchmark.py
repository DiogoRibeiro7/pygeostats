# benchmarks/comprehensive_large_dataset_benchmark.py
"""Comprehensive benchmarks for large dataset optimizations."""

import gc
import os
import time
import warnings
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pygeostats.variogram import EmpiricalVariogram, Variogram
from pygeostats.variogram.streaming import (
    StreamingEmpiricalVariogram,
    streaming_variogram,
)
from pygeostats.kriging import OrdinaryKriging
from pygeostats.kriging.neighbor_search import (
    ApproximateNeighborIndex,
    LocalKrigingPredictor,
)
from pygeostats.kriging.executor import ParallelKrigingExecutor
from pygeostats.optimization.memory import (
    MemoryManager,
    SparseDistanceMatrix,
    MemoryEfficientKriging,
    estimate_kriging_memory,
    memory_profile,
)
from pygeostats.acceleration.gpu import get_gpu_accelerator, is_gpu_available


class LargeDatasetBenchmark:
    """Comprehensive benchmark suite for large dataset performance."""

    def __init__(
        self, output_dir: str = "large_dataset_benchmarks", enable_gpu: bool = True
    ):
        """
        Initialize benchmark suite.

        Parameters
        ----------
        output_dir : str, default="large_dataset_benchmarks"
            Directory to save benchmark results.
        enable_gpu : bool, default=True
            Whether to enable GPU acceleration if available.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.memory_manager = MemoryManager()
        self.gpu_accelerator = get_gpu_accelerator(enable_gpu=enable_gpu)

        self.results = []

        # Test sizes (up to 1M points)
        self.test_sizes = [
            1_000,
            2_500,
            5_000,
            10_000,
            25_000,
            50_000,
            100_000,
            250_000,
            500_000,
            1_000_000,
        ]

        # Determine which sizes to actually test based on system memory
        system_memory_gb = self.memory_manager.get_memory_info()["system_total_gb"]
        if system_memory_gb < 16:
            self.test_sizes = [s for s in self.test_sizes if s <= 100_000]
            print(
                f"Limited testing to ≤100k points due to system memory ({system_memory_gb:.1f} GB)"
            )
        elif system_memory_gb < 32:
            self.test_sizes = [s for s in self.test_sizes if s <= 500_000]
            print(
                f"Limited testing to ≤500k points due to system memory ({system_memory_gb:.1f} GB)"
            )

        print(f"Testing with sizes: {[f'{s:,}' for s in self.test_sizes]}")

    def generate_large_synthetic_dataset(
        self,
        n_samples: int,
        domain_size: float = 100.0,
        correlation_range: float = 10.0,
        nugget: float = 0.1,
        sill: float = 1.0,
        random_seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate large synthetic spatial dataset with known structure."""
        np.random.seed(random_seed)

        print(f"Generating {n_samples:,} synthetic spatial points...")

        # Generate coordinates with slight clustering for realism
        coords = np.random.uniform(0, domain_size, size=(n_samples, 2))

        # Add some clustering
        n_clusters = max(1, n_samples // 10000)
        cluster_centers = np.random.uniform(
            domain_size * 0.2, domain_size * 0.8, size=(n_clusters, 2)
        )

        # Assign some points to clusters
        cluster_fraction = 0.3
        n_clustered = int(n_samples * cluster_fraction)
        cluster_indices = np.random.choice(n_samples, n_clustered, replace=False)

        for i in cluster_indices:
            center_idx = np.random.randint(n_clusters)
            cluster_noise = np.random.normal(0, domain_size * 0.05, 2)
            coords[i] = cluster_centers[center_idx] + cluster_noise
            coords[i] = np.clip(coords[i], 0, domain_size)

        # Generate spatially correlated values efficiently
        if n_samples <= 50_000:
            # Full covariance approach for smaller datasets
            values = self._generate_correlated_values_full(
                coords, correlation_range, nugget, sill
            )
        else:
            # Approximation approach for larger datasets
            values = self._generate_correlated_values_approximate(
                coords, correlation_range, nugget, sill
            )

        return coords, values

    def _generate_correlated_values_full(
        self, coords: np.ndarray, range_param: float, nugget: float, sill: float
    ) -> np.ndarray:
        """Generate correlated values using full covariance matrix."""
        n = len(coords)

        # Compute distance matrix in chunks to manage memory
        chunk_size = min(5000, n)
        cov_matrix = np.zeros((n, n))

        for i in range(0, n, chunk_size):
            i_end = min(i + chunk_size, n)
            for j in range(0, n, chunk_size):
                j_end = min(j + chunk_size, n)

                coords_i = coords[i:i_end]
                coords_j = coords[j:j_end]

                # Compute distances for this block
                from scipy.spatial.distance import cdist

                distances = cdist(coords_i, coords_j)

                # Exponential covariance
                gamma = nugget + (sill - nugget) * (
                    1 - np.exp(-distances / range_param)
                )
                cov_block = sill - gamma

                # Add nugget effect on diagonal
                if i == j:
                    np.fill_diagonal(cov_block, sill)

                cov_matrix[i:i_end, j:j_end] = cov_block

        # Generate correlated values
        return np.random.multivariate_normal(np.zeros(n), cov_matrix)

    def _generate_correlated_values_approximate(
        self, coords: np.ndarray, range_param: float, nugget: float, sill: float
    ) -> np.ndarray:
        """Generate approximately correlated values for large datasets."""
        n = len(coords)

        # Use moving average approach with spatial weighting
        values = np.random.normal(0, np.sqrt(sill), n)

        # Apply spatial smoothing using k-nearest neighbors
        from sklearn.neighbors import NearestNeighbors

        k_neighbors = min(50, n // 100)  # Adaptive number of neighbors
        nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, algorithm="auto")
        nbrs.fit(coords)

        distances, indices = nbrs.kneighbors(coords)

        # Apply spatial correlation through weighted averaging
        smoothed_values = np.zeros(n)
        for i in range(n):
            neighbor_dists = distances[i, 1:]  # Exclude self
            neighbor_indices = indices[i, 1:]

            # Exponential weights
            weights = np.exp(-neighbor_dists / range_param)
            weights /= np.sum(weights)

            # Weighted average
            smoothed_values[i] = np.sum(weights * values[neighbor_indices])

        # Combine original and smoothed values
        correlation_strength = 0.7
        final_values = (
            correlation_strength * smoothed_values + (1 - correlation_strength) * values
        )

        return final_values

    @memory_profile
    def benchmark_streaming_variogram(self) -> pd.DataFrame:
        """Benchmark streaming variogram computation."""
        print("\n=== Streaming Variogram Benchmark ===")

        results = []

        for n_samples in self.test_sizes:
            print(f"\nTesting streaming variogram with {n_samples:,} samples")

            # Generate data
            coords, values = self.generate_large_synthetic_dataset(n_samples)

            # Test both regular and streaming approaches
            for approach in ["regular", "streaming", "streaming_sparse"]:
                if approach == "regular" and n_samples > 50_000:
                    # Skip regular approach for very large datasets
                    continue

                memory_before = self.memory_manager.get_memory_info()

                try:
                    start_time = time.time()

                    if approach == "regular":
                        # Regular empirical variogram
                        emp_vario = EmpiricalVariogram(
                            coords, values, max_distance=20.0, n_bins=20
                        )
                        emp_vario.compute()
                        success = True
                        n_pairs = np.sum(emp_vario.counts_)

                    elif approach == "streaming":
                        # Streaming variogram with chunks
                        streaming_vario = StreamingEmpiricalVariogram(
                            max_distance=20.0,
                            n_bins=20,
                            chunk_size=min(10000, n_samples // 4),
                            use_spatial_indexing=False,
                        )
                        streaming_vario.compute(coords, values, show_progress=False)
                        success = True
                        n_pairs = streaming_vario.statistics_["total_pairs_processed"]

                    elif approach == "streaming_sparse":
                        # Streaming variogram with spatial indexing
                        streaming_vario = StreamingEmpiricalVariogram(
                            max_distance=20.0,
                            n_bins=20,
                            chunk_size=min(5000, n_samples // 8),
                            use_spatial_indexing=True,
                            max_neighbors=100,
                        )
                        streaming_vario.compute(coords, values, show_progress=False)
                        success = True
                        n_pairs = streaming_vario.statistics_["total_pairs_processed"]

                    elapsed_time = time.time() - start_time

                except Exception as e:
                    print(f"    {approach} failed: {e}")
                    elapsed_time = np.nan
                    success = False
                    n_pairs = 0

                memory_after = self.memory_manager.get_memory_info()
                memory_used = memory_after["process_gb"] - memory_before["process_gb"]

                results.append(
                    {
                        "n_samples": n_samples,
                        "approach": approach,
                        "time_s": elapsed_time,
                        "memory_used_gb": memory_used,
                        "success": success,
                        "pairs_processed": n_pairs,
                        "pairs_per_second": n_pairs / elapsed_time
                        if success and elapsed_time > 0
                        else 0,
                    }
                )

                print(
                    f"    {approach}: {elapsed_time:.2f}s, {memory_used:.2f}GB, {n_pairs:,} pairs"
                )

            # Cleanup
            del coords, values
            gc.collect()

        return pd.DataFrame(results)

    @memory_profile
    def benchmark_neighbor_kriging(self) -> pd.DataFrame:
        """Benchmark neighbor-based kriging approaches."""
        print("\n=== Neighbor-based Kriging Benchmark ===")

        results = []

        for n_samples in self.test_sizes:
            if n_samples < 5_000:  # Skip small datasets for neighbor methods
                continue

            print(f"\nTesting neighbor kriging with {n_samples:,} samples")

            # Generate training data
            coords, values = self.generate_large_synthetic_dataset(n_samples)

            # Create simple variogram for testing
            variogram = Variogram(model="exponential")
            variogram.parameters = [0.1, 1.0, 10.0]
            variogram.is_fitted_ = True
            variogram.nugget_ = 0.1
            variogram.sill_ = 1.0
            variogram.range_ = 10.0

            # Generate prediction points
            n_pred = min(1000, n_samples // 20)
            pred_coords = np.random.uniform(0, 100, size=(n_pred, 2))

            # Test different approaches
            approaches = {
                "standard_kriging": self._test_standard_kriging,
                "neighbor_kriging_64": partial(
                    self._test_neighbor_kriging, max_neighbors=64
                ),
                "neighbor_kriging_32": partial(
                    self._test_neighbor_kriging, max_neighbors=32
                ),
                "neighbor_kriging_16": partial(
                    self._test_neighbor_kriging, max_neighbors=16
                ),
            }

            for approach_name, test_func in approaches.items():
                if approach_name == "standard_kriging" and n_samples > 25_000:
                    # Skip standard kriging for very large datasets
                    continue

                memory_before = self.memory_manager.get_memory_info()

                try:
                    start_time = time.time()
                    predictions = test_func(variogram, coords, values, pred_coords)
                    elapsed_time = time.time() - start_time

                    memory_after = self.memory_manager.get_memory_info()
                    memory_used = (
                        memory_after["process_gb"] - memory_before["process_gb"]
                    )

                    success = True
                    rmse = np.sqrt(
                        np.mean((predictions - predictions.mean()) ** 2)
                    )  # Rough quality measure

                except Exception as e:
                    print(f"    {approach_name} failed: {e}")
                    elapsed_time = np.nan
                    memory_used = np.nan
                    success = False
                    rmse = np.nan

                results.append(
                    {
                        "n_samples": n_samples,
                        "n_pred": n_pred,
                        "approach": approach_name,
                        "time_s": elapsed_time,
                        "memory_used_gb": memory_used,
                        "success": success,
                        "rmse": rmse,
                        "predictions_per_second": n_pred / elapsed_time
                        if success and elapsed_time > 0
                        else 0,
                    }
                )

                print(f"    {approach_name}: {elapsed_time:.2f}s, {memory_used:.2f}GB")

            # Cleanup
            del coords, values, pred_coords
            gc.collect()

        return pd.DataFrame(results)

    @memory_profile
    def benchmark_parallel_execution(self) -> pd.DataFrame:
        """Benchmark parallel execution strategies."""
        print("\n=== Parallel Execution Benchmark ===")

        results = []

        # Test with moderately sized datasets for parallel comparison
        test_sizes = [s for s in self.test_sizes if 10_000 <= s <= 100_000]

        for n_samples in test_sizes:
            print(f"\nTesting parallel execution with {n_samples:,} samples")

            # Generate training data
            coords, values = self.generate_large_synthetic_dataset(n_samples)

            # Create variogram
            variogram = Variogram(model="exponential")
            variogram.parameters = [0.1, 1.0, 10.0]
            variogram.is_fitted_ = True
            variogram.nugget_ = 0.1
            variogram.sill_ = 1.0
            variogram.range_ = 10.0

            # Fit kriging model
            kriging = OrdinaryKriging(variogram)
            kriging.fit(coords, values)

            # Generate prediction grid
            n_pred = min(2000, n_samples // 10)
            pred_coords = np.random.uniform(0, 100, size=(n_pred, 2))

            # Test different parallel approaches
            parallel_configs = [
                ("sequential", 1, "sequential"),
                ("thread_2", 2, "thread"),
                ("thread_4", 4, "thread"),
                ("process_2", 2, "process"),
                ("process_4", 4, "process"),
            ]

            for config_name, n_workers, method in parallel_configs:
                memory_before = self.memory_manager.get_memory_info()

                try:
                    executor = ParallelKrigingExecutor(
                        n_workers=n_workers,
                        execution_method=method,
                        chunk_size=min(500, n_pred // n_workers),
                    )

                    start_time = time.time()
                    predictions = executor.predict_parallel(
                        kriging, pred_coords, strategy="chunk"
                    )
                    elapsed_time = time.time() - start_time

                    memory_after = self.memory_manager.get_memory_info()
                    memory_used = (
                        memory_after["process_gb"] - memory_before["process_gb"]
                    )

                    success = True

                except Exception as e:
                    print(f"    {config_name} failed: {e}")
                    elapsed_time = np.nan
                    memory_used = np.nan
                    success = False

                results.append(
                    {
                        "n_samples": n_samples,
                        "n_pred": n_pred,
                        "config": config_name,
                        "n_workers": n_workers,
                        "method": method,
                        "time_s": elapsed_time,
                        "memory_used_gb": memory_used,
                        "success": success,
                        "speedup": None,  # Will calculate later
                    }
                )

                print(f"    {config_name}: {elapsed_time:.2f}s, {memory_used:.2f}GB")

        # Calculate speedups relative to sequential
        df = pd.DataFrame(results)
        for n_samples in df["n_samples"].unique():
            subset = df[df["n_samples"] == n_samples]
            sequential_time = subset[subset["config"] == "sequential"]["time_s"].iloc[0]

            if not np.isnan(sequential_time):
                for idx in subset.index:
                    if df.loc[idx, "success"]:
                        df.loc[idx, "speedup"] = sequential_time / df.loc[idx, "time_s"]

        return df

    @memory_profile
    def benchmark_memory_optimization(self) -> pd.DataFrame:
        """Benchmark memory optimization techniques."""
        print("\n=== Memory Optimization Benchmark ===")

        results = []

        for n_samples in self.test_sizes:
            if n_samples < 10_000:  # Focus on larger datasets
                continue

            print(f"\nTesting memory optimization with {n_samples:,} samples")

            # Generate data
            coords, values = self.generate_large_synthetic_dataset(n_samples)

            # Test different memory approaches
            approaches = {
                "dense_matrix": self._test_dense_matrix_memory,
                "sparse_matrix": self._test_sparse_matrix_memory,
                "chunked_processing": self._test_chunked_processing_memory,
                "memory_mapped": self._test_memory_mapped_arrays,
            }

            for approach_name, test_func in approaches.items():
                memory_before = self.memory_manager.get_memory_info()

                try:
                    start_time = time.time()
                    result = test_func(coords, values)
                    elapsed_time = time.time() - start_time

                    memory_after = self.memory_manager.get_memory_info()
                    memory_used = (
                        memory_after["process_gb"] - memory_before["process_gb"]
                    )

                    success = True
                    efficiency_score = (
                        result.get("efficiency", 1.0)
                        if isinstance(result, dict)
                        else 1.0
                    )

                except Exception as e:
                    print(f"    {approach_name} failed: {e}")
                    elapsed_time = np.nan
                    memory_used = np.nan
                    success = False
                    efficiency_score = 0.0

                results.append(
                    {
                        "n_samples": n_samples,
                        "approach": approach_name,
                        "time_s": elapsed_time,
                        "memory_used_gb": memory_used,
                        "success": success,
                        "efficiency_score": efficiency_score,
                    }
                )

                print(f"    {approach_name}: {elapsed_time:.2f}s, {memory_used:.2f}GB")

            # Cleanup
            del coords, values
            gc.collect()

        return pd.DataFrame(results)

    def _test_standard_kriging(self, variogram, coords, values, pred_coords):
        """Test standard kriging approach."""
        kriging = OrdinaryKriging(variogram)
        kriging.fit(coords, values)
        return kriging.predict(pred_coords)

    def _test_neighbor_kriging(
        self, variogram, coords, values, pred_coords, max_neighbors=64
    ):
        """Test neighbor-based kriging approach."""
        # Build neighbor index
        neighbor_index = ApproximateNeighborIndex(
            method="auto", max_neighbors=max_neighbors
        )
        neighbor_index.fit(coords)

        # Create local kriging predictor
        local_kriging = LocalKrigingPredictor(
            variogram=variogram,
            neighbor_index=neighbor_index,
            max_neighbors=max_neighbors,
            max_distance=30.0,
        )
        local_kriging.fit(coords, values)

        return local_kriging.predict(pred_coords)

    def _test_dense_matrix_memory(self, coords, values):
        """Test dense distance matrix memory usage."""
        from scipy.spatial.distance import pdist, squareform

        distances = pdist(coords)
        dense_matrix = squareform(distances)

        return {"efficiency": 1.0, "matrix_size": dense_matrix.size}

    def _test_sparse_matrix_memory(self, coords, values):
        """Test sparse distance matrix memory usage."""
        sparse_matrix = SparseDistanceMatrix(
            coords, max_distance=20.0, chunk_size=min(5000, len(coords))
        )
        sparse_matrix.compute(show_progress=False)

        sparsity = sparse_matrix.sparsity_ratio()
        memory_usage = sparse_matrix.memory_usage_gb()

        return {
            "efficiency": sparsity,
            "memory_gb": memory_usage,
            "sparsity_ratio": sparsity,
        }

    def _test_chunked_processing_memory(self, coords, values):
        """Test chunked processing memory efficiency."""
        from pygeostats.optimization.memory import ChunkedArrayProcessor

        processor = ChunkedArrayProcessor(max_memory_gb=2.0)

        def dummy_process(chunk):
            # Simulate some processing
            return np.mean(chunk, axis=0)

        results = list(
            processor.process_chunks(coords, dummy_process, show_progress=False)
        )

        return {
            "efficiency": len(results) / (len(coords) // 1000 + 1),
            "chunks_processed": len(results),
        }

    def _test_memory_mapped_arrays(self, coords, values):
        """Test memory-mapped array efficiency."""
        from pygeostats.optimization.memory import create_memory_mapped_array

        # Create temporary memory-mapped file
        temp_file = self.output_dir / f"temp_coords_{len(coords)}.dat"

        try:
            memmap_coords = create_memory_mapped_array(
                temp_file, coords.shape, coords.dtype
            )
            memmap_coords[:] = coords

            # Simulate processing
            result = np.mean(memmap_coords, axis=0)

            efficiency = 1.0  # Memory mapping successful

        finally:
            if temp_file.exists():
                temp_file.unlink()

        return {"efficiency": efficiency, "file_size_gb": coords.nbytes / (1024**3)}

    def run_comprehensive_benchmark(self) -> Dict[str, pd.DataFrame]:
        """Run the complete benchmark suite."""
        print("=" * 60)
        print("PySpatialStats Large Dataset Comprehensive Benchmark")
        print("=" * 60)
        print(
            f"System: {self.memory_manager.get_memory_info()['system_total_gb']:.1f} GB RAM"
        )
        print(
            f"GPU: {'Available' if self.gpu_accelerator.is_available else 'Not Available'}"
        )
        print(f"CPU Cores: {os.cpu_count()}")
        print()

        results = {}

        # Run individual benchmarks
        results["streaming_variogram"] = self.benchmark_streaming_variogram()
        results["neighbor_kriging"] = self.benchmark_neighbor_kriging()
        results["parallel_execution"] = self.benchmark_parallel_execution()
        results["memory_optimization"] = self.benchmark_memory_optimization()

        # Save results
        self.save_results(results)

        # Generate comprehensive plots
        self.generate_comprehensive_plots(results)

        # Generate performance report
        self.generate_performance_report(results)

        return results

    def save_results(self, results: Dict[str, pd.DataFrame]) -> None:
        """Save benchmark results to files."""
        print("\nSaving benchmark results...")

        for benchmark_name, df in results.items():
            output_file = self.output_dir / f"{benchmark_name}_results.csv"
            df.to_csv(output_file, index=False)
            print(f"  Saved {benchmark_name} to {output_file}")

        # Save system information
        system_info = {
            "system_memory_gb": self.memory_manager.get_memory_info()[
                "system_total_gb"
            ],
            "gpu_available": self.gpu_accelerator.is_available,
            "cpu_count": os.cpu_count(),
            "test_sizes": self.test_sizes,
            "timestamp": pd.Timestamp.now().isoformat(),
        }

        import json

        system_file = self.output_dir / "system_info.json"
        with open(system_file, "w") as f:
            json.dump(system_info, f, indent=2)

    def generate_comprehensive_plots(self, results: Dict[str, pd.DataFrame]) -> None:
        """Generate comprehensive visualization plots."""
        print("Generating comprehensive plots...")

        plt.style.use("seaborn-v0_8")

        # Create a large figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))

        # Plot 1: Streaming variogram performance
        ax1 = plt.subplot(3, 3, 1)
        if "streaming_variogram" in results:
            df = results["streaming_variogram"]
            for approach in df["approach"].unique():
                subset = df[(df["approach"] == approach) & df["success"]]
                if not subset.empty:
                    ax1.loglog(
                        subset["n_samples"],
                        subset["time_s"],
                        "o-",
                        label=approach,
                        alpha=0.7,
                    )
            ax1.set_xlabel("Number of Samples")
            ax1.set_ylabel("Computation Time (s)")
            ax1.set_title("Streaming Variogram Performance")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # Plot 2: Memory usage comparison
        ax2 = plt.subplot(3, 3, 2)
        if "memory_optimization" in results:
            df = results["memory_optimization"]
            for approach in df["approach"].unique():
                subset = df[(df["approach"] == approach) & df["success"]]
                if not subset.empty:
                    ax2.loglog(
                        subset["n_samples"],
                        subset["memory_used_gb"],
                        "s-",
                        label=approach,
                        alpha=0.7,
                    )
            ax2.set_xlabel("Number of Samples")
            ax2.set_ylabel("Memory Usage (GB)")
            ax2.set_title("Memory Optimization")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        # Plot 3: Parallel execution speedup
        ax3 = plt.subplot(3, 3, 3)
        if "parallel_execution" in results:
            df = results["parallel_execution"]
            df_valid = df[df["success"] & ~df["speedup"].isna()]
            for config in df_valid["config"].unique():
                if config != "sequential":
                    subset = df_valid[df_valid["config"] == config]
                    if not subset.empty:
                        ax3.semilogx(
                            subset["n_samples"],
                            subset["speedup"],
                            "o-",
                            label=config,
                            alpha=0.7,
                        )
            ax3.axhline(
                y=1, color="black", linestyle="--", alpha=0.5, label="No speedup"
            )
            ax3.set_xlabel("Number of Samples")
            ax3.set_ylabel("Speedup Factor")
            ax3.set_title("Parallel Execution Speedup")
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # Plot 4: Neighbor kriging comparison
        ax4 = plt.subplot(3, 3, 4)
        if "neighbor_kriging" in results:
            df = results["neighbor_kriging"]
            for approach in df["approach"].unique():
                subset = df[(df["approach"] == approach) & df["success"]]
                if not subset.empty:
                    ax4.loglog(
                        subset["n_samples"],
                        subset["time_s"],
                        "o-",
                        label=approach,
                        alpha=0.7,
                    )
            ax4.set_xlabel("Number of Samples")
            ax4.set_ylabel("Prediction Time (s)")
            ax4.set_title("Neighbor Kriging Performance")
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        # Plot 5: Throughput comparison (predictions per second)
        ax5 = plt.subplot(3, 3, 5)
        if "neighbor_kriging" in results:
            df = results["neighbor_kriging"]
            for approach in df["approach"].unique():
                subset = df[(df["approach"] == approach) & df["success"]]
                if not subset.empty:
                    ax5.loglog(
                        subset["n_samples"],
                        subset["predictions_per_second"],
                        "o-",
                        label=approach,
                        alpha=0.7,
                    )
            ax5.set_xlabel("Number of Samples")
            ax5.set_ylabel("Predictions per Second")
            ax5.set_title("Kriging Throughput")
            ax5.legend()
            ax5.grid(True, alpha=0.3)

        # Plot 6: Memory efficiency by approach
        ax6 = plt.subplot(3, 3, 6)
        if "memory_optimization" in results:
            df = results["memory_optimization"]
            df_recent = df[df["n_samples"] >= 50000]  # Focus on larger datasets
            if not df_recent.empty:
                approaches = df_recent["approach"].unique()
                efficiencies = [
                    df_recent[df_recent["approach"] == app]["efficiency_score"].mean()
                    for app in approaches
                ]
                bars = ax6.bar(approaches, efficiencies, alpha=0.7)
                ax6.set_ylabel("Efficiency Score")
                ax6.set_title("Memory Efficiency (50k+ samples)")
                ax6.tick_params(axis="x", rotation=45)

                # Add value labels on bars
                for bar, eff in zip(bars, efficiencies):
                    if not np.isnan(eff):
                        ax6.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            bar.get_height() + 0.01,
                            f"{eff:.2f}",
                            ha="center",
                            va="bottom",
                        )

        # Plot 7: Scalability comparison
        ax7 = plt.subplot(3, 3, 7)
        # Combine all timing results for scalability comparison
        all_timing_data = []

        for benchmark, df in results.items():
            if "time_s" in df.columns and "n_samples" in df.columns:
                df_clean = df[df["success"] & ~df["time_s"].isna()].copy()
                df_clean["benchmark"] = benchmark
                all_timing_data.append(df_clean[["n_samples", "time_s", "benchmark"]])

        if all_timing_data:
            combined_df = pd.concat(all_timing_data, ignore_index=True)
            for benchmark in combined_df["benchmark"].unique():
                subset = combined_df[combined_df["benchmark"] == benchmark]
                ax7.loglog(
                    subset["n_samples"],
                    subset["time_s"],
                    "o-",
                    label=benchmark,
                    alpha=0.7,
                )

            # Add theoretical scaling lines
            x_theory = np.array([1000, 1000000])
            ax7.loglog(x_theory, x_theory * 1e-6, "k--", alpha=0.5, label="O(n)")
            ax7.loglog(x_theory, (x_theory**2) * 1e-12, "k:", alpha=0.5, label="O(n²)")

            ax7.set_xlabel("Number of Samples")
            ax7.set_ylabel("Computation Time (s)")
            ax7.set_title("Overall Scalability Comparison")
            ax7.legend()
            ax7.grid(True, alpha=0.3)

        # Plot 8: Success rate by dataset size
        ax8 = plt.subplot(3, 3, 8)
        success_data = []
        for benchmark, df in results.items():
            if "success" in df.columns:
                success_rates = df.groupby("n_samples")["success"].mean().reset_index()
                success_rates["benchmark"] = benchmark
                success_data.append(success_rates)

        if success_data:
            success_df = pd.concat(success_data, ignore_index=True)
            for benchmark in success_df["benchmark"].unique():
                subset = success_df[success_df["benchmark"] == benchmark]
                ax8.semilogx(
                    subset["n_samples"],
                    subset["success"],
                    "o-",
                    label=benchmark,
                    alpha=0.7,
                )

            ax8.set_xlabel("Number of Samples")
            ax8.set_ylabel("Success Rate")
            ax8.set_title("Algorithm Success Rate")
            ax8.set_ylim(0, 1.1)
            ax8.legend()
            ax8.grid(True, alpha=0.3)

        # Plot 9: Memory vs Performance tradeoff
        ax9 = plt.subplot(3, 3, 9)
        if "neighbor_kriging" in results:
            df = results["neighbor_kriging"]
            df_clean = df[
                df["success"] & ~df["time_s"].isna() & ~df["memory_used_gb"].isna()
            ]

            if not df_clean.empty:
                scatter = ax9.scatter(
                    df_clean["memory_used_gb"],
                    df_clean["predictions_per_second"],
                    c=df_clean["n_samples"],
                    s=50,
                    alpha=0.7,
                    cmap="viridis",
                )

                # Add approach labels
                for approach in df_clean["approach"].unique():
                    subset = df_clean[df_clean["approach"] == approach]
                    if len(subset) > 0:
                        ax9.annotate(
                            approach.replace("_", " ").title(),
                            xy=(
                                subset["memory_used_gb"].mean(),
                                subset["predictions_per_second"].mean(),
                            ),
                            xytext=(5, 5),
                            textcoords="offset points",
                            fontsize=8,
                            alpha=0.8,
                        )

                plt.colorbar(scatter, ax=ax9, label="Dataset Size")
                ax9.set_xlabel("Memory Usage (GB)")
                ax9.set_ylabel("Predictions per Second")
                ax9.set_title("Memory vs Performance Tradeoff")
                ax9.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "comprehensive_benchmark_results.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        print(
            f"Comprehensive plots saved to {self.output_dir / 'comprehensive_benchmark_results.png'}"
        )

    def generate_performance_report(self, results: Dict[str, pd.DataFrame]) -> str:
        """Generate a detailed performance report."""
        report_lines = [
            "# PySpatialStats Large Dataset Performance Report",
            f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## System Configuration",
            f"- System Memory: {self.memory_manager.get_memory_info()['system_total_gb']:.1f} GB",
            f"- CPU Cores: {os.cpu_count()}",
            f"- GPU Available: {self.gpu_accelerator.is_available}",
            f"- Test Sizes: {', '.join(f'{s:,}' for s in self.test_sizes)}",
            "",
            "## Performance Summary",
            "",
        ]

        # Streaming Variogram Results
        if "streaming_variogram" in results:
            df = results["streaming_variogram"]
            max_size_regular = (
                df[df["approach"] == "regular"]["n_samples"].max()
                if "regular" in df["approach"].values
                else 0
            )
            max_size_streaming = (
                df[df["approach"] == "streaming"]["n_samples"].max()
                if "streaming" in df["approach"].values
                else 0
            )

            report_lines.extend(
                [
                    "### Streaming Variogram",
                    f"- Regular approach maximum: {max_size_regular:,} samples",
                    f"- Streaming approach maximum: {max_size_streaming:,} samples",
                    f"- Streaming enables processing of datasets {max_size_streaming / max_size_regular:.1f}x larger"
                    if max_size_regular > 0
                    else "",
                    "",
                ]
            )

        # Neighbor Kriging Results
        if "neighbor_kriging" in results:
            df = results["neighbor_kriging"]
            successful = df[df["success"]]

            if not successful.empty:
                best_approach = successful.loc[
                    successful["predictions_per_second"].idxmax(), "approach"
                ]
                best_throughput = successful["predictions_per_second"].max()
                max_dataset = successful["n_samples"].max()

                report_lines.extend(
                    [
                        "### Neighbor-based Kriging",
                        f"- Maximum dataset size: {max_dataset:,} samples",
                        f"- Best approach: {best_approach.replace('_', ' ').title()}",
                        f"- Peak throughput: {best_throughput:.0f} predictions/second",
                        "",
                    ]
                )

        # Parallel Execution Results
        if "parallel_execution" in results:
            df = results["parallel_execution"]
            df_valid = df[df["success"] & ~df["speedup"].isna()]

            if not df_valid.empty:
                max_speedup = df_valid["speedup"].max()
                best_config = df_valid.loc[df_valid["speedup"].idxmax(), "config"]

                report_lines.extend(
                    [
                        "### Parallel Execution",
                        f"- Maximum speedup: {max_speedup:.1f}x",
                        f"- Best configuration: {best_config}",
                        f"- Parallel efficiency: {max_speedup / df_valid.loc[df_valid['speedup'].idxmax(), 'n_workers'] * 100:.1f}%",
                        "",
                    ]
                )

        # Memory Optimization Results
        if "memory_optimization" in results:
            df = results["memory_optimization"]
            successful = df[df["success"]]

            if not successful.empty:
                most_efficient = successful.loc[
                    successful["efficiency_score"].idxmax(), "approach"
                ]
                efficiency = successful["efficiency_score"].max()

                report_lines.extend(
                    [
                        "### Memory Optimization",
                        f"- Most efficient approach: {most_efficient.replace('_', ' ').title()}",
                        f"- Peak efficiency score: {efficiency:.2f}",
                        "",
                    ]
                )

        # Scalability Analysis
        report_lines.extend(["## Scalability Analysis", "", "### Key Findings:", ""])

        # Calculate overall scalability metrics
        largest_successful = {}
        for benchmark, df in results.items():
            if "n_samples" in df.columns and "success" in df.columns:
                successful = df[df["success"]]
                if not successful.empty:
                    largest_successful[benchmark] = successful["n_samples"].max()

        if largest_successful:
            max_overall = max(largest_successful.values())
            best_benchmark = max(
                largest_successful.keys(), key=lambda k: largest_successful[k]
            )

            report_lines.extend(
                [
                    f"- Largest dataset successfully processed: {max_overall:,} samples",
                    f"- Best performing benchmark: {best_benchmark.replace('_', ' ').title()}",
                    "",
                ]
            )

        # Recommendations
        report_lines.extend(
            [
                "## Recommendations",
                "",
                "### For different dataset sizes:",
                "- **< 10,000 samples**: Use standard algorithms",
                "- **10,000 - 50,000 samples**: Consider neighbor-based kriging",
                "- **50,000 - 500,000 samples**: Use streaming variogram + neighbor kriging",
                "- **> 500,000 samples**: Use streaming variogram + parallel neighbor kriging",
                "",
                "### Memory optimization:",
                "- Use sparse matrices for datasets > 25,000 samples",
                "- Enable memory-mapped arrays for very large coordinate datasets",
                "- Consider GPU acceleration when available for > 100,000 samples",
                "",
                "### Performance optimization:",
                "- Use parallel execution for prediction grids > 1,000 points",
                "- Limit neighbor search to 32-64 neighbors for best throughput",
                "- Enable spatial indexing for sparse datasets",
                "",
            ]
        )

        report_text = "\n".join(report_lines)

        # Save report
        report_file = self.output_dir / "performance_report.md"
        with open(report_file, "w") as f:
            f.write(report_text)

        print(f"Performance report saved to {report_file}")
        return report_text


def main():
    """Run the comprehensive large dataset benchmark."""
    import argparse

    parser = argparse.ArgumentParser(
        description="PySpatialStats Large Dataset Benchmark"
    )
    parser.add_argument(
        "--output-dir",
        default="large_dataset_benchmarks",
        help="Output directory for results",
    )
    parser.add_argument(
        "--max-size", type=int, default=None, help="Maximum dataset size to test"
    )
    parser.add_argument(
        "--enable-gpu",
        action="store_true",
        default=True,
        help="Enable GPU acceleration if available",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmark with smaller dataset sizes",
    )

    args = parser.parse_args()

    # Initialize benchmark
    benchmark = LargeDatasetBenchmark(
        output_dir=args.output_dir, enable_gpu=args.enable_gpu
    )

    # Adjust test sizes if requested
    if args.quick:
        benchmark.test_sizes = [1_000, 5_000, 10_000, 25_000, 50_000]
        print("Running quick benchmark with reduced dataset sizes")

    if args.max_size:
        benchmark.test_sizes = [s for s in benchmark.test_sizes if s <= args.max_size]
        print(f"Limited testing to datasets ≤ {args.max_size:,} samples")

    # Run comprehensive benchmark
    results = benchmark.run_comprehensive_benchmark()

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {benchmark.output_dir}")
    print("\nFiles generated:")
    for file_path in benchmark.output_dir.glob("*"):
        print(f"  {file_path.name}")


if __name__ == "__main__":
    main()
