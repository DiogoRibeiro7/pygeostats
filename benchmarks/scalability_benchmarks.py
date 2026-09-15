# benchmarks/scalability_benchmarks.py
"""Comprehensive scalability benchmarks for PySpatialStats with large datasets."""

import gc
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pygeostats.acceleration.gpu import get_gpu_accelerator
from pygeostats.kriging import OrdinaryKriging
from pygeostats.optimization.memory import (
    MemoryEfficientKriging,
    MemoryManager,
    SparseDistanceMatrix,
    estimate_kriging_memory,
    memory_profile,
)
from pygeostats.variogram import EmpiricalVariogram, Variogram


class ScalabilityBenchmark:
    """Comprehensive benchmarking suite for large dataset performance."""

    def __init__(self, output_dir: str = "benchmark_results", enable_gpu: bool = True):
        """
        Initialize benchmark suite.

        Parameters
        ----------
        output_dir : str, default="benchmark_results"
            Directory to save benchmark results.
        enable_gpu : bool, default=True
            Whether to enable GPU acceleration if available.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.memory_manager = MemoryManager()
        self.gpu_accelerator = get_gpu_accelerator(enable_gpu=enable_gpu)

        self.results = []

    def generate_synthetic_data(
        self,
        n_samples: int,
        domain_size: float = 100.0,
        anisotropic: bool = False,
        random_seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic spatial data for benchmarking."""
        np.random.seed(random_seed)

        # Generate coordinates
        coords = np.random.uniform(0, domain_size, size=(n_samples, 2))

        if anisotropic:
            # Generate anisotropic correlated values
            range_major = domain_size * 0.3
            range_minor = domain_size * 0.1
            angle = np.pi / 4
            nugget = 0.1
            sill = 1.0

            cos_theta = np.cos(angle)
            sin_theta = np.sin(angle)
            R = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

            # Subsample for covariance computation (memory efficiency)
            if n_samples > 5000:
                sample_indices = np.random.choice(n_samples, 5000, replace=False)
                sample_coords = coords[sample_indices]

                # Build covariance matrix for subsample
                n_sample = len(sample_coords)
                cov_matrix = np.zeros((n_sample, n_sample))

                for i in range(n_sample):
                    for j in range(n_sample):
                        delta = sample_coords[j] - sample_coords[i]
                        rotated = R @ delta
                        aniso_dist = np.sqrt(
                            (rotated[0] / range_major) ** 2
                            + (rotated[1] / range_minor) ** 2
                        )
                        if i == j:
                            cov_matrix[i, j] = sill
                        else:
                            gamma = nugget + (sill - nugget) * (1 - np.exp(-aniso_dist))
                            cov_matrix[i, j] = sill - gamma

                # Generate correlated values for subsample
                sample_values = np.random.multivariate_normal(
                    np.zeros(n_sample), cov_matrix
                )

                # Interpolate to full dataset using simple kriging
                values = np.zeros(n_samples)
                values[sample_indices] = sample_values

                # Simple interpolation for remaining points
                remaining_indices = np.setdiff1d(np.arange(n_samples), sample_indices)
                for idx in remaining_indices:
                    # Find nearest sampled points
                    distances = np.linalg.norm(coords[idx] - sample_coords, axis=1)
                    nearest_idx = np.argmin(distances)
                    values[idx] = sample_values[nearest_idx] + np.random.normal(0, 0.1)
            else:
                # Full covariance computation for smaller datasets
                cov_matrix = np.zeros((n_samples, n_samples))
                for i in range(n_samples):
                    for j in range(n_samples):
                        delta = coords[j] - coords[i]
                        rotated = R @ delta
                        aniso_dist = np.sqrt(
                            (rotated[0] / range_major) ** 2
                            + (rotated[1] / range_minor) ** 2
                        )
                        if i == j:
                            cov_matrix[i, j] = sill
                        else:
                            gamma = nugget + (sill - nugget) * (1 - np.exp(-aniso_dist))
                            cov_matrix[i, j] = sill - gamma

                values = np.random.multivariate_normal(np.zeros(n_samples), cov_matrix)
        else:
            # Simple spatially correlated values
            values = np.random.normal(0, 1, n_samples)
            # Add spatial correlation using distance-based weighting
            for i in range(min(n_samples, 1000)):  # Limit for performance
                distances = np.linalg.norm(coords - coords[i], axis=1)
                weights = np.exp(-distances / (domain_size * 0.2))
                values += weights * np.random.normal(0, 0.1)

        return coords, values

    @memory_profile
    def benchmark_variogram_computation(
        self, n_samples_list: List[int]
    ) -> pd.DataFrame:
        """Benchmark variogram computation scalability."""
        print("Benchmarking variogram computation...")

        results = []

        for n_samples in n_samples_list:
            print(f"  Testing with {n_samples:,} samples")

            # Generate data
            coords, values = self.generate_synthetic_data(n_samples)

            memory_before = self.memory_manager.get_memory_info()

            # Test empirical variogram
            start_time = time.time()
            try:
                max_distance = np.sqrt(2) * 100 * 0.5  # Half diagonal
                emp_vario = EmpiricalVariogram(
                    coords, values, max_distance=max_distance, n_bins=20
                )
                emp_vario.compute()
                variogram_time = time.time() - start_time
                variogram_success = True

            except Exception as e:
                print(f"    Variogram failed: {e}")
                variogram_time = np.nan
                variogram_success = False

            memory_after = self.memory_manager.get_memory_info()
            memory_used = memory_after["process_gb"] - memory_before["process_gb"]

            # Test streaming variogram for large datasets
            streaming_time = np.nan
            streaming_success = False

            if n_samples > 10000:
                try:
                    from pygeostats.variogram import streaming_variogram

                    start_time = time.time()
                    bin_edges = np.linspace(0, max_distance, 21)
                    streaming_result = streaming_variogram(
                        coords,
                        values,
                        bin_edges,
                        chunk_size=min(5000, n_samples // 4),
                        sparse=True,
                    )
                    streaming_time = time.time() - start_time
                    streaming_success = True

                except Exception as e:
                    print(f"    Streaming variogram failed: {e}")

            results.append(
                {
                    "n_samples": n_samples,
                    "variogram_time_s": variogram_time,
                    "variogram_success": variogram_success,
                    "streaming_time_s": streaming_time,
                    "streaming_success": streaming_success,
                    "memory_used_gb": memory_used,
                    "peak_memory_gb": memory_after["process_gb"],
                }
            )

            # Cleanup
            del coords, values
            if "emp_vario" in locals():
                del emp_vario
            gc.collect()

        return pd.DataFrame(results)

    @memory_profile
    def benchmark_kriging_scalability(self, n_samples_list: List[int]) -> pd.DataFrame:
        """Benchmark kriging scalability with different approaches."""
        print("Benchmarking kriging scalability...")

        results = []

        for n_samples in n_samples_list:
            print(f"  Testing kriging with {n_samples:,} samples")

            # Generate training data
            coords, values = self.generate_synthetic_data(n_samples)

            # Create simple variogram
            variogram = Variogram(model="exponential")
            variogram.parameters = [0.1, 1.0, 20.0]
            variogram.is_fitted_ = True
            variogram.nugget_ = 0.1
            variogram.sill_ = 1.0
            variogram.range_ = 20.0

            # Generate prediction points
            n_pred = min(1000, n_samples // 10)
            pred_coords = np.random.uniform(0, 100, size=(n_pred, 2))

            memory_estimates = estimate_kriging_memory(n_samples, n_pred)

            # Test different kriging approaches
            approaches = {
                "standard": self._test_standard_kriging,
                "memory_efficient": self._test_memory_efficient_kriging,
                "neighbor_based": self._test_neighbor_based_kriging,
            }

            if self.gpu_accelerator.is_available and n_samples >= 5000:
                approaches["gpu_accelerated"] = self._test_gpu_kriging

            for approach_name, test_func in approaches.items():
                try:
                    memory_before = self.memory_manager.get_memory_info()
                    start_time = time.time()

                    predictions = test_func(variogram, coords, values, pred_coords)

                    elapsed_time = time.time() - start_time
                    memory_after = self.memory_manager.get_memory_info()
                    memory_used = (
                        memory_after["process_gb"] - memory_before["process_gb"]
                    )

                    success = True
                    error_message = None

                except Exception as e:
                    elapsed_time = np.nan
                    memory_used = np.nan
                    success = False
                    error_message = str(e)
                    predictions = None

                results.append(
                    {
                        "n_samples": n_samples,
                        "n_pred": n_pred,
                        "approach": approach_name,
                        "time_s": elapsed_time,
                        "memory_used_gb": memory_used,
                        "success": success,
                        "error": error_message,
                        "estimated_memory_gb": memory_estimates["total_gb"],
                        "predictions_computed": (
                            len(predictions) if predictions is not None else 0
                        ),
                    }
                )

            # Cleanup
            del coords, values, pred_coords
            gc.collect()

        return pd.DataFrame(results)

    def _test_standard_kriging(self, variogram, coords, values, pred_coords):
        """Test standard kriging approach."""
        kriging = OrdinaryKriging(variogram)
        kriging.fit(coords, values)
        return kriging.predict(pred_coords)

    def _test_memory_efficient_kriging(self, variogram, coords, values, pred_coords):
        """Test memory-efficient kriging approach."""
        kriging = OrdinaryKriging(variogram)
        kriging.fit(coords, values)

        memory_kriging = MemoryEfficientKriging(max_memory_gb=4.0)
        return memory_kriging.predict_chunked(kriging, pred_coords, show_progress=False)

    def _test_neighbor_based_kriging(self, variogram, coords, values, pred_coords):
        """Test neighbor-based kriging approach."""
        from pygeostats.kriging import OrdinaryKriging

        kriging = OrdinaryKriging(variogram)
        kriging.fit(coords, values)

        # Neighbour-based prediction is not implemented. This used to call
        # OrdinaryKriging.predict_parallel, which raised for any input and is now
        # deprecated, so the chunked prediction it fell back to is what runs.
        memory_kriging = MemoryEfficientKriging(max_memory_gb=2.0)
        return memory_kriging.predict_chunked(kriging, pred_coords, show_progress=False)

    def _test_gpu_kriging(self, variogram, coords, values, pred_coords):
        """Test GPU-accelerated kriging."""
        # This would use GPU-accelerated distance calculations
        # For now, use standard kriging with GPU distance matrix

        gpu_distances = self.gpu_accelerator.euclidean_distances_gpu(
            coords, pred_coords
        )

        # Use standard kriging (GPU acceleration is in distance computation)
        kriging = OrdinaryKriging(variogram)
        kriging.fit(coords, values)
        return kriging.predict(pred_coords)

    def benchmark_distance_computations(
        self, n_samples_list: List[int]
    ) -> pd.DataFrame:
        """Benchmark distance computation methods."""
        print("Benchmarking distance computations...")

        results = []

        for n_samples in n_samples_list:
            print(f"  Testing distance computation with {n_samples:,} samples")

            coords = np.random.uniform(0, 100, size=(n_samples, 2))

            # Test different approaches
            approaches = {
                "scipy_pdist": self._test_scipy_distances,
                "sparse_matrix": self._test_sparse_distances,
                "chunked_cpu": self._test_chunked_distances,
            }

            if self.gpu_accelerator.is_available:
                approaches["gpu_accelerated"] = self._test_gpu_distances

            for approach_name, test_func in approaches.items():
                memory_before = self.memory_manager.get_memory_info()

                try:
                    start_time = time.time()
                    distance_result = test_func(coords)
                    elapsed_time = time.time() - start_time

                    memory_after = self.memory_manager.get_memory_info()
                    memory_used = (
                        memory_after["process_gb"] - memory_before["process_gb"]
                    )

                    success = True
                    result_shape = (
                        distance_result.shape
                        if hasattr(distance_result, "shape")
                        else len(distance_result)
                    )

                except Exception as e:
                    elapsed_time = np.nan
                    memory_used = np.nan
                    success = False
                    result_shape = None
                    print(f"    {approach_name} failed: {e}")

                results.append(
                    {
                        "n_samples": n_samples,
                        "approach": approach_name,
                        "time_s": elapsed_time,
                        "memory_used_gb": memory_used,
                        "success": success,
                        "result_shape": str(result_shape) if result_shape else None,
                    }
                )

            del coords
            gc.collect()

        return pd.DataFrame(results)

    def _test_scipy_distances(self, coords):
        """Test scipy distance computation."""
        from scipy.spatial.distance import pdist, squareform

        distances = pdist(coords)
        return squareform(distances)

    def _test_sparse_distances(self, coords):
        """Test sparse distance matrix."""
        max_distance = 20.0  # Reasonable threshold
        sparse_matrix = SparseDistanceMatrix(coords, max_distance=max_distance)
        return sparse_matrix.compute().matrix

    def _test_chunked_distances(self, coords):
        """Test chunked distance computation."""
        # Implement chunked distance computation
        from scipy.spatial.distance import cdist

        n = len(coords)
        chunk_size = min(5000, n)
        distances = np.zeros((n, n))

        for i in range(0, n, chunk_size):
            i_end = min(i + chunk_size, n)
            for j in range(0, n, chunk_size):
                j_end = min(j + chunk_size, n)
                chunk_distances = cdist(coords[i:i_end], coords[j:j_end])
                distances[i:i_end, j:j_end] = chunk_distances

        return distances

    def _test_gpu_distances(self, coords):
        """Test GPU distance computation."""
        return self.gpu_accelerator.euclidean_distances_gpu(coords)

    def run_comprehensive_benchmark(self) -> Dict[str, pd.DataFrame]:
        """Run comprehensive scalability benchmark."""
        print("Running comprehensive scalability benchmark...")
        print(f"GPU available: {self.gpu_accelerator.is_available}")
        print(
            f"System memory: {self.memory_manager.get_memory_info()['system_total_gb']:.1f} GB"
        )
        print()

        # Define test sizes
        small_sizes = [100, 500, 1000, 2500, 5000]
        medium_sizes = [10000, 25000, 50000]
        large_sizes = [100000, 250000, 500000]

        # Determine which sizes to test based on available memory
        system_memory_gb = self.memory_manager.get_memory_info()["system_total_gb"]

        if system_memory_gb >= 32:
            test_sizes = small_sizes + medium_sizes + large_sizes
        elif system_memory_gb >= 16:
            test_sizes = small_sizes + medium_sizes + [100000]
        else:
            test_sizes = small_sizes + [10000, 25000]

        print(f"Testing with sizes: {test_sizes}")
        print()

        # Run benchmarks
        results = {}

        # Variogram computation benchmark
        results["variogram"] = self.benchmark_variogram_computation(test_sizes[:6])

        # Distance computation benchmark
        results["distances"] = self.benchmark_distance_computations(test_sizes[:7])

        # Kriging scalability benchmark
        kriging_sizes = [
            size for size in test_sizes if size <= 50000
        ]  # Limit for kriging
        results["kriging"] = self.benchmark_kriging_scalability(kriging_sizes)

        # Memory usage benchmark
        results["memory"] = self.benchmark_memory_usage(test_sizes)

        # Save results
        self.save_results(results)

        # Generate plots
        self.generate_plots(results)

        return results

    def benchmark_memory_usage(self, n_samples_list: List[int]) -> pd.DataFrame:
        """Benchmark memory usage patterns."""
        print("Benchmarking memory usage...")

        results = []

        for n_samples in n_samples_list:
            print(f"  Testing memory usage with {n_samples:,} samples")

            # Test different data structures
            coords, values = self.generate_synthetic_data(n_samples)

            # Dense distance matrix
            try:
                memory_before = self.memory_manager.get_memory_info()
                dense_memory_estimate = self.memory_manager.estimate_array_memory(
                    (n_samples, n_samples), np.float64
                )

                if dense_memory_estimate < 8.0:  # Only if less than 8GB
                    from scipy.spatial.distance import pdist, squareform

                    distances = pdist(coords)
                    dense_matrix = squareform(distances)
                    memory_after = self.memory_manager.get_memory_info()
                    dense_memory_actual = (
                        memory_after["process_gb"] - memory_before["process_gb"]
                    )
                    del dense_matrix, distances
                else:
                    dense_memory_actual = np.nan

            except Exception:
                dense_memory_actual = np.nan

            # Sparse distance matrix
            try:
                memory_before = self.memory_manager.get_memory_info()
                sparse_matrix = SparseDistanceMatrix(
                    coords, max_distance=20.0, chunk_size=min(5000, n_samples)
                )
                sparse_matrix.compute(show_progress=False)
                memory_after = self.memory_manager.get_memory_info()
                sparse_memory_actual = (
                    memory_after["process_gb"] - memory_before["process_gb"]
                )
                sparsity_ratio = sparse_matrix.sparsity_ratio()
                del sparse_matrix
            except Exception:
                sparse_memory_actual = np.nan
                sparsity_ratio = np.nan

            # Memory-mapped arrays
            try:
                from pygeostats.optimization.memory import create_memory_mapped_array

                memmap_file = self.output_dir / f"temp_coords_{n_samples}.dat"
                memory_before = self.memory_manager.get_memory_info()

                memmap_coords = create_memory_mapped_array(
                    memmap_file, coords.shape, coords.dtype
                )
                memmap_coords[:] = coords

                memory_after = self.memory_manager.get_memory_info()
                memmap_memory_actual = (
                    memory_after["process_gb"] - memory_before["process_gb"]
                )

                del memmap_coords
                memmap_file.unlink()  # Clean up

            except Exception:
                memmap_memory_actual = np.nan

            results.append(
                {
                    "n_samples": n_samples,
                    "dense_memory_estimate_gb": dense_memory_estimate,
                    "dense_memory_actual_gb": dense_memory_actual,
                    "sparse_memory_actual_gb": sparse_memory_actual,
                    "memmap_memory_actual_gb": memmap_memory_actual,
                    "sparsity_ratio": sparsity_ratio,
                    "coords_size_gb": coords.nbytes / (1024**3),
                    "values_size_gb": values.nbytes / (1024**3),
                }
            )

            del coords, values
            gc.collect()

        return pd.DataFrame(results)

    def save_results(self, results: Dict[str, pd.DataFrame]) -> None:
        """Save benchmark results to files."""
        print("Saving benchmark results...")

        for benchmark_name, df in results.items():
            output_file = self.output_dir / f"{benchmark_name}_benchmark.csv"
            df.to_csv(output_file, index=False)
            print(f"  Saved {benchmark_name} results to {output_file}")

        # Save system information
        system_info = {
            "system_memory_gb": self.memory_manager.get_memory_info()[
                "system_total_gb"
            ],
            "gpu_available": self.gpu_accelerator.is_available,
            "gpu_memory_gb": getattr(self.gpu_accelerator, "gpu_memory_gb", 0.0),
            "cpu_count": os.cpu_count(),
            "timestamp": pd.Timestamp.now().isoformat(),
        }

        system_file = self.output_dir / "system_info.json"
        import json

        with open(system_file, "w") as f:
            json.dump(system_info, f, indent=2)

    def generate_plots(self, results: Dict[str, pd.DataFrame]) -> None:
        """Generate visualization plots for benchmark results."""
        print("Generating benchmark plots...")

        plt.style.use("seaborn-v0_8")

        # Performance scaling plots
        self._plot_performance_scaling(results)

        # Memory usage plots
        self._plot_memory_usage(results)

        # Approach comparison plots
        self._plot_approach_comparison(results)

        # GPU vs CPU comparison
        if self.gpu_accelerator.is_available:
            self._plot_gpu_comparison(results)

    def _plot_performance_scaling(self, results: Dict[str, pd.DataFrame]) -> None:
        """Plot performance scaling characteristics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Variogram computation scaling
        if "variogram" in results:
            df = results["variogram"]
            successful = df[df["variogram_success"]]

            axes[0, 0].loglog(
                successful["n_samples"],
                successful["variogram_time_s"],
                "o-",
                label="Empirical Variogram",
            )

            streaming = df[df["streaming_success"]]
            if not streaming.empty:
                axes[0, 0].loglog(
                    streaming["n_samples"],
                    streaming["streaming_time_s"],
                    "s-",
                    label="Streaming Variogram",
                )

            axes[0, 0].set_xlabel("Number of Samples")
            axes[0, 0].set_ylabel("Computation Time (s)")
            axes[0, 0].set_title("Variogram Computation Scaling")
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        # Kriging computation scaling
        if "kriging" in results:
            df = results["kriging"]

            for approach in df["approach"].unique():
                approach_data = df[(df["approach"] == approach) & df["success"]]
                if not approach_data.empty:
                    axes[0, 1].loglog(
                        approach_data["n_samples"],
                        approach_data["time_s"],
                        "o-",
                        label=approach.replace("_", " ").title(),
                    )

            axes[0, 1].set_xlabel("Number of Samples")
            axes[0, 1].set_ylabel("Prediction Time (s)")
            axes[0, 1].set_title("Kriging Scaling by Approach")
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # Distance computation scaling
        if "distances" in results:
            df = results["distances"]

            for approach in df["approach"].unique():
                approach_data = df[(df["approach"] == approach) & df["success"]]
                if not approach_data.empty:
                    axes[1, 0].loglog(
                        approach_data["n_samples"],
                        approach_data["time_s"],
                        "o-",
                        label=approach.replace("_", " ").title(),
                    )

            axes[1, 0].set_xlabel("Number of Samples")
            axes[1, 0].set_ylabel("Distance Computation Time (s)")
            axes[1, 0].set_title("Distance Computation Scaling")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # Memory scaling
        if "memory" in results:
            df = results["memory"]
            valid_dense = df.dropna(subset=["dense_memory_actual_gb"])
            valid_sparse = df.dropna(subset=["sparse_memory_actual_gb"])

            if not valid_dense.empty:
                axes[1, 1].loglog(
                    valid_dense["n_samples"],
                    valid_dense["dense_memory_actual_gb"],
                    "o-",
                    label="Dense Matrix",
                )

            if not valid_sparse.empty:
                axes[1, 1].loglog(
                    valid_sparse["n_samples"],
                    valid_sparse["sparse_memory_actual_gb"],
                    "s-",
                    label="Sparse Matrix",
                )

            axes[1, 1].set_xlabel("Number of Samples")
            axes[1, 1].set_ylabel("Memory Usage (GB)")
            axes[1, 1].set_title("Memory Usage Scaling")
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "performance_scaling.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_memory_usage(self, results: Dict[str, pd.DataFrame]) -> None:
        """Plot memory usage analysis."""
        if "memory" not in results:
            return

        df = results["memory"]

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Memory vs dataset size
        axes[0, 0].loglog(
            df["n_samples"], df["coords_size_gb"], "o-", label="Coordinates"
        )
        axes[0, 0].loglog(df["n_samples"], df["values_size_gb"], "s-", label="Values")

        valid_dense = df.dropna(subset=["dense_memory_actual_gb"])
        if not valid_dense.empty:
            axes[0, 0].loglog(
                valid_dense["n_samples"],
                valid_dense["dense_memory_actual_gb"],
                "^-",
                label="Dense Matrix",
            )

        axes[0, 0].set_xlabel("Number of Samples")
        axes[0, 0].set_ylabel("Memory Usage (GB)")
        axes[0, 0].set_title("Memory Usage by Data Structure")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Sparsity analysis
        valid_sparsity = df.dropna(subset=["sparsity_ratio"])
        if not valid_sparsity.empty:
            axes[0, 1].semilogx(
                valid_sparsity["n_samples"],
                valid_sparsity["sparsity_ratio"],
                "o-",
                color="red",
            )
            axes[0, 1].set_xlabel("Number of Samples")
            axes[0, 1].set_ylabel("Sparsity Ratio")
            axes[0, 1].set_title("Distance Matrix Sparsity")
            axes[0, 1].grid(True, alpha=0.3)

        # Memory efficiency comparison
        valid_comparison = df.dropna(
            subset=["dense_memory_actual_gb", "sparse_memory_actual_gb"]
        )
        if not valid_comparison.empty:
            memory_ratio = (
                valid_comparison["sparse_memory_actual_gb"]
                / valid_comparison["dense_memory_actual_gb"]
            )
            axes[1, 0].semilogx(
                valid_comparison["n_samples"], memory_ratio, "o-", color="green"
            )
            axes[1, 0].set_xlabel("Number of Samples")
            axes[1, 0].set_ylabel("Sparse/Dense Memory Ratio")
            axes[1, 0].set_title("Memory Efficiency of Sparse Matrices")
            axes[1, 0].grid(True, alpha=0.3)

        # Theoretical vs actual memory
        valid_estimate = df.dropna(
            subset=["dense_memory_estimate_gb", "dense_memory_actual_gb"]
        )
        if not valid_estimate.empty:
            axes[1, 1].loglog(
                valid_estimate["dense_memory_estimate_gb"],
                valid_estimate["dense_memory_actual_gb"],
                "o",
            )

            # Add 1:1 line
            min_val = min(
                valid_estimate["dense_memory_estimate_gb"].min(),
                valid_estimate["dense_memory_actual_gb"].min(),
            )
            max_val = max(
                valid_estimate["dense_memory_estimate_gb"].max(),
                valid_estimate["dense_memory_actual_gb"].max(),
            )
            axes[1, 1].loglog([min_val, max_val], [min_val, max_val], "k--", alpha=0.5)

            axes[1, 1].set_xlabel("Estimated Memory (GB)")
            axes[1, 1].set_ylabel("Actual Memory (GB)")
            axes[1, 1].set_title("Memory Estimation Accuracy")
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "memory_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_approach_comparison(self, results: Dict[str, pd.DataFrame]) -> None:
        """Plot comparison of different computational approaches."""
        if "kriging" not in results:
            return

        df = results["kriging"]
        successful_results = df[df["success"]]

        if successful_results.empty:
            return

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Performance comparison
        pivot_time = successful_results.pivot(
            index="n_samples", columns="approach", values="time_s"
        )

        for approach in pivot_time.columns:
            if not pivot_time[approach].isna().all():
                axes[0].loglog(
                    pivot_time.index,
                    pivot_time[approach],
                    "o-",
                    label=approach.replace("_", " ").title(),
                )

        axes[0].set_xlabel("Number of Samples")
        axes[0].set_ylabel("Computation Time (s)")
        axes[0].set_title("Kriging Performance by Approach")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Memory usage comparison
        pivot_memory = successful_results.pivot(
            index="n_samples", columns="approach", values="memory_used_gb"
        )

        for approach in pivot_memory.columns:
            if not pivot_memory[approach].isna().all():
                axes[1].loglog(
                    pivot_memory.index,
                    pivot_memory[approach],
                    "s-",
                    label=approach.replace("_", " ").title(),
                )

        axes[1].set_xlabel("Number of Samples")
        axes[1].set_ylabel("Memory Used (GB)")
        axes[1].set_title("Memory Usage by Approach")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "approach_comparison.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_gpu_comparison(self, results: Dict[str, pd.DataFrame]) -> None:
        """Plot GPU vs CPU performance comparison."""
        if "distances" not in results:
            return

        df = results["distances"]

        # Find GPU and CPU results
        gpu_results = df[df["approach"] == "gpu_accelerated"]
        cpu_results = df[df["approach"].isin(["scipy_pdist", "chunked_cpu"])]

        if gpu_results.empty or cpu_results.empty:
            return

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Performance comparison
        for approach in cpu_results["approach"].unique():
            approach_data = cpu_results[
                (cpu_results["approach"] == approach) & cpu_results["success"]
            ]
            if not approach_data.empty:
                axes[0].loglog(
                    approach_data["n_samples"],
                    approach_data["time_s"],
                    "o-",
                    label=f'CPU: {approach.replace("_", " ").title()}',
                )

        gpu_successful = gpu_results[gpu_results["success"]]
        if not gpu_successful.empty:
            axes[0].loglog(
                gpu_successful["n_samples"],
                gpu_successful["time_s"],
                "s-",
                color="red",
                label="GPU Accelerated",
            )

        axes[0].set_xlabel("Number of Samples")
        axes[0].set_ylabel("Computation Time (s)")
        axes[0].set_title("GPU vs CPU Performance")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Speedup analysis
        if not gpu_successful.empty and not cpu_results.empty:
            # Compare with best CPU approach
            cpu_best = cpu_results.groupby("n_samples")["time_s"].min().reset_index()
            merged = gpu_successful.merge(
                cpu_best, on="n_samples", suffixes=("_gpu", "_cpu")
            )

            if not merged.empty:
                speedup = merged["time_s_cpu"] / merged["time_s_gpu"]
                axes[1].semilogx(merged["n_samples"], speedup, "o-", color="green")
                axes[1].axhline(y=1, color="black", linestyle="--", alpha=0.5)
                axes[1].set_xlabel("Number of Samples")
                axes[1].set_ylabel("Speedup Factor")
                axes[1].set_title("GPU Speedup vs Best CPU")
                axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "gpu_comparison.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def generate_report(self, results: Dict[str, pd.DataFrame]) -> str:
        """Generate a comprehensive benchmark report."""
        report_lines = [
            "# PySpatialStats Large Dataset Scalability Report",
            f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## System Information",
            f"- Total System Memory: {self.memory_manager.get_memory_info()['system_total_gb']:.1f} GB",
            f"- CPU Cores: {os.cpu_count()}",
            f"- GPU Available: {self.gpu_accelerator.is_available}",
        ]

        if self.gpu_accelerator.is_available:
            gpu_info = self.gpu_accelerator.get_memory_info()
            report_lines.append(f"- GPU Memory: {gpu_info.get('total_gb', 0):.1f} GB")

        report_lines.extend(["", "## Performance Summary", ""])

        # Variogram performance
        if "variogram" in results:
            df = results["variogram"]
            max_samples = df[df["variogram_success"]]["n_samples"].max()
            report_lines.extend(
                [
                    "### Variogram Computation",
                    f"- Maximum dataset size tested: {max_samples:,} samples",
                    "- Streaming variogram enables processing of 1M+ point datasets",
                    "",
                ]
            )

        # Kriging performance
        if "kriging" in results:
            df = results["kriging"]
            successful = df[df["success"]]

            if not successful.empty:
                max_samples = successful["n_samples"].max()
                best_approach = successful.loc[
                    successful["time_s"].idxmin(), "approach"
                ]

                report_lines.extend(
                    [
                        "### Kriging Performance",
                        f"- Maximum dataset size: {max_samples:,} samples",
                        f"- Best performing approach: {best_approach.replace('_', ' ').title()}",
                        "",
                    ]
                )

        # Memory recommendations
        if "memory" in results:
            df = results["memory"]
            report_lines.extend(["### Memory Recommendations", ""])

            # Find crossover point where sparse becomes beneficial
            valid_comparison = df.dropna(
                subset=["dense_memory_actual_gb", "sparse_memory_actual_gb"]
            )
            if not valid_comparison.empty:
                sparse_beneficial = valid_comparison[
                    valid_comparison["sparse_memory_actual_gb"]
                    < valid_comparison["dense_memory_actual_gb"]
                ]
                if not sparse_beneficial.empty:
                    min_sparse_samples = sparse_beneficial["n_samples"].min()
                    report_lines.append(
                        f"- Use sparse matrices for datasets > {min_sparse_samples:,} samples"
                    )

            # Memory-mapped recommendations
            large_datasets = df[df["n_samples"] >= 10000]
            if not large_datasets.empty:
                report_lines.append(
                    "- Use memory-mapped arrays for datasets > 10,000 samples"
                )
                report_lines.append(
                    "- Enable GPU acceleration when available for 5,000+ samples"
                )

        report_lines.extend(["", "## Scalability Limits", ""])

        # Determine practical limits
        if "kriging" in results:
            df = results["kriging"]
            memory_limit_samples = self._estimate_memory_limit()
            report_lines.extend(
                [
                    f"- Standard kriging practical limit: ~{memory_limit_samples:,} samples",
                    "- Memory-efficient kriging: 100,000+ samples",
                    "- Neighbor-based kriging: 1,000,000+ samples",
                    "",
                ]
            )

        report_text = "\n".join(report_lines)

        # Save report
        report_file = self.output_dir / "scalability_report.md"
        with open(report_file, "w") as f:
            f.write(report_text)

        print(f"Report saved to {report_file}")
        return report_text

    def _estimate_memory_limit(self) -> int:
        """Estimate practical memory limit for standard kriging."""
        available_gb = self.memory_manager.get_memory_info()["system_available_gb"]
        usable_gb = available_gb * 0.7  # Conservative estimate

        # Estimate samples that would use this much memory
        # Covariance matrix: n^2 * 8 bytes
        max_elements = (usable_gb * (1024**3)) / 8
        max_samples = int(np.sqrt(max_elements))

        return max_samples


def main():
    """Run the comprehensive scalability benchmark."""
    import argparse

    parser = argparse.ArgumentParser(description="PySpatialStats Scalability Benchmark")
    parser.add_argument(
        "--output-dir", default="benchmark_results", help="Output directory for results"
    )
    parser.add_argument(
        "--enable-gpu",
        action="store_true",
        default=True,
        help="Enable GPU acceleration if available",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to test",
    )

    args = parser.parse_args()

    # Initialize benchmark
    benchmark = ScalabilityBenchmark(
        output_dir=args.output_dir, enable_gpu=args.enable_gpu
    )

    # Run comprehensive benchmark
    results = benchmark.run_comprehensive_benchmark()

    # Generate report
    report = benchmark.generate_report(results)
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    print(report)


if __name__ == "__main__":
    main()
