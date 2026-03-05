# src/python/pyspatialstats/__init__.py
"""
PySpatialStats: High-performance spatial statistics for Python
==============================================================

A comprehensive spatial statistics library with Rust-accelerated core algorithms.
"""

__version__ = "0.1.0"

from . import point_patterns
from . import utils

from .point_patterns import (
    cluster_validation_metrics,
    f_function,
    g_function,
    getis_ord_gi_star,
    kernel_density_estimate,
    nearest_neighbor_distances,
    pair_correlation_function,
    ripley_k_function,
    ripley_l_function,
    compute_spatial_segregation_indices,
    simulate_cox_process,
    simulate_marked_poisson_process,
    simulate_poisson_process,
    spatial_dbscan,
)

kriging = None
validation = None
variogram = None

try:
    from . import kriging
    from .kriging import OrdinaryKriging, SimpleKriging, UniversalKriging
except ModuleNotFoundError:
    pass

try:
    from . import validation
    from .validation import (
        CrossValidationResult,
        block_cross_validation,
        compute_kriging_residuals,
        leave_one_out_cross_validation,
        normality_test,
        select_best_variogram_model,
        spatial_kfold_cross_validation,
        standardized_residuals,
        variogram_aic,
        variogram_bic,
        variogram_cloud,
    )
except ModuleNotFoundError:
    pass

try:
    from . import variogram
    from .variogram import (
        AnisotropyInitializationCandidate,
        AnisotropyInitializationSummary,
        AnisotropyResult,
        DirectionalVariogram,
        EmpiricalVariogram,
        Variogram,
    )
except ModuleNotFoundError:
    pass

__all__ = [
    "utils",
    "point_patterns",
    "nearest_neighbor_distances",
    "ripley_k_function",
    "ripley_l_function",
    "g_function",
    "f_function",
    "pair_correlation_function",
    "spatial_dbscan",
    "getis_ord_gi_star",
    "kernel_density_estimate",
    "cluster_validation_metrics",
    "simulate_cox_process",
    "simulate_marked_poisson_process",
    "simulate_poisson_process",
    "compute_spatial_segregation_indices",
]

if kriging is not None:
    __all__.extend(
        [
            "kriging",
            "OrdinaryKriging",
            "SimpleKriging",
            "UniversalKriging",
        ]
    )

if validation is not None:
    __all__.extend(
        [
            "validation",
            "CrossValidationResult",
            "block_cross_validation",
            "leave_one_out_cross_validation",
            "spatial_kfold_cross_validation",
            "variogram_aic",
            "variogram_bic",
            "select_best_variogram_model",
            "compute_kriging_residuals",
            "variogram_cloud",
            "standardized_residuals",
            "normality_test",
        ]
    )

if variogram is not None:
    __all__.extend(
        [
            "variogram",
            "Variogram",
            "EmpiricalVariogram",
            "DirectionalVariogram",
            "AnisotropyResult",
            "AnisotropyInitializationCandidate",
            "AnisotropyInitializationSummary",
        ]
    )
