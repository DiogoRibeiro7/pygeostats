# src/python/pygeostats/__init__.py
"""
PySpatialStats: High-performance spatial statistics for Python
==============================================================

A comprehensive spatial statistics library with Rust-accelerated core algorithms.
"""

__version__ = "0.1.0a2"

from . import point_patterns, spatial_autocorrelation, utils
from .point_patterns import (
    cluster_validation_metrics,
    compute_spatial_segregation_indices,
    f_function,
    g_function,
    getis_ord_gi_star,
    kernel_density_estimate,
    nearest_neighbor_distances,
    pair_correlation_function,
    ripley_k_function,
    ripley_l_function,
    simulate_cox_process,
    simulate_marked_poisson_process,
    simulate_poisson_process,
    spatial_dbscan,
)
from .spatial_autocorrelation import (
    gearys_c,
    global_getis_ord_g,
    local_gearys_c,
    local_getis_ord_g,
    local_morans_i,
    morans_i,
    row_standardize_weights,
    spatial_weights_distance_band,
    spatial_weights_inverse_distance,
    spatial_weights_knn,
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
        AnisotropyResult,
        DirectionalVariogram,
        EmpiricalVariogram,
        Variogram,
    )
except ModuleNotFoundError:
    pass

__all__ = [
    "cluster_validation_metrics",
    "compute_spatial_segregation_indices",
    "f_function",
    "g_function",
    "gearys_c",
    "getis_ord_gi_star",
    "global_getis_ord_g",
    "kernel_density_estimate",
    "local_gearys_c",
    "local_getis_ord_g",
    "local_morans_i",
    "morans_i",
    "nearest_neighbor_distances",
    "pair_correlation_function",
    "point_patterns",
    "ripley_k_function",
    "ripley_l_function",
    "row_standardize_weights",
    "simulate_cox_process",
    "simulate_marked_poisson_process",
    "simulate_poisson_process",
    "spatial_autocorrelation",
    "spatial_dbscan",
    "spatial_weights_distance_band",
    "spatial_weights_inverse_distance",
    "spatial_weights_knn",
    "utils",
]

if kriging is not None:
    __all__.extend(
        [
            "OrdinaryKriging",
            "SimpleKriging",
            "UniversalKriging",
            "kriging",
        ]
    )

if validation is not None:
    __all__.extend(
        [
            "CrossValidationResult",
            "block_cross_validation",
            "compute_kriging_residuals",
            "leave_one_out_cross_validation",
            "normality_test",
            "select_best_variogram_model",
            "spatial_kfold_cross_validation",
            "standardized_residuals",
            "validation",
            "variogram_aic",
            "variogram_bic",
            "variogram_cloud",
        ]
    )

if variogram is not None:
    __all__.extend(
        [
            "AnisotropyResult",
            "DirectionalVariogram",
            "EmpiricalVariogram",
            "Variogram",
            "variogram",
        ]
    )
