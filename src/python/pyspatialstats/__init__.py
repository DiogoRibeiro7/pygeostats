# src/python/pyspatialstats/__init__.py
"""
PySpatialStats: High-performance spatial statistics for Python
==============================================================

A comprehensive spatial statistics library with Rust-accelerated core algorithms.
"""

__version__ = "0.1.0"

from . import variogram
from . import kriging
from . import utils
from . import validation

from .variogram import (
    Variogram,
    EmpiricalVariogram,
    DirectionalVariogram,
    AnisotropyResult,
    AnisotropyInitializationCandidate,
    AnisotropyInitializationSummary,
)
from .kriging import OrdinaryKriging, SimpleKriging, UniversalKriging
from .validation import (
    CrossValidationResult,
    block_cross_validation,
    leave_one_out_cross_validation,
    spatial_kfold_cross_validation,
    variogram_aic,
    variogram_bic,
    select_best_variogram_model,
    compute_kriging_residuals,
    variogram_cloud,
    standardized_residuals,
    normality_test,
)

__all__ = [
    "variogram",
    "kriging",
    "utils",
    "validation",
    "Variogram",
    "EmpiricalVariogram",
    "DirectionalVariogram",
    "AnisotropyResult",
    "AnisotropyInitializationCandidate",
    "AnisotropyInitializationSummary",
    "OrdinaryKriging",
    "SimpleKriging",
    "UniversalKriging",
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
