"""Model validation and diagnostic tools for PySpatialStats."""

from .cross_validation import (
    CrossValidationResult,
    block_cross_validation,
    leave_one_out_cross_validation,
    spatial_kfold_cross_validation,
    default_variogram_builder,
    default_kriging_builder,
)
from .model_selection import (
    variogram_aic,
    variogram_bic,
    select_best_variogram_model,
)
from .diagnostics import (
    compute_kriging_residuals,
    variogram_cloud,
    standardized_residuals,
    normality_test,
)

__all__ = [
    "CrossValidationResult",
    "block_cross_validation",
    "leave_one_out_cross_validation",
    "spatial_kfold_cross_validation",
    "default_variogram_builder",
    "default_kriging_builder",
    "variogram_aic",
    "variogram_bic",
    "select_best_variogram_model",
    "compute_kriging_residuals",
    "variogram_cloud",
    "standardized_residuals",
    "normality_test",
]
