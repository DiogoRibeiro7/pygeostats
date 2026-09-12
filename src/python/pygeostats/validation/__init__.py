"""Model validation and diagnostic tools for PySpatialStats."""

from .cross_validation import (
    CrossValidationResult,
    block_cross_validation,
    default_kriging_builder,
    default_variogram_builder,
    leave_one_out_cross_validation,
    spatial_kfold_cross_validation,
)
from .diagnostics import (
    compute_kriging_residuals,
    normality_test,
    standardized_residuals,
    variogram_cloud,
)
from .model_selection import (
    select_best_variogram_model,
    variogram_aic,
    variogram_bic,
)

__all__ = [
    "CrossValidationResult",
    "block_cross_validation",
    "compute_kriging_residuals",
    "default_kriging_builder",
    "default_variogram_builder",
    "leave_one_out_cross_validation",
    "normality_test",
    "select_best_variogram_model",
    "spatial_kfold_cross_validation",
    "standardized_residuals",
    "variogram_aic",
    "variogram_bic",
    "variogram_cloud",
]
