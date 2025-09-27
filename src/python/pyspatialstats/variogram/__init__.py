# src/python/pyspatialstats/variogram/__init__.py
"""Variogram modeling and analysis."""

from .empirical import EmpiricalVariogram
from .models import Variogram
from .directional import DirectionalVariogram, AnisotropyResult
from .initialization import (
    AnisotropyInitializationCandidate,
    AnisotropyInitializationSummary,
)
from .streaming import (
    StreamingVariogramBuilder,
    StreamingVariogramDenseResult,
    StreamingVariogramSparseResult,
    streaming_variogram,
    streaming_variogram_memmap,
    memory_map_array,
    chunk_indices,
)

__all__ = [
    "EmpiricalVariogram",
    "Variogram",
    "DirectionalVariogram",
    "AnisotropyResult",
    "AnisotropyInitializationCandidate",
    "AnisotropyInitializationSummary",
    "StreamingVariogramBuilder",
    "StreamingVariogramDenseResult",
    "StreamingVariogramSparseResult",
    "streaming_variogram",
    "streaming_variogram_memmap",
    "memory_map_array",
    "chunk_indices",
]
