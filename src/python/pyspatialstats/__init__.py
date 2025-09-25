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

from .variogram import Variogram, EmpiricalVariogram
from .kriging import OrdinaryKriging, SimpleKriging

__all__ = [
    "variogram",
    "kriging", 
    "utils",
    "Variogram",
    "EmpiricalVariogram",
    "OrdinaryKriging",
    "SimpleKriging",
]
