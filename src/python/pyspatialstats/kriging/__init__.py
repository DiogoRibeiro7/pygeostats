# src/python/pyspatialstats/kriging/__init__.py
"""Kriging interpolation methods."""

from .ordinary import OrdinaryKriging
from .simple import SimpleKriging

__all__ = ["OrdinaryKriging", "SimpleKriging"]
