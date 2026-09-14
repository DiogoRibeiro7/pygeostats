# src/python/pygeostats/variogram/geometry_analysis.py
"""Spatial geometry analysis utilities for anisotropy initialization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy.spatial import cKDTree

from .directional import DirectionalResult

__all__ = [
    "GeometryDiagnostics",
    "SpatialGeometryAnalyzer",
]


@dataclass(frozen=True)
class GeometryDiagnostics:
    """Summary of geometric characteristics relevant for anisotropy."""

    primary_angle_deg: float
    secondary_angle_deg: float
    pca_eigenvalues: np.ndarray
    anisotropy_ratio: float
    strength_label: str
    sampling_pattern: str
    coefficient_of_variation: float
    diagnostics: Dict[str, float]


class SpatialGeometryAnalyzer:
    """Analyze spatial geometry to assist anisotropy parameter initialization."""

    def __init__(
        self,
        coordinates: np.ndarray,
        directional_results: Optional[Dict[float, DirectionalResult]] = None,
    ) -> None:
        coords = np.asarray(coordinates, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("Coordinates must be (n_samples, 2)")
        if coords.shape[0] < 3:
            raise ValueError("At least three coordinate samples are required")
        self.coordinates = coords
        self.directional_results = directional_results or {}

    def analyze(self) -> GeometryDiagnostics:
        pca_angle, eigvals = _principal_axes(self.coordinates)
        ranges = _collect_directional_ranges(self.directional_results)

        if ranges.size >= 2:
            range_max = float(ranges.max())
            range_min = float(ranges.min())
        else:
            range_max = range_min = float(eigvals.max())

        ratio = range_max / max(range_min, 1e-12)
        strength_label = _classify_strength(ratio, ranges)
        sampling_pattern = _sampling_pattern(self.coordinates)
        coeff_var = float(np.std(ranges) / np.mean(ranges)) if ranges.size > 1 else 0.0

        combined_angle = pca_angle
        if ranges.size > 1:
            best_dir = _best_direction(self.directional_results)
            if best_dir is not None:
                combined_angle = 0.5 * (combined_angle + best_dir)

        diagnostics = {
            "range_max": range_max,
            "range_min": range_min,
            "num_ranges": float(ranges.size),
            "pca_ratio": float(eigvals.max() / max(eigvals.min(), 1e-12)),
        }

        return GeometryDiagnostics(
            primary_angle_deg=combined_angle % 180.0,
            secondary_angle_deg=(combined_angle + 90.0) % 180.0,
            pca_eigenvalues=eigvals,
            anisotropy_ratio=ratio,
            strength_label=strength_label,
            sampling_pattern=sampling_pattern,
            coefficient_of_variation=coeff_var,
            diagnostics=diagnostics,
        )


def _principal_axes(coords: np.ndarray) -> tuple[float, np.ndarray]:
    centered = coords - coords.mean(axis=0, keepdims=True)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvec = eigvecs[:, order[0]]
    angle = float(np.degrees(np.arctan2(eigvec[1], eigvec[0]))) % 180.0
    return angle, eigvals


def _collect_directional_ranges(results: Dict[float, DirectionalResult]) -> np.ndarray:
    ranges = []
    for res in results.values():
        valid = (res.counts > 0) & np.isfinite(res.gamma)
        if not np.any(valid):
            continue
        gamma = res.gamma[valid]
        distances = res.bin_centers[valid]
        if gamma.size == 0:
            continue
        sill = float(np.nanmax(gamma))
        if sill <= 0.0:
            ranges.append(float(distances[-1]))
            continue
        threshold = sill * 0.9
        idx = np.argmax(gamma >= threshold)
        ranges.append(float(distances[idx]))
    return np.asarray(ranges, dtype=float)


def _classify_strength(ratio: float, ranges: np.ndarray) -> str:
    if ratio >= 3.0:
        return "strong"
    if ratio >= 2.0:
        return "moderate"
    if ratio < 1.5:
        return "isotropic"
    if ranges.size > 1:
        coeff_var = float(np.std(ranges) / np.mean(ranges))
        return "weak" if coeff_var > 0.1 else "isotropic"
    return "weak"


def _sampling_pattern(coords: np.ndarray) -> str:
    """Classify how evenly the sampling locations are spaced.

    Uses the coefficient of variation of each location's distance to its nearest
    neighbour: 0 on any lattice and about 0.52 for uniformly random locations.
    Below 0.15 is regular, which covers a grid jittered by up to about 20% of its
    spacing; below 0.35 is quasi-regular; anything higher is irregular. Repeated
    locations are counted once, and fewer than two distinct ones are insufficient.
    """
    # This used distances between consecutive rows, so the label depended on the
    # order the points were listed in: a perfect 10x10 grid listed row by row came
    # out irregular, with a coefficient of variation of 1.34.
    locations = np.unique(coords, axis=0)
    if locations.shape[0] < 2:
        return "insufficient"
    distances, _ = cKDTree(locations).query(locations, k=2)
    nearest = distances[:, 1]
    cv = float(np.std(nearest) / np.mean(nearest))
    if cv < 0.15:
        return "regular"
    if cv < 0.35:
        return "quasi-regular"
    return "irregular"


def _best_direction(results: Dict[float, DirectionalResult]) -> Optional[float]:
    if not results:
        return None
    candidates = []
    for angle, res in results.items():
        valid = (res.counts > 0) & np.isfinite(res.gamma)
        if not np.any(valid):
            continue
        gamma = res.gamma[valid]
        if gamma.size == 0:
            continue
        ranges = res.bin_centers[valid]
        sill = float(np.nanmax(gamma))
        if sill <= 0.0:
            continue
        threshold = sill * 0.9
        idx = np.argmax(gamma >= threshold)
        candidates.append((float(ranges[idx]), float(angle % 180.0)))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]
