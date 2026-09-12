# src/python/pygeostats/variogram/directional.py
"""Directional variogram analysis and anisotropy detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, NoReturn, Optional, Sequence, Tuple

import numpy as np

from ..utils.validation import validate_coordinates, validate_values


DEFAULT_DIRECTIONS = (0.0, 45.0, 90.0, 135.0)


@dataclass
class DirectionalResult:
    """Container for directional variogram outputs."""

    angle: float
    bin_centers: np.ndarray
    gamma: np.ndarray
    counts: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray


@dataclass
class AnisotropyResult:
    """Summary of detected geometric anisotropy."""

    is_anisotropic: bool
    major_direction: float
    minor_direction: float
    anisotropy_ratio: float
    ranges: Dict[float, float]
    sill: float
    diagnostics: Dict[str, float]


class DirectionalVariogram:
    """Directional empirical variogram computation.

    Parameters
    ----------
    coordinates : array-like, shape (n_samples, 2)
        Sampling locations. Only planar (2D) coordinates are supported.
    values : array-like, shape (n_samples,)
        Sample values.
    directions : sequence of float, optional
        Directions (degrees) for which directional variograms are computed.
        Defaults to ``(0, 45, 90, 135)`` if not provided.
    tolerance : float, default=22.5
        Angular tolerance (degrees) defining the half-window around each
        direction.
    bandwidth : float, optional
        Maximum perpendicular distance (same units as coordinates) allowed when
        classifying pairs within a directional sector. If ``None`` no limit is
        applied.
    max_distance : float, optional
        Maximum separation distance considered. Defaults to half of the maximum
        pairwise distance.
    n_bins : int, default=12
        Number of lag bins.
    bin_edges : array-like, optional
        Custom bin edges. If provided ``n_bins`` is ignored.
    """

    def __init__(
        self,
        coordinates: np.ndarray,
        values: np.ndarray,
        directions: Optional[Sequence[float]] = None,
        tolerance: float = 22.5,
        bandwidth: Optional[float] = None,
        max_distance: Optional[float] = None,
        n_bins: int = 12,
        bin_edges: Optional[np.ndarray] = None,
    ) -> None:
        self.coordinates = validate_coordinates(coordinates)
        self.values = validate_values(values)
        if self.coordinates.shape[1] != 2:
            raise ValueError("Directional variograms currently require 2D coordinates")
        if len(self.coordinates) != len(self.values):
            raise ValueError("coordinates and values must have same length")

        if directions is None:
            self.directions = np.asarray(DEFAULT_DIRECTIONS, dtype=float)
        else:
            arr = np.asarray(directions, dtype=float)
            if arr.size == 0:
                raise ValueError("directions must contain at least one angle")
            self.directions = arr

        self.tolerance = float(tolerance)
        if not (0 < self.tolerance <= 90):
            raise ValueError("tolerance must be in (0, 90]")
        self.bandwidth = bandwidth
        self.max_distance = max_distance
        self.n_bins = int(n_bins)
        if self.n_bins < 1:
            raise ValueError("n_bins must be positive")
        self.bin_edges = None if bin_edges is None else np.asarray(bin_edges, dtype=float)

        self._pairs_cache: Optional[Dict[str, np.ndarray]] = None
        self.bin_edges_: Optional[np.ndarray] = None
        self.directional_results_: Dict[float, DirectionalResult] = {}
        self.max_distance_: Optional[float] = None
        self._is_fitted = False

    @staticmethod
    def automatic_direction_set(coords: np.ndarray, n_directions: int = 4) -> List[float]:
        """Return evenly spaced directions aligned with principal axes."""

        coords = validate_coordinates(coords)
        if coords.shape[1] != 2:
            raise ValueError("automatic_direction_set requires 2D coordinates")
        if n_directions < 1:
            raise ValueError("n_directions must be >= 1")
        centered = coords - coords.mean(axis=0)
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        major_vec = eigvecs[:, np.argmax(eigvals)]
        base_angle = (np.degrees(np.arctan2(major_vec[1], major_vec[0])) + 360.0) % 180.0
        step = 180.0 / n_directions
        return [float((base_angle + k * step) % 180.0) for k in range(n_directions)]

    def _prepare_bins(self) -> None:
        if self.bin_edges is not None:
            self.bin_edges_ = self.bin_edges
            self.max_distance_ = float(self.bin_edges_[-1]) if self.bin_edges_.size else None
            return
        if self.max_distance is None:
            distances = np.linalg.norm(
                self.coordinates[:, None, :] - self.coordinates[None, :, :], axis=-1
            )
            max_dist = np.max(distances) / 2.0
        else:
            max_dist = float(self.max_distance)
        self.bin_edges_ = np.linspace(0.0, max_dist, self.n_bins + 1)
        self.max_distance_ = float(self.bin_edges_[-1])

    def _prepare_pairs(self) -> None:
        if self._pairs_cache is not None:
            return
        coords = self.coordinates
        values = self.values
        n = coords.shape[0]
        iu, ju = np.triu_indices(n, k=1)
        vectors = coords[ju] - coords[iu]
        distances = np.linalg.norm(vectors, axis=1)
        mask = distances > 0
        vectors = vectors[mask]
        distances = distances[mask]
        semivariances = 0.5 * (values[ju][mask] - values[iu][mask]) ** 2
        self._pairs_cache = {
            "vectors": vectors,
            "distances": distances,
            "semivariances": semivariances,
        }

    def compute(self) -> "DirectionalVariogram":
        """Compute directional variograms for the configured directions."""

        self._prepare_bins()
        self._prepare_pairs()
        assert self.bin_edges_ is not None
        bin_centers = 0.5 * (self.bin_edges_[:-1] + self.bin_edges_[1:])
        cos_tolerance = np.cos(np.deg2rad(self.tolerance))

        vectors = self._pairs_cache["vectors"]
        distances = self._pairs_cache["distances"]
        semivariances = self._pairs_cache["semivariances"]

        for angle in self.directions:
            rad = np.deg2rad(angle)
            direction = np.array([np.cos(rad), np.sin(rad)])
            projections = np.dot(vectors, direction)
            with np.errstate(invalid="ignore", divide="ignore"):
                cos_angles = np.abs(projections / distances)
            mask = cos_angles >= cos_tolerance
            if self.bandwidth is not None:
                perp_sq = np.maximum(distances**2 - projections**2, 0.0)
                mask &= np.sqrt(perp_sq) <= self.bandwidth
            mask &= np.isfinite(distances)

            sel_distances = distances[mask]
            sel_semivar = semivariances[mask]
            result = self._reduce_direction(float(angle), sel_distances, sel_semivar, bin_centers)
            self.directional_results_[float(angle)] = result

        self._is_fitted = True
        return self

    def _reduce_direction(
        self,
        angle: float,
        distances: np.ndarray,
        semivariances: np.ndarray,
        bin_centers: np.ndarray,
    ) -> DirectionalResult:
        n_bins = len(bin_centers)
        counts = np.zeros(n_bins, dtype=int)
        mean = np.full(n_bins, np.nan)
        ci_lower = np.full(n_bins, np.nan)
        ci_upper = np.full(n_bins, np.nan)
        if len(distances) == 0:
            return DirectionalResult(angle, bin_centers, mean, counts, ci_lower, ci_upper)

        indices = np.digitize(distances, self.bin_edges_) - 1
        valid = (indices >= 0) & (indices < n_bins)
        indices = indices[valid]
        if len(indices) == 0:
            return DirectionalResult(angle, bin_centers, mean, counts, ci_lower, ci_upper)

        counts = np.bincount(indices, minlength=n_bins).astype(int)
        sums = np.bincount(indices, weights=semivariances[valid], minlength=n_bins)
        sums_sq = np.bincount(indices, weights=(semivariances[valid] ** 2), minlength=n_bins)

        with np.errstate(invalid="ignore"):
            mean = sums / counts
            variance = sums_sq / counts - mean**2
            variance = np.where(variance < 0, 0, variance)
            se = np.sqrt(variance / counts)
            ci_lower = mean - 1.96 * se
            ci_upper = mean + 1.96 * se

        zero_mask = counts == 0
        mean[zero_mask] = np.nan
        ci_lower[zero_mask] = np.nan
        ci_upper[zero_mask] = np.nan

        return DirectionalResult(angle, bin_centers, mean, counts, ci_lower, ci_upper)

    @property
    def is_fitted(self) -> bool:
        """Whether ``compute`` has been executed."""

        return self._is_fitted

    def directional_summary(self) -> Dict[float, Dict[str, np.ndarray]]:
        """Return a dictionary summarising directional variograms."""

        if not self.is_fitted:
            raise RuntimeError("compute() must be called first")
        summary: Dict[float, Dict[str, np.ndarray]] = {}
        for angle, res in self.directional_results_.items():
            summary[angle] = {
                "bin_centers": res.bin_centers,
                "gamma": res.gamma,
                "counts": res.counts,
                "ci_lower": res.ci_lower,
                "ci_upper": res.ci_upper,
            }
        return summary

    def detect_anisotropy(
        self,
        sill_fraction: float = 0.95,
        ratio_threshold: float = 1.2,
        range_difference: float = 0.0,
    ) -> AnisotropyResult:
        """Detect geometric anisotropy from directional variograms."""

        if not self.is_fitted:
            raise RuntimeError("compute() must be called before anisotropy detection")

        gamma_maxes: List[float] = []
        for res in self.directional_results_.values():
            valid = res.counts > 0
            data = res.gamma[valid]
            if data.size > 0 and not np.all(np.isnan(data)):
                gamma_maxes.append(float(np.nanmax(data)))
        if not gamma_maxes:
            diagnostics = {"max_range": np.nan, "min_range": np.nan}
            return AnisotropyResult(False, np.nan, np.nan, 1.0, {}, np.nan, diagnostics)
        sill = float(np.nanmax(gamma_maxes))
        threshold = sill * sill_fraction

        ranges: Dict[float, float] = {}
        for angle, res in self.directional_results_.items():
            valid = res.counts > 0
            distances = res.bin_centers[valid]
            gamma = res.gamma[valid]
            if distances.size == 0 or np.all(np.isnan(gamma)):
                ranges[angle] = np.nan
                continue
            if np.all(gamma < threshold):
                ranges[angle] = float(distances[-1])
            else:
                idx = np.argmax(gamma >= threshold)
                ranges[angle] = float(distances[idx])

        finite_ranges = {k: v for k, v in ranges.items() if np.isfinite(v)}
        if len(finite_ranges) < 2:
            diagnostics = {"max_range": np.nan, "min_range": np.nan}
            return AnisotropyResult(False, np.nan, np.nan, 1.0, ranges, sill, diagnostics)

        major_direction = max(finite_ranges, key=finite_ranges.get)
        minor_direction = min(finite_ranges, key=finite_ranges.get)
        max_range = finite_ranges[major_direction]
        min_range = finite_ranges[minor_direction]
        ratio = max_range / max(min_range, 1e-9)
        is_aniso = (ratio >= ratio_threshold) and (
            (max_range - min_range) >= range_difference
        )
        diagnostics = {
            "max_range": max_range,
            "min_range": min_range,
            "range_difference": max_range - min_range,
        }
        return AnisotropyResult(
            is_aniso,
            float(major_direction),
            float(minor_direction),
            float(ratio),
            ranges,
            sill,
            diagnostics,
        )

    def estimate_initial_parameters(
        self,
        strategies: Optional[Sequence[str]] = None,
        min_weight: int = 5,
        sill_fraction: float = 0.95,
    ) -> "NoReturn":
        """Generate anisotropic variogram initialisation candidates.

        Not implemented. This method was committed in feca441 calling
        ``estimate_anisotropy_initialization``, and returning an
        ``AnisotropyInitializationSummary``, neither of which was ever
        written. Use :class:`~.initialization.InitializationEnsemble` or
        :class:`~.initialization.RangeInitializer` directly instead; both
        are implemented and cover the same ground.
        """

        raise NotImplementedError(
            "estimate_initial_parameters() is not implemented: the "
            "estimate_anisotropy_initialization() helper it calls has never "
            "existed. Use InitializationEnsemble or RangeInitializer from "
            "pygeostats.variogram.initialization instead."
        )

    def anisotropy_rose_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return angles and ranges suitable for rose-diagram plotting."""

        if not self.is_fitted:
            raise RuntimeError("compute() must be called first")
        result = self.detect_anisotropy()
        angles = np.array(sorted(result.ranges.keys()))
        ranges = np.array([result.ranges[a] for a in angles])
        return angles, ranges
