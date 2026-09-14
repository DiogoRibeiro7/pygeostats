# src/python/pygeostats/variogram/initialization.py
"""Initialization utilities for anisotropic variogram parameter estimation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .directional import DirectionalResult
from .geometry_analysis import SpatialGeometryAnalyzer

__all__ = [
    "AngleEstimationResult",
    "EnsembleResult",
    "InitializationEnsemble",
    "RangeInitializationResult",
    "RangeInitializer",
    "estimate_rotation_angle",
]

_EPS = 1e-12
_PI = np.pi


@dataclass(frozen=True)
class AngleEstimationResult:
    """Simple container for anisotropy angle estimation outputs."""

    angle_deg: float
    angle_confidence: Tuple[float, float]
    ratio: float
    confidence: str
    significant: bool
    diagnostics: Dict[str, float]


@dataclass(frozen=True)
class RangeInitializationResult:
    """Container for range, sill, and nugget initial estimates."""

    range_major: float
    range_minor: float
    ratio: float
    range_major_se: float
    range_minor_se: float
    range_major_bounds: Tuple[float, float]
    range_minor_bounds: Tuple[float, float]
    sill: float
    nugget: float
    quality_score: float
    diagnostics: Dict[str, float]


@dataclass(frozen=True)
class EnsembleResult:
    """Combined initialization result delivered by the ensemble."""

    angle_deg: float
    angle_confidence: Tuple[float, float]
    ratio: float
    confidence: str
    range_major: float
    range_minor: float
    sill: float
    nugget: float
    top_candidates: List[Tuple[float, float]]
    quality: float
    diagnostics: Dict[str, float]


class RangeInitializer:
    """Derive range, sill, and nugget initial values from directional variograms."""

    REQUIRED_DIRECTIONS: Tuple[float, ...] = (0.0, 30.0, 60.0, 90.0, 120.0, 150.0)

    def __init__(
        self,
        directional_results: Dict[float, DirectionalResult],
    ) -> None:
        self.directional_results = directional_results

    def estimate(self) -> RangeInitializationResult:
        ranges: List[float] = []
        sill_samples: List[float] = []
        nugget_candidates: List[float] = []
        angles_seen: List[float] = []
        missing = [
            angle
            for angle in self.REQUIRED_DIRECTIONS
            if angle not in self.directional_results
        ]

        for angle in sorted(self.directional_results.keys()):
            result = self.directional_results[angle]
            rng = _extract_range(result)
            if np.isfinite(rng):
                ranges.append(float(rng))
                angles_seen.append(float(angle % 180.0))
            sill_samples.extend(_tail_samples(result))
            nugget_candidates.extend(_nugget_samples(result))

        if not ranges:
            raise ValueError("No valid directional ranges supplied")

        range_values = np.asarray(ranges, dtype=float)
        major = float(np.max(range_values))
        minor = float(np.min(range_values))
        ratio = major / max(minor, _EPS)

        se = (
            float(np.std(range_values, ddof=1) / np.sqrt(len(range_values)))
            if len(range_values) > 1
            else 0.0
        )
        major_bounds = (major - se, major + se)
        minor_bounds = (minor - se, minor + se)

        sill_array = np.asarray(
            [v for v in sill_samples if np.isfinite(v) and v > 0.0], dtype=float
        )
        if sill_array.size:
            sill_array.sort()
            take = max(int(np.ceil(sill_array.size * 0.3)), 1)
            sill = float(np.mean(sill_array[-take:]))
        else:
            sill = major

        nugget_array = np.asarray(
            [v for v in nugget_candidates if np.isfinite(v) and v >= 0.0], dtype=float
        )
        nugget = float(nugget_array.min()) if nugget_array.size else 0.0
        if nugget > sill:
            nugget = float(max(sill * 0.99, 0.0))

        quality_score = ratio / (1.0 + max(se, 1e-12))
        diagnostics = {
            "sample_size": float(len(range_values)),
            "angle_span": (
                float(np.max(angles_seen) - np.min(angles_seen)) if angles_seen else 0.0
            ),
            "range_std": (
                float(np.std(range_values, ddof=1)) if len(range_values) > 1 else 0.0
            ),
            "se": se,
            "missing_directions": float(len(missing)),
        }

        return RangeInitializationResult(
            range_major=major,
            range_minor=minor,
            ratio=ratio,
            range_major_se=se,
            range_minor_se=se,
            range_major_bounds=major_bounds,
            range_minor_bounds=minor_bounds,
            sill=sill,
            nugget=nugget,
            quality_score=quality_score,
            diagnostics=diagnostics,
        )


class InitializationEnsemble:
    """Combine geometric and directional methods into a robust initialization ensemble."""

    def __init__(
        self,
        coordinates: np.ndarray,
        directional_results: Dict[float, DirectionalResult],
    ) -> None:
        coords = np.asarray(coordinates, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("coordinates must be of shape (n_samples, 2)")
        if coords.shape[0] < 3:
            raise ValueError("At least three coordinate samples are required")
        self.coordinates = coords
        self.directional_results = directional_results
        self.range_initializer = RangeInitializer(directional_results)

    def run(self) -> EnsembleResult:
        range_init = self.range_initializer.estimate()

        angles = np.array(sorted(self.directional_results.keys()), dtype=float)
        ranges = _collect_directional_ranges(self.directional_results)
        if ranges.size < 2:
            ranges = np.asarray(
                [range_init.range_major, range_init.range_minor], dtype=float
            )
            angles = np.asarray([0.0, 90.0], dtype=float)

        weights = np.ones_like(ranges)
        angle_result = estimate_rotation_angle(angles, ranges, weights)
        geom = SpatialGeometryAnalyzer(
            self.coordinates, self.directional_results
        ).analyze()

        method_angles = [
            (angle_result.angle_deg, _confidence_weight(angle_result.confidence)),
            (geom.primary_angle_deg, _strength_weight(geom.strength_label)),
            (geom.primary_angle_deg, max(0.5, geom.diagnostics.get("pca_ratio", 1.0))),
        ]

        combined_angle_rad = _weighted_circular_mean(method_angles)
        combined_angle = (np.degrees(combined_angle_rad) + 180.0) % 180.0

        vote_angles = [
            int(round(angle / 15.0)) * 15 % 180 for angle, _ in method_angles
        ]
        vote_counts: Dict[int, int] = {}
        for candidate in vote_angles:
            vote_counts[candidate] = vote_counts.get(candidate, 0) + 1
        voted_angle = max(
            vote_counts.items(),
            key=lambda item: (item[1], -abs(item[0] - combined_angle)),
        )[0]
        ensemble_angle = (0.6 * combined_angle + 0.4 * voted_angle) % 180.0

        grid_angles = np.arange(0.0, 180.0, 15.0)
        grid_records: List[Tuple[float, float]] = []
        angles_rad = np.deg2rad(angles % 180.0)
        for angle_deg in grid_angles:
            fit = _fit_ellipse_for_phi(
                np.deg2rad(angle_deg), angles_rad, ranges, np.ones_like(ranges)
            )
            if fit is None:
                continue
            grid_records.append((float(angle_deg), float(fit["sse"])))
        grid_records.sort(key=lambda item: item[1])
        top_candidates = (
            grid_records[:3] if grid_records else [(float(ensemble_angle), 0.0)]
        )

        quality = float(
            0.4 * _quality_from_confidence(angle_result.confidence)
            + 0.3 * _quality_from_strength(geom.strength_label)
            + 0.3 * range_init.quality_score
        )

        diagnostics = {
            "angle_confidence_low": angle_result.angle_confidence[0],
            "angle_confidence_high": angle_result.angle_confidence[1],
            "geom_ratio": geom.anisotropy_ratio,
            "range_quality": range_init.quality_score,
            "votes": float(len(vote_angles)),
        }

        return EnsembleResult(
            angle_deg=float(ensemble_angle),
            angle_confidence=angle_result.angle_confidence,
            ratio=range_init.ratio,
            confidence=angle_result.confidence,
            range_major=range_init.range_major,
            range_minor=range_init.range_minor,
            sill=range_init.sill,
            nugget=range_init.nugget,
            top_candidates=top_candidates,
            quality=quality,
            diagnostics=diagnostics,
        )


def estimate_rotation_angle(
    directions_deg: Sequence[float],
    ranges: Sequence[float],
    weights: Optional[Sequence[float]] = None,
    phi_step_deg: float = 0.5,
) -> AngleEstimationResult:
    """Estimate the primary anisotropy angle from directional variogram ranges.

    An ellipse is fitted through the ranges. ``angle_deg`` is the angle of its major
    axis, in degrees counter-clockwise from the x-axis in [0, 180), and ``ratio`` is
    its major range over its minor one, so never below 1. ``angle_confidence`` is
    ``(low, high)`` in the same range; when the interval crosses 0 degrees, ``low``
    is greater than ``high``.
    """

    directions = np.asarray(list(directions_deg), dtype=float)
    ranges = np.asarray(list(ranges), dtype=float)
    assert (
        directions.size == ranges.size
    ), "directions and ranges must share the same length"
    assert directions.size >= 3, "At least three directional samples required"
    assert np.all(np.isfinite(ranges)), "Directional ranges must be finite"

    weights_arr = (
        np.asarray(list(weights), dtype=float)
        if weights is not None
        else np.ones_like(ranges, dtype=float)
    )
    assert weights_arr.size == ranges.size, "Weights must match number of directions"

    angles_rad = np.deg2rad(directions % 180.0)
    ranges = np.clip(ranges, _EPS, None)
    weights_arr = np.clip(weights_arr, _EPS, None)

    phi_step = max(phi_step_deg, 0.1)
    grid = np.deg2rad(np.arange(0.0, 180.0 + phi_step, phi_step))

    best: Optional[Dict[str, float]] = None
    records: List[Dict[str, float]] = []
    weight_sum = float(weights_arr.sum())

    for phi in grid:
        fit = _fit_ellipse_for_phi(phi, angles_rad, ranges, weights_arr)
        if fit is None:
            continue
        records.append(fit)
        if best is None or fit["sse"] < best["sse"]:
            best = fit

    if not records or best is None:
        angle = float(directions.mean() % 180.0)
        return AngleEstimationResult(
            angle_deg=angle,
            angle_confidence=(max(angle - 10.0, 0.0), min(angle + 10.0, 180.0)),
            ratio=1.0,
            confidence="low",
            significant=False,
            diagnostics={"fallback": 1.0},
        )

    angle_deg = float(np.degrees(best["phi"]) % 180.0)
    threshold = best["sse"] * 1.05 + 1e-12
    # Offsets of every near-best fit from the estimate, taken axially so they lie in
    # [-90, 90). The minimum and maximum of the raw angles ignored the wrap at 0 and
    # 180 degrees.
    offsets = [
        (float(np.degrees(rec["phi"])) - angle_deg + 90.0) % 180.0 - 90.0
        for rec in records
        if rec["sse"] <= threshold
    ]
    if len(offsets) >= 2:
        conf_interval = (
            (angle_deg + min(offsets)) % 180.0,
            (angle_deg + max(offsets)) % 180.0,
        )
    else:
        conf_interval = ((angle_deg - 10.0) % 180.0, (angle_deg + 10.0) % 180.0)
    ratio = float(best["ratio"])

    max_range = float(ranges.max())
    min_range = float(ranges.min())
    delta = max_range - min_range
    pooled_std = float(np.std(ranges, ddof=1)) if ranges.size > 1 else 0.0
    z_score = delta / (pooled_std + _EPS)
    significant = bool(delta > 0.0 and z_score > 1.5)

    if ratio > 1.5 and significant:
        confidence = "high"
    elif ratio > 1.2:
        confidence = "medium" if significant else "low"
    else:
        confidence = "low"

    diagnostics = {
        "major_range": best["major"],
        "minor_range": best["minor"],
        "rmse": best["rmse"],
        "sse": best["sse"],
        "weight_sum": weight_sum,
        "z_score": z_score,
        "max_range": max_range,
        "min_range": min_range,
    }

    return AngleEstimationResult(
        angle_deg=angle_deg,
        angle_confidence=conf_interval,
        ratio=ratio,
        confidence=confidence,
        significant=significant,
        diagnostics=diagnostics,
    )


def _confidence_weight(level: str) -> float:
    return {"high": 2.0, "medium": 1.2, "low": 0.7}.get(level.lower(), 0.5)


def _strength_weight(label: str) -> float:
    return {"strong": 2.0, "moderate": 1.3, "weak": 0.8, "isotropic": 0.4}.get(
        label.lower(), 0.6
    )


def _quality_from_confidence(level: str) -> float:
    return {"high": 1.0, "medium": 0.7, "low": 0.4}.get(level.lower(), 0.5)


def _quality_from_strength(label: str) -> float:
    return {"strong": 1.0, "moderate": 0.8, "weak": 0.6, "isotropic": 0.3}.get(
        label.lower(), 0.5
    )


def _weighted_circular_mean(angle_weights: Iterable[Tuple[float, float]]) -> float:
    sin_sum = 0.0
    cos_sum = 0.0
    for angle_deg, weight in angle_weights:
        angle_rad = np.deg2rad(angle_deg)
        sin_sum += weight * np.sin(2.0 * angle_rad)
        cos_sum += weight * np.cos(2.0 * angle_rad)
    return (
        0.5 * np.arctan2(sin_sum, cos_sum)
        if (sin_sum != 0.0 or cos_sum != 0.0)
        else 0.0
    )


def _collect_directional_ranges(results: Dict[float, DirectionalResult]) -> np.ndarray:
    ranges = []
    for res in results.values():
        rng = _extract_range(res)
        if np.isfinite(rng):
            ranges.append(float(rng))
    return np.asarray(ranges, dtype=float)


def _fit_ellipse_for_phi(
    phi: float,
    angles_rad: np.ndarray,
    ranges: np.ndarray,
    weights: np.ndarray,
) -> Optional[Dict[str, float]]:
    cos_t = np.cos(angles_rad - phi)
    sin_t = np.sin(angles_rad - phi)
    x1 = cos_t * cos_t
    x2 = sin_t * sin_t
    y = 1.0 / np.clip(ranges, _EPS, None) ** 2

    w = np.clip(weights, _EPS, None)
    w_sum = float(np.sum(w))
    if not np.isfinite(w_sum) or w_sum <= _EPS:
        return None

    xtwx00 = float(np.sum(w * x1 * x1))
    xtwx01 = float(np.sum(w * x1 * x2))
    xtwx11 = float(np.sum(w * x2 * x2))
    xtwy0 = float(np.sum(w * x1 * y))
    xtwy1 = float(np.sum(w * x2 * y))

    det = xtwx00 * xtwx11 - xtwx01 * xtwx01
    if abs(det) < 1e-12:
        return None

    alpha = (xtwy0 * xtwx11 - xtwx01 * xtwy1) / det
    beta = (xtwx00 * xtwy1 - xtwx01 * xtwy0) / det
    if alpha <= 0.0 or beta <= 0.0 or not np.isfinite(alpha) or not np.isfinite(beta):
        return None

    predicted_inv_sq = alpha * x1 + beta * x2
    if np.any(predicted_inv_sq <= _EPS):
        return None

    predicted_ranges = 1.0 / np.sqrt(predicted_inv_sq)
    residuals = predicted_ranges - ranges
    sse = float(np.sum(w * residuals**2))
    rmse = float(np.sqrt(sse / w_sum))

    # alpha belongs to the axis at phi and beta to the one 90 degrees round. The fit
    # at phi + 90 degrees is the same ellipse with the roles swapped and the same
    # error, so the grid search reported whichever it met first: from exact ranges
    # the minor axis came back as the major one in 3 to 5 of 8 orientations, with
    # a ratio below 1. Report the longer axis.
    range_along = 1.0 / np.sqrt(alpha)
    range_across = 1.0 / np.sqrt(beta)
    if range_along >= range_across:
        major_phi, major, minor = phi, range_along, range_across
    else:
        major_phi, major, minor = (phi + 0.5 * _PI) % _PI, range_across, range_along

    return {
        "phi": float(major_phi),
        "major": float(max(major, _EPS)),
        "minor": float(max(minor, _EPS)),
        "ratio": float(major / max(minor, _EPS)),
        "rmse": rmse,
        "sse": sse,
    }


def _extract_range(result: DirectionalResult) -> float:
    valid = (result.counts > 0) & np.isfinite(result.gamma)
    if not np.any(valid):
        return np.nan
    gamma = result.gamma[valid]
    distances = result.bin_centers[valid]
    if gamma.size == 0:
        return np.nan
    sill = float(np.nanmax(gamma))
    if sill <= 0.0:
        return float(distances[-1])
    threshold = sill * 0.9
    idx = np.argmax(gamma >= threshold)
    return float(distances[idx])


def _tail_samples(result: DirectionalResult) -> List[float]:
    valid = (result.counts > 0) & np.isfinite(result.gamma)
    if not np.any(valid):
        return []
    gamma = result.gamma[valid]
    take = max(int(np.ceil(gamma.size * 0.3)), 1)
    return [float(v) for v in gamma[-take:]]


def _nugget_samples(result: DirectionalResult) -> List[float]:
    valid = (result.counts > 0) & np.isfinite(result.gamma)
    if not np.any(valid):
        return []
    return [float(result.gamma[valid][0])]
