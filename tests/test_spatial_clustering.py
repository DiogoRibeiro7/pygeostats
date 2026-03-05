"""Tests for Phase 3.2 spatial clustering tools."""

import numpy as np

from pyspatialstats.point_patterns import (
    cluster_validation_metrics,
    getis_ord_gi_star,
    kernel_density_estimate,
    spatial_dbscan,
)


def _two_cluster_points(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cluster_a = rng.normal(loc=[0.0, 0.0], scale=0.08, size=(60, 2))
    cluster_b = rng.normal(loc=[1.0, 1.0], scale=0.08, size=(60, 2))
    return np.vstack([cluster_a, cluster_b])


def test_spatial_dbscan_finds_multiple_clusters() -> None:
    coords = _two_cluster_points()
    labels = spatial_dbscan(coords, eps=0.18, min_samples=5)
    cluster_ids = np.unique(labels[labels != -1])
    assert len(cluster_ids) >= 2


def test_getis_ord_gi_star_detects_hotspot_center() -> None:
    coords = _two_cluster_points()
    hotspot_values = np.concatenate(
        [
            np.full(60, 10.0),  # hot cluster
            np.full(60, 1.0),   # cool cluster
        ]
    )
    gi = getis_ord_gi_star(coords, hotspot_values, distance_threshold=0.2)
    assert gi.shape == (len(coords),)
    assert np.isfinite(gi).all()
    assert np.mean(gi[:60]) > np.mean(gi[60:])


def test_kernel_density_estimate_returns_grid() -> None:
    coords = _two_cluster_points()
    kde = kernel_density_estimate(coords, grid_size=40, bounds=(-0.5, 1.5, -0.5, 1.5))
    assert set(kde.keys()) == {"x", "y", "density"}
    assert kde["x"].shape == (40, 40)
    assert kde["y"].shape == (40, 40)
    assert kde["density"].shape == (40, 40)
    assert np.all(kde["density"] >= 0.0)


def test_cluster_validation_metrics_returns_scores() -> None:
    coords = _two_cluster_points()
    labels = spatial_dbscan(coords, eps=0.2, min_samples=5)
    metrics = cluster_validation_metrics(coords, labels)
    assert set(metrics.keys()) == {
        "n_clusters",
        "silhouette",
        "calinski_harabasz",
        "davies_bouldin",
    }
    assert metrics["n_clusters"] >= 2
    assert np.isfinite(metrics["silhouette"])
    assert np.isfinite(metrics["calinski_harabasz"])
    assert np.isfinite(metrics["davies_bouldin"])
