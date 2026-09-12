# src/python/pygeostats/utils/plotting.py
"""Visualization utilities for variogram and kriging diagnostics."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import stats

try:
    import matplotlib.pyplot as plt
    from matplotlib import cm

    _HAS_MPL = True
except ImportError:  # pragma: no cover - optional dependency
    plt = None
    cm = None
    _HAS_MPL = False

try:
    import plotly.graph_objects as go

    _HAS_PLOTLY = True
except ImportError:  # pragma: no cover - optional dependency
    go = None
    _HAS_PLOTLY = False

__all__ = [
    "plot_anisotropy_rose",
    "plot_directional_variograms",
    "plot_kriging_cross_section",
    "plot_kriging_results",
    "plot_kriging_uncertainty",
    "plot_prediction_comparison",
    "plot_residuals_qq",
    "plot_spatial_correlation",
    "plot_variogram",
    "plot_variogram_cloud",
    "plot_variogram_rose",
]


def _ensure_backend(backend: str) -> str:
    backend = backend.lower()
    if backend == "matplotlib" and not _HAS_MPL:
        raise RuntimeError(
            "matplotlib is not installed; install it to use this backend"
        )
    if backend == "plotly" and not _HAS_PLOTLY:
        raise RuntimeError("plotly is not installed; install it to use this backend")
    if backend not in {"matplotlib", "plotly"}:
        raise ValueError("backend must be either 'matplotlib' or 'plotly'")
    return backend


def _export_figure(fig, backend: str, save_path: Optional[str]) -> None:
    if not save_path:
        return
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if backend == "matplotlib":
        fig.savefig(path, bbox_inches="tight")
    else:
        try:
            fig.write_image(str(path))
        except ValueError as exc:  # pragma: no cover - kaleido missing
            raise RuntimeError(
                "plotly static export requires the 'kaleido' package"
            ) from exc


def plot_variogram(
    distances: Sequence[float],
    gamma: Sequence[float],
    counts: Optional[Sequence[int]] = None,
    model_curves: Optional[
        Iterable[Tuple[str, Sequence[float], Sequence[float]]]
    ] = None,
    confidence_interval: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
    backend: str = "matplotlib",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """Plot an empirical variogram with optional model overlays and uncertainty."""

    backend = _ensure_backend(backend)
    distances = np.asarray(distances)
    gamma = np.asarray(gamma)
    mask = np.isfinite(distances) & np.isfinite(gamma)
    distances = distances[mask]
    gamma = gamma[mask]

    order = np.argsort(distances)
    distances = distances[order]
    gamma = gamma[order]
    weights = np.asarray(counts)[mask][order] if counts is not None else None

    if confidence_interval is not None:
        lower = np.asarray(confidence_interval[0])[mask][order]
        upper = np.asarray(confidence_interval[1])[mask][order]
    else:
        lower = upper = None

    if backend == "matplotlib":
        fig, ax = plt.subplots(figsize=(8, 5))
        size = None
        if weights is not None and weights.size > 0:
            size = 40 * (weights / weights.max()) + 20
        ax.scatter(distances, gamma, s=size, color="tab:blue", label="Empirical")
        if lower is not None and upper is not None:
            ax.fill_between(
                distances, lower, upper, color="tab:blue", alpha=0.2, label="CI"
            )
        if model_curves:
            for label, model_dist, model_gamma in model_curves:
                ax.plot(model_dist, model_gamma, linewidth=2, label=label)
        ax.set_xlabel("Distance")
        ax.set_ylabel("Semivariance")
        ax.set_title(title or "Empirical Variogram")
        ax.grid(True, alpha=0.3)
        ax.legend()
        _export_figure(fig, backend, save_path)
        if show:
            plt.show(block=False)
        return fig

    fig = go.Figure()
    hovertext = None
    if weights is not None:
        hovertext = [
            f"Distance: {d:.4f}<br>Gamma: {g:.4f}<br>Pairs: {int(w)}"
            for d, g, w in zip(distances, gamma, weights)
        ]
    fig.add_trace(
        go.Scatter(
            x=distances,
            y=gamma,
            mode="markers",
            marker=dict(size=10, color="#1f77b4"),
            hovertext=hovertext,
            name="Empirical",
        )
    )
    if lower is not None and upper is not None:
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([distances, distances[::-1]]),
                y=np.concatenate([upper, lower[::-1]]),
                fill="toself",
                fillcolor="rgba(31, 119, 180, 0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                name="Confidence interval",
            )
        )
    if model_curves:
        for label, model_dist, model_gamma in model_curves:
            fig.add_trace(
                go.Scatter(x=model_dist, y=model_gamma, mode="lines", name=label)
            )
    fig.update_layout(
        title=title or "Empirical Variogram",
        xaxis_title="Distance",
        yaxis_title="Semivariance",
        template="plotly_white",
    )
    _export_figure(fig, backend, save_path)
    if show:
        fig.show()
    return fig


def plot_directional_variograms(
    directional_results: (
        Dict[float, Dict[str, np.ndarray]] | Iterable[Dict[str, np.ndarray]]
    ),
    backend: str = "matplotlib",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """Plot multiple directional variograms for anisotropy assessment."""

    backend = _ensure_backend(backend)
    if isinstance(directional_results, dict):
        items = directional_results.items()
    else:
        items = [
            (
                (res["angle"], res)
                if isinstance(res, dict) and "angle" in res
                else (
                    res.angle,
                    {
                        "bin_centers": res.bin_centers,
                        "gamma": res.gamma,
                        "counts": res.counts,
                        "ci_lower": res.ci_lower,
                        "ci_upper": res.ci_upper,
                    },
                )
            )
            for res in directional_results
        ]
    curves = []
    for angle, payload in items:
        centers = np.asarray(payload["bin_centers"])
        gamma = np.asarray(payload["gamma"])
        counts = np.asarray(payload.get("counts", np.ones_like(centers)))
        valid = (counts > 0) & np.isfinite(centers) & np.isfinite(gamma)
        if np.any(valid):
            curves.append(
                (
                    float(angle),
                    centers[valid],
                    gamma[valid],
                    np.asarray(payload.get("ci_lower", np.full_like(centers, np.nan)))[
                        valid
                    ],
                    np.asarray(payload.get("ci_upper", np.full_like(centers, np.nan)))[
                        valid
                    ],
                )
            )
    curves.sort(key=lambda item: item[0])

    if backend == "matplotlib":
        fig, ax = plt.subplots(figsize=(8, 5))
        for angle, centers, gamma, lower, upper in curves:
            label = f"{angle:.0f} deg"
            ax.plot(centers, gamma, label=label)
            if np.any(np.isfinite(lower)) and np.any(np.isfinite(upper)):
                ax.fill_between(centers, lower, upper, alpha=0.15)
        ax.set_xlabel("Distance")
        ax.set_ylabel("Semivariance")
        ax.set_title(title or "Directional variograms")
        ax.grid(True, alpha=0.3)
        ax.legend(title="Direction")
        _export_figure(fig, backend, save_path)
        if show:
            plt.show(block=False)
        return fig

    fig = go.Figure()
    for angle, centers, gamma, lower, upper in curves:
        label = f"{angle:.0f} deg"
        fig.add_trace(go.Scatter(x=centers, y=gamma, mode="lines+markers", name=label))
        if np.any(np.isfinite(lower)) and np.any(np.isfinite(upper)):
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([centers, centers[::-1]]),
                    y=np.concatenate([upper, lower[::-1]]),
                    fill="toself",
                    fillcolor="rgba(0,0,0,0.08)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name=f"{label} CI",
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
    fig.update_layout(
        title=title or "Directional variograms",
        xaxis_title="Distance",
        yaxis_title="Semivariance",
        template="plotly_white",
    )
    _export_figure(fig, backend, save_path)
    if show:
        fig.show()
    return fig


def plot_variogram_rose(
    azimuths: Sequence[float],
    values: Sequence[float],
    backend: str = "matplotlib",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """Plot a directional variogram rose diagram."""

    backend = _ensure_backend(backend)
    azimuths = np.asarray(azimuths)
    values = np.asarray(values)

    if backend == "matplotlib":
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="polar")
        width = np.deg2rad(360.0 / len(azimuths)) if len(azimuths) > 0 else 0.1
        ax.bar(np.deg2rad(azimuths), values, width=width, color="tab:blue", alpha=0.6)
        ax.set_title(title or "Directional Variogram Rose")
        _export_figure(fig, backend, save_path)
        if show:
            plt.show(block=False)
        return fig

    fig = go.Figure(
        data=go.Barpolar(
            theta=azimuths,
            r=values,
            marker=dict(color=values, colorscale="Viridis", showscale=True),
        )
    )
    fig.update_layout(
        template="plotly_white",
        title=title or "Directional Variogram Rose",
    )
    _export_figure(fig, backend, save_path)
    if show:
        fig.show()
    return fig


def plot_kriging_results(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    predictions: np.ndarray,
    sample_coords: Optional[np.ndarray] = None,
    sample_values: Optional[np.ndarray] = None,
    backend: str = "matplotlib",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """Plot kriging prediction surfaces with optional sampling points."""

    backend = _ensure_backend(backend)
    if backend == "matplotlib":
        fig, ax = plt.subplots(figsize=(8, 6))
        contour = ax.contourf(grid_x, grid_y, predictions, levels=20, cmap=cm.viridis)
        fig.colorbar(contour, ax=ax, label="Prediction")
        if sample_coords is not None and sample_values is not None:
            sc = ax.scatter(
                sample_coords[:, 0],
                sample_coords[:, 1],
                c=sample_values,
                cmap=cm.inferno,
                edgecolor="k",
                label="Samples",
            )
            fig.colorbar(sc, ax=ax, label="Sample values")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(title or "Kriging predictions")
        ax.grid(True, alpha=0.2)
        if sample_coords is not None:
            ax.legend(loc="upper right")
        _export_figure(fig, backend, save_path)
        if show:
            plt.show(block=False)
        return fig

    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            x=grid_x[0, :],
            y=grid_y[:, 0],
            z=predictions,
            contours_coloring="heatmap",
            colorbar=dict(title="Prediction"),
            name="Predictions",
        )
    )
    if sample_coords is not None and sample_values is not None:
        fig.add_trace(
            go.Scatter(
                x=sample_coords[:, 0],
                y=sample_coords[:, 1],
                mode="markers",
                marker=dict(
                    color=sample_values,
                    colorscale="Inferno",
                    size=8,
                    showscale=True,
                    colorbar=dict(title="Sample values"),
                ),
                name="Samples",
            )
        )
    fig.update_layout(
        title=title or "Kriging predictions",
        xaxis_title="X",
        yaxis_title="Y",
        template="plotly_white",
    )
    _export_figure(fig, backend, save_path)
    if show:
        fig.show()
    return fig


def plot_kriging_uncertainty(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    variance: np.ndarray,
    backend: str = "matplotlib",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """Visualize kriging uncertainty (variance) maps."""

    backend = _ensure_backend(backend)
    if backend == "matplotlib":
        fig, ax = plt.subplots(figsize=(8, 6))
        contour = ax.contourf(grid_x, grid_y, variance, levels=20, cmap=cm.cividis)
        fig.colorbar(contour, ax=ax, label="Variance")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(title or "Kriging variance")
        ax.grid(True, alpha=0.2)
        _export_figure(fig, backend, save_path)
        if show:
            plt.show(block=False)
        return fig

    fig = go.Figure(
        go.Contour(
            x=grid_x[0, :],
            y=grid_y[:, 0],
            z=variance,
            contours_coloring="heatmap",
            colorscale="Cividis",
            colorbar=dict(title="Variance"),
        )
    )
    fig.update_layout(
        title=title or "Kriging variance",
        xaxis_title="X",
        yaxis_title="Y",
        template="plotly_white",
    )
    _export_figure(fig, backend, save_path)
    if show:
        fig.show()
    return fig


def plot_kriging_cross_section(
    distances: Sequence[float],
    predictions: Sequence[float],
    observations: Optional[Sequence[float]] = None,
    backend: str = "matplotlib",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """Plot a cross-section through a kriging prediction surface."""

    backend = _ensure_backend(backend)
    distances = np.asarray(distances)
    predictions = np.asarray(predictions)

    if backend == "matplotlib":
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(distances, predictions, label="Prediction", color="tab:blue")
        if observations is not None:
            ax.scatter(distances, observations, color="tab:orange", label="Observed")
        ax.set_xlabel("Distance along section")
        ax.set_ylabel("Value")
        ax.set_title(title or "Kriging cross-section")
        ax.grid(True, alpha=0.3)
        ax.legend()
        _export_figure(fig, backend, save_path)
        if show:
            plt.show(block=False)
        return fig

    fig = go.Figure(
        go.Scatter(x=distances, y=predictions, mode="lines", name="Prediction")
    )
    if observations is not None:
        fig.add_trace(
            go.Scatter(x=distances, y=observations, mode="markers", name="Observed")
        )
    fig.update_layout(
        title=title or "Kriging cross-section",
        xaxis_title="Distance",
        yaxis_title="Value",
        template="plotly_white",
    )
    _export_figure(fig, backend, save_path)
    if show:
        fig.show()
    return fig


def plot_prediction_comparison(
    predicted: Sequence[float],
    observed: Sequence[float],
    backend: str = "matplotlib",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """Scatter plot comparing predicted and observed values."""

    backend = _ensure_backend(backend)
    predicted = np.asarray(predicted)
    observed = np.asarray(observed)
    identity = np.linspace(
        min(observed.min(), predicted.min()),
        max(observed.max(), predicted.max()),
        num=100,
    )

    if backend == "matplotlib":
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(
            observed, predicted, color="tab:blue", alpha=0.7, label="Predictions"
        )
        ax.plot(identity, identity, color="black", linestyle="--", label="1:1 line")
        ax.set_xlabel("Observed")
        ax.set_ylabel("Predicted")
        ax.set_title(title or "Prediction vs Observed")
        ax.grid(True, alpha=0.3)
        ax.legend()
        _export_figure(fig, backend, save_path)
        if show:
            plt.show(block=False)
        return fig

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=observed, y=predicted, mode="markers", name="Predictions")
    )
    fig.add_trace(
        go.Scatter(
            x=identity,
            y=identity,
            mode="lines",
            name="1:1 line",
            line=dict(color="black", dash="dash"),
        )
    )
    fig.update_layout(
        title=title or "Prediction vs Observed",
        xaxis_title="Observed",
        yaxis_title="Predicted",
        template="plotly_white",
    )
    _export_figure(fig, backend, save_path)
    if show:
        fig.show()
    return fig


def plot_residuals_qq(
    residuals: Sequence[float],
    backend: str = "matplotlib",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """Generate a QQ plot for kriging residuals."""

    backend = _ensure_backend(backend)
    residuals = np.asarray(residuals)
    n = len(residuals)
    if n < 3:
        raise ValueError("plot_residuals_qq requires at least three residuals")

    ordered = np.sort(residuals)
    probs = (np.arange(1, n + 1) - 0.5) / n
    theoretical = stats.norm.ppf(probs)

    lo = min(theoretical.min(), ordered.min())
    hi = max(theoretical.max(), ordered.max())

    if backend == "matplotlib":
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(theoretical, ordered, color="tab:blue")
        ax.plot([lo, hi], [lo, hi], color="black", linestyle="--")
        ax.set_xlabel("Theoretical quantiles")
        ax.set_ylabel("Sample quantiles")
        ax.set_title(title or "Residual QQ plot")
        ax.grid(True, alpha=0.3)
        _export_figure(fig, backend, save_path)
        if show:
            plt.show(block=False)
        return fig

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=theoretical, y=ordered, mode="markers", name="Residuals")
    )
    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            name="1:1",
            line=dict(color="black", dash="dash"),
        )
    )
    fig.update_layout(
        title=title or "Residual QQ plot",
        xaxis_title="Theoretical quantiles",
        yaxis_title="Sample quantiles",
        template="plotly_white",
    )
    _export_figure(fig, backend, save_path)
    if show:
        fig.show()
    return fig


def plot_variogram_cloud(
    distances: Sequence[float],
    gamma: Sequence[float],
    backend: str = "matplotlib",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """Plot a variogram cloud scatter plot."""

    backend = _ensure_backend(backend)
    distances = np.asarray(distances)
    gamma = np.asarray(gamma)

    if backend == "matplotlib":
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(distances, gamma, color="tab:blue", alpha=0.4)
        ax.set_xlabel("Distance")
        ax.set_ylabel("Semivariance")
        ax.set_title(title or "Variogram cloud")
        ax.grid(True, alpha=0.3)
        _export_figure(fig, backend, save_path)
        if show:
            plt.show(block=False)
        return fig

    fig = go.Figure(go.Scatter(x=distances, y=gamma, mode="markers", opacity=0.6))
    fig.update_layout(
        title=title or "Variogram cloud",
        xaxis_title="Distance",
        yaxis_title="Semivariance",
        template="plotly_white",
    )
    _export_figure(fig, backend, save_path)
    if show:
        fig.show()
    return fig


def plot_spatial_correlation(
    lags: Sequence[float],
    correlations: Sequence[float],
    backend: str = "matplotlib",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """Plot spatial correlation statistics against lag distance."""

    backend = _ensure_backend(backend)
    lags = np.asarray(lags)
    correlations = np.asarray(correlations)

    if backend == "matplotlib":
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(lags, correlations, marker="o", color="tab:green")
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Lag distance")
        ax.set_ylabel("Correlation")
        ax.set_title(title or "Spatial correlation by lag")
        ax.grid(True, alpha=0.3)
        _export_figure(fig, backend, save_path)
        if show:
            plt.show(block=False)
        return fig

    fig = go.Figure(
        go.Scatter(x=lags, y=correlations, mode="markers+lines", name="Correlation")
    )
    fig.add_hline(y=0.0, line=dict(color="black", dash="dash"))
    fig.update_layout(
        title=title or "Spatial correlation by lag",
        xaxis_title="Lag distance",
        yaxis_title="Correlation",
        template="plotly_white",
    )
    _export_figure(fig, backend, save_path)
    if show:
        fig.show()
    return fig


def plot_anisotropy_rose(
    angles: Sequence[float],
    ranges: Sequence[float],
    backend: str = "matplotlib",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """Plot anisotropy range estimates as a rose diagram."""

    backend = _ensure_backend(backend)
    angles = np.asarray(angles)
    ranges = np.asarray(ranges)

    if backend == "matplotlib":
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="polar")
        width = np.deg2rad(360.0 / len(angles)) if len(angles) > 0 else 0.1
        ax.bar(np.deg2rad(angles), ranges, width=width, color="tab:purple", alpha=0.6)
        ax.set_title(title or "Anisotropy rose")
        _export_figure(fig, backend, save_path)
        if show:
            plt.show(block=False)
        return fig

    fig = go.Figure(
        data=go.Barpolar(
            theta=angles,
            r=ranges,
            marker=dict(color=ranges, colorscale="Plasma", showscale=True),
        )
    )
    fig.update_layout(
        title=title or "Anisotropy rose",
        template="plotly_white",
    )
    _export_figure(fig, backend, save_path)
    if show:
        fig.show()
    return fig
