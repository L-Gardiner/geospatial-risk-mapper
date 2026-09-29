"""Evaluation and visualization for flood risk predictions.

Generates metrics, confusion matrices, and choropleth maps.
"""

import logging
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from .config import settings

logger = logging.getLogger(__name__)

RISK_LABELS = {0: "Low", 1: "Medium", 2: "High"}
RISK_COLORS = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}  # Green, Orange, Red


def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    output_path: str | Path | None = None,
) -> Path:
    """Plot confusion matrix for flood risk classification.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        output_path: Path to save figure. If None, uses default.

    Returns:
        Path to saved figure.
    """
    if output_path is None:
        output_path = Path(settings.models_dir) / "confusion_matrix.png"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")

    # Labels
    classes = [RISK_LABELS[i] for i in range(3)]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    # Text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )

    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_title("Confusion Matrix: Flood Risk Classification")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved confusion matrix to {output_path}")
    return output_path


def plot_predictions_on_map(
    nuts_gdf: gpd.GeoDataFrame,
    predictions: pd.Series,
    title: str = "Predicted Flood Risk",
    output_path: str | Path | None = None,
) -> Path:
    """Plot predicted flood risk as choropleth map.

    Args:
        nuts_gdf: GeoDataFrame with NUTS regions.
        predictions: Series with predicted risk classes (index=NUTS_ID).
        title: Map title.
        output_path: Path to save figure. If None, uses default.

    Returns:
        Path to saved figure.
    """
    if output_path is None:
        output_path = Path(settings.models_dir) / "predicted_risk_map.png"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Merge predictions with geometry
    map_data = nuts_gdf[["NUTS_ID", "geometry"]].copy()
    map_data["risk_class"] = map_data["NUTS_ID"].map(predictions)
    map_data["risk_label"] = map_data["risk_class"].map(RISK_LABELS)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot each risk class separately for custom colors
    for risk_class in [0, 1, 2]:
        subset = map_data[map_data["risk_class"] == risk_class]
        subset.plot(
            ax=ax,
            color=RISK_COLORS[risk_class],
            edgecolor="black",
            linewidth=0.5,
            label=RISK_LABELS[risk_class],
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(title="Risk Class", loc="upper right")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved prediction map to {output_path}")
    return output_path


def plot_actual_vs_predicted(
    nuts_gdf: gpd.GeoDataFrame,
    actual: pd.Series,
    predicted: pd.Series,
    output_path: str | Path | None = None,
) -> Path:
    """Plot actual vs predicted flood risk side-by-side.

    Args:
        nuts_gdf: GeoDataFrame with NUTS regions.
        actual: Series with actual risk classes.
        predicted: Series with predicted risk classes.
        output_path: Path to save figure. If None, uses default.

    Returns:
        Path to saved figure.
    """
    if output_path is None:
        output_path = Path(settings.models_dir) / "actual_vs_predicted.png"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Actual
    map_data = nuts_gdf[["NUTS_ID", "geometry"]].copy()
    map_data["risk_class"] = map_data["NUTS_ID"].map(actual)

    for risk_class in [0, 1, 2]:
        subset = map_data[map_data["risk_class"] == risk_class]
        subset.plot(
            ax=axes[0],
            color=RISK_COLORS[risk_class],
            edgecolor="black",
            linewidth=0.5,
            label=RISK_LABELS[risk_class],
        )

    axes[0].set_title("Actual Flood Risk", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    axes[0].legend(title="Risk Class", loc="upper right")

    # Predicted
    map_data["risk_class"] = map_data["NUTS_ID"].map(predicted)

    for risk_class in [0, 1, 2]:
        subset = map_data[map_data["risk_class"] == risk_class]
        subset.plot(
            ax=axes[1],
            color=RISK_COLORS[risk_class],
            edgecolor="black",
            linewidth=0.5,
            label=RISK_LABELS[risk_class],
        )

    axes[1].set_title("Predicted Flood Risk", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")
    axes[1].legend(title="Risk Class", loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved actual vs predicted map to {output_path}")
    return output_path


def plot_moran_scatter(
    values: pd.Series | np.ndarray,
    coords_or_geo: gpd.GeoDataFrame | pd.DataFrame | np.ndarray,
    title: str = "Moran's I Spatial Autocorrelation Scatter Plot",
    output_path: str | Path | None = None,
) -> Path:
    """Plot Moran scatter plot showing standardized value vs spatial lag.

    Args:
        values: Series or array of values across geographic regions.
        coords_or_geo: Geographic geometries or coordinates.
        title: Title for the scatter plot.
        output_path: Output file path.

    Returns:
        Path to the saved figure.
    """
    from .spatial_cv import (
        build_spatial_weight_matrix,
        compute_global_morans_i,
        extract_coordinates,
    )

    if output_path is None:
        output_path = Path(settings.models_dir) / "moran_scatter.png"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    x = np.asarray(values, dtype=float)
    coords = extract_coordinates(coords_or_geo)
    w = build_spatial_weight_matrix(coords, weight_type="inverse_distance", row_standardize=True)

    z = (x - np.mean(x)) / (np.std(x) if np.std(x) > 0 else 1.0)
    wz = w @ z

    moran = compute_global_morans_i(x, coords)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(z, wz, color="#2980b9", alpha=0.8, edgecolors="k", s=60)

    # Linear fit line with slope = Moran's I
    slope, intercept = np.polyfit(z, wz, 1)
    x_vals = np.linspace(min(z), max(z), 100)
    ax.plot(x_vals, slope * x_vals + intercept, color="#e74c3c", linestyle="--", linewidth=2,
            label=f"Slope (Moran's I) = {moran.i_stat:.3f} (p={moran.p_value:.3f})")

    ax.axhline(0, color="gray", linestyle=":", alpha=0.6)
    ax.axvline(0, color="gray", linestyle=":", alpha=0.6)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Standardized Value (z)", fontsize=10)
    ax.set_ylabel("Spatial Lag (Wz)", fontsize=10)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved Moran scatter plot to {output_path}")
    return output_path


def plot_lisa_clusters(
    nuts_gdf: gpd.GeoDataFrame,
    values: pd.Series | np.ndarray,
    title: str = "Local Indicators of Spatial Association (LISA)",
    output_path: str | Path | None = None,
) -> Path:
    """Plot Local Moran's I (LISA) spatial clusters on geographic map.

    Args:
        nuts_gdf: GeoDataFrame with region geometries.
        values: Series or array of regional flood risk exposure.
        title: Title of map.
        output_path: Filepath to save plot.

    Returns:
        Path to saved figure.
    """
    from .spatial_cv import compute_local_morans_i

    if output_path is None:
        output_path = Path(settings.models_dir) / "lisa_clusters.png"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    lisa = compute_local_morans_i(values, nuts_gdf)

    gdf = nuts_gdf.copy()
    gdf["quadrant"] = lisa.quadrants

    color_map = {
        "HH": "#e74c3c",  # Hotspot (High-High)
        "LL": "#3498db",  # Coldspot (Low-Low)
        "HL": "#e67e22",  # Spatial outlier
        "LH": "#9b59b6",  # Spatial outlier
    }

    fig, ax = plt.subplots(figsize=(10, 8))
    for quad, color in color_map.items():
        subset = gdf[gdf["quadrant"] == quad]
        if not subset.empty:
            subset.plot(ax=ax, color=color, edgecolor="black", linewidth=0.5, label=quad)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(title="LISA Cluster", loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved LISA cluster map to {output_path}")
    return output_path

