"""Spatial autocorrelation analysis and spatial cross-validation for flood risk.

Addresses spatial autocorrelation (Tobler's First Law of Geography) and prevents
spatial data leakage across contiguous geographic regions.

Research Grounding:
- Moran, P. A. P. (1950). "Notes on Continuous Stochastic Phenomena." Biometrika.
- Anselin, L. (1995). "Local Indicators of Spatial Association—LISA." Geographical Analysis.
- Roberts, D. R. et al. (2017). "Cross-validation strategies for data with spatial,
  temporal, hierarchical or phylogenetic structure." Ecography, 40(8), 913-929.
- Wadoux, A. M. J. C. et al. (2021). "Spatial cross-validation is not the right way
  to evaluate map accuracy." Ecological Modelling, 457, 109692 (arXiv:2303.07334).
- Meyer, H. et al. (2019). "Importance of spatial predictor variable selection in
  machine learning applications." Ecological Modelling, 411, 108815 (arXiv:1908.07805).
"""

from __future__ import annotations

import logging
from collections.abc import Generator, Sequence
from dataclasses import dataclass
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)


@dataclass
class MoransIResult:
    """Statistical summary of Global Moran's I spatial autocorrelation test."""

    i_stat: float
    expected_i: float
    variance: float
    z_score: float
    p_value: float
    weights_type: str
    n_samples: int


@dataclass
class LocalMoranResult:
    """Summary of Local Indicators of Spatial Association (LISA)."""

    local_i: np.ndarray
    quadrants: list[str]  # "HH", "LL", "HL", "LH"
    z_scores: np.ndarray


@dataclass
class SpatialCVComparisonResult:
    """Comparison metrics between standard random CV and spatial CV."""

    task: str
    n_splits: int
    standard_cv: dict[str, float]
    spatial_cv: dict[str, float]
    spatial_gap: dict[str, float]
    morans_i: MoransIResult | None = None


def extract_coordinates(
    geo_data: gpd.GeoDataFrame | pd.DataFrame | np.ndarray,
) -> np.ndarray:
    """Extract 2D representative coordinates (x, y) from geometries or DataFrame.

    Args:
        geo_data: GeoDataFrame with polygon geometries, DataFrame with lon/lat
            or x/y columns, or an array of shape (N, 2).

    Returns:
        np.ndarray of shape (N, 2).
    """
    if isinstance(geo_data, gpd.GeoDataFrame):
        gdf = geo_data
        try:
            if gdf.crs is not None and gdf.crs.is_geographic:
                utm_crs = gdf.estimate_utm_crs()
                projected = gdf.to_crs(utm_crs)
                centroids = projected.geometry.centroid.to_crs(gdf.crs)
            else:
                centroids = gdf.geometry.centroid
        except Exception:
            centroids = gdf.geometry.centroid
        coords = np.column_stack([centroids.x.to_numpy(), centroids.y.to_numpy()])
        if not np.all(np.isfinite(coords)):
            raise ValueError("Extracted coordinates contain NaN or Inf.")
        return coords
    if isinstance(geo_data, pd.DataFrame):
        for pair in [("lon", "lat"), ("longitude", "latitude"), ("x", "y"), ("X", "Y")]:
            if pair[0] in geo_data.columns and pair[1] in geo_data.columns:
                coords = geo_data[[pair[0], pair[1]]].to_numpy()
                if not np.all(np.isfinite(coords)):
                    raise ValueError("Coordinates contain NaN or Inf.")
                return coords
        raise ValueError(
            f"DataFrame must contain coordinate columns (e.g. lon/lat or x/y). Found: {list(geo_data.columns)}"
        )
    if isinstance(geo_data, np.ndarray):
        if geo_data.ndim != 2 or geo_data.shape[1] != 2:
            raise ValueError(f"Array of coordinates must have shape (N, 2), got {geo_data.shape}")
        if not np.all(np.isfinite(geo_data)):
            raise ValueError("Coordinates contain NaN or Inf.")
        return geo_data.copy()
    raise TypeError(f"Unsupported geographic data type: {type(geo_data)}")


def compute_haversine_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Compute pairwise great-circle haversine distances in kilometers for (lon, lat) points in degrees."""
    lon = np.radians(coords[:, 0])
    lat = np.radians(coords[:, 1])
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return 6371.0 * c


def build_spatial_weight_matrix(
    coords: np.ndarray,
    weight_type: str = "inverse_distance",
    k_neighbors: int | None = None,
    row_standardize: bool = True,
    metric: str = "euclidean",
) -> np.ndarray:
    """Build spatial weight matrix W with zero diagonal.

    Args:
        coords: (N, 2) array of spatial points.
        weight_type: 'inverse_distance', 'k_nearest', or 'gaussian'.
        k_neighbors: Number of nearest neighbors if using 'k_nearest'.
        row_standardize: If True, row-standardizes W so each row sums to 1.
        metric: 'euclidean' (coordinate units) or 'haversine' (kilometers for lon/lat).

    Returns:
        (N, N) spatial weight matrix.
    """
    n = len(coords)
    if n < 2:
        raise ValueError("At least 2 spatial samples required to construct a spatial weight matrix.")

    if metric == "haversine":
        dist = compute_haversine_distance_matrix(coords)
    elif metric == "euclidean":
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=-1))
    else:
        raise ValueError(f"Unknown metric '{metric}'. Choose from 'euclidean', 'haversine'.")

    # Set self-distance to infinity to prevent division by zero on diagonal
    np.fill_diagonal(dist, np.inf)

    if weight_type == "inverse_distance":
        with np.errstate(divide="ignore"):
            w = 1.0 / dist
        np.fill_diagonal(w, 0.0)

    elif weight_type == "k_nearest":
        k = k_neighbors if k_neighbors is not None else min(4, n - 1)
        k = max(1, min(k, n - 1))
        w = np.zeros((n, n), dtype=float)
        for i in range(n):
            nn_idx = np.argsort(dist[i])[:k]
            w[i, nn_idx] = 1.0

    elif weight_type == "gaussian":
        sigma = np.median(dist[np.isfinite(dist)])
        if sigma <= 0:
            sigma = 1.0
        w = np.exp(-(dist**2) / (2.0 * sigma**2))
        np.fill_diagonal(w, 0.0)

    else:
        raise ValueError(f"Unknown weight_type '{weight_type}'. Choose from 'inverse_distance', 'k_nearest', 'gaussian'.")

    if row_standardize:
        row_sums = w.sum(axis=1, keepdims=True)
        # Avoid division by zero if a point is completely isolated
        row_sums[row_sums == 0] = 1.0
        w = w / row_sums

    return w


def compute_global_morans_i(
    values: Sequence[float] | np.ndarray | pd.Series,
    coords_or_geo: gpd.GeoDataFrame | pd.DataFrame | np.ndarray,
    weight_type: str = "inverse_distance",
    k_neighbors: int | None = None,
) -> MoransIResult:
    """Compute Global Moran's I statistic with analytical expectation and variance.

    Formulation (Moran 1950):
        I = (N / S0) * sum_i sum_j [w_ij * (x_i - x_mean) * (x_j - x_mean)] / sum_i (x_i - x_mean)^2
    under the randomization hypothesis for inference against spatial randomness.

    Args:
        values: Numerical feature or exposure values across N regions.
        coords_or_geo: Spatial coordinates or GeoDataFrame for spatial weights.
        weight_type: 'inverse_distance' or 'k_nearest'.
        k_neighbors: Optional neighbor count for k-nearest weights.

    Returns:
        MoransIResult with i_stat, expected_i, variance, z_score, p_value.
    """
    x = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(x)):
        raise ValueError("Input values contain NaN or Inf.")
    n = len(x)
    if n < 3:
        raise ValueError("Moran's I requires at least 3 spatial samples.")

    coords = extract_coordinates(coords_or_geo)
    w = build_spatial_weight_matrix(
        coords, weight_type=weight_type, k_neighbors=k_neighbors, row_standardize=True
    )

    z = x - np.mean(x)
    ss = np.sum(z**2)
    if ss == 0.0:
        return MoransIResult(
            i_stat=0.0,
            expected_i=-1.0 / (n - 1),
            variance=0.0,
            z_score=0.0,
            p_value=1.0,
            weights_type=weight_type,
            n_samples=n,
        )

    s0 = np.sum(w)
    # Global Moran's I
    numerator = np.sum(w * np.outer(z, z))
    i_stat = float((n / s0) * (numerator / ss))

    # Expectation under null hypothesis of no spatial pattern
    expected_i = float(-1.0 / (n - 1))

    # Variance under randomization assumption (Cliff & Ord 1981, Anselin 1995)
    s1 = 0.5 * np.sum((w + w.T) ** 2)
    s2 = np.sum((w.sum(axis=1) + w.sum(axis=0)) ** 2)
    m4 = np.sum(z**4) / n
    m2 = ss / n
    b2 = m4 / (m2**2) if m2 > 0 else 0.0

    a = n * ((n**2 - 3 * n + 3) * s1 - n * s2 + 3 * (s0**2))
    b = b2 * ((n**2 - n) * s1 - 2 * n * s2 + 6 * (s0**2))
    c = (n - 1) * (n - 2) * (n - 3) * (s0**2)

    var_i = float((a - b) / c - (expected_i**2)) if c > 0 else 1.0 / (n - 1)
    var_i = max(var_i, 1e-12)

    z_score = float((i_stat - expected_i) / np.sqrt(var_i))
    p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(z_score))))

    return MoransIResult(
        i_stat=i_stat,
        expected_i=expected_i,
        variance=var_i,
        z_score=z_score,
        p_value=p_value,
        weights_type=weight_type,
        n_samples=n,
    )


def compute_local_morans_i(
    values: Sequence[float] | np.ndarray | pd.Series,
    coords_or_geo: gpd.GeoDataFrame | pd.DataFrame | np.ndarray,
    weight_type: str = "inverse_distance",
) -> LocalMoranResult:
    """Compute Local Indicators of Spatial Association (LISA, Anselin 1995).

    Identifies localized High-High (hotspots), Low-Low (coldspots), and
    spatial outlier regions (High-Low, Low-High).
    """
    x = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(x)):
        raise ValueError("Input values contain NaN or Inf.")
    n = len(x)
    coords = extract_coordinates(coords_or_geo)
    w = build_spatial_weight_matrix(coords, weight_type=weight_type, row_standardize=True)

    z = x - np.mean(x)
    m2 = np.sum(z**2) / n
    if m2 == 0:
        return LocalMoranResult(
            local_i=np.zeros(n),
            quadrants=["LL"] * n,
            z_scores=np.zeros(n),
        )

    # Local Moran's I: I_i = (z_i / m2) * sum_j w_ij * z_j
    spatial_lag = w @ z
    local_i = (z / m2) * spatial_lag

    # Quadrant classification
    quadrants = []
    for zi, lag_i in zip(z, spatial_lag, strict=True):
        if zi >= 0 and lag_i >= 0:
            quadrants.append("HH")
        elif zi < 0 and lag_i < 0:
            quadrants.append("LL")
        elif zi >= 0 and lag_i < 0:
            quadrants.append("HL")
        else:
            quadrants.append("LH")

    std_li = np.std(local_i) if np.std(local_i) > 0 else 1.0
    z_scores = (local_i - np.mean(local_i)) / std_li

    return LocalMoranResult(
        local_i=local_i,
        quadrants=quadrants,
        z_scores=z_scores,
    )


class SpatialKFold:
    """Spatial K-Fold cross-validator using spatial cluster partitioning.

    Partitions spatial coordinates into K spatially contiguous clusters (via KMeans),
    ensuring contiguous spatial units are kept together in test folds to prevent
    spatial information leakage.
    """

    def __init__(self, n_splits: int = 5, random_state: int = 42):
        if n_splits < 2:
            raise ValueError(f"n_splits must be at least 2, got {n_splits}")
        self.n_splits = n_splits
        self.random_state = random_state

    def split(
        self,
        X: pd.DataFrame | np.ndarray,  # noqa: N803
        y: pd.Series | np.ndarray | None = None,
        coords_or_geo: gpd.GeoDataFrame | pd.DataFrame | np.ndarray | None = None,
    ) -> Generator[tuple[np.ndarray, np.ndarray]]:
        """Generate indices to split data into training and test sets."""
        n_samples = len(X)
        if n_samples < 2:
            raise ValueError("Need at least 2 samples for spatial cross-validation.")

        if coords_or_geo is None:
            # Fall back to using first two feature dimensions if coords not provided
            if isinstance(X, pd.DataFrame):
                coords = X.iloc[:, :2].to_numpy()
            else:
                coords = np.asarray(X)[:, :2]
        else:
            coords = extract_coordinates(coords_or_geo)

        if len(coords) != n_samples:
            raise ValueError(f"Coords length ({len(coords)}) does not match sample count ({n_samples})")

        unique_coords = np.unique(coords, axis=0)
        if len(unique_coords) < 2:
            raise ValueError(
                f"SpatialKFold requires at least 2 distinct spatial locations, but found {len(unique_coords)}."
            )

        effective_splits = min(self.n_splits, len(unique_coords))

        # Cluster spatial coordinates
        kmeans = KMeans(
            n_clusters=effective_splits,
            random_state=self.random_state,
            n_init=10,
        )
        cluster_labels = kmeans.fit_predict(coords)

        # Ensure every cluster has samples
        unique_clusters = np.unique(cluster_labels)
        all_indices = np.arange(n_samples)

        for cluster_id in unique_clusters:
            test_mask = cluster_labels == cluster_id
            test_indices = all_indices[test_mask]
            train_indices = all_indices[~test_mask]
            if len(train_indices) == 0:
                continue
            yield train_indices, test_indices


class SpatialBufferCV:
    """Leave-one-region-out cross-validator with spatial buffer exclusion.

    For each test region i, excludes all regions j whose centroid distance
    dist(i, j) <= buffer_distance to eliminate proximate spatial autocorrelation leakage.
    Supports 'euclidean' (coordinate units) or 'haversine' (kilometers for lon/lat).
    """

    def __init__(self, buffer_distance: float = 0.0, metric: str = "euclidean"):
        self.buffer_distance = max(0.0, float(buffer_distance))
        if metric not in ("euclidean", "haversine"):
            raise ValueError(f"Unknown metric '{metric}'. Choose from 'euclidean', 'haversine'.")
        self.metric = metric

    def split(
        self,
        X: pd.DataFrame | np.ndarray,  # noqa: N803
        y: pd.Series | np.ndarray | None = None,
        coords_or_geo: gpd.GeoDataFrame | pd.DataFrame | np.ndarray | None = None,
    ) -> Generator[tuple[np.ndarray, np.ndarray]]:
        """Generate leave-one-out splits with spatial buffer exclusion."""
        n_samples = len(X)
        if n_samples < 2:
            raise ValueError("SpatialBufferCV requires at least 2 samples.")
        if coords_or_geo is None:
            raise ValueError("coords_or_geo is required for SpatialBufferCV.")

        coords = extract_coordinates(coords_or_geo)
        all_indices = np.arange(n_samples)

        if self.metric == "haversine":
            dist_matrix = compute_haversine_distance_matrix(coords)
        else:
            diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
            dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))

        for i in range(n_samples):
            test_idx = np.array([i])
            # Exclude self and any region within buffer distance
            excluded = (dist_matrix[i] <= self.buffer_distance) | (all_indices == i)
            train_idx = all_indices[~excluded]

            if len(train_idx) == 0:
                # If buffer is too large and excludes everything, fallback to leave-one-out
                train_idx = all_indices[all_indices != i]

            yield train_idx, test_idx


def evaluate_spatial_cv(
    features_df: pd.DataFrame,
    target: pd.Series,
    coords_or_geo: gpd.GeoDataFrame | pd.DataFrame | np.ndarray,
    task: str = "classification",
    n_splits: int = 5,
    random_state: int = 42,
    drop_leakage_features: list[str] | None = None,
) -> SpatialCVComparisonResult:
    """Evaluate models under both Standard K-Fold and Spatial K-Fold cross validation.

    Reports the generalization gap, revealing spatial overfitting caused by
    spatial autocorrelation.

    Args:
        features_df: Engineered features DataFrame.
        target: Target Series (discrete for classification, continuous for regression).
        coords_or_geo: GeoDataFrame or spatial coordinates.
        task: 'classification' or 'regression'.
        n_splits: Number of cross-validation folds.
        random_state: Random state for model and splitting.
        drop_leakage_features: Optional feature names to drop to evaluate leakage-free
            performance (e.g. ['flood_fraction'] when predicting tertile risk class).

    Returns:
        SpatialCVComparisonResult containing comparative metrics and spatial gap.
    """
    X = features_df.copy()
    if drop_leakage_features:
        valid_drops = [f for f in drop_leakage_features if f in X.columns]
        if valid_drops:
            X = X.drop(columns=valid_drops)
            logger.info(f"Dropped target-leakage features for honest evaluation: {valid_drops}")

    y = target.copy()
    coords = extract_coordinates(coords_or_geo)

    # Compute Moran's I on the target variable
    morans_result = None
    try:
        morans_result = compute_global_morans_i(y.to_numpy(), coords)
    except Exception as e:
        logger.warning(f"Failed to compute Moran's I on target: {e}")

    # Standard Random K-Fold
    standard_kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    spatial_kf = SpatialKFold(n_splits=n_splits, random_state=random_state)

    def _run_cv(splitter, is_spatial: bool) -> dict[str, float]:
        fold_metrics: list[dict[str, float]] = []

        if is_spatial:
            splits_gen = splitter.split(X, y, coords_or_geo=coords)
        else:
            splits_gen = splitter.split(X, y)

        for train_idx, test_idx in splits_gen:
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

            if task == "classification":
                from sklearn.ensemble import RandomForestClassifier

                # If train set is missing a class, skip or handle gracefully
                model: Any = RandomForestClassifier(
                    n_estimators=50,
                    max_depth=8,
                    min_samples_split=3,
                    random_state=random_state,
                    n_jobs=-1,
                )
                model.fit(X_tr, y_tr)
                preds = model.predict(X_te)

                acc = float(accuracy_score(y_te, preds))
                f1_w = float(f1_score(y_te, preds, average="weighted", zero_division=0))
                fold_metrics.append({"accuracy": acc, "f1_weighted": f1_w})

            else:
                from sklearn.ensemble import RandomForestRegressor

                model = RandomForestRegressor(
                    n_estimators=50,
                    max_depth=8,
                    min_samples_split=3,
                    random_state=random_state,
                    n_jobs=-1,
                )
                model.fit(X_tr, y_tr)
                preds = model.predict(X_te)

                mae = float(mean_absolute_error(y_te, preds))
                rmse = float(np.sqrt(mean_squared_error(y_te, preds)))
                r2 = float(r2_score(y_te, preds)) if len(y_te) > 1 and np.var(y_te) > 0 else 0.0
                fold_metrics.append({"mae": mae, "rmse": rmse, "r2": r2})

        if not fold_metrics:
            return {}

        metrics_df = pd.DataFrame(fold_metrics)
        return {f"mean_{k}": float(metrics_df[k].mean()) for k in metrics_df.columns}

    std_res = _run_cv(standard_kf, is_spatial=False)
    spat_res = _run_cv(spatial_kf, is_spatial=True)

    # Compute spatial performance gap (overoptimism of random CV)
    gap: dict[str, float] = {}
    for k in std_res:
        if k in spat_res:
            gap[f"{k}_diff"] = float(std_res[k] - spat_res[k])

    return SpatialCVComparisonResult(
        task=task,
        n_splits=n_splits,
        standard_cv=std_res,
        spatial_cv=spat_res,
        spatial_gap=gap,
        morans_i=morans_result,
    )
