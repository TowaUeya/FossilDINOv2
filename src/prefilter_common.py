from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import RobustScaler


LOGGER = logging.getLogger(__name__)
SUPPORTED_PREFILTER_EXTENSIONS = {".ply", ".obj", ".stl", ".off"}
EPS = 1e-8
PLANAR_DEGENERATE_THRESHOLD = 0.02
LINEAR_DEGENERATE_THRESHOLD = 0.02


class MetadataExtractionError(RuntimeError):
    def __init__(self, message: str, timings: dict[str, float], specimen_id: str, source_path: str):
        super().__init__(message)
        self.timings = timings
        self.specimen_id = specimen_id
        self.source_path = source_path


def list_source_files(input_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_PREFILTER_EXTENSIONS
    )


def to_specimen_id(path: Path, root_dir: Path) -> str:
    rel = path.relative_to(root_dir)
    no_suffix = rel.with_suffix("")
    return no_suffix.as_posix()


def _safe_bbox_extent(points: np.ndarray) -> tuple[float, float, float]:
    min_b = points.min(axis=0)
    max_b = points.max(axis=0)
    ext = max_b - min_b
    if np.any(~np.isfinite(ext)):
        raise ValueError("invalid bbox extent")
    return float(ext[0]), float(ext[1]), float(ext[2])


def _normalize_color_range(colors: np.ndarray) -> np.ndarray:
    colors = np.asarray(colors, dtype=np.float64)
    if colors.size == 0:
        return colors
    max_val = float(np.nanmax(colors))
    if max_val > 1.0:
        colors = colors / 255.0
    return np.clip(colors, 0.0, 1.0)


def _extract_colors_if_any(geom: Any) -> tuple[int, float, float, float, float, float, float]:
    colors = None
    if hasattr(geom, "has_vertex_colors") and geom.has_vertex_colors():
        colors = np.asarray(geom.vertex_colors)
    elif hasattr(geom, "has_colors") and geom.has_colors():
        colors = np.asarray(geom.colors)

    if colors is None or colors.size == 0:
        return 0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    colors = _normalize_color_range(colors)
    if colors.ndim != 2 or colors.shape[1] < 3:
        return 0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    rgb = colors[:, :3]
    return (
        1,
        float(np.nanmean(rgb[:, 0])),
        float(np.nanmean(rgb[:, 1])),
        float(np.nanmean(rgb[:, 2])),
        float(np.nanstd(rgb[:, 0])),
        float(np.nanstd(rgb[:, 1])),
        float(np.nanstd(rgb[:, 2])),
    )


def _safe_equiv_diameter(volume: float) -> float:
    if not np.isfinite(volume) or volume <= 0:
        return np.nan
    return float((6.0 * volume / math.pi) ** (1.0 / 3.0))


def _safe_mesh_surface_area(geom: Any) -> float:
    if not hasattr(geom, "get_surface_area"):
        return np.nan
    try:
        return float(geom.get_surface_area())
    except Exception:
        return np.nan


def _safe_convex_hull_metrics(geom: Any, joggle_inputs: bool = False) -> tuple[float, float, bool]:
    if not hasattr(geom, "compute_convex_hull"):
        return np.nan, np.nan, True
    try:
        hull, _ = geom.compute_convex_hull(joggle_inputs=joggle_inputs)
    except Exception:
        return np.nan, np.nan, True

    hull_area = np.nan
    hull_volume = np.nan
    failed = False
    try:
        hull_area = float(hull.get_surface_area())
    except Exception:
        failed = True
    try:
        hull_volume = float(hull.get_volume())
    except Exception:
        failed = True
    return hull_volume, hull_area, failed


def _safe_mesh_volume(geom: Any, has_triangles: bool, is_watertight: int) -> tuple[float, bool]:
    if not has_triangles or is_watertight != 1 or not hasattr(geom, "get_volume"):
        return np.nan, False
    try:
        return float(geom.get_volume()), False
    except Exception:
        return np.nan, True


def _pca_axis_lengths(points: np.ndarray) -> tuple[float, float, float]:
    if points.shape[0] < 3:
        return np.nan, np.nan, np.nan
    centered = points - np.mean(points, axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    if cov.shape != (3, 3) or np.any(~np.isfinite(cov)):
        return np.nan, np.nan, np.nan
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(np.sort(eigvals)[::-1], 0.0, None)
    lengths = 2.0 * np.sqrt(eigvals)
    return float(lengths[0]), float(lengths[1]), float(lengths[2])


def _sample_points_for_hull(
    points: np.ndarray,
    max_points: int,
    sampling: str,
    bbox_longest: float,
    seed: int,
) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points

    if sampling == "random":
        rng = np.random.default_rng(seed)
        idx = rng.choice(points.shape[0], size=max_points, replace=False)
        return points[idx]

    try:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        denom = max(max_points, 8)
        voxel_size = max(float(bbox_longest) / (denom ** (1.0 / 3.0)), 1e-9)
        down = pcd.voxel_down_sample(voxel_size)
        sampled = np.asarray(down.points)
        if sampled.shape[0] == 0:
            return points[:max_points]
        if sampled.shape[0] > max_points:
            rng = np.random.default_rng(seed)
            idx = rng.choice(sampled.shape[0], size=max_points, replace=False)
            return sampled[idx]
        return sampled
    except Exception:
        rng = np.random.default_rng(seed)
        idx = rng.choice(points.shape[0], size=max_points, replace=False)
        return points[idx]


def extract_metadata_record(
    mesh_path: Path,
    root_dir: Path,
    hull_max_points: int = 5000,
    hull_sampling: str = "random",
    hull_joggle_inputs: bool = False,
    planar_degenerate_thresh: float = PLANAR_DEGENERATE_THRESHOLD,
    linear_degenerate_thresh: float = LINEAR_DEGENERATE_THRESHOLD,
    size_compute_mode: str = "fast",
    enable_hull_features: bool = False,
    enable_mesh_volume: bool = False,
) -> dict[str, Any]:
    from src.utils.geometry import load_geometry

    specimen_id = to_specimen_id(mesh_path, root_dir)
    timings = {
        "mesh_load_sec": np.nan,
        "aabb_sec": np.nan,
        "hull_sec": np.nan,
        "mesh_volume_sec": np.nan,
        "total_sec": np.nan,
    }
    total_start = time.perf_counter()
    try:
        load_start = time.perf_counter()
        geom = load_geometry(mesh_path)
        if hasattr(geom, "vertices"):
            points = np.asarray(geom.vertices)
        elif hasattr(geom, "points"):
            points = np.asarray(geom.points)
        else:
            raise ValueError(f"unsupported geometry type: {type(geom)}")
        if points.size == 0:
            raise ValueError("geometry has no points")
        timings["mesh_load_sec"] = float(time.perf_counter() - load_start)

        warnings: list[str] = []
        aabb_start = time.perf_counter()
        sx, sy, sz = _safe_bbox_extent(points)
        max_extent = max(sx, sy, sz)
        bbox_volume = sx * sy * sz
        bbox_longest = max_extent
        log_bbox_longest = float(np.log10(max(float(bbox_longest), 1e-12)))
        log_bbox_volume = float(np.log10(max(float(bbox_volume), 1e-12)))
        timings["aabb_sec"] = float(time.perf_counter() - aabb_start)
        surface_area = _safe_mesh_surface_area(geom)
        n_points_original = int(points.shape[0])
        is_triangle_mesh = hasattr(geom, "triangles") and hasattr(geom, "vertices")
        is_point_cloud_like = hasattr(geom, "points")
        has_triangles = False
        if hasattr(geom, "triangles"):
            try:
                triangles = np.asarray(geom.triangles)
                has_triangles = triangles.ndim == 2 and triangles.shape[0] > 0
            except Exception:
                has_triangles = False
        is_watertight = np.nan

        pc1_length, pc2_length, pc3_length = _pca_axis_lengths(points)
        flatness_ratio = (
            (pc3_length / max(pc1_length, EPS))
            if np.isfinite(pc1_length) and np.isfinite(pc3_length)
            else np.nan
        )
        thinness_ratio = (
            (pc2_length / max(pc1_length, EPS))
            if np.isfinite(pc1_length) and np.isfinite(pc2_length)
            else np.nan
        )
        is_planar_degenerate = int(bool(np.isfinite(flatness_ratio) and flatness_ratio < planar_degenerate_thresh))
        is_linear_degenerate = int(bool(np.isfinite(thinness_ratio) and thinness_ratio < linear_degenerate_thresh))
        # backward compatibility: deprecated aliases
        is_planar_like = is_planar_degenerate
        is_linear_like = is_linear_degenerate

        full_mode = str(size_compute_mode).strip().lower() == "full"
        use_hull_features = bool(full_mode and enable_hull_features)
        use_mesh_volume = bool(full_mode and enable_mesh_volume)

        mesh_volume = np.nan
        volume_failed = False
        if use_mesh_volume:
            mesh_volume_start = time.perf_counter()
            if hasattr(geom, "is_watertight"):
                try:
                    is_watertight = int(bool(geom.is_watertight()))
                except Exception:
                    warnings.append("watertight_check_failed")
                    is_watertight = 0
            else:
                is_watertight = 0
            mesh_volume, volume_failed = _safe_mesh_volume(geom, has_triangles=has_triangles, is_watertight=is_watertight)
            if not has_triangles:
                warnings.append("no_triangles")
            elif is_watertight != 1:
                warnings.append("non_watertight")
            if volume_failed:
                warnings.append("mesh_volume_failed")
            timings["mesh_volume_sec"] = float(time.perf_counter() - mesh_volume_start)
        equiv_diameter_mesh = _safe_equiv_diameter(mesh_volume)

        hull_start = time.perf_counter()
        convex_hull_volume = np.nan
        convex_hull_area = np.nan
        n_points_used_for_hull = 0
        hull_failed = False
        skip_hull = is_planar_degenerate == 1 or is_linear_degenerate == 1
        if use_hull_features and skip_hull:
            warnings.append("degenerate_shape")
        if use_hull_features and not is_triangle_mesh and not is_point_cloud_like:
            skip_hull = True
            warnings.append("unsupported_geometry_for_hull")
        if use_hull_features and not skip_hull:
            sampled_points = _sample_points_for_hull(
                points=points,
                max_points=max(int(hull_max_points), 16),
                sampling=hull_sampling,
                bbox_longest=bbox_longest,
                seed=abs(hash(specimen_id)) % (2**32),
            )
            n_points_used_for_hull = int(sampled_points.shape[0])
            try:
                import open3d as o3d

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(sampled_points.astype(np.float64))
                convex_hull_volume, convex_hull_area, hull_failed = _safe_convex_hull_metrics(
                    pcd,
                    joggle_inputs=hull_joggle_inputs,
                )
            except Exception:
                convex_hull_volume, convex_hull_area, hull_failed = np.nan, np.nan, True
            if hull_failed:
                warnings.append("hull_failed")
        equiv_diameter_hull = _safe_equiv_diameter(convex_hull_volume)
        timings["hull_sec"] = float(time.perf_counter() - hull_start)

        has_color, mean_r, mean_g, mean_b, std_r, std_g, std_b = _extract_colors_if_any(geom)

        size_metric = "bbox_longest"
        size_scalar = bbox_longest
        log_size_scalar = log_bbox_longest
        if full_mode:
            if use_mesh_volume and (
                is_watertight == 1
                and is_planar_degenerate != 1
                and is_linear_degenerate != 1
                and np.isfinite(mesh_volume)
                and mesh_volume > 0.0
            ):
                size_metric = "mesh_volume"
                size_scalar = mesh_volume
            elif use_hull_features and (
                is_planar_degenerate != 1
                and is_linear_degenerate != 1
                and np.isfinite(convex_hull_volume)
                and convex_hull_volume > 0.0
            ):
                size_metric = "hull_volume"
                size_scalar = convex_hull_volume
            elif np.isfinite(surface_area) and surface_area > 0.0:
                size_metric = "surface_area"
                size_scalar = surface_area
            log_size_scalar = float(np.log10(max(float(size_scalar), 1e-12)))

        if warnings:
            LOGGER.warning("metadata warnings for %s: %s", mesh_path, ",".join(sorted(set(warnings))))

        timings["total_sec"] = float(time.perf_counter() - total_start)
        return {
            "specimen_id": specimen_id,
            "source_path": str(mesh_path.as_posix()),
            "has_color": has_color,
            "size_x": sx,
            "size_y": sy,
            "size_z": sz,
            "max_extent": max_extent,
            "bbox_longest": bbox_longest,
            "bbox_volume": bbox_volume,
            "log_bbox_longest": log_bbox_longest,
            "log_bbox_volume": log_bbox_volume,
            "surface_area": surface_area,
            "convex_hull_volume": convex_hull_volume,
            "convex_hull_area": convex_hull_area,
            "is_watertight": is_watertight,
            "mesh_volume": mesh_volume,
            "equiv_diameter_hull": equiv_diameter_hull,
            "equiv_diameter_mesh": equiv_diameter_mesh,
            "aspect_xy": sx / max(sy, EPS),
            "aspect_xz": sx / max(sz, EPS),
            "aspect_yz": sy / max(sz, EPS),
            "pc1_length": pc1_length,
            "pc2_length": pc2_length,
            "pc3_length": pc3_length,
            "flatness_ratio": flatness_ratio,
            "thinness_ratio": thinness_ratio,
            "elongation_12": (pc1_length / max(pc2_length, EPS)) if np.isfinite(pc1_length) and np.isfinite(pc2_length) else np.nan,
            "elongation_13": (pc1_length / max(pc3_length, EPS)) if np.isfinite(pc1_length) and np.isfinite(pc3_length) else np.nan,
            "elongation_23": (pc2_length / max(pc3_length, EPS)) if np.isfinite(pc2_length) and np.isfinite(pc3_length) else np.nan,
            "size_metric": size_metric,
            "size_scalar": float(size_scalar),
            "log_size_scalar": log_size_scalar,
            "is_planar_degenerate": is_planar_degenerate,
            "is_linear_degenerate": is_linear_degenerate,
            "is_planar_like": is_planar_like,
            "is_linear_like": is_linear_like,
            "n_points_original": n_points_original,
            "n_points_used_for_hull": n_points_used_for_hull,
            "hull_failed": int(hull_failed),
            "volume_failed": int(volume_failed),
            "mean_r": mean_r,
            "mean_g": mean_g,
            "mean_b": mean_b,
            "std_r": std_r,
            "std_g": std_g,
            "std_b": std_b,
            "mesh_load_sec": timings["mesh_load_sec"],
            "aabb_sec": timings["aabb_sec"],
            "hull_sec": timings["hull_sec"],
            "mesh_volume_sec": timings["mesh_volume_sec"],
            "total_sec": timings["total_sec"],
        }
    except Exception as exc:
        raise MetadataExtractionError(str(exc), timings=timings, specimen_id=specimen_id, source_path=str(mesh_path.as_posix())) from exc
    finally:
        try:
            total_val = float(timings["total_sec"])
            has_total = np.isfinite(total_val)
        except Exception:
            has_total = False
        if not has_total:
            timings["total_sec"] = float(time.perf_counter() - total_start)


def resolve_use_color(mode: str, color_count: int, total_count: int, threshold: float = 0.2) -> bool:
    if mode == "off":
        return False
    if mode == "force":
        return True
    if total_count == 0:
        return False
    return (color_count / total_count) >= threshold


def build_feature_matrix(df: pd.DataFrame, use_color: bool, apply_log1p: bool = True) -> tuple[np.ndarray, list[str]]:
    base_cols = [
        "bbox_longest",
        "surface_area",
        "convex_hull_volume",
        "convex_hull_area",
        "equiv_diameter_hull",
        "aspect_xy",
        "aspect_xz",
        "aspect_yz",
        "elongation_12",
        "elongation_13",
        "elongation_23",
    ]
    color_cols = ["mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b"]

    mat = df[base_cols].astype(float).replace([np.inf, -np.inf], np.nan)
    mat = mat.fillna(mat.median(numeric_only=True)).fillna(0.0)
    if apply_log1p:
        for col in ["bbox_longest", "surface_area", "convex_hull_volume", "convex_hull_area", "equiv_diameter_hull"]:
            mat[col] = np.log1p(np.clip(mat[col].values, 0.0, None))

    used_cols = list(base_cols)

    if use_color:
        cdf = df[color_cols].astype(float).fillna(0.0)
        mat = pd.concat([mat, cdf], axis=1)
        used_cols.extend(color_cols)

    scaler = RobustScaler()
    scaled = scaler.fit_transform(mat.values)
    return scaled.astype(np.float64), used_cols


def run_hdbscan_labels(
    X: np.ndarray,
    min_cluster_size: int = 10,
    min_samples: int = 1,
    metric: str = "euclidean",
) -> tuple[np.ndarray, np.ndarray, str]:
    if X.shape[0] == 0:
        return np.array([], dtype=int), np.array([], dtype=float), "empty"
    if X.shape[0] == 1:
        return np.array([-1], dtype=int), np.array([0.0], dtype=float), "singleton"

    try:
        import hdbscan

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            cluster_selection_method="leaf",
        )
        labels = clusterer.fit_predict(X)
        probs = getattr(clusterer, "probabilities_", np.ones(X.shape[0], dtype=float))
        return labels.astype(int), np.asarray(probs, dtype=float), "hdbscan"
    except Exception as exc:
        LOGGER.warning("python-hdbscan unavailable/failed (%s). Trying sklearn fallback.", exc)

    try:
        from sklearn.cluster import HDBSCAN as SKHDBSCAN

        clusterer = SKHDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            cluster_selection_method="leaf",
        )
        labels = clusterer.fit_predict(X)
        probs = getattr(clusterer, "probabilities_", np.ones(X.shape[0], dtype=float))
        return labels.astype(int), np.asarray(probs, dtype=float), "sklearn_hdbscan"
    except Exception as exc:
        LOGGER.warning("sklearn HDBSCAN fallback failed (%s). assigning noise labels.", exc)

    return np.full(X.shape[0], -1, dtype=int), np.zeros(X.shape[0], dtype=float), "noise_fallback"


def save_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def prefilter_lookup(prefilter_df: pd.DataFrame) -> dict[str, dict[str, float | int | str]]:
    table: dict[str, dict[str, float | int | str]] = {}
    for row in prefilter_df.itertuples(index=False):
        sid = str(getattr(row, "specimen_id"))
        pregroup_id_raw = getattr(row, "pregroup_id")
        if pd.isna(pregroup_id_raw):
            pregroup_id: int | str = "-1"
        else:
            pregroup_id = str(pregroup_id_raw)
        table[sid] = {
            "pregroup_id": pregroup_id,
            "pregroup_prob": float(getattr(row, "pregroup_prob")),
        }
    return table
