from __future__ import annotations

import argparse
import os
import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from src.prefilter_common import (
    MetadataExtractionError,
    build_feature_matrix,
    extract_metadata_record,
    list_source_files,
    resolve_use_color,
    run_hdbscan_labels,
    save_yaml,
)
from src.utils.io import ensure_dir, set_seed, setup_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract optional prefilter metadata (size/color) from raw 3D files and build coarse pregroups. "
            "This is independent from the shape embedding pipeline."
        )
    )
    parser.add_argument("--in", dest="input_dir", type=Path, required=True, help="Input mesh directory")
    parser.add_argument("--out", type=Path, default=Path("data/prefilter"), help="Output directory")
    parser.add_argument("--use_color", choices=["auto", "off", "force"], default="off")
    parser.add_argument(
        "--grouping_method",
        choices=["physical_bins", "hdbscan"],
        default="physical_bins",
    )
    parser.add_argument("--min_cluster_size", type=int, default=10)
    parser.add_argument("--min_samples", type=int, default=1)
    parser.add_argument("--size_source", choices=["hull", "mesh", "auto"], default="auto")
    parser.add_argument("--size_compute_mode", choices=["fast", "full"], default="fast")
    parser.add_argument("--enable_hull_features", action="store_true")
    parser.add_argument("--enable_mesh_volume", action="store_true")
    parser.add_argument("--volume_bins", type=int, default=5)
    parser.add_argument("--hull_max_points", type=int, default=5000)
    parser.add_argument("--hull_sampling", choices=["random", "voxel"], default="random")
    parser.add_argument("--hull_joggle_inputs", action="store_true")
    parser.add_argument("--shape_split", choices=["none", "aspect_xy", "elongation_12", "flatness_ratio"], default="none")
    parser.add_argument("--planar_degenerate_thresh", type=float, default=0.02)
    parser.add_argument("--linear_degenerate_thresh", type=float, default=0.02)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--progress_every", type=int, default=5)
    parser.add_argument(
        "--slow_log_seconds",
        type=float,
        default=0.0,
        help="If > 0, completion logs slower than this threshold are emitted as WARNING.",
    )
    parser.add_argument(
        "--slow_threshold_sec",
        type=float,
        default=60.0,
        help="If total specimen processing time >= this threshold, timing logs are emitted as WARNING.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _extract_one(
    fp_str: str,
    root_str: str,
    hull_max_points: int,
    hull_sampling: str,
    hull_joggle_inputs: bool,
    planar_degenerate_thresh: float,
    linear_degenerate_thresh: float,
    size_compute_mode: str,
    enable_hull_features: bool,
    enable_mesh_volume: bool,
) -> dict:
    try:
        return {
            "ok": True,
            "record": extract_metadata_record(
                Path(fp_str),
                Path(root_str),
                hull_max_points=hull_max_points,
                hull_sampling=hull_sampling,
                hull_joggle_inputs=hull_joggle_inputs,
                planar_degenerate_thresh=planar_degenerate_thresh,
                linear_degenerate_thresh=linear_degenerate_thresh,
                size_compute_mode=size_compute_mode,
                enable_hull_features=enable_hull_features,
                enable_mesh_volume=enable_mesh_volume,
            ),
            "error": None,
        }
    except MetadataExtractionError as exc:
        return {
            "ok": False,
            "record": None,
            "error": {
                "specimen_id": exc.specimen_id,
                "source_path": exc.source_path,
                "error": str(exc),
                **exc.timings,
            },
        }
    except Exception as exc:
        return {
            "ok": False,
            "record": None,
            "error": {"source_path": fp_str, "error": str(exc)},
        }


def _fmt_sec(value: float | int | None) -> str:
    try:
        v = float(value)
    except Exception:
        return "nan"
    if np.isnan(v):
        return "nan"
    return f"{v:.3f}s"


def _resolve_size_volume(df: pd.DataFrame, size_source: str) -> pd.Series:
    bbox = df["bbox_volume"].astype(float)
    hull = df["convex_hull_volume"].astype(float)
    mesh = df["mesh_volume"].astype(float)
    is_watertight = df["is_watertight"].astype(float).fillna(0.0) >= 0.5

    if size_source == "hull":
        return hull
    if size_source == "mesh":
        return mesh.where(mesh > 0.0, np.nan).fillna(hull)

    # auto: default は AABB ベースを優先し、full 計算時のみ hull/mesh を補完で利用
    valid_bbox = np.isfinite(bbox) & (bbox > 0.0)
    valid_mesh = is_watertight & np.isfinite(mesh) & (mesh > 0.0)
    valid_hull = np.isfinite(hull) & (hull > 0.0)
    return bbox.where(valid_bbox, np.nan).fillna(hull.where(valid_hull, np.nan)).fillna(mesh.where(valid_mesh, np.nan))


def _with_log_volume(df: pd.DataFrame, size_source: str) -> pd.DataFrame:
    out = df.copy()
    out["size_volume"] = _resolve_size_volume(out, size_source=size_source).astype(float)
    safe_vol = out["size_volume"].replace([np.inf, -np.inf], np.nan)
    median_vol = float(safe_vol.dropna().median()) if safe_vol.notna().any() else 1.0
    safe_vol = safe_vol.fillna(max(median_vol, 1e-12))
    out["log_size_volume"] = out["log_bbox_volume"].astype(float).replace([np.inf, -np.inf], np.nan)
    fallback_log_vol = pd.Series(
        np.log10(np.clip(safe_vol.to_numpy(dtype=float), 1e-12, None)),
        index=out.index,
    )
    out["log_size_volume"] = out["log_size_volume"].fillna(fallback_log_vol)
    return out


def _physical_groups(df: pd.DataFrame, volume_bins: int, shape_split: str, size_source: str) -> pd.DataFrame:
    out = _with_log_volume(df, size_source=size_source)
    n_bins = max(2, int(volume_bins))
    vol_bins = pd.qcut(out["log_size_volume"], q=n_bins, labels=False, duplicates="drop")
    vol_bins = vol_bins.fillna(0).astype(int)

    out["volume_bin"] = vol_bins
    out["shape_flag"] = "none"
    out["pregroup_prob"] = 1.0
    out["grouping_method"] = "physical_bins"

    group_names = [f"vol_q{v}" for v in vol_bins.tolist()]
    if shape_split != "none":
        shape_col = shape_split
        shape_values = out[shape_col].astype(float).replace([np.inf, -np.inf], np.nan)
        threshold = float(shape_values.dropna().median()) if shape_values.notna().any() else 0.0
        shape_values = shape_values.fillna(threshold)
        flags = np.where(shape_values.values <= threshold, "low", "high")
        out["shape_flag"] = flags
        if shape_col == "aspect_xy":
            split_prefix = "aspect"
        elif shape_col == "elongation_12":
            split_prefix = "elong"
        else:
            split_prefix = "flat"
        group_names = [f"vol_q{v}__{split_prefix}_{f}" for v, f in zip(vol_bins.tolist(), flags)]

    out["pregroup_id"] = group_names
    return out


def main() -> None:
    args = parse_args()
    setup_logging()
    set_seed(args.seed)
    ensure_dir(args.out)

    files = list_source_files(args.input_dir)
    LOGGER.info("Found %d source files", len(files))
    jobs = max(1, int(args.jobs))
    progress_every = max(1, int(args.progress_every))
    slow_log_seconds = max(0.0, float(args.slow_log_seconds))
    slow_threshold_sec = max(0.0, float(args.slow_threshold_sec))
    if slow_log_seconds > 0:
        slow_threshold_sec = slow_log_seconds
    LOGGER.info("Using jobs=%d", jobs)
    LOGGER.info("progress_every=%d", progress_every)
    LOGGER.info("slow_threshold_sec=%.2f", slow_threshold_sec)

    records: list[dict] = []
    errors: list[dict] = []

    done = 0
    if jobs == 1:
        for fp in files:
            start_time = time.perf_counter()
            LOGGER.info("Start: %s", fp)
            try:
                rec = extract_metadata_record(
                    fp,
                    args.input_dir,
                    hull_max_points=args.hull_max_points,
                    hull_sampling=args.hull_sampling,
                    hull_joggle_inputs=args.hull_joggle_inputs,
                    planar_degenerate_thresh=args.planar_degenerate_thresh,
                    linear_degenerate_thresh=args.linear_degenerate_thresh,
                    size_compute_mode=args.size_compute_mode,
                    enable_hull_features=bool(args.enable_hull_features),
                    enable_mesh_volume=bool(args.enable_mesh_volume),
                )
                records.append(rec)
                specimen_id = rec.get("specimen_id", fp.as_posix())
                total_sec = float(rec.get("total_sec", np.nan))
                done_logger = LOGGER.warning if slow_threshold_sec > 0 and np.isfinite(total_sec) and total_sec >= slow_threshold_sec else LOGGER.info
                done_logger(
                    "[%d/%d] specimen_id=%s total=%s load=%s aabb=%s hull=%s mesh_volume=%s",
                    done + 1,
                    len(files),
                    specimen_id,
                    _fmt_sec(total_sec),
                    _fmt_sec(rec.get("mesh_load_sec")),
                    _fmt_sec(rec.get("aabb_sec")),
                    _fmt_sec(rec.get("hull_sec")),
                    _fmt_sec(rec.get("mesh_volume_sec")),
                )
                elapsed = time.perf_counter() - start_time
                LOGGER.debug("Done wall-clock: %s (%.3fs)", fp, elapsed)
            except MetadataExtractionError as exc:
                elapsed = time.perf_counter() - start_time
                err = {
                    "specimen_id": exc.specimen_id,
                    "source_path": exc.source_path,
                    "error": str(exc),
                    **exc.timings,
                }
                errors.append(err)
                total_sec = float(exc.timings.get("total_sec", np.nan))
                done_logger = LOGGER.warning if slow_threshold_sec > 0 and np.isfinite(total_sec) and total_sec >= slow_threshold_sec else LOGGER.info
                done_logger(
                    "[%d/%d] specimen_id=%s total=%s load=%s aabb=%s hull=%s mesh_volume=%s (FAILED: %s)",
                    done + 1,
                    len(files),
                    exc.specimen_id,
                    _fmt_sec(total_sec),
                    _fmt_sec(exc.timings.get("mesh_load_sec")),
                    _fmt_sec(exc.timings.get("aabb_sec")),
                    _fmt_sec(exc.timings.get("hull_sec")),
                    _fmt_sec(exc.timings.get("mesh_volume_sec")),
                    exc,
                )
                LOGGER.debug("Failed wall-clock: %s (%.3fs)", fp, elapsed)
            except Exception as exc:
                elapsed = time.perf_counter() - start_time
                LOGGER.warning("Skip failed file: %s (%s) [%.2fs]", fp, exc, elapsed)
                errors.append({"source_path": str(fp), "error": str(exc)})
            finally:
                done += 1
                if done % progress_every == 0 or done == len(files):
                    LOGGER.info("Progress: %d / %d", done, len(files))
    else:
        max_default_jobs = max(1, min(4, (os.cpu_count() or 2) - 1))
        worker_jobs = min(jobs, max_default_jobs) if args.jobs <= 0 else jobs
        with ProcessPoolExecutor(max_workers=worker_jobs) as ex:
            future_to_fp = {}
            future_start = {}
            for fp in files:
                fut = ex.submit(
                    _extract_one,
                    str(fp),
                    str(args.input_dir),
                    int(args.hull_max_points),
                    str(args.hull_sampling),
                    bool(args.hull_joggle_inputs),
                    float(args.planar_degenerate_thresh),
                    float(args.linear_degenerate_thresh),
                    str(args.size_compute_mode),
                    bool(args.enable_hull_features),
                    bool(args.enable_mesh_volume),
                )
                future_to_fp[fut] = fp
                future_start[fut] = time.perf_counter()
                LOGGER.info("Start: %s", fp)

            for fut in as_completed(future_to_fp):
                fp = future_to_fp[fut]
                elapsed = time.perf_counter() - future_start[fut]
                try:
                    result = fut.result()
                    if result.get("ok", False):
                        rec = result["record"]
                        records.append(rec)
                        specimen_id = rec.get("specimen_id", fp.as_posix())
                        total_sec = float(rec.get("total_sec", np.nan))
                        done_logger = LOGGER.warning if slow_threshold_sec > 0 and np.isfinite(total_sec) and total_sec >= slow_threshold_sec else LOGGER.info
                        done_logger(
                            "[%d/%d] specimen_id=%s total=%s load=%s aabb=%s hull=%s mesh_volume=%s",
                            done + 1,
                            len(files),
                            specimen_id,
                            _fmt_sec(total_sec),
                            _fmt_sec(rec.get("mesh_load_sec")),
                            _fmt_sec(rec.get("aabb_sec")),
                            _fmt_sec(rec.get("hull_sec")),
                            _fmt_sec(rec.get("mesh_volume_sec")),
                        )
                    else:
                        err = result.get("error") or {"source_path": str(fp), "error": "unknown error"}
                        errors.append(err)
                        total_sec = float(err.get("total_sec", np.nan))
                        specimen_id = str(err.get("specimen_id", fp.as_posix()))
                        done_logger = LOGGER.warning if slow_threshold_sec > 0 and np.isfinite(total_sec) and total_sec >= slow_threshold_sec else LOGGER.info
                        done_logger(
                            "[%d/%d] specimen_id=%s total=%s load=%s aabb=%s hull=%s mesh_volume=%s (FAILED: %s)",
                            done + 1,
                            len(files),
                            specimen_id,
                            _fmt_sec(total_sec),
                            _fmt_sec(err.get("mesh_load_sec")),
                            _fmt_sec(err.get("aabb_sec")),
                            _fmt_sec(err.get("hull_sec")),
                            _fmt_sec(err.get("mesh_volume_sec")),
                            err.get("error", "unknown error"),
                        )
                except Exception as exc:
                    LOGGER.warning("Skip failed file: %s (%s) [%.2fs]", fp, exc, elapsed)
                    errors.append({"source_path": str(fp), "error": str(exc)})
                finally:
                    done += 1
                    if done % progress_every == 0 or done == len(files):
                        LOGGER.info("Progress: %d / %d", done, len(files))

    metadata_csv = args.out / "metadata_features.csv"
    summary_json = args.out / "pregroup_summary.json"
    counts_csv = args.out / "pregroup_counts.csv"
    config_yaml = args.out / "run_config.yaml"

    if not records:
        empty_columns = [
            "specimen_id",
            "source_path",
            "has_color",
            "size_x",
            "size_y",
            "size_z",
            "max_extent",
            "bbox_longest",
            "bbox_volume",
            "log_bbox_longest",
            "log_bbox_volume",
            "surface_area",
            "convex_hull_volume",
            "convex_hull_area",
            "is_watertight",
            "mesh_volume",
            "equiv_diameter_hull",
            "equiv_diameter_mesh",
            "aspect_xy",
            "aspect_xz",
            "aspect_yz",
            "pc1_length",
            "pc2_length",
            "pc3_length",
            "elongation_12",
            "elongation_13",
            "elongation_23",
            "flatness_ratio",
            "thinness_ratio",
            "size_metric",
            "size_scalar",
            "log_size_scalar",
            "is_planar_degenerate",
            "is_linear_degenerate",
            "is_planar_like",
            "is_linear_like",
            "n_points_original",
            "n_points_used_for_hull",
            "hull_failed",
            "volume_failed",
            "mean_r",
            "mean_g",
            "mean_b",
            "std_r",
            "std_g",
            "std_b",
            "size_volume",
            "log_size_volume",
            "volume_bin",
            "shape_flag",
            "grouping_method",
            "pregroup_id",
            "pregroup_prob",
            "mesh_load_sec",
            "aabb_sec",
            "hull_sec",
            "mesh_volume_sec",
            "total_sec",
        ]
        pd.DataFrame(columns=empty_columns).to_csv(metadata_csv, index=False)
        summary = {
            "n_input_files": len(files),
            "n_success": 0,
            "n_skipped": len(errors),
            "jobs": jobs,
            "progress_every": progress_every,
            "errors": errors,
            "use_color_mode": args.use_color,
            "use_color_resolved": False,
            "grouping_method": args.grouping_method,
            "grouping_backend": "none",
        }
        summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        empty_cfg = vars(args).copy()
        empty_cfg["input_dir"] = str(args.input_dir)
        empty_cfg["out"] = str(args.out)
        save_yaml(config_yaml, empty_cfg)
        LOGGER.info("No valid files. Saved empty outputs to %s", args.out)
        return

    df = pd.DataFrame(records).sort_values("specimen_id").reset_index(drop=True)
    color_count = int((df["has_color"] == 1).sum())
    resolved_use_color = resolve_use_color(args.use_color, color_count, len(df))

    if args.grouping_method == "hdbscan":
        X, used_cols = build_feature_matrix(df, use_color=resolved_use_color, apply_log1p=True)
        labels, probs, backend = run_hdbscan_labels(
            X,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            metric="euclidean",
        )
        df = _with_log_volume(df, size_source=args.size_source)
        df["volume_bin"] = np.nan
        df["shape_flag"] = "none"
        df["grouping_method"] = "hdbscan"
        df["pregroup_id"] = labels.astype(str)
        df["pregroup_prob"] = probs
    else:
        used_cols = ["bbox_longest", "bbox_volume", "log_bbox_longest", "log_bbox_volume", "aspect_xy", "elongation_12"]
        backend = "physical_bins"
        df = _physical_groups(df, volume_bins=args.volume_bins, shape_split=args.shape_split, size_source=args.size_source)

    timing_cols = ["mesh_load_sec", "aabb_sec", "hull_sec", "mesh_volume_sec", "total_sec"]
    ordered_cols = [c for c in df.columns if c not in timing_cols] + [c for c in timing_cols if c in df.columns]
    df = df[ordered_cols]
    df.to_csv(metadata_csv, index=False)

    counts = (
        df.groupby("pregroup_id", dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    counts.to_csv(counts_csv, index=False)

    n_noise = int((df["pregroup_id"].astype(str) == "-1").sum())
    summary = {
        "n_input_files": len(files),
        "n_success": int(len(df)),
        "n_skipped": len(errors),
        "jobs": jobs,
        "progress_every": progress_every,
        "n_color": color_count,
        "color_ratio": float(color_count / len(df)),
        "use_color_mode": args.use_color,
        "use_color_resolved": bool(resolved_use_color),
        "used_feature_columns": used_cols,
        "grouping_method": args.grouping_method,
        "grouping_backend": backend,
        "size_source": args.size_source,
        "shape_split": args.shape_split,
        "planar_degenerate_thresh": float(args.planar_degenerate_thresh),
        "linear_degenerate_thresh": float(args.linear_degenerate_thresh),
        "n_pregroups_excluding_noise": int(len(set(df["pregroup_id"].astype(str).tolist()) - {"-1"})),
        "n_noise": n_noise,
        "noise_ratio": float(n_noise / len(df)),
        "n_planar_degenerate": int(df["is_planar_degenerate"].astype(int).sum()),
        "n_linear_degenerate": int(df["is_linear_degenerate"].astype(int).sum()),
        "n_planar_like": int(df["is_planar_like"].astype(int).sum()),
        "n_linear_like": int(df["is_linear_like"].astype(int).sum()),
        "n_flatness_lt_0_02": int((df["flatness_ratio"].astype(float) < 0.02).sum()),
        "n_flatness_lt_0_05": int((df["flatness_ratio"].astype(float) < 0.05).sum()),
        "n_flatness_lt_0_10": int((df["flatness_ratio"].astype(float) < 0.10).sum()),
        "n_thinness_lt_0_02": int((df["thinness_ratio"].astype(float) < 0.02).sum()),
        "n_thinness_lt_0_05": int((df["thinness_ratio"].astype(float) < 0.05).sum()),
        "n_thinness_lt_0_10": int((df["thinness_ratio"].astype(float) < 0.10).sum()),
        "size_metric_counts": df["size_metric"].astype(str).value_counts(dropna=False).to_dict(),
        "n_hull_failed": int(df["hull_failed"].astype(int).sum()),
        "n_volume_failed": int(df["volume_failed"].astype(int).sum()),
        "mean_points_used_for_hull": float(df["n_points_used_for_hull"].astype(float).mean()),
        "mean_mesh_load_sec": float(df["mesh_load_sec"].astype(float).mean()),
        "mean_aabb_sec": float(df["aabb_sec"].astype(float).mean()),
        "mean_hull_sec": float(df["hull_sec"].astype(float).mean()),
        "mean_mesh_volume_sec": float(df["mesh_volume_sec"].astype(float).mean()),
        "mean_total_sec": float(df["total_sec"].astype(float).mean()),
        "max_total_sec": float(df["total_sec"].astype(float).max()),
        "slowest_specimen_id": (
            str(df.loc[df["total_sec"].astype(float).idxmax(), "specimen_id"])
            if len(df) > 0 and df["total_sec"].astype(float).notna().any()
            else None
        ),
        "n_slow_specimens": int((df["total_sec"].astype(float) >= slow_threshold_sec).sum()) if slow_threshold_sec > 0 else 0,
        "slow_threshold_sec": slow_threshold_sec,
        "errors": errors,
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    run_config = vars(args).copy()
    run_config["input_dir"] = str(args.input_dir)
    run_config["out"] = str(args.out)
    save_yaml(config_yaml, run_config)

    LOGGER.info("Saved metadata CSV: %s", metadata_csv)
    LOGGER.info("Saved summary JSON: %s", summary_json)
    LOGGER.info("Saved pregroup counts CSV: %s", counts_csv)
    LOGGER.info("Saved run config YAML: %s", config_yaml)


if __name__ == "__main__":
    main()
