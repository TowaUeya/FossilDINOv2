"""旧実験コード。本研究の主張には使わない。"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.fusion_common import SIZE_FEATURE_CHOICES
from src.utils.io import ensure_dir

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run systematic late-fusion (shape+size) ablation against shape_only.")
    p.add_argument("--emb", type=Path, required=True)
    p.add_argument("--ids", type=Path, required=True)
    p.add_argument("--prefilter_csv", type=Path, required=True)
    p.add_argument("--labels_csv", type=Path, required=True)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--rerank_topk", type=int, default=50)
    p.add_argument("--metric", choices=["cosine", "l2"], default="cosine")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--lambda_grid", type=str, default="0.02,0.05,0.1,0.2")
    p.add_argument(
        "--size_features",
        type=str,
        default="bbox_longest,log_bbox_longest,log_bbox_volume,apparent_size_proxy,apparent_size_volume_proxy",
    )
    p.add_argument("--size_penalty_modes", type=str, default="plain_distance,ratio_penalty,margin_gate")
    p.add_argument("--size_ratio_threshold", type=float, default=1.5)
    p.add_argument("--size_gate_penalty", type=float, default=0.05)
    p.add_argument("--size_eps", type=float, default=1e-8)
    p.add_argument("--prefilter_mode", choices=["off", "soft", "strict"], default="soft")
    p.add_argument("--expand_strategy", choices=["adjacent", "global"], default="adjacent")
    p.add_argument("--size_distance_norm", choices=["none", "zscore", "minmax", "robust"], default="robust")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _parse_lambda_grid(text: str) -> list[float]:
    vals = []
    for tok in [t.strip() for t in text.split(",") if t.strip()]:
        vals.append(float(tok))
    if not vals:
        raise ValueError("--lambda_grid is empty")
    return vals


def _parse_size_features(text: str) -> list[str]:
    feats = [t.strip() for t in text.split(",") if t.strip()]
    if not feats:
        raise ValueError("--size_features is empty")
    unknown = [f for f in feats if f not in SIZE_FEATURE_CHOICES]
    if unknown:
        raise ValueError(f"Unknown size features: {unknown}, allowed={SIZE_FEATURE_CHOICES}")
    return feats


def _run_cmd(cmd: list[str]) -> None:
    LOGGER.info("RUN: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _search_cmd(
    args: argparse.Namespace,
    out_dir: Path,
    lambda_size: float,
    size_feature: str,
    size_penalty_mode: str,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "src.search_all_fusion",
        "--emb",
        str(args.emb),
        "--ids",
        str(args.ids),
        "--prefilter_csv",
        str(args.prefilter_csv),
        "--topk",
        str(args.topk),
        "--rerank_topk",
        str(args.rerank_topk),
        "--lambda_size",
        str(lambda_size),
        "--lambda_tex",
        "0.0",
        "--size_feature",
        size_feature,
        "--size_penalty_mode",
        size_penalty_mode,
        "--size_ratio_threshold",
        str(args.size_ratio_threshold),
        "--size_gate_penalty",
        str(args.size_gate_penalty),
        "--size_eps",
        str(args.size_eps),
        "--size_distance_norm",
        args.size_distance_norm,
        "--prefilter_mode",
        args.prefilter_mode,
        "--expand_strategy",
        args.expand_strategy,
        "--metric",
        args.metric,
        "--out",
        str(out_dir),
        "--seed",
        str(args.seed),
    ]


def _evaluate_pair(shape_csv: Path, run_csv: Path, labels_csv: Path, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dir(out_dir)
    combined_csv = out_dir / "knn_all_fusion_compare.csv"
    _run_cmd(
        [
            sys.executable,
            "-m",
            "src.combine_fusion_runs",
            "--shape_only_csv",
            str(shape_csv),
            "--shape_size_csv",
            str(run_csv),
            "--out",
            str(combined_csv),
        ]
    )
    _run_cmd(
        [
            sys.executable,
            "-m",
            "src.evaluate_fusion",
            "--knn_csv",
            str(combined_csv),
            "--labels_csv",
            str(labels_csv),
            "--out",
            str(out_dir),
            "--strict_query_set_check",
        ]
    )
    overall = pd.read_csv(out_dir / "fusion_eval_overall.csv")
    per_cat = pd.read_csv(out_dir / "fusion_eval_per_category.csv")
    return overall, per_cat


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    ensure_dir(args.out)

    lambdas = _parse_lambda_grid(args.lambda_grid)
    size_features = _parse_size_features(args.size_features)
    size_penalty_modes = [t.strip() for t in str(args.size_penalty_modes).split(",") if t.strip()]
    allowed_penalty_modes = {"plain_distance", "ratio_penalty", "margin_gate"}
    unknown_modes = [m for m in size_penalty_modes if m not in allowed_penalty_modes]
    if unknown_modes:
        raise ValueError(f"Unknown --size_penalty_modes: {unknown_modes}, allowed={sorted(allowed_penalty_modes)}")

    runs_root = args.out / "runs"
    eval_root = args.out / "eval"
    ensure_dir(runs_root)
    ensure_dir(eval_root)

    shape_run_dir = runs_root / "shape_only"
    _run_cmd(_search_cmd(args, shape_run_dir, lambda_size=0.0, size_feature="log_size_scalar", size_penalty_mode="ratio_penalty"))
    shape_csv = shape_run_dir / "knn_all_fusion.csv"
    if not shape_csv.exists():
        raise FileNotFoundError(f"shape_only output not found: {shape_csv}")

    records: list[dict[str, object]] = []
    per_cat_records: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    baseline_mean: float | None = None

    for sf in size_features:
        for spm in size_penalty_modes:
            for lam in lambdas:
                tag = f"size_{sf}__mode_{spm}__lambda_{lam:g}"
                run_dir = runs_root / tag
                eval_dir = eval_root / tag
                try:
                    _run_cmd(_search_cmd(args, run_dir, lambda_size=lam, size_feature=sf, size_penalty_mode=spm))
                    run_csv = run_dir / "knn_all_fusion.csv"
                    overall, per_cat = _evaluate_pair(shape_csv, run_csv, args.labels_csv, eval_dir)
                except Exception as e:  # noqa: BLE001
                    LOGGER.exception("Failed config size_feature=%s size_penalty_mode=%s lambda_size=%s", sf, spm, lam)
                    failures.append({"size_feature": sf, "size_penalty_mode": spm, "lambda_size": str(lam), "error": str(e)})
                    continue

                for _, row in overall.iterrows():
                    fm = str(row["fusion_mode"])
                    rec = {
                        "fusion_mode": fm,
                        "size_feature": sf if fm == "shape+size" else "baseline",
                        "size_penalty_mode": spm if fm == "shape+size" else "baseline",
                        "lambda_size": float(lam) if fm == "shape+size" else 0.0,
                        "size_ratio_threshold": float(args.size_ratio_threshold) if fm == "shape+size" else np.nan,
                        "size_gate_penalty": float(args.size_gate_penalty) if fm == "shape+size" else np.nan,
                        "n_queries": int(row["n_queries"]),
                        "mean_match_rate": float(row["mean_match_rate"]),
                        "std_match_rate": float(row["std_match_rate"]) if pd.notna(row["std_match_rate"]) else 0.0,
                    }
                    records.append(rec)
                    if fm == "shape_only" and baseline_mean is None:
                        baseline_mean = float(row["mean_match_rate"])

                for _, row in per_cat.iterrows():
                    fm = str(row["fusion_mode"])
                    per_cat_records.append(
                        {
                            "fusion_mode": fm,
                            "size_feature": sf if fm == "shape+size" else "baseline",
                            "size_penalty_mode": spm if fm == "shape+size" else "baseline",
                            "lambda_size": float(lam) if fm == "shape+size" else 0.0,
                            "size_ratio_threshold": float(args.size_ratio_threshold) if fm == "shape+size" else np.nan,
                            "size_gate_penalty": float(args.size_gate_penalty) if fm == "shape+size" else np.nan,
                            "category": str(row["query_label"]),
                            "mean_match_rate": float(row["mean_match_rate"]),
                        }
                    )

    overall_df = pd.DataFrame(records)
    per_cat_df = pd.DataFrame(per_cat_records)
    if overall_df.empty:
        raise RuntimeError("No successful ablation runs. Check logs.")

    overall_df = (
        overall_df.groupby(
            ["fusion_mode", "size_feature", "size_penalty_mode", "lambda_size", "size_ratio_threshold", "size_gate_penalty"],
            as_index=False,
        )
        .agg(n_queries=("n_queries", "max"), mean_match_rate=("mean_match_rate", "mean"), std_match_rate=("std_match_rate", "mean"))
        .sort_values(["fusion_mode", "mean_match_rate"], ascending=[True, False], kind="stable")
    )
    per_cat_df = (
        per_cat_df.groupby(
            ["fusion_mode", "size_feature", "size_penalty_mode", "lambda_size", "size_ratio_threshold", "size_gate_penalty", "category"],
            as_index=False,
        )
        .agg(n_queries=("mean_match_rate", "size"), mean_match_rate=("mean_match_rate", "mean"))
        .sort_values(["fusion_mode", "category", "mean_match_rate"], ascending=[True, True, False], kind="stable")
    )

    if baseline_mean is None:
        base_rows = overall_df[overall_df["fusion_mode"] == "shape_only"]
        baseline_mean = float(base_rows["mean_match_rate"].mean()) if not base_rows.empty else 0.0

    shape_size_df = overall_df[overall_df["fusion_mode"] == "shape+size"].copy()
    shape_size_df["delta_mean_match_rate"] = shape_size_df["mean_match_rate"] - float(baseline_mean)
    shape_size_df.sort_values("delta_mean_match_rate", ascending=False, inplace=True)

    per_cat_base = per_cat_df[per_cat_df["fusion_mode"] == "shape_only"][["category", "mean_match_rate"]].rename(
        columns={"mean_match_rate": "baseline_mean_match_rate"}
    )
    delta_by_cat = (
        per_cat_df[per_cat_df["fusion_mode"] == "shape+size"]
        .merge(per_cat_base, on="category", how="left")
        .assign(delta_mean_match_rate=lambda d: d["mean_match_rate"] - d["baseline_mean_match_rate"])
    )
    delta_by_cat.to_csv(args.out / "delta_vs_shape_only_per_category.csv", index=False)

    rank_df = shape_size_df.merge(
        delta_by_cat.groupby(["size_feature", "size_penalty_mode", "lambda_size"], as_index=False).agg(
            n_category_worse=("delta_mean_match_rate", lambda s: int((s < 0).sum()))
        ),
        on=["size_feature", "size_penalty_mode", "lambda_size"],
        how="left",
    )
    rank_df = rank_df.sort_values(
        ["mean_match_rate", "delta_mean_match_rate", "n_category_worse"],
        ascending=[False, False, True],
        kind="stable",
    )
    best = rank_df.iloc[0].to_dict() if not rank_df.empty else {}
    best_json = {
        "baseline_shape_only_mean_match_rate": float(baseline_mean),
        "best_config": best,
        "n_successful_shape_size_runs": int(len(shape_size_df)),
        "n_failed_runs": int(len(failures)),
        "failures": failures,
    }

    overall_df.to_csv(args.out / "fusion_ablation_overall.csv", index=False)
    per_cat_df.to_csv(args.out / "fusion_ablation_per_category.csv", index=False)
    shape_size_df[["size_feature", "size_penalty_mode", "lambda_size", "size_ratio_threshold", "size_gate_penalty", "delta_mean_match_rate"]].to_csv(
        args.out / "delta_vs_shape_only.csv", index=False
    )
    (args.out / "best_fusion_config.json").write_text(json.dumps(best_json, ensure_ascii=False, indent=2), encoding="utf-8")

    LOGGER.info("Saved: %s", args.out / "fusion_ablation_overall.csv")
    LOGGER.info("Saved: %s", args.out / "fusion_ablation_per_category.csv")
    LOGGER.info("Saved: %s", args.out / "delta_vs_shape_only.csv")
    LOGGER.info("Saved: %s", args.out / "best_fusion_config.json")


if __name__ == "__main__":
    main()
