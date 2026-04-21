from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src.utils.io import ensure_dir

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate shape-first / late-fusion retrieval quality by label match rate.")
    p.add_argument("--knn_csv", type=Path, required=True)
    p.add_argument("--labels_csv", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--topk", type=int, default=None, help="Optional top-k cutoff per query/mode (uses rank columns when available).")
    p.add_argument(
        "--strict_query_set_check",
        action="store_true",
        help="Warn when per-mode query sets are inconsistent after top-k selection.",
    )
    return p.parse_args()


def _label_col(df: pd.DataFrame) -> str:
    for col in ["Fossil category", "label", "category"]:
        if col in df.columns:
            return col
    raise ValueError("labels_csv must contain one of: Fossil category / label / category")


def _normalize_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def _normalize_mode_value(value: object) -> str:
    s = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "shapeonly": "shape_only",
        "shape_only": "shape_only",
        "shape_size": "shape+size",
        "shape+size": "shape+size",
        "shape_texture": "shape+texture",
        "shape+texture": "shape+texture",
        "shape_size_texture": "shape+size+texture",
        "shape+size+texture": "shape+size+texture",
    }
    return aliases.get(s, s)


def _attach_fusion_mode(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "fusion_mode" in out.columns:
        out["fusion_mode"] = out["fusion_mode"].map(_normalize_mode_value)
        return out

    # `search_all_fusion` の単一実行出力では mode 列がないため、CSV 全体で 1 つの mode を付与する。
    has_size_signal = bool(out["distance_size"].astype(float).abs().gt(0).any())
    has_tex_signal = bool(out["distance_tex"].astype(float).abs().gt(0).any())
    if has_size_signal and has_tex_signal:
        mode = "shape+size+texture"
    elif has_size_signal:
        mode = "shape+size"
    elif has_tex_signal:
        mode = "shape+texture"
    else:
        mode = "shape_only"

    out["fusion_mode"] = mode
    LOGGER.warning("fusion_mode 列が無いため、CSV 全体を '%s' として評価します。", mode)
    return out


def _log_mode_stats(stage: str, df: pd.DataFrame) -> None:
    if df.empty:
        LOGGER.info("[%s] rows=0", stage)
        return
    mode_col = "fusion_mode" if "fusion_mode" in df.columns else None
    query_col = "query_id" if "query_id" in df.columns else None
    LOGGER.info("[%s] rows=%d columns=%s", stage, len(df), list(df.columns))
    if mode_col:
        mode_rows = df.groupby(mode_col, dropna=False).size().to_dict()
        if query_col:
            mode_queries = df.groupby(mode_col, dropna=False)[query_col].nunique(dropna=False).to_dict()
        else:
            mode_queries = {}
        LOGGER.info("[%s] rows_by_mode=%s", stage, mode_rows)
        if query_col:
            LOGGER.info("[%s] unique_queries_by_mode=%s", stage, mode_queries)


def _apply_topk(df: pd.DataFrame, topk: int | None) -> pd.DataFrame:
    if topk is None:
        return df
    out = df.copy()
    rank_cols = [c for c in ["rank_final", "rank", "neighbor_rank"] if c in out.columns]
    if rank_cols:
        rank_col = rank_cols[0]
        out = out[out[rank_col].astype(float) <= float(topk)].copy()
        return out
    out = (
        out.sort_values(["fusion_mode", "query_id"], kind="stable")
        .groupby(["fusion_mode", "query_id"], dropna=False, as_index=False, group_keys=False)
        .head(topk)
    )
    return out


def _warn_query_set_mismatch(df: pd.DataFrame) -> None:
    query_sets = {
        mode: set(group["query_id"].dropna().astype(str).tolist())
        for mode, group in df.groupby("fusion_mode", dropna=False)
    }
    if len(query_sets) <= 1:
        return
    sizes = {k: len(v) for k, v in query_sets.items()}
    if len(set(sizes.values())) > 1:
        LOGGER.warning("mode 間で query 数が不一致です: %s", sizes)
    modes = list(query_sets.keys())
    baseline = query_sets[modes[0]]
    for m in modes[1:]:
        if query_sets[m] != baseline:
            only_base = len(baseline - query_sets[m])
            only_m = len(query_sets[m] - baseline)
            LOGGER.warning("query 集合不一致: %s vs %s (base_only=%d, mode_only=%d)", modes[0], m, only_base, only_m)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    ensure_dir(args.out)

    knn = pd.read_csv(args.knn_csv)
    _log_mode_stats("knn_loaded_raw", knn)
    labels = pd.read_csv(args.labels_csv)
    LOGGER.info(
        "[labels_loaded] rows=%d unique_specimen_id=%d columns=%s",
        len(labels),
        labels["specimen_id"].astype(str).nunique(dropna=False),
        list(labels.columns),
    )
    lcol = _label_col(labels)

    required_cols = {"query_id", "neighbor_id"}
    if not required_cols.issubset(set(knn.columns)):
        raise ValueError("knn_csv must contain query_id and neighbor_id")

    if "distance_size" not in knn.columns:
        knn["distance_size"] = 0.0
    if "distance_tex" not in knn.columns:
        knn["distance_tex"] = 0.0

    knn["query_id"] = _normalize_id(knn["query_id"])
    knn["neighbor_id"] = _normalize_id(knn["neighbor_id"])
    knn = _attach_fusion_mode(knn)
    _log_mode_stats("knn_with_mode", knn)

    ldf = labels[["specimen_id", lcol]].rename(columns={lcol: "label"})
    ldf["specimen_id"] = _normalize_id(ldf["specimen_id"])
    merged = knn.merge(ldf, left_on="query_id", right_on="specimen_id", how="left").rename(columns={"label": "query_label"})
    merged = merged.drop(columns=["specimen_id"])
    _log_mode_stats("after_query_label_join", merged)
    merged = merged.merge(ldf, left_on="neighbor_id", right_on="specimen_id", how="left").rename(columns={"label": "neighbor_label"})
    merged = merged.drop(columns=["specimen_id"])
    _log_mode_stats("after_neighbor_label_join", merged)

    before_topk = len(merged)
    merged = _apply_topk(merged, args.topk)
    _log_mode_stats("after_topk", merged)
    if len(merged) < before_topk * 0.5:
        LOGGER.warning("topk 適用で行数が大きく減少しました: %d -> %d", before_topk, len(merged))
    if args.strict_query_set_check:
        _warn_query_set_mismatch(merged)

    merged["label_match"] = (merged["query_label"].astype(str) == merged["neighbor_label"].astype(str)).astype(int)
    pre_summary = (
        merged.groupby("fusion_mode", dropna=False)
        .agg(rows=("query_id", "size"), unique_queries=("query_id", "nunique"), mean_label_match=("label_match", "mean"))
        .reset_index()
    )
    LOGGER.info("[before_aggregate] %s", pre_summary.to_dict(orient="records"))
    merged.to_csv(args.out / "fusion_eval_rows.csv", index=False)

    per_query = (
        merged.groupby(["fusion_mode", "query_id", "query_label"], dropna=False)["label_match"]
        .mean()
        .reset_index(name="topk_match_rate")
    )
    per_query.to_csv(args.out / "fusion_eval_per_query.csv", index=False)

    per_category = (
        per_query.groupby(["fusion_mode", "query_label"], dropna=False)["topk_match_rate"]
        .mean()
        .reset_index(name="mean_match_rate")
    )
    per_category.to_csv(args.out / "fusion_eval_per_category.csv", index=False)

    overall = (
        per_query.groupby("fusion_mode", dropna=False)["topk_match_rate"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_match_rate", "std": "std_match_rate", "count": "n_queries"})
    )
    overall.to_csv(args.out / "fusion_eval_overall.csv", index=False)

    summary = {
        "n_rows": int(len(merged)),
        "n_queries": int(per_query["query_id"].nunique()),
        "modes": overall.to_dict(orient="records"),
    }
    (args.out / "fusion_eval_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
