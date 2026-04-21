"""旧実験コード。本研究の主張には使わない。"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.utils.io import ensure_dir

LOGGER = logging.getLogger(__name__)


MODE_COLUMNS = {
    "shape_only_csv": "shape_only",
    "shape_size_csv": "shape+size",
    "shape_size_texture_csv": "shape+size+texture",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Combine multiple fusion run CSV files into one CSV with fusion_mode column."
    )
    p.add_argument("--shape_only_csv", type=Path, default=None)
    p.add_argument("--shape_size_csv", type=Path, default=None)
    p.add_argument("--shape_size_texture_csv", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True, help="Output path for combined CSV.")
    return p.parse_args()


def _normalize_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def _load_one(csv_path: Path, mode: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_cols = {"query_id", "neighbor_id"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"{csv_path} must contain query_id and neighbor_id")

    out = df.copy()
    out["query_id"] = _normalize_id(out["query_id"])
    out["neighbor_id"] = _normalize_id(out["neighbor_id"])
    out["fusion_mode"] = mode
    return out


def _log_mode_stats(df: pd.DataFrame) -> None:
    if df.empty:
        LOGGER.info("[combined] rows=0")
        return

    rows_by_mode = df.groupby("fusion_mode", dropna=False).size().to_dict()
    uniq_query_by_mode = df.groupby("fusion_mode", dropna=False)["query_id"].nunique(dropna=False).to_dict()
    query_set_size_by_mode = {
        mode: len(set(group["query_id"].dropna().astype(str).tolist()))
        for mode, group in df.groupby("fusion_mode", dropna=False)
    }

    LOGGER.info("[combined] rows=%d", len(df))
    LOGGER.info("[combined] rows_by_mode=%s", rows_by_mode)
    LOGGER.info("[combined] unique_queries_by_mode=%s", uniq_query_by_mode)
    LOGGER.info("[combined] query_set_size_by_mode=%s", query_set_size_by_mode)

    query_sets = {
        mode: set(group["query_id"].dropna().astype(str).tolist())
        for mode, group in df.groupby("fusion_mode", dropna=False)
    }
    if len(query_sets) > 1:
        modes = list(query_sets.keys())
        baseline_mode = modes[0]
        baseline_set = query_sets[baseline_mode]
        mismatch = False
        for mode in modes[1:]:
            if query_sets[mode] != baseline_set:
                mismatch = True
                only_base = len(baseline_set - query_sets[mode])
                only_mode = len(query_sets[mode] - baseline_set)
                LOGGER.warning(
                    "query 集合不一致: %s vs %s (base_only=%d, mode_only=%d)",
                    baseline_mode,
                    mode,
                    only_base,
                    only_mode,
                )
        if not mismatch:
            LOGGER.info("全 mode で query 集合は一致しています。")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    inputs: list[tuple[Path, str]] = []
    for arg_name, mode in MODE_COLUMNS.items():
        csv_path = getattr(args, arg_name)
        if csv_path is not None:
            inputs.append((csv_path, mode))

    if not inputs:
        raise ValueError("At least one input CSV must be provided.")

    frames: list[pd.DataFrame] = []
    for csv_path, mode in inputs:
        LOGGER.info("Loading mode=%s from %s", mode, csv_path)
        frames.append(_load_one(csv_path, mode))

    combined = pd.concat(frames, axis=0, ignore_index=True)
    _log_mode_stats(combined)

    ensure_dir(args.out.parent)
    combined.to_csv(args.out, index=False)
    LOGGER.info("Saved combined CSV to %s", args.out)


if __name__ == "__main__":
    main()
