from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate prefilter grouping quality with labels.")
    p.add_argument("--prefilter_csv", type=Path, required=True)
    p.add_argument("--labels_csv", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.out)

    pf = pd.read_csv(args.prefilter_csv)
    lb = pd.read_csv(args.labels_csv)

    if "specimen_id" not in pf.columns or "pregroup_id" not in pf.columns:
        raise ValueError("prefilter_csv must contain specimen_id and pregroup_id")
    if "specimen_id" not in lb.columns or "Fossil category" not in lb.columns:
        raise ValueError("labels_csv must contain specimen_id and Fossil category")

    merged = pf.merge(lb[["specimen_id", "Fossil category"]], on="specimen_id", how="inner")
    merged["pregroup_id"] = merged["pregroup_id"].astype(str)
    if "Fossil category" in merged.columns:
        merged = merged.rename(columns={"Fossil category": "label"})
    elif "Fossil category_y" in merged.columns:
        merged = merged.rename(columns={"Fossil category_y": "label"})
    elif "Fossil_category" in merged.columns:
        merged = merged.rename(columns={"Fossil_category": "label"})
    elif "Fossil_category_y" in merged.columns:
        merged = merged.rename(columns={"Fossil_category_y": "label"})
    else:
        raise ValueError(
            "Could not find a label column after merge. "
            "Expected one of: Fossil category, Fossil category_y, Fossil_category, Fossil_category_y."
        )

    total = max(len(merged), 1)
    rows: list[dict[str, float | int | str]] = []
    weighted_purity_sum = 0.0

    for gid, gdf in merged.groupby("pregroup_id", dropna=False):
        counts = gdf["label"].value_counts(dropna=False)
        n = int(len(gdf))
        dominant = str(counts.index[0]) if not counts.empty else ""
        dom_count = int(counts.iloc[0]) if not counts.empty else 0
        purity = float(dom_count / max(n, 1))
        candidate_fraction = float(n / total)
        weighted_purity_sum += purity * n

        rows.append(
            {
                "pregroup_id": str(gid),
                "n_samples": n,
                "dominant_label": dominant,
                "dominant_label_ratio": purity,
                "n_labels": int(counts.shape[0]),
                "candidate_fraction": candidate_fraction,
            }
        )

    groups_df = pd.DataFrame(rows).sort_values(["n_samples", "pregroup_id"], ascending=[False, True])
    groups_df.to_csv(args.out / "prefilter_eval_groups.csv", index=False)

    candidate_reduction_rate = float(1.0 - groups_df["candidate_fraction"].mean()) if not groups_df.empty else 0.0

    method_comparison: dict[str, dict[str, float | int]] = {}
    if "grouping_method" in merged.columns:
        for method, mdf in merged.groupby("grouping_method", dropna=False):
            n_method = int(len(mdf))
            if n_method == 0:
                continue
            purity_sum = 0.0
            for _, gdf in mdf.groupby("pregroup_id", dropna=False):
                counts = gdf["label"].value_counts(dropna=False)
                dom = int(counts.iloc[0]) if not counts.empty else 0
                purity_sum += float(dom / max(len(gdf), 1)) * len(gdf)
            method_comparison[str(method)] = {
                "n_samples": n_method,
                "weighted_purity": float(purity_sum / max(n_method, 1)),
                "n_groups": int(mdf["pregroup_id"].astype(str).nunique()),
            }

    summary = {
        "n_joined_samples": int(len(merged)),
        "n_groups": int(groups_df.shape[0]),
        "weighted_purity": float(weighted_purity_sum / total),
        "candidate_reduction_rate": candidate_reduction_rate,
        "grouping_method_comparison": method_comparison,
    }
    (args.out / "prefilter_eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
