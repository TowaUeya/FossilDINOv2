from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from src.utils.io import ensure_dir, setup_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze best_clusters.csv for paper-ready diagnostics")
    parser.add_argument("--clusters", type=Path, required=True, help="Path to best_clusters.csv")
    parser.add_argument("--labels", type=Path, default=None, help="Optional labels file")
    parser.add_argument("--out", type=Path, default=Path("results"), help="Output directory")
    parser.add_argument("--mixed_threshold", type=float, default=0.4, help="majority_fraction threshold for mixed flag")
    return parser.parse_args()


def infer_labels_from_ids(ids: pd.Series) -> pd.Series | None:
    vals = ids.astype(str).tolist()
    if any("/" not in sid for sid in vals):
        return None
    return pd.Series([sid.split("/")[0] for sid in vals], index=ids.index, dtype="object")


def load_labels(path: Path | None, ids: pd.Series) -> pd.Series | None:
    if path is None:
        inferred = infer_labels_from_ids(ids)
        if inferred is not None:
            LOGGER.info("No --labels provided. Inferred labels from specimen_id prefix.")
        return inferred

    if not path.exists():
        LOGGER.warning("labels not found: %s", path)
        return infer_labels_from_ids(ids)

    rows = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return infer_labels_from_ids(ids)

    if "\t" in rows[0] or "," in rows[0]:
        sep = "\t" if "\t" in rows[0] else ","
        mapping: dict[str, str] = {}
        for row in rows:
            cols = row.split(sep)
            if len(cols) < 2:
                continue
            mapping[cols[0]] = cols[1]
        out = ids.map(mapping).fillna("")
        if (out != "").any():
            return out
        return infer_labels_from_ids(ids)

    if len(rows) != len(ids):
        LOGGER.warning("labels count mismatch (%d vs %d). Fallback to ID inference.", len(rows), len(ids))
        return infer_labels_from_ids(ids)

    return pd.Series(rows, index=ids.index, dtype="object")


def overall_purity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    correct = 0
    for cluster_id in np.unique(y_pred):
        members = y_true[y_pred == cluster_id]
        if len(members) == 0:
            continue
        _, counts = np.unique(members, return_counts=True)
        correct += int(np.max(counts))
    return float(correct / len(y_true))


def build_cluster_purity_table(df_non_noise: pd.DataFrame, label_col: str | None) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for cluster_id, group in df_non_noise.groupby("cluster_id"):
        row: dict[str, Any] = {
            "cluster_id": int(cluster_id),
            "size": int(len(group)),
            "mean_prob": float(group["prob"].mean()) if "prob" in group else float("nan"),
        }
        if label_col is not None:
            vc = group[label_col].value_counts()
            if len(vc) > 0:
                row["top_label"] = str(vc.index[0])
                row["purity"] = float(vc.iloc[0] / len(group))
            else:
                row["top_label"] = ""
                row["purity"] = float("nan")
        else:
            row["top_label"] = ""
            row["purity"] = float("nan")
        records.append(row)
    if not records:
        return pd.DataFrame(columns=["cluster_id", "size", "top_label", "purity", "mean_prob"])
    return pd.DataFrame(records).sort_values("size", ascending=False)


def main() -> None:
    args = parse_args()
    setup_logging()
    ensure_dir(args.out)

    df = pd.read_csv(args.clusters)
    required_cols = {"specimen_id", "cluster_id"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"clusters csv must include {required_cols}, got {set(df.columns)}")
    if "prob" not in df.columns:
        df["prob"] = np.nan

    labels = load_labels(args.labels, df["specimen_id"])
    label_col: str | None = None
    if labels is not None:
        df["label"] = labels
        label_col = "label"

    n_total = int(len(df))
    n_noise = int((df["cluster_id"] == -1).sum())
    noise_ratio = float(n_noise / n_total) if n_total else 0.0

    df_non_noise = df[df["cluster_id"] != -1].copy()
    cluster_sizes = df_non_noise.groupby("cluster_id").size().sort_values(ascending=False)
    n_clusters = int(cluster_sizes.shape[0])

    summary: dict[str, Any] = {
        "n_total": n_total,
        "n_noise": n_noise,
        "noise_ratio": noise_ratio,
        "n_clusters": n_clusters,
        "cluster_size_stats": {
            "min": int(cluster_sizes.min()) if n_clusters else None,
            "median": float(cluster_sizes.median()) if n_clusters else None,
            "max": int(cluster_sizes.max()) if n_clusters else None,
            "top10_sizes": cluster_sizes.head(10).astype(int).tolist() if n_clusters else [],
        },
        "mean_prob_overall": float(df["prob"].mean()) if len(df) else float("nan"),
    }

    cluster_prob = df_non_noise.groupby("cluster_id")["prob"].mean().reset_index().rename(columns={"prob": "mean_prob"})

    ari = float("nan")
    nmi = float("nan")
    purity = float("nan")
    category_noise_rate: dict[str, float] = {}

    if label_col is not None:
        eval_mask = (df["cluster_id"] != -1) & (df[label_col] != "")
        try:
            if eval_mask.sum() > 1:
                ari = float(adjusted_rand_score(df.loc[eval_mask, label_col], df.loc[eval_mask, "cluster_id"]))
                nmi = float(normalized_mutual_info_score(df.loc[eval_mask, label_col], df.loc[eval_mask, "cluster_id"]))
                purity = float(overall_purity(df.loc[eval_mask, label_col].to_numpy(), df.loc[eval_mask, "cluster_id"].to_numpy()))
        except Exception:
            LOGGER.exception("Failed to compute ARI/NMI/purity.")

        grouped = df.groupby(label_col)
        for label, g in grouped:
            if label == "":
                continue
            category_noise_rate[str(label)] = float((g["cluster_id"] == -1).mean())

    summary["external_metrics"] = {
        "ari": ari,
        "nmi": nmi,
        "overall_purity": purity,
        "category_noise_rate": category_noise_rate,
    }

    largest_records: list[dict[str, Any]] = []
    for cluster_id, size in cluster_sizes.head(10).items():
        row = {"cluster_id": int(cluster_id), "size": int(size)}
        group = df_non_noise[df_non_noise["cluster_id"] == cluster_id]
        if label_col is not None:
            vc = group[label_col].value_counts()
            if len(vc) > 0:
                row["top_label"] = str(vc.index[0])
                row["majority_fraction"] = float(vc.iloc[0] / len(group))
                row["mixed"] = bool(row["majority_fraction"] < args.mixed_threshold)
            else:
                row["top_label"] = ""
                row["majority_fraction"] = float("nan")
                row["mixed"] = False
        else:
            row["top_label"] = ""
            row["majority_fraction"] = float("nan")
            row["mixed"] = False
        largest_records.append(row)

    cluster_purity_df = build_cluster_purity_table(df_non_noise, label_col)
    largest_df = pd.DataFrame(largest_records)

    summary["largest_clusters"] = largest_records
    summary["cluster_mean_prob"] = {
        str(int(r.cluster_id)): float(r.mean_prob)
        for r in cluster_prob.itertuples(index=False)
    }

    report_json = args.out / "cluster_report.json"
    report_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    purity_csv = args.out / "cluster_purity.csv"
    cluster_purity_df.to_csv(purity_csv, index=False)

    largest_csv = args.out / "largest_clusters.csv"
    largest_df.to_csv(largest_csv, index=False)

    LOGGER.info("Saved report to %s", report_json)
    LOGGER.info("Saved cluster purity table to %s", purity_csv)
    LOGGER.info("Saved largest cluster diagnostics to %s", largest_csv)


if __name__ == "__main__":
    main()
