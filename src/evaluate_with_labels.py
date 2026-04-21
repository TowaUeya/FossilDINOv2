from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    v_measure_score,
)

from src.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate clusters with labels (post-hoc only)")
    p.add_argument("--clusters", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def load_labels(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if {"specimen_id", "label"}.issubset(df.columns):
            return df[["specimen_id", "label"]]
        raise ValueError("labels.csv must contain specimen_id,label")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            raise ValueError("labels.txt format: <specimen_id> <label>")
        rows.append((parts[0], " ".join(parts[1:])))
    return pd.DataFrame(rows, columns=["specimen_id", "label"])


def purity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    total = len(df)
    if total == 0:
        return 0.0
    correct = 0
    for _, g in df.groupby("y_pred"):
        correct += g["y_true"].value_counts().iloc[0]
    return float(correct / total)


def main() -> None:
    args = parse_args()
    ensure_dir(args.out)

    c = pd.read_csv(args.clusters)
    l = load_labels(args.labels)
    df = c.merge(l, on="specimen_id", how="inner")

    y_true = df["label"].to_numpy()
    y_pred = df["cluster_id"].to_numpy()

    metrics = {
        "ARI": adjusted_rand_score(y_true, y_pred),
        "NMI": normalized_mutual_info_score(y_true, y_pred),
        "AMI": adjusted_mutual_info_score(y_true, y_pred),
        "Purity": purity_score(y_true, y_pred),
        "Homogeneity": homogeneity_score(y_true, y_pred),
        "Completeness": completeness_score(y_true, y_pred),
        "V-measure": v_measure_score(y_true, y_pred),
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    pd.crosstab(df["label"], df["cluster_id"]).to_csv(args.out / "class_confusion.csv")

    pur_rows = []
    for cid, g in df[df["cluster_id"] != -1].groupby("cluster_id"):
        vc = g["label"].value_counts(normalize=True)
        pur_rows.append({"cluster_id": int(cid), "purity": float(vc.iloc[0]), "major_label": vc.index[0], "size": int(len(g))})
    pd.DataFrame(pur_rows).to_csv(args.out / "cluster_purity.csv", index=False)

    noise_rows = []
    for label, g in df.groupby("label"):
        n = len(g)
        z = int((g["cluster_id"] == -1).sum())
        noise_rows.append({"label": label, "n": n, "n_noise": z, "noise_rate": float(z / max(1, n))})
    pd.DataFrame(noise_rows).to_csv(args.out / "category_noise_rate.csv", index=False)


if __name__ == "__main__":
    main()
