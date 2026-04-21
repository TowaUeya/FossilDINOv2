from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.utils.io import ensure_dir, load_ids


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize embedding space (for interpretation only)")
    p.add_argument("--emb", type=Path, required=True)
    p.add_argument("--ids", type=Path, required=True)
    p.add_argument("--clusters", type=Path, default=None)
    p.add_argument("--labels", type=Path, default=None)
    p.add_argument("--method", choices=["pca", "umap"], default="pca")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def _load_optional_labels(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        sp = line.strip().split()
        if len(sp) >= 2:
            rows.append((sp[0], " ".join(sp[1:])))
    return pd.DataFrame(rows, columns=["specimen_id", "label"])


def main() -> None:
    args = parse_args()
    ensure_dir(args.out)

    x = np.load(args.emb)
    ids = load_ids(args.ids)
    df = pd.DataFrame({"specimen_id": ids})

    if args.clusters is not None:
        df = df.merge(pd.read_csv(args.clusters)[["specimen_id", "cluster_id"]], on="specimen_id", how="left")
    labels_df = _load_optional_labels(args.labels)
    if labels_df is not None:
        df = df.merge(labels_df, on="specimen_id", how="left")

    if args.method == "pca":
        xy = PCA(n_components=2, random_state=42).fit_transform(x)
        out_path = args.out / "embedding_space_pca.png"
    else:
        import umap

        xy = umap.UMAP(n_components=2, random_state=42).fit_transform(x)
        out_path = args.out / "embedding_space_umap.png"

    df["x"] = xy[:, 0]
    df["y"] = xy[:, 1]

    plt.figure(figsize=(8, 6))
    color_col = "cluster_id" if "cluster_id" in df.columns else ("label" if "label" in df.columns else None)
    if color_col is None:
        plt.scatter(df["x"], df["y"], s=18)
    else:
        for k, g in df.groupby(color_col):
            plt.scatter(g["x"], g["y"], s=18, label=str(k), alpha=0.8)
        if df[color_col].nunique() <= 20:
            plt.legend(fontsize=8)
    plt.title(f"Embedding space ({args.method})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)


if __name__ == "__main__":
    main()
