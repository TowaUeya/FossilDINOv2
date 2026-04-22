from __future__ import annotations

import argparse
from pathlib import Path

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.io import ensure_dir, load_ids
from src.utils.vision import l2_normalize


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot HDBSCAN trees for explanation (not model selection)")
    p.add_argument("--emb", type=Path, required=True)
    p.add_argument("--ids", type=Path, required=True)
    p.add_argument("--clusters", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--min_cluster_size", type=int, default=10)
    p.add_argument("--min_samples", type=int, default=None)
    p.add_argument("--selection_method", choices=["eom"], default="eom")
    p.add_argument(
        "--single_linkage_truncate_mode",
        choices=["lastp", "level", "none"],
        default="lastp",
        help="Dendrogram truncation mode for single linkage tree plot. Use 'none' for full tree.",
    )
    p.add_argument(
        "--single_linkage_p",
        type=int,
        default=50,
        help="Number of leaves/levels to show when single linkage truncation is enabled.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.out)
    x = l2_normalize(np.load(args.emb))
    ids = load_ids(args.ids)

    clusterer = hdbscan.HDBSCAN(
        metric="euclidean",
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        cluster_selection_method=args.selection_method,
    ).fit(x)

    truncate_mode = None if args.single_linkage_truncate_mode == "none" else args.single_linkage_truncate_mode
    fig, ax = plt.subplots(figsize=(10, 6))
    try:
        clusterer.single_linkage_tree_.plot(axis=ax, truncate_mode=truncate_mode, p=args.single_linkage_p)
    except RecursionError:
        ax.clear()
        clusterer.single_linkage_tree_.plot(axis=ax, truncate_mode="lastp", p=min(args.single_linkage_p, 20))
    plt.tight_layout()
    fig.savefig(args.out / "single_linkage_tree.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    clusterer.condensed_tree_.plot(axis=ax, select_clusters=False)
    plt.tight_layout()
    fig.savefig(args.out / "condensed_tree.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    clusterer.condensed_tree_.plot(axis=ax, select_clusters=True)
    plt.tight_layout()
    fig.savefig(args.out / "condensed_tree_selected.png", dpi=200)
    plt.close(fig)

    pd.DataFrame(clusterer.single_linkage_tree_.to_numpy(), columns=["left", "right", "distance", "size"]).to_csv(
        args.out / "single_linkage_tree.csv", index=False
    )
    pd.DataFrame(clusterer.condensed_tree_.to_numpy()).to_csv(args.out / "condensed_tree.csv", index=False)

    if args.clusters is not None:
        c = pd.read_csv(args.clusters)
        c = c.set_index("specimen_id").reindex(ids)
        c.to_csv(args.out / "clusters_reindexed.csv")


if __name__ == "__main__":
    main()
