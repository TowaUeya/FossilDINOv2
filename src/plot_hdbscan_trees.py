from __future__ import annotations

import argparse
import sys
from pathlib import Path

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram

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
        original_limit = sys.getrecursionlimit()
        required_limit = min(1_000_000, max(original_limit, int(4 * len(clusterer.single_linkage_tree_._linkage) + 1_000)))
        sys.setrecursionlimit(required_limit)
        try:
            ax.clear()
            clusterer.single_linkage_tree_.plot(axis=ax, truncate_mode=truncate_mode, p=args.single_linkage_p)
        except RecursionError:
            ax.clear()
            ax.text(
                0.5,
                0.5,
                "single_linkage_tree_ plot failed due to recursion depth.\nCSV was still exported.",
                ha="center",
                va="center",
            )
            ax.set_axis_off()
        finally:
            sys.setrecursionlimit(original_limit)
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
    export_single_linkage_html(clusterer=clusterer, ids=ids, out_dir=args.out)

    if args.clusters is not None:
        c = pd.read_csv(args.clusters)
        c = c.set_index("specimen_id").reindex(ids)
        c.to_csv(args.out / "clusters_reindexed.csv")


def export_single_linkage_html(clusterer: hdbscan.HDBSCAN, ids: list[str], out_dir: Path) -> None:
    """Export an interactive dendrogram where each leaf hover shows specimen/model ID."""
    linkage = clusterer.single_linkage_tree_.to_numpy()
    if linkage.size == 0:
        return

    # scipy dendrogram expects (n-1, 4) linkage format.
    # For HTML we intentionally use full tree (no truncation), so each model/specimen leaf is traceable.
    dendro = dendrogram(linkage, labels=ids, no_plot=True)

    fig = go.Figure()
    for xs, ys in zip(dendro["icoord"], dendro["dcoord"]):
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color="#4B5563", width=1),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    leaf_positions = [5 + 10 * i for i in range(len(dendro["ivl"]))]
    fig.add_trace(
        go.Scatter(
            x=leaf_positions,
            y=[0.0] * len(leaf_positions),
            mode="markers",
            marker=dict(size=5, color="#2563EB"),
            text=dendro["ivl"],
            customdata=np.array(dendro["ivl"], dtype=object),
            hovertemplate="model/specimen: %{customdata}<extra></extra>",
            showlegend=False,
        )
    )

    fig.update_layout(
        title="HDBSCAN Single Linkage Tree (hover leaves to identify each model/specimen)",
        xaxis_title="Leaf order",
        yaxis_title="Distance",
        template="plotly_white",
        hovermode="closest",
        height=800,
    )

    fig.write_html(out_dir / "single_linkage_tree.html", include_plotlyjs="cdn")


if __name__ == "__main__":
    main()
