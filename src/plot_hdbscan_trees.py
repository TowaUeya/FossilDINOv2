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
    cluster_colors: dict[str, int] | None = None
    if args.clusters is not None:
        c = pd.read_csv(args.clusters)
        c = c.set_index("specimen_id").reindex(ids)
        c.to_csv(args.out / "clusters_reindexed.csv")
        if "cluster_id" in c.columns:
            cluster_colors = {
                specimen_id: int(cluster_id)
                for specimen_id, cluster_id in c["cluster_id"].items()
                if pd.notna(cluster_id)
            }

    export_single_linkage_html(
        clusterer=clusterer,
        ids=ids,
        out_dir=args.out,
        truncate_mode=truncate_mode,
        p=args.single_linkage_p,
        cluster_labels_by_id=cluster_colors,
    )


def export_single_linkage_html(
    clusterer: hdbscan.HDBSCAN,
    ids: list[str],
    out_dir: Path,
    truncate_mode: str | None = None,
    p: int = 50,
    cluster_labels_by_id: dict[str, int] | None = None,
) -> None:
    """Export an interactive dendrogram where each leaf hover shows specimen/model ID."""
    linkage = clusterer.single_linkage_tree_.to_numpy()
    if linkage.size == 0:
        return

    # scipy dendrogram expects (n-1, 4) linkage format.
    # We first attempt the full tree. If recursion depth is exceeded, retry with
    # a larger recursion limit and finally fall back to the CLI truncation mode.
    try:
        dendro = dendrogram(linkage, labels=ids, no_plot=True)
    except RecursionError:
        original_limit = sys.getrecursionlimit()
        required_limit = min(1_000_000, max(original_limit, int(4 * len(linkage) + 1_000)))
        sys.setrecursionlimit(required_limit)
        try:
            dendro = dendrogram(linkage, labels=ids, no_plot=True)
        except RecursionError:
            try:
                dendro = dendrogram(
                    linkage,
                    labels=ids,
                    no_plot=True,
                    truncate_mode=truncate_mode,
                    p=p,
                )
            except RecursionError:
                fig = go.Figure()
                fig.add_annotation(
                    text=(
                        "Failed to build dendrogram due to recursion depth.<br>"
                        "Try a more aggressive truncation setting or fewer samples."
                    ),
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                )
                fig.update_layout(
                    title="HDBSCAN Single Linkage Tree",
                    template="plotly_white",
                )
                fig.write_html(out_dir / "single_linkage_tree.html", include_plotlyjs="cdn")
                return
        finally:
            sys.setrecursionlimit(original_limit)

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
    cluster_palette = [
        "#2563EB",
        "#DC2626",
        "#059669",
        "#D97706",
        "#7C3AED",
        "#0891B2",
        "#DB2777",
        "#65A30D",
        "#EA580C",
        "#4338CA",
    ]
    noise_color = "#9CA3AF"
    unknown_color = "#111827"

    leaf_cluster_ids: list[int | None] = []
    for specimen_id in dendro["ivl"]:
        if cluster_labels_by_id is None:
            leaf_cluster_ids.append(None)
        else:
            leaf_cluster_ids.append(cluster_labels_by_id.get(specimen_id))

    leaf_colors: list[str] = []
    hover_texts: list[str] = []
    for specimen_id, cluster_id in zip(dendro["ivl"], leaf_cluster_ids):
        if cluster_labels_by_id is None:
            leaf_colors.append("#2563EB")
            hover_texts.append(f"model/specimen: {specimen_id}")
            continue
        if cluster_id is None:
            leaf_colors.append(unknown_color)
            cluster_text = "N/A"
        elif cluster_id == -1:
            leaf_colors.append(noise_color)
            cluster_text = str(cluster_id)
        else:
            leaf_colors.append(cluster_palette[cluster_id % len(cluster_palette)])
            cluster_text = str(cluster_id)
        hover_texts.append(f"model/specimen: {specimen_id}<br>fixed cluster: {cluster_text}")

    fig.add_trace(
        go.Scatter(
            x=leaf_positions,
            y=[0.0] * len(leaf_positions),
            mode="markers",
            marker=dict(size=5, color=leaf_colors),
            text=dendro["ivl"],
            customdata=np.array(hover_texts, dtype=object),
            hovertemplate="%{customdata}<extra></extra>",
            showlegend=False,
        )
    )

    if cluster_labels_by_id is not None:
        present_clusters = sorted(
            {
                cid
                for cid in leaf_cluster_ids
                if cid is not None and cid != -1
            }
        )
        for cid in present_clusters:
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(size=8, color=cluster_palette[cid % len(cluster_palette)]),
                    name=f"cluster {cid}",
                    showlegend=True,
                    hoverinfo="skip",
                )
            )
        if -1 in leaf_cluster_ids:
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(size=8, color=noise_color),
                    name="cluster -1 (noise)",
                    showlegend=True,
                    hoverinfo="skip",
                )
            )
        if None in leaf_cluster_ids:
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(size=8, color=unknown_color),
                    name="cluster N/A",
                    showlegend=True,
                    hoverinfo="skip",
                )
            )

    fig.update_layout(
        title="HDBSCAN Single Linkage Tree (hover leaves to identify each model/specimen)",
        xaxis_title="Leaf order",
        yaxis_title="Distance",
        template="plotly_white",
        hovermode="closest",
        height=800,
        legend_title="Fixed clustering",
    )

    fig.write_html(out_dir / "single_linkage_tree.html", include_plotlyjs="cdn")


if __name__ == "__main__":
    main()
