from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
    p.add_argument(
        "--format",
        choices=["png", "html", "both"],
        default="both",
        help="Output format: static PNG, interactive HTML, or both",
    )
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
        proj_2d = PCA(n_components=2, random_state=42).fit_transform(x)
        proj_3d = PCA(n_components=3, random_state=42).fit_transform(x)
    else:
        import umap

        proj_2d = umap.UMAP(n_components=2, random_state=42).fit_transform(x)
        proj_3d = umap.UMAP(n_components=3, random_state=42).fit_transform(x)

    df["x"] = proj_2d[:, 0]
    df["y"] = proj_2d[:, 1]
    df["x3"] = proj_3d[:, 0]
    df["y3"] = proj_3d[:, 1]
    df["z3"] = proj_3d[:, 2]

    color_col = "cluster_id" if "cluster_id" in df.columns else ("label" if "label" in df.columns else None)

    if args.format in {"png", "both"}:
        out_png = args.out / f"embedding_space_{args.method}.png"
        plt.figure(figsize=(8, 6))
        if color_col is None:
            plt.scatter(df["x"], df["y"], s=18)
        else:
            for k, g in df.groupby(color_col):
                plt.scatter(g["x"], g["y"], s=18, label=str(k), alpha=0.8)
            if df[color_col].nunique() <= 20:
                plt.legend(fontsize=8)
        plt.title(f"Embedding space ({args.method}, 2D)")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)

    if args.format in {"html", "both"}:
        out_html = args.out / f"embedding_space_{args.method}_3d.html"
        if color_col == "cluster_id":
            fig = _build_cluster_toggle_3d_figure(df, args.method)
        else:
            hover_cols = [c for c in ["cluster_id", "label"] if c in df.columns]
            fig = px.scatter_3d(
                df,
                x="x3",
                y="y3",
                z="z3",
                color=color_col,
                hover_name="specimen_id",
                hover_data=hover_cols if hover_cols else None,
                opacity=0.85,
                title=f"Embedding space ({args.method}, 3D interactive)",
            )
        fig.update_traces(marker={"size": 4})
        fig.update_layout(margin={"l": 0, "r": 0, "t": 40, "b": 0})
        fig.write_html(out_html, include_plotlyjs="cdn")


def _build_cluster_toggle_3d_figure(df: pd.DataFrame, method: str) -> go.Figure:
    hover_cols = [c for c in ["cluster_id", "label"] if c in df.columns]
    cluster_values = sorted(df["cluster_id"].dropna().unique().tolist(), key=lambda x: (str(type(x)), x))
    if not cluster_values:
        return px.scatter_3d(
            df,
            x="x3",
            y="y3",
            z="z3",
            hover_name="specimen_id",
            hover_data=hover_cols if hover_cols else None,
            opacity=0.85,
            title=f"Embedding space ({method}, 3D interactive)",
        )

    palette = px.colors.qualitative.Alphabet
    color_map = {cluster: palette[i % len(palette)] for i, cluster in enumerate(cluster_values)}

    fig = go.Figure()
    for cluster in cluster_values:
        group = df[df["cluster_id"] == cluster]
        customdata = np.stack([group[c].fillna("-").astype(str).to_numpy() for c in hover_cols], axis=1) if hover_cols else None
        hovertemplate = "<b>%{hovertext}</b><br>"
        if hover_cols:
            hovertemplate += "<br>".join(f"{c}: %{{customdata[{i}]}}" for i, c in enumerate(hover_cols)) + "<br>"
        hovertemplate += "x: %{x:.3f}<br>y: %{y:.3f}<br>z: %{z:.3f}<extra></extra>"
        fig.add_trace(
            go.Scatter3d(
                x=group["x3"],
                y=group["y3"],
                z=group["z3"],
                mode="markers",
                name=str(cluster),
                legendgroup=str(cluster),
                marker={"size": 4, "opacity": 0.85, "color": color_map[cluster]},
                hovertext=group["specimen_id"],
                customdata=customdata,
                hovertemplate=hovertemplate,
            )
        )

    visible_all = [True] * len(cluster_values)
    visible_none = ["legendonly"] * len(cluster_values)
    visible_without_minus1 = [cluster != -1 for cluster in cluster_values]
    visible_only_minus1 = [cluster == -1 for cluster in cluster_values]

    fig.update_layout(
        title=f"Embedding space ({method}, 3D interactive)",
        scene={"xaxis_title": "x3", "yaxis_title": "y3", "zaxis_title": "z3"},
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.0,
                "y": 1.15,
                "showactive": True,
                "buttons": [
                    {"label": "All", "method": "update", "args": [{"visible": visible_all}]},
                    {"label": "None", "method": "update", "args": [{"visible": visible_none}]},
                    {"label": "Hide cluster -1", "method": "update", "args": [{"visible": visible_without_minus1}]},
                    {"label": "Only cluster -1", "method": "update", "args": [{"visible": visible_only_minus1}]},
                ],
            }
        ],
    )
    return fig


if __name__ == "__main__":
    main()
