from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
import yaml

from src.cluster import infer_ids, preprocess_for_clustering
from src.utils.io import ensure_dir, resolve_file_or_recursive_search, set_seed, setup_logging

LOGGER = logging.getLogger(__name__)

DEFAULT_EMB = Path("data/embeddings/embeddings.npy")
DEFAULT_IDS = Path("data/embeddings/ids.txt")
DEFAULT_SWEEP_CSV = Path("results/sweep/sweep_results.csv")
DEFAULT_OUT = Path("results/trees")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot HDBSCAN single-linkage/condensed trees from cluster_sweep results"
    )
    parser.add_argument("--emb", type=Path, default=DEFAULT_EMB, help="Path to embeddings.npy")
    parser.add_argument("--ids", type=Path, default=DEFAULT_IDS, help="Path to ids.txt")
    parser.add_argument("--sweep_csv", type=Path, default=DEFAULT_SWEEP_CSV, help="Path to sweep_results.csv")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output directory")

    parser.add_argument("--list", action="store_true", help="List top-ranked sweep configurations")
    parser.add_argument("--top", type=int, default=20, help="Rows to show when using --list")
    parser.add_argument("--pick", type=int, default=None, help="Pick config by rank (0-based)")

    parser.add_argument("--normalize", choices=["none", "l2"], default="l2")
    parser.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--umap_metric", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--umap_n_components", type=int, default=15)
    parser.add_argument("--umap_n_neighbors", type=int, default=30)
    parser.add_argument("--umap_min_dist", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--skip_single", action="store_true", help="Skip single_linkage_tree plotting")
    parser.add_argument(
        "--skip_single_leaf_labels",
        action="store_true",
        help="Skip single_linkage_tree plotting with all leaf labels (3D model names)",
    )
    return parser.parse_args()


def _parse_bool(value: Any, field: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer, float, np.floating)) and not pd.isna(value):
        return bool(int(value))
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes", "y", "on"}:
            return True
        if v in {"false", "0", "no", "n", "off"}:
            return False
    raise ValueError(f"Invalid bool for {field}: {value!r}")


def _parse_pca(value: Any) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    pca = float(value)
    return pca


def _choose_ranked(df: pd.DataFrame, pick: int | None) -> tuple[pd.Series, int, pd.DataFrame]:
    ranked = df.sort_values("final_score", ascending=False, na_position="last").reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(len(ranked), dtype=int))

    if pick is not None:
        if pick < 0 or pick >= len(ranked):
            raise ValueError(f"--pick must be in [0, {len(ranked)-1}], got {pick}")
        return ranked.iloc[pick], pick, ranked

    valid = ranked.dropna(subset=["final_score"])
    if valid.empty:
        raise ValueError("sweep_results.csv has no valid final_score (all NaN)")
    chosen = valid.iloc[0]
    return chosen, int(chosen["rank"]), ranked


def _format_list_table(ranked: pd.DataFrame, top: int) -> str:
    cols = [
        "rank",
        "final_score",
        "noise_ratio",
        "n_clusters",
        "largest_cluster_fraction",
        "pca",
        "umap",
        "min_cluster_size",
        "min_samples",
        "selection_method",
    ]
    for optional in ["ari", "nmi", "purity"]:
        if optional in ranked.columns:
            cols.append(optional)
    shown = ranked.loc[:, [c for c in cols if c in ranked.columns]].head(top)
    return shown.to_string(index=False)


def _save_tree_plot(plot_fn: Any, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    try:
        plot_fn(ax)
        ax.set_title(title)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        LOGGER.info("Saved plot: %s", out_path)
    finally:
        plt.close(fig)


def _save_single_linkage_leaf_labeled(
    clusterer: Any,
    leaf_labels: list[str],
    out_dir: Path,
    title: str,
) -> None:
    linkage = np.asarray(clusterer.single_linkage_tree_.to_numpy())
    if linkage.ndim != 2 or linkage.shape[1] != 4:
        raise ValueError(f"single linkage matrix must have shape [N-1,4], got {linkage.shape}")

    n_leaves = len(leaf_labels)
    if linkage.shape[0] != max(0, n_leaves - 1):
        raise ValueError(
            f"single linkage row count and label count mismatch: linkage_rows={linkage.shape[0]}, labels={n_leaves}"
        )

    fig_width = float(np.clip(n_leaves * 0.25, 24, 200))
    fig, ax = plt.subplots(figsize=(fig_width, 10))
    try:
        dendrogram(
            linkage,
            labels=leaf_labels,
            leaf_rotation=90,
            leaf_font_size=4,
            ax=ax,
            distance_sort=False,
            count_sort=False,
        )
        ax.set_title(title)
        ax.set_ylabel("distance")
        fig.tight_layout()

        for ext in ["png", "pdf"]:
            out_path = out_dir / f"single_linkage_tree_with_leaf_labels.{ext}"
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            LOGGER.info("Saved plot: %s", out_path)
    finally:
        plt.close(fig)

    try:
        import plotly.figure_factory as ff

        # create_dendrogram のXは観測行列であり、linkageはlinkagefunで注入する。
        dummy_X = np.arange(n_leaves, dtype=float).reshape(-1, 1)
        fig_html = ff.create_dendrogram(
            dummy_X,
            labels=leaf_labels,
            orientation="bottom",
            linkagefun=lambda _: linkage,
            distfun=lambda x: x,
            color_threshold=None,
        )

        # Plotly の対数軸では 0 を表示できないため、全トレースを微小量だけ持ち上げる。
        positive_dist = linkage[:, 2][linkage[:, 2] > 0]
        min_positive = float(np.min(positive_dist)) if positive_dist.size > 0 else 1e-6
        y_shift = max(min_positive * 1e-3, 1e-12)
        for trace in fig_html.data:
            y = getattr(trace, "y", None)
            if y is None:
                continue
            y_arr = np.asarray(y, dtype=float)
            trace.y = (y_arr + y_shift).tolist()

        fig_html.update_layout(
            title=title,
            width=int(np.clip(n_leaves * 25, 1600, 22000)),
            height=900,
            xaxis={"tickangle": 90, "tickfont": {"size": 8}},
            yaxis={"type": "log", "title": "distance (log scale)"},
        )
        out_html = out_dir / "single_linkage_tree_with_leaf_labels.html"
        fig_html.write_html(str(out_html), include_plotlyjs="directory")
        LOGGER.info("Saved plot: %s", out_html)
    except ImportError as exc:
        LOGGER.warning("Plotly is not installed; skipping single_linkage_tree_with_leaf_labels.html: %s", exc)
    except Exception as exc:
        LOGGER.warning("Failed to save single_linkage_tree_with_leaf_labels.html: %s", exc)


def _save_condensed_tree_selected_safe(
    clusterer: Any,
    out_path: Path,
    title: str,
    *,
    label_clusters: bool = True,
    leaf_separation: float = 1.0,
    log_size: bool = False,
    max_rectangles_per_icicle: int = 20,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    try:
        clusterer.condensed_tree_.plot(
            axis=ax,
            select_clusters=False,
            leaf_separation=leaf_separation,
            log_size=log_size,
            max_rectangles_per_icicle=max_rectangles_per_icicle,
        )

        plot_data = clusterer.condensed_tree_.get_plot_data(
            leaf_separation=leaf_separation,
            log_size=log_size,
            max_rectangle_per_icicle=max_rectangles_per_icicle,
        )
        chosen_clusters = clusterer.condensed_tree_._select_clusters()

        plot_range = np.hstack([plot_data["bar_tops"], plot_data["bar_bottoms"]])
        finite_range = plot_range[np.isfinite(plot_range)]
        if finite_range.size == 0:
            finite_range = np.array([0.0, 1.0], dtype=float)

        mean_y_center = float(np.mean([np.max(finite_range), np.min(finite_range)]))
        max_height = float(np.percentile(finite_range, 90) - np.percentile(finite_range, 10))
        if not np.isfinite(max_height) or max_height <= 0:
            max_height = 1.0
        min_height = 0.1 * max_height

        for c in chosen_clusters:
            c_bounds = plot_data["cluster_bounds"][c]
            left, right, bottom, top = map(float, c_bounds)
            width = right - left
            height = top - bottom
            center = (0.5 * (left + right), 0.5 * (top + bottom))

            if not np.isfinite(center[1]):
                center = (center[0], mean_y_center)
            if not np.isfinite(height):
                height = max_height
            if height < min_height:
                height = min_height
            if not np.isfinite(width) or width <= 0:
                width = 1.0

            ellipse = Ellipse(
                xy=center,
                width=width,
                height=height,
                facecolor="none",
                edgecolor="red",
                linewidth=1.5,
                alpha=0.9,
            )
            ax.add_artist(ellipse)

            if label_clusters:
                ax.annotate(
                    f"cluster_id={c}",
                    xy=center,
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="red",
                )

        ax.set_title(title)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        LOGGER.info("Saved plot: %s", out_path)
    finally:
        plt.close(fig)


def run(args: argparse.Namespace) -> int:
    setup_logging()
    set_seed(args.seed)

    try:
        sweep_csv = resolve_file_or_recursive_search(
            args.sweep_csv,
            patterns=["sweep_results.csv"],
            fallback_patterns=["*.csv"],
            label="sweep_csv",
        )
        sweep_df = pd.read_csv(sweep_csv)
        if sweep_df.empty:
            raise ValueError(f"sweep_results.csv is empty: {args.sweep_csv}")
        if "final_score" not in sweep_df.columns:
            raise ValueError("sweep_results.csv must contain 'final_score'")

        selected, selected_rank, ranked = _choose_ranked(sweep_df, args.pick)

        if args.list:
            print(_format_list_table(ranked, args.top))
            if args.pick is None:
                LOGGER.info("--list specified without --pick; exiting without plotting.")
                return 0

        pca = _parse_pca(selected.get("pca", 0.0))
        use_umap = _parse_bool(selected.get("umap", False), "umap")
        min_cluster_size = int(selected["min_cluster_size"])
        min_samples = int(selected["min_samples"])
        selection_method = str(selected["selection_method"])
        if selection_method not in {"leaf", "eom"}:
            raise ValueError(f"Invalid selection_method: {selection_method}")

        pca_to_use = pca
        if use_umap and pca_to_use <= 0:
            pca_to_use = 50
            LOGGER.info("UMAP is enabled with pca<=0; forcing PCA to 50 for consistency with sweep")

        LOGGER.info(
            "Selected rank=%d with final_score=%s (pca=%s, umap=%s, min_cluster_size=%d, min_samples=%d, selection_method=%s)",
            selected_rank,
            selected.get("final_score"),
            pca,
            use_umap,
            min_cluster_size,
            min_samples,
            selection_method,
        )

        emb_path = resolve_file_or_recursive_search(
            args.emb,
            patterns=["embeddings.npy"],
            fallback_patterns=["*.npy"],
            label="embeddings",
        )
        X = np.load(emb_path)
        if X.ndim != 2:
            raise ValueError(f"embeddings must be 2D [N,D], got {X.shape}")
        ids = infer_ids(emb_path, X.shape[0], args.ids)

        X_proc, _ = preprocess_for_clustering(
            X,
            normalize=args.normalize,
            pca=pca_to_use,
            pca_report=None,
            use_umap=use_umap,
            umap_n_components=args.umap_n_components,
            umap_n_neighbors=args.umap_n_neighbors,
            umap_min_dist=args.umap_min_dist,
            umap_metric=args.umap_metric,
            seed=args.seed,
        )

        import hdbscan

        hdbscan_kwargs: dict[str, Any] = {}
        if args.metric == "cosine":
            hdbscan_kwargs["algorithm"] = "generic"

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=args.metric,
            cluster_selection_method=selection_method,
            allow_single_cluster=False,
            **hdbscan_kwargs,
        )
        clusterer.fit(np.asarray(X_proc, dtype=np.float64))

        labels = np.asarray(clusterer.labels_)
        probs = getattr(clusterer, "probabilities_", np.zeros(labels.shape[0], dtype=float))
        if probs is None:
            probs = np.zeros(labels.shape[0], dtype=float)

        out_dir = ensure_dir(args.out)

        selected_payload = {
            "selected_rank": selected_rank,
            "selected_from": str(args.sweep_csv),
            "selected_row": {
                k: (None if pd.isna(v) else v.item() if isinstance(v, np.generic) else v)
                for k, v in selected.to_dict().items()
                if k != "rank"
            },
            "global_settings": {
                "normalize": args.normalize,
                "metric": args.metric,
                "umap_metric": args.umap_metric,
                "umap_n_components": args.umap_n_components,
                "umap_n_neighbors": args.umap_n_neighbors,
                "umap_min_dist": args.umap_min_dist,
                "seed": args.seed,
            },
            "resolved": {
                "pca_from_sweep": pca,
                "pca_used": pca_to_use,
                "use_umap": use_umap,
            },
            "paths": {
                "emb": str(args.emb),
                "ids": str(args.ids),
                "sweep_csv": str(args.sweep_csv),
                "out": str(out_dir),
            },
        }
        selected_yaml = out_dir / "selected_config.yaml"
        selected_yaml.write_text(
            yaml.safe_dump(selected_payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        LOGGER.info("Saved selected config: %s", selected_yaml)

        labels_csv = out_dir / "labels.csv"
        pd.DataFrame({"specimen_id": ids, "cluster_id": labels, "prob": probs}).to_csv(labels_csv, index=False)
        LOGGER.info("Saved labels: %s", labels_csv)

        try:
            condensed_csv = out_dir / "condensed_tree.csv"
            clusterer.condensed_tree_.to_pandas().to_csv(condensed_csv, index=False)
            LOGGER.info("Saved condensed tree csv: %s", condensed_csv)
        except Exception as exc:
            LOGGER.warning("Failed to save condensed_tree.csv: %s", exc)

        try:
            single_csv = out_dir / "single_linkage_tree.csv"
            clusterer.single_linkage_tree_.to_pandas().to_csv(single_csv, index=False)
            LOGGER.info("Saved single linkage tree csv: %s", single_csv)
        except Exception as exc:
            LOGGER.warning("Failed to save single_linkage_tree.csv: %s", exc)

        if not args.skip_single:
            try:
                _save_tree_plot(
                    lambda axis: clusterer.single_linkage_tree_.plot(axis=axis),
                    out_dir / "single_linkage_tree.png",
                    "HDBSCAN Single Linkage Tree",
                )
            except Exception as exc:
                LOGGER.warning("Failed to plot single_linkage_tree.png (use --skip_single to disable): %s", exc)

        if not args.skip_single_leaf_labels:
            try:
                _save_single_linkage_leaf_labeled(
                    clusterer,
                    leaf_labels=[str(x) for x in ids],
                    out_dir=out_dir,
                    title="HDBSCAN Single Linkage Tree (All Leaf Labels)",
                )
            except Exception as exc:
                LOGGER.warning(
                    "Failed to plot single_linkage_tree_with_leaf_labels.* (use --skip_single_leaf_labels to disable): %s",
                    exc,
                )

        _save_tree_plot(
            lambda axis: clusterer.condensed_tree_.plot(axis=axis),
            out_dir / "condensed_tree.png",
            "HDBSCAN Condensed Tree",
        )

        selected_png = out_dir / "condensed_tree_selected.png"
        _save_condensed_tree_selected_safe(
            clusterer,
            selected_png,
            "HDBSCAN Condensed Tree (Selected Clusters)",
            label_clusters=True,
        )

        LOGGER.info("Done. Output directory: %s", out_dir)
        return 0

    except Exception:
        LOGGER.exception("Failed to generate HDBSCAN trees")
        return 1


def main() -> None:
    args = parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
