"""旧実験コード。本研究の主張には使わない。"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.cluster import preprocess_for_clustering, run_hdbscan
from src.prefilter_common import save_yaml
from src.utils.io import ensure_dir, load_ids, resolve_file_or_recursive_search, set_seed, setup_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optional prefilter-assisted clustering: split by pregroup_id, then cluster shape embeddings in each group. "
            "This is a routing helper, not a replacement for the main clustering pipeline."
        )
    )
    parser.add_argument("--emb", type=Path, required=True)
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--prefilter_csv", type=Path, required=True)
    parser.add_argument("--method", choices=["hdbscan"], default="hdbscan")
    parser.add_argument("--normalize", choices=["none", "l2"], default="l2")
    parser.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--pca", type=float, default=64)
    parser.add_argument("--min_cluster_size", type=int, default=10)
    parser.add_argument("--min_samples", type=int, default=1)
    parser.add_argument("--cluster_selection_method", choices=["eom", "leaf"], default="leaf")
    parser.add_argument("--allow_single_cluster", action="store_true")
    parser.add_argument("--cluster_selection_epsilon", type=float, default=None)
    parser.add_argument("--noise_group_mode", choices=["separate", "merge"], default="separate")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _global_cluster_id(pregroup_id: str, local_cluster_id: int) -> str:
    return f"{pregroup_id}__{local_cluster_id}"


def _cluster_one_group(
    X_group: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, bool]:
    if X_group.shape[0] == 0:
        return np.array([], dtype=int), np.array([], dtype=float), False
    if X_group.shape[0] < max(2, args.min_cluster_size):
        return np.full(X_group.shape[0], -1, dtype=int), np.zeros(X_group.shape[0], dtype=float), True

    labels, probs = run_hdbscan(
        X_group,
        metric=args.metric,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        cluster_selection_method=args.cluster_selection_method,
        allow_single_cluster=args.allow_single_cluster,
        cluster_selection_epsilon=args.cluster_selection_epsilon,
    )
    return labels.astype(int), probs.astype(float), False


def main() -> None:
    args = parse_args()
    setup_logging()
    set_seed(args.seed)
    ensure_dir(args.out)

    emb_path = resolve_file_or_recursive_search(
        args.emb,
        patterns=["embeddings.npy"],
        fallback_patterns=["*.npy"],
        label="embeddings",
    )
    ids_path = resolve_file_or_recursive_search(
        args.ids,
        patterns=["ids.txt"],
        fallback_patterns=["*.txt"],
        label="ids",
    )
    prefilter_csv = resolve_file_or_recursive_search(
        args.prefilter_csv,
        patterns=["prefilter_metadata.csv"],
        fallback_patterns=["*.csv"],
        label="prefilter_csv",
    )
    X = np.load(emb_path)
    ids = load_ids(ids_path)
    if len(ids) != X.shape[0]:
        raise ValueError("ids and embeddings row mismatch")

    pf = pd.read_csv(prefilter_csv)
    if "specimen_id" not in pf.columns or "pregroup_id" not in pf.columns:
        raise ValueError("prefilter_csv must include specimen_id and pregroup_id")

    grouping_method = "unknown"
    if "grouping_method" in pf.columns and not pf["grouping_method"].dropna().empty:
        grouping_method = str(pf["grouping_method"].dropna().iloc[0])

    pregroup_map = {str(r.specimen_id): str(r.pregroup_id) for r in pf.itertuples(index=False)}
    default_group = "-1"
    group_values = [pregroup_map.get(sid, default_group) for sid in ids]
    if args.noise_group_mode == "merge":
        group_values = [("merged_noise" if g == "-1" else g) for g in group_values]

    X_proc, preprocess_summary = preprocess_for_clustering(
        X,
        normalize=args.normalize,
        pca=args.pca,
        pca_report=None,
        use_umap=False,
        umap_n_components=15,
        umap_n_neighbors=30,
        umap_min_dist=0.0,
        umap_metric=args.metric,
        seed=args.seed,
    )

    df = pd.DataFrame({"specimen_id": ids, "pregroup_id": group_values})
    results = []
    group_summary = []
    group_sizes: dict[str, int] = {}
    small_groups_skipped: list[str] = []

    for pregroup_id, gdf in df.groupby("pregroup_id", sort=True):
        idx = gdf.index.to_numpy()
        labels, probs, skipped = _cluster_one_group(X_proc[idx], args)

        if skipped:
            small_groups_skipped.append(str(pregroup_id))

        for local_i, specimen_id in enumerate(gdf["specimen_id"].tolist()):
            local_label = int(labels[local_i]) if local_i < len(labels) else -1
            prob = float(probs[local_i]) if local_i < len(probs) else 0.0
            results.append(
                {
                    "specimen_id": specimen_id,
                    "pregroup_id": str(pregroup_id),
                    "cluster_id": _global_cluster_id(str(pregroup_id), local_label),
                    "prob": prob,
                }
            )

        n_noise = int(np.sum(labels == -1))
        group_sizes[str(pregroup_id)] = int(len(idx))
        group_summary.append(
            {
                "pregroup_id": str(pregroup_id),
                "n_samples": int(len(idx)),
                "n_noise": n_noise,
                "n_clusters": int(len(set(labels.tolist()) - {-1})),
                "skipped": bool(skipped),
            }
        )

    out_csv = args.out / "clusters_prefilter.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False)

    summary = {
        "n_total": len(ids),
        "grouping_method": grouping_method,
        "n_groups": len(group_summary),
        "group_sizes": group_sizes,
        "small_groups_skipped": small_groups_skipped,
        "noise_group_mode": args.noise_group_mode,
        "group_summary": group_summary,
    }
    summary.update(preprocess_summary)
    summary_path = args.out / "cluster_prefilter_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    config = vars(args).copy()
    config["emb"] = str(args.emb)
    config["ids"] = str(args.ids)
    config["prefilter_csv"] = str(args.prefilter_csv)
    config["out"] = str(args.out)
    save_yaml(args.out / "cluster_prefilter_config.yaml", config)

    LOGGER.info("Saved clusters to %s", out_csv)
    LOGGER.info("Saved summary to %s", summary_path)


if __name__ == "__main__":
    main()
