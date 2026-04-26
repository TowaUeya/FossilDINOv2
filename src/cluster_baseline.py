from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import hdbscan
import numpy as np
import pandas as pd

from src.cluster_branch_detector import run_branch_detector
from src.cluster_recursive_hdbscan import run_recursive_hdbscan
from src.utils.io import ensure_dir, load_ids, resolve_file_or_recursive_search, set_seed, setup_logging
from src.utils.vision import l2_normalize

LOGGER = logging.getLogger(__name__)


def optional_int(value: str) -> int | None:
    lowered = value.lower()
    if lowered in {"none", "null"}:
        return None
    return int(value)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fixed baseline clustering with optional auxiliary tree-based analyses")
    p.add_argument("--emb", type=Path, required=True)
    p.add_argument("--ids", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--normalize", choices=["none", "l2"], default="l2")
    p.add_argument("--method", choices=["hdbscan"], default="hdbscan")
    p.add_argument("--metric", default="euclidean")
    p.add_argument("--min_cluster_size", type=int, default=10)
    p.add_argument("--min_samples", type=optional_int, default=None)
    p.add_argument("--selection_method", choices=["eom", "leaf"], default="eom")

    p.add_argument("--auxiliary_method", choices=["baseline", "recursive", "branch"], default="baseline")

    p.add_argument("--sub_min_cluster_size", type=optional_int, default=None)
    p.add_argument("--sub_min_samples", type=optional_int, default=None)
    p.add_argument("--sub_selection_method", choices=["eom", "leaf"], default="leaf")
    p.add_argument("--split_size_frac", type=float, default=0.20)
    p.add_argument("--split_median_multiplier", type=float, default=5.0)
    p.add_argument("--min_child_clusters", type=int, default=2)
    p.add_argument("--max_largest_child_fraction", type=float, default=0.8)
    p.add_argument("--max_sub_noise_ratio", type=float, default=0.5)
    p.add_argument("--max_depth", type=int, default=1)
    p.add_argument("--sub_noise_policy", choices=["parent", "noise"], default="parent")
    p.add_argument("--random_seed", type=int, default=0)
    p.add_argument("--max_intra_distance_samples", type=int, default=2000)

    p.add_argument("--branch_min_cluster_size", type=optional_int, default=None)
    p.add_argument("--branch_selection_method", choices=["eom", "leaf"], default="eom")
    p.add_argument("--branch_detection_method", choices=["core", "full"], default="core")
    p.add_argument("--label_sides_as_branches", action="store_true")

    return p.parse_args()


def _size_stats(labels: np.ndarray) -> dict[str, float | None]:
    valid = labels[labels != -1]
    if valid.size == 0:
        return {"min": None, "max": None, "mean": None, "median": None}
    _, counts = np.unique(valid, return_counts=True)
    return {
        "min": float(counts.min()),
        "max": float(counts.max()),
        "mean": float(counts.mean()),
        "median": float(np.median(counts)),
    }


def _cluster_centroids(x: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, list[int]]:
    centroids = []
    cids = []
    for cid in sorted(int(c) for c in np.unique(labels) if c != -1):
        members = x[labels == cid]
        centroids.append(members.mean(axis=0))
        cids.append(cid)
    if not centroids:
        return np.zeros((0, x.shape[1]), dtype=np.float32), []
    return np.stack(centroids).astype(np.float32), cids


def _representatives(ids: list[str], x: np.ndarray, labels: np.ndarray, centroids: np.ndarray, cids: list[int]) -> pd.DataFrame:
    rows = []
    for idx, cid in enumerate(cids):
        m = np.where(labels == cid)[0]
        if m.size == 0:
            continue
        d = np.linalg.norm(x[m] - centroids[idx], axis=1)
        best = m[int(np.argmin(d))]
        rows.append({"cluster_id": cid, "specimen_id": ids[best], "distance_to_centroid": float(d.min())})
    return pd.DataFrame(rows)


def _run_baseline(x: np.ndarray, ids: list[str], args: argparse.Namespace) -> None:
    clusterer = hdbscan.HDBSCAN(
        metric=args.metric,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        cluster_selection_method=args.selection_method,
        prediction_data=True,
    )
    labels = clusterer.fit_predict(x)
    probs = getattr(clusterer, "probabilities_", np.ones_like(labels, dtype=float))

    pd.DataFrame({"specimen_id": ids, "cluster_id": labels, "prob": probs}).to_csv(args.out / "clusters.csv", index=False)

    centroids, cids = _cluster_centroids(x, labels)
    np.save(args.out / "cluster_centroids.npy", centroids)
    _representatives(ids, x, labels, centroids, cids).to_csv(args.out / "representative_specimens.csv", index=False)

    n_noise = int((labels == -1).sum())
    n_total = int(labels.shape[0])
    summary = {
        "n_total": n_total,
        "n_clusters": int(len(set(labels.tolist()) - {-1})),
        "n_noise": n_noise,
        "noise_ratio": float(n_noise / max(1, n_total)),
        "cluster_size_stats": _size_stats(labels),
        "metric": args.metric,
        "normalize": args.normalize,
        "min_cluster_size": args.min_cluster_size,
        "min_samples": args.min_samples,
        "selection_method": args.selection_method,
        "auxiliary_method": args.auxiliary_method,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    setup_logging()
    set_seed(args.random_seed)
    ensure_dir(args.out)

    emb_path = resolve_file_or_recursive_search(args.emb, patterns=["embeddings.npy"], fallback_patterns=["*.npy"], label="emb")
    x = np.load(emb_path)
    ids = load_ids(args.ids)
    if len(ids) != x.shape[0]:
        raise ValueError("ids length mismatch")
    if args.normalize == "l2":
        x = l2_normalize(x)

    raw_config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}

    if args.auxiliary_method == "baseline":
        _run_baseline(x, ids, args)
        LOGGER.info("Saved baseline clustering outputs to %s", args.out)
        return

    if args.auxiliary_method == "recursive":
        run_recursive_hdbscan(
            x,
            ids,
            args.out,
            metric=args.metric,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            selection_method=args.selection_method,
            sub_min_cluster_size=args.sub_min_cluster_size,
            sub_min_samples=args.sub_min_samples,
            sub_selection_method=args.sub_selection_method,
            split_size_frac=args.split_size_frac,
            split_median_multiplier=args.split_median_multiplier,
            min_child_clusters=args.min_child_clusters,
            max_largest_child_fraction=args.max_largest_child_fraction,
            max_sub_noise_ratio=args.max_sub_noise_ratio,
            max_depth=args.max_depth,
            sub_noise_policy=args.sub_noise_policy,
            random_seed=args.random_seed,
            max_intra_distance_samples=args.max_intra_distance_samples,
            raw_config=raw_config,
        )
        LOGGER.info("Saved recursive auxiliary outputs to %s", args.out)
        return

    if args.auxiliary_method == "branch":
        run_branch_detector(
            x,
            ids,
            args.out,
            metric=args.metric,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            selection_method=args.selection_method,
            branch_min_cluster_size=args.branch_min_cluster_size,
            branch_selection_method=args.branch_selection_method,
            branch_detection_method=args.branch_detection_method,
            label_sides_as_branches=args.label_sides_as_branches,
            raw_config=raw_config,
        )
        LOGGER.info("Saved branch auxiliary outputs to %s", args.out)
        return

    raise ValueError(f"Unsupported auxiliary_method: {args.auxiliary_method}")


if __name__ == "__main__":
    main()
