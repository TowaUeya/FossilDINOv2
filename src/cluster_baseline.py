from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import hdbscan
import numpy as np
import pandas as pd

from src.utils.io import ensure_dir, load_ids, resolve_file_or_recursive_search, setup_logging
from src.utils.vision import l2_normalize

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fixed baseline clustering (no sweep)")
    p.add_argument("--emb", type=Path, required=True)
    p.add_argument("--ids", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--normalize", choices=["none", "l2"], default="l2")
    p.add_argument("--method", choices=["hdbscan"], default="hdbscan")
    p.add_argument("--metric", choices=["euclidean"], default="euclidean")
    p.add_argument("--min_cluster_size", type=int, default=10)
    p.add_argument("--min_samples", type=int, default=None)
    p.add_argument("--selection_method", choices=["eom"], default="eom")
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


def main() -> None:
    args = parse_args()
    setup_logging()
    ensure_dir(args.out)

    emb_path = resolve_file_or_recursive_search(args.emb, patterns=["embeddings.npy"], fallback_patterns=["*.npy"], label="emb")
    x = np.load(emb_path)
    ids = load_ids(args.ids)
    if len(ids) != x.shape[0]:
        raise ValueError("ids length mismatch")
    if args.normalize == "l2":
        x = l2_normalize(x)

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
    }
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Saved baseline clustering outputs to %s", args.out)


if __name__ == "__main__":
    main()
