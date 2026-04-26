from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import hdbscan
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances, silhouette_score

LOGGER = logging.getLogger(__name__)


def _cluster_stats(labels: np.ndarray) -> dict[str, float | int]:
    n_total = int(labels.size)
    non_noise = labels[labels != -1]
    n_clusters = int(np.unique(non_noise).size)
    n_noise = int((labels == -1).sum())
    largest = 0
    if non_noise.size > 0:
        _, counts = np.unique(non_noise, return_counts=True)
        largest = int(counts.max())
    return {
        "n_samples": n_total,
        "n_clusters": n_clusters,
        "noise_ratio": float(n_noise / max(1, n_total)),
        "largest_cluster_fraction": float(largest / max(1, n_total)),
    }


def _median_non_noise_size(labels: np.ndarray) -> float:
    non_noise = labels[labels != -1]
    if non_noise.size == 0:
        return 0.0
    _, counts = np.unique(non_noise, return_counts=True)
    return float(np.median(counts))


def _mean_intra_distance(x: np.ndarray, rng: np.random.Generator, max_samples: int, metric: str) -> float:
    n = x.shape[0]
    if n < 2:
        return float("nan")
    k = min(n, max_samples)
    if k < n:
        idx = rng.choice(n, size=k, replace=False)
        x_use = x[idx]
    else:
        x_use = x
    d = pairwise_distances(x_use, metric=metric)
    tri = d[np.triu_indices_from(d, k=1)]
    if tri.size == 0:
        return float("nan")
    return float(np.mean(tri))


def _child_weighted_intra_distance(
    x: np.ndarray,
    child_labels: np.ndarray,
    rng: np.random.Generator,
    max_samples: int,
    metric: str,
) -> float:
    values: list[float] = []
    weights: list[int] = []
    for cid in sorted(int(c) for c in np.unique(child_labels) if c != -1):
        members = x[child_labels == cid]
        dist = _mean_intra_distance(members, rng, max_samples, metric)
        if np.isnan(dist):
            continue
        values.append(dist)
        weights.append(int(members.shape[0]))
    if not values or sum(weights) == 0:
        return float("nan")
    return float(np.average(np.array(values, dtype=float), weights=np.array(weights, dtype=float)))


def _child_silhouette(x: np.ndarray, labels: np.ndarray, metric: str) -> float:
    mask = labels != -1
    x_n = x[mask]
    y_n = labels[mask]
    if x_n.shape[0] < 3:
        return float("nan")
    uniq = np.unique(y_n)
    if uniq.size < 2 or uniq.size >= x_n.shape[0]:
        return float("nan")
    try:
        return float(silhouette_score(x_n, y_n, metric=metric))
    except Exception:
        return float("nan")


def _fit_hdbscan(
    x: np.ndarray,
    metric: str,
    min_cluster_size: int,
    min_samples: int | None,
    selection_method: str,
) -> tuple[np.ndarray, np.ndarray]:
    clusterer = hdbscan.HDBSCAN(
        metric=metric,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=selection_method,
        prediction_data=True,
    )
    labels = clusterer.fit_predict(x)
    probs = np.asarray(getattr(clusterer, "probabilities_", np.ones(labels.shape[0], dtype=float)), dtype=float)
    return labels.astype(int), probs


def run_recursive_hdbscan(
    x: np.ndarray,
    ids: list[str],
    out_dir: Path,
    *,
    metric: str,
    min_cluster_size: int,
    min_samples: int | None,
    selection_method: str,
    sub_min_cluster_size: int | None,
    sub_min_samples: int | None,
    sub_selection_method: str,
    split_size_frac: float,
    split_median_multiplier: float,
    min_child_clusters: int,
    max_largest_child_fraction: float,
    max_sub_noise_ratio: float,
    max_depth: int,
    sub_noise_policy: str,
    random_seed: int,
    max_intra_distance_samples: int,
    raw_config: dict[str, Any],
) -> None:
    if len(ids) != x.shape[0]:
        raise ValueError("ids length mismatch")

    base_labels, base_probs = _fit_hdbscan(
        x,
        metric=metric,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        selection_method=selection_method,
    )

    final_labels = base_labels.copy()
    final_probs = base_probs.copy()
    recursive_depth = np.zeros(x.shape[0], dtype=int)
    was_recursively_split = np.zeros(x.shape[0], dtype=bool)

    initial_stats = _cluster_stats(base_labels)
    split_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(random_seed)

    next_cluster_id = int(np.max(base_labels[base_labels != -1]) + 1) if np.any(base_labels != -1) else 0

    for depth in range(1, max_depth + 1):
        current = final_labels.copy()
        non_noise = current[current != -1]
        if non_noise.size == 0:
            break
        median_non_noise = _median_non_noise_size(current)
        threshold = max(split_size_frac * x.shape[0], split_median_multiplier * median_non_noise)

        unique_ids, counts = np.unique(non_noise, return_counts=True)
        candidate_ids = [int(cid) for cid, c in zip(unique_ids, counts) if c > threshold]
        LOGGER.info("Depth %d: %d candidate clusters (threshold=%.2f)", depth, len(candidate_ids), threshold)

        if not candidate_ids:
            break

        for parent_cid in candidate_ids:
            parent_idx = np.where(current == parent_cid)[0]
            parent_x = x[parent_idx]
            parent_size = int(parent_idx.size)

            sub_labels, sub_probs = _fit_hdbscan(
                parent_x,
                metric=metric,
                min_cluster_size=sub_min_cluster_size or min_cluster_size,
                min_samples=sub_min_samples if sub_min_samples is not None else min_samples,
                selection_method=sub_selection_method,
            )

            child_non_noise = sub_labels[sub_labels != -1]
            n_child_clusters = int(np.unique(child_non_noise).size)
            sub_noise_ratio = float((sub_labels == -1).sum() / max(1, parent_size))

            largest_child_fraction = 0.0
            if child_non_noise.size > 0:
                _, child_counts = np.unique(child_non_noise, return_counts=True)
                largest_child_fraction = float(child_counts.max() / max(1, parent_size))

            parent_mean_intra = _mean_intra_distance(parent_x, rng, max_intra_distance_samples, metric)
            child_weighted_intra = _child_weighted_intra_distance(
                parent_x, sub_labels, rng, max_intra_distance_samples, metric
            )
            silhouette = _child_silhouette(parent_x, sub_labels, metric=metric)
            intra_improved = bool(
                np.isfinite(parent_mean_intra)
                and np.isfinite(child_weighted_intra)
                and child_weighted_intra < parent_mean_intra
            )

            hard_fail_reasons: list[str] = []
            if n_child_clusters < min_child_clusters:
                hard_fail_reasons.append("too_few_child_clusters")
            if largest_child_fraction >= max_largest_child_fraction:
                hard_fail_reasons.append("largest_child_fraction_too_high")
            if sub_noise_ratio >= max_sub_noise_ratio:
                hard_fail_reasons.append("sub_noise_ratio_too_high")

            adopted = len(hard_fail_reasons) == 0
            reject_reason = "" if adopted else ";".join(hard_fail_reasons)

            if adopted:
                local_map: dict[int, int] = {}
                for local_cid in sorted(int(c) for c in np.unique(sub_labels) if c != -1):
                    local_map[local_cid] = next_cluster_id
                    next_cluster_id += 1

                for j, global_idx in enumerate(parent_idx):
                    local = int(sub_labels[j])
                    if local == -1:
                        if sub_noise_policy == "parent":
                            final_labels[global_idx] = parent_cid
                            final_probs[global_idx] = 1.0
                        else:
                            final_labels[global_idx] = -1
                            final_probs[global_idx] = float(sub_probs[j])
                    else:
                        final_labels[global_idx] = local_map[local]
                        final_probs[global_idx] = float(sub_probs[j])
                        recursive_depth[global_idx] = depth
                        was_recursively_split[global_idx] = True

            split_rows.append(
                {
                    "depth": depth,
                    "parent_cluster": parent_cid,
                    "parent_size": parent_size,
                    "split_threshold": float(threshold),
                    "median_non_noise_cluster_size": float(median_non_noise),
                    "n_child_clusters": n_child_clusters,
                    "largest_child_fraction": largest_child_fraction,
                    "sub_noise_ratio": sub_noise_ratio,
                    "parent_mean_intra_distance": parent_mean_intra,
                    "child_weighted_mean_intra_distance": child_weighted_intra,
                    "intra_distance_improved": intra_improved,
                    "child_silhouette": silhouette,
                    "adopted": adopted,
                    "reject_reason": reject_reason,
                }
            )

    pd.DataFrame(split_rows).to_csv(out_dir / "recursive_split_report.csv", index=False)

    pd.DataFrame(
        {
            "specimen_id": ids,
            "cluster_id": final_labels,
            "prob": final_probs,
            "base_cluster": base_labels,
            "recursive_cluster": final_labels,
            "recursive_depth": recursive_depth,
            "was_recursively_split": was_recursively_split,
        }
    ).to_csv(out_dir / "clusters.csv", index=False)

    final_stats = _cluster_stats(final_labels)
    summary = {
        **final_stats,
        "initial_n_clusters": initial_stats["n_clusters"],
        "initial_noise_ratio": initial_stats["noise_ratio"],
        "initial_largest_cluster_fraction": initial_stats["largest_cluster_fraction"],
        "n_split_candidates": int(len(split_rows)),
        "n_adopted_splits": int(sum(bool(row["adopted"]) for row in split_rows)),
        "config": raw_config,
    }
    (out_dir / "cluster_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    (out_dir / "config.json").write_text(
        json.dumps(raw_config, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
