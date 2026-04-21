"""旧実験コード。本研究の主張には使わない。"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize as sk_normalize

from src.utils.io import ensure_dir, load_ids, resolve_file_or_recursive_search, set_seed, setup_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster specimen embeddings")
    parser.add_argument("--emb", type=Path, required=True, help="Path to embeddings.npy")
    parser.add_argument("--out", type=Path, required=True, help="Output results directory")
    parser.add_argument("--ids", type=Path, default=None, help="Optional ids.txt path")
    parser.add_argument("--method", choices=["hdbscan", "kmeans"], default="hdbscan")
    parser.add_argument(
        "--pca",
        type=float,
        default=0.0,
        help=(
            "PCA setting: 0 (or omitted) disables PCA, int-like value >=1 uses fixed dimensions, "
            "0<float<=1 uses explained-variance target"
        ),
    )
    parser.add_argument(
        "--pca_report",
        type=Path,
        default=None,
        help="Optional CSV path for PCA explained variance report",
    )
    parser.add_argument("--normalize", choices=["none", "l2"], default="l2")
    parser.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--min_cluster_size", type=int, default=10)
    parser.add_argument("--min_samples", type=int, default=1)
    parser.add_argument("--cluster_selection_method", choices=["eom", "leaf"], default="leaf")
    parser.add_argument("--allow_single_cluster", action="store_true")
    parser.add_argument("--cluster_selection_epsilon", type=float, default=None)
    parser.add_argument("--umap", action="store_true", help="Enable UMAP before clustering")
    parser.add_argument("--umap_n_components", type=int, default=15)
    parser.add_argument("--umap_n_neighbors", type=int, default=30)
    parser.add_argument("--umap_min_dist", type=float, default=0.0)
    parser.add_argument("--umap_metric", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--k", type=int, default=20, help="K for kmeans")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def infer_ids(emb_path: Path, n: int, ids_path: Path | None) -> list[str]:
    if ids_path is not None and ids_path.exists():
        ids = load_ids(
            resolve_file_or_recursive_search(
                ids_path,
                patterns=["ids.txt"],
                fallback_patterns=["*.txt"],
                label="ids",
            )
        )
    else:
        candidate = emb_path.parent / "ids.txt"
        ids = load_ids(candidate) if candidate.exists() else [f"{i:03d}" for i in range(n)]
    if len(ids) != n:
        raise ValueError(f"ids count ({len(ids)}) does not match embeddings ({n})")
    return ids


def resolve_pca_setting(pca_value: float) -> int | float | None:
    if pca_value <= 0:
        return None
    if 0 < pca_value <= 1:
        return pca_value

    if float(pca_value).is_integer():
        return int(pca_value)

    raise ValueError("--pca must be 0, an integer >= 1, or 0 < float <= 1")


def write_pca_report(report_path: Path, pca_model: PCA) -> None:
    ensure_dir(report_path.parent)
    ratios = pca_model.explained_variance_ratio_
    cumulative = np.cumsum(ratios)
    df = pd.DataFrame(
        {
            "component": np.arange(1, len(ratios) + 1),
            "explained_variance_ratio": ratios,
            "cumulative": cumulative,
        }
    )
    df.to_csv(report_path, index=False)


def preprocess_for_clustering(
    X: np.ndarray,
    *,
    normalize: str,
    pca: float,
    pca_report: Path | None,
    use_umap: bool,
    umap_n_components: int,
    umap_n_neighbors: int,
    umap_min_dist: float,
    umap_metric: str,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    summary: dict[str, Any] = {
        "normalize": normalize,
        "pca": pca,
        "umap": use_umap,
    }

    if normalize == "l2":
        LOGGER.info("Applying L2 normalization")
        X = sk_normalize(X, norm="l2")

    pca_setting = resolve_pca_setting(pca)
    if pca_setting is not None:
        if isinstance(pca_setting, int):
            n_components: int | float = min(pca_setting, X.shape[0], X.shape[1])
            LOGGER.info("Applying PCA with fixed n_components=%d", n_components)
        else:
            n_components = pca_setting
            LOGGER.info("Applying PCA with explained variance target=%.4f", n_components)

        pca_model = PCA(n_components=n_components, random_state=seed)
        X = pca_model.fit_transform(X)

        final_cumulative = float(np.cumsum(pca_model.explained_variance_ratio_)[-1])
        summary["pca_components"] = int(pca_model.n_components_)
        summary["pca_cumulative_explained_variance"] = final_cumulative

        LOGGER.info("PCA final dimensions: %d", int(pca_model.n_components_))
        LOGGER.info("PCA cumulative explained variance: %.6f", final_cumulative)

        if pca_report is not None:
            write_pca_report(pca_report, pca_model)
            LOGGER.info("Saved PCA report to %s", pca_report)

    if use_umap:
        try:
            import umap
        except Exception as exc:
            LOGGER.warning("UMAP is not available (%s). Skipping UMAP.", exc)
        else:
            LOGGER.info(
                "Applying UMAP(n_components=%d, n_neighbors=%d, min_dist=%.4f, metric=%s)",
                umap_n_components,
                umap_n_neighbors,
                umap_min_dist,
                umap_metric,
            )
            umap_model = umap.UMAP(
                n_components=umap_n_components,
                n_neighbors=umap_n_neighbors,
                min_dist=umap_min_dist,
                metric=umap_metric,
                random_state=seed,
            )
            X = umap_model.fit_transform(X)
            summary["umap_applied"] = True
            summary["umap_output_dim"] = int(X.shape[1])

    return X, summary


def _cluster_size_stats(labels: np.ndarray) -> dict[str, float] | dict[str, None]:
    valid = labels[labels != -1]
    if valid.size == 0:
        return {"min": None, "max": None, "mean": None, "median": None}

    _, counts = np.unique(valid, return_counts=True)
    return {
        "min": float(np.min(counts)),
        "max": float(np.max(counts)),
        "mean": float(np.mean(counts)),
        "median": float(np.median(counts)),
    }


def run_hdbscan(
    X: np.ndarray,
    *,
    metric: str,
    min_cluster_size: int,
    min_samples: int,
    cluster_selection_method: str,
    allow_single_cluster: bool,
    cluster_selection_epsilon: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    import hdbscan

    X = np.asarray(X, dtype=np.float64)

    hdbscan_kwargs: dict[str, Any] = {}
    if metric == "cosine":
        hdbscan_kwargs["algorithm"] = "generic"

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
        allow_single_cluster=allow_single_cluster,
        cluster_selection_epsilon=cluster_selection_epsilon if cluster_selection_epsilon is not None else 0.0,
        prediction_data=(metric != "cosine"),
        **hdbscan_kwargs,
    )
    labels = clusterer.fit_predict(X)
    probs = getattr(clusterer, "probabilities_", np.ones_like(labels, dtype=float))
    return labels, probs


def save_summary(out_dir: Path, summary: dict[str, Any]) -> None:
    ensure_dir(out_dir)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Saved summary to %s", summary_path)


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
    X = np.load(emb_path)
    if X.ndim != 2:
        raise ValueError(f"embeddings must be 2D [N,D], got {X.shape}")

    ids = infer_ids(emb_path, X.shape[0], args.ids)

    if args.metric == "cosine" and args.normalize == "none" and args.method == "hdbscan":
        LOGGER.warning("metric=cosine is usually paired with --normalize l2")

    try:
        X_proc, preprocess_summary = preprocess_for_clustering(
            X,
            normalize=args.normalize,
            pca=args.pca,
            pca_report=args.pca_report,
            use_umap=args.umap,
            umap_n_components=args.umap_n_components,
            umap_n_neighbors=args.umap_n_neighbors,
            umap_min_dist=args.umap_min_dist,
            umap_metric=args.umap_metric,
            seed=args.seed,
        )
    except Exception:
        LOGGER.exception("Preprocessing failed; fallback to raw embeddings")
        X_proc = X
        preprocess_summary = {}

    if args.method == "hdbscan":
        try:
            labels, scores = run_hdbscan(
                X_proc,
                metric=args.metric,
                min_cluster_size=args.min_cluster_size,
                min_samples=args.min_samples,
                cluster_selection_method=args.cluster_selection_method,
                allow_single_cluster=args.allow_single_cluster,
                cluster_selection_epsilon=args.cluster_selection_epsilon,
            )
            score_col = "prob"
        except Exception:
            LOGGER.exception("HDBSCAN failed; assigning all samples to noise")
            labels = np.full(X_proc.shape[0], -1, dtype=int)
            scores = np.zeros(X_proc.shape[0], dtype=float)
            score_col = "prob"
    else:
        clusterer = KMeans(n_clusters=args.k, random_state=args.seed, n_init="auto")
        labels = clusterer.fit_predict(X_proc)
        dists = clusterer.transform(X_proc)
        scores = dists.min(axis=1)
        score_col = "score"

    df = pd.DataFrame({"specimen_id": ids, "cluster_id": labels, score_col: scores})
    out_csv = args.out / "clusters.csv"
    df.to_csv(out_csv, index=False)
    LOGGER.info("Saved clustering results to %s", out_csv)

    n_total = int(len(labels))
    n_noise = int(np.sum(labels == -1))
    n_clusters = int(len(set(labels.tolist()) - {-1}))
    summary = {
        "n_total": n_total,
        "n_noise": n_noise,
        "noise_ratio": float(n_noise / n_total if n_total else 0.0),
        "n_clusters": n_clusters,
        "cluster_size_stats": _cluster_size_stats(labels),
        "mean_prob": float(np.mean(scores)) if len(scores) else 0.0,
        "method": args.method,
        "metric": args.metric,
        "min_cluster_size": args.min_cluster_size,
        "min_samples": args.min_samples,
        "cluster_selection_method": args.cluster_selection_method,
        "allow_single_cluster": args.allow_single_cluster,
        "cluster_selection_epsilon": args.cluster_selection_epsilon,
    }
    summary.update(preprocess_summary)
    save_summary(args.out, summary)


if __name__ == "__main__":
    main()
