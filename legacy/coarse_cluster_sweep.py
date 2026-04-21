"""旧実験コード。本研究の主張には使わない。"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from tqdm import tqdm

from src.cluster import infer_ids, preprocess_for_clustering, run_hdbscan
from src.utils.io import ensure_dir, resolve_file_or_recursive_search, set_seed, setup_logging

LOGGER = logging.getLogger(__name__)

DEFAULT_PCA_VALUES = [0.0, 64.0, 0.95]
DEFAULT_MIN_CLUSTER_SIZES = [5, 10, 20, 40]
DEFAULT_MIN_SAMPLES_VALUES = [1, 2, 5]
DEFAULT_SELECTION_METHODS = ["leaf", "eom"]
DEFAULT_UMAP_OPTIONS = ["off", "on"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Coarse clustering sweep: re-rank HDBSCAN configs with coarse_score or "
            "fallback to internal grid search when sweep CSV is unavailable"
        )
    )
    parser.add_argument("--sweep_csv", type=Path, default=None, help="Optional sweep_results.csv from src.cluster_sweep")
    parser.add_argument("--emb", type=Path, required=True, help="Path to embeddings.npy")
    parser.add_argument("--ids", type=Path, required=True, help="Path to ids.txt")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--labels", type=Path, default=None, help="Optional labels file (diagnostics only)")

    parser.add_argument("--normalize", choices=["none", "l2"], default="l2")
    parser.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--umap_metric", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--umap_n_components", type=int, default=15)
    parser.add_argument("--umap_n_neighbors", type=int, default=30)
    parser.add_argument("--umap_min_dist", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)

    # fallback grid values (mode B)
    parser.add_argument("--pca_values", type=float, nargs="+", default=DEFAULT_PCA_VALUES)
    parser.add_argument("--min_cluster_sizes", type=int, nargs="+", default=DEFAULT_MIN_CLUSTER_SIZES)
    parser.add_argument("--min_samples_values", type=int, nargs="+", default=DEFAULT_MIN_SAMPLES_VALUES)
    parser.add_argument("--selection_methods", choices=["leaf", "eom"], nargs="+", default=DEFAULT_SELECTION_METHODS)
    parser.add_argument("--umap_options", choices=["off", "on"], nargs="+", default=DEFAULT_UMAP_OPTIONS)

    parser.add_argument("--list", action="store_true", help="List top coarse-ranked configs and exit")
    parser.add_argument("--topn", type=int, default=20, help="Top-N rows to print in --list mode")
    parser.add_argument("--pick", type=int, default=None, help="Choose config by coarse_rank (1-based)")
    return parser.parse_args()


def load_labels(path: Path | None, ids: list[str]) -> np.ndarray | None:
    """Load labels only for diagnostics. Never used in coarse_score."""
    if path is None:
        LOGGER.info("No --labels specified. ARI/NMI/purity diagnostics are skipped.")
        return None

    if not path.exists():
        LOGGER.warning("labels file not found: %s", path)
        return None

    rows = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        LOGGER.warning("labels file is empty: %s", path)
        return None

    if "\t" in rows[0] or "," in rows[0]:
        sep = "\t" if "\t" in rows[0] else ","
        mapping: dict[str, str] = {}
        for row in rows:
            cols = row.split(sep)
            if len(cols) >= 2:
                mapping[cols[0]] = cols[1]
        labels = np.asarray([mapping.get(sid, "") for sid in ids], dtype=object)
        if np.all(labels == ""):
            LOGGER.warning("No ids matched mapping labels: %s", path)
            return None
        return labels

    if len(rows) != len(ids):
        LOGGER.warning("labels count mismatch (%d vs %d): %s", len(rows), len(ids), path)
        return None

    return np.asarray(rows, dtype=object)


def purity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    correct = 0
    for cluster_id in np.unique(y_pred):
        members = y_true[y_pred == cluster_id]
        if len(members) == 0:
            continue
        _, counts = np.unique(members, return_counts=True)
        correct += int(np.max(counts))
    return float(correct / len(y_true))


def run_config(
    X: np.ndarray,
    ids: list[str],
    *,
    normalize: str,
    metric: str,
    pca: float,
    use_umap: bool,
    min_cluster_size: int,
    min_samples: int,
    selection_method: str,
    umap_n_components: int,
    umap_n_neighbors: int,
    umap_min_dist: float,
    umap_metric: str,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, Any]]:
    """Run preprocessing + HDBSCAN for one config."""
    pca_to_use = pca
    if use_umap and pca_to_use <= 0:
        pca_to_use = 50

    X_proc, preprocess_summary = preprocess_for_clustering(
        X,
        normalize=normalize,
        pca=pca_to_use,
        pca_report=None,
        use_umap=use_umap,
        umap_n_components=umap_n_components,
        umap_n_neighbors=umap_n_neighbors,
        umap_min_dist=umap_min_dist,
        umap_metric=umap_metric,
        seed=seed,
    )

    labels, probs = run_hdbscan(
        X_proc,
        metric=metric,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=selection_method,
        allow_single_cluster=False,
        cluster_selection_epsilon=None,
    )

    df = pd.DataFrame({"specimen_id": ids, "cluster_id": labels, "prob": probs})
    return df, labels, probs, preprocess_summary


def compute_coarse_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    min_cluster_size: int,
    selection_method: str,
) -> dict[str, Any]:
    n_total = len(labels)
    non_noise_mask = labels != -1
    non_noise = labels[non_noise_mask]
    n_non_noise = int(np.sum(non_noise_mask))
    n_noise = int(n_total - n_non_noise)

    n_clusters = int(len(set(non_noise.tolist()))) if n_non_noise > 0 else 0
    noise_ratio = float(n_noise / n_total) if n_total > 0 else 1.0

    largest_cluster_fraction = 0.0
    second_largest_cluster_fraction = 0.0
    size_entropy = 0.0
    effective_num_clusters = 0.0

    if n_non_noise > 0 and n_clusters > 0:
        _, counts = np.unique(non_noise, return_counts=True)
        counts = np.sort(counts)[::-1]
        largest_cluster_fraction = float(counts[0] / n_non_noise)
        second_largest_cluster_fraction = float(counts[1] / n_non_noise) if len(counts) >= 2 else 0.0
        small_cluster_count = int(np.sum(counts < int(min_cluster_size)))

        p = counts.astype(float) / float(n_non_noise)
        p = p[p > 0.0]
        entropy_raw = float(-np.sum(p * np.log(p)))
        size_entropy = float(entropy_raw / np.log(n_clusters)) if n_clusters >= 2 else 0.0
        effective_num_clusters = float(np.exp(entropy_raw))
    else:
        small_cluster_count = 0

    mean_prob_non_noise = float(np.mean(probs[non_noise_mask])) if n_non_noise > 0 else 0.0

    giant_cluster_penalty = max(0.0, (largest_cluster_fraction - 0.55) / 0.15)
    fragmentation_penalty = max(0.0, (n_clusters - 6) / 10.0)

    invalid_conditions = [
        n_clusters <= 1,
        noise_ratio > 0.10,
        largest_cluster_fraction > 0.70,
        second_largest_cluster_fraction < 0.15,
        effective_num_clusters < 1.8,
    ]
    invalid = bool(any(invalid_conditions))

    reject_reasons: list[str] = []
    if n_clusters <= 1:
        reject_reasons.append("n_clusters<=1")
    if noise_ratio > 0.10:
        reject_reasons.append("noise_ratio>0.10")
    if largest_cluster_fraction > 0.70:
        reject_reasons.append("largest_cluster_fraction>0.70")
    if second_largest_cluster_fraction < 0.15:
        reject_reasons.append("second_largest_cluster_fraction<0.15")
    if effective_num_clusters < 1.8:
        reject_reasons.append("effective_num_clusters<1.8")

    return {
        "noise_ratio": noise_ratio,
        "n_clusters": float(n_clusters),
        "largest_cluster_fraction": largest_cluster_fraction,
        "second_largest_cluster_fraction": second_largest_cluster_fraction,
        "size_entropy": size_entropy,
        "effective_num_clusters": effective_num_clusters,
        "mean_prob_non_noise": mean_prob_non_noise,
        "min_cluster_size": float(min_cluster_size),
        "selection_method": selection_method,
        "small_cluster_count": float(small_cluster_count),
        "giant_cluster_penalty": giant_cluster_penalty,
        "fragmentation_penalty": fragmentation_penalty,
        "invalid": invalid,
        "invalid_reason": ";".join(reject_reasons),
    }


def compute_coarse_score(metrics: dict[str, Any]) -> float:
    if bool(metrics["invalid"]):
        return float("-inf")

    second = float(metrics["second_largest_cluster_fraction"])
    size_entropy = float(metrics["size_entropy"])
    effective_k = float(metrics["effective_num_clusters"])
    noise_ratio = float(metrics["noise_ratio"])
    mean_prob = float(metrics["mean_prob_non_noise"])
    min_cluster_size = float(metrics["min_cluster_size"])
    giant_cluster_penalty = float(metrics["giant_cluster_penalty"])
    n_clusters = float(metrics["n_clusters"])
    selection_method = str(metrics["selection_method"])
    cluster_count_penalty = max(0.0, abs(n_clusters - 2.0) - 0.5)

    return float(
        1.5 * min(second / 0.40, 1.0)
        + 1.2 * max(0.0, 1.0 - abs(effective_k - 2.0) / 1.0)
        + 0.8 * size_entropy
        + 0.6 * (1.0 - noise_ratio)
        + 0.4 * mean_prob
        + 0.3 * min(min_cluster_size / 40.0, 1.0)
        + 0.2 * (1.0 if selection_method == "eom" else 0.0)
        - 1.2 * giant_cluster_penalty
        - 0.3 * cluster_count_penalty
    )


def compute_diagnostic_metrics(labels: np.ndarray, y_true: np.ndarray | None) -> dict[str, float]:
    if y_true is None:
        return {"ari": float("nan"), "nmi": float("nan"), "purity": float("nan")}

    mask = (labels != -1) & (y_true != "")
    if int(np.sum(mask)) <= 1:
        return {"ari": float("nan"), "nmi": float("nan"), "purity": float("nan")}

    try:
        ari = float(adjusted_rand_score(y_true[mask], labels[mask]))
        nmi = float(normalized_mutual_info_score(y_true[mask], labels[mask]))
        purity = float(purity_score(y_true[mask], labels[mask]))
        return {"ari": ari, "nmi": nmi, "purity": purity}
    except Exception:
        LOGGER.exception("Diagnostic metric calculation failed; using NaN")
        return {"ari": float("nan"), "nmi": float("nan"), "purity": float("nan")}


def normalize_candidate_row(row: pd.Series) -> dict[str, Any]:
    """Normalize CSV row into concrete config values."""
    def _bool_from_any(v: Any) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, np.integer)):
            return bool(v)
        text = str(v).strip().lower()
        return text in {"true", "1", "on", "yes", "y"}

    return {
        "pca": float(row["pca"]),
        "umap": _bool_from_any(row["umap"]),
        "min_cluster_size": int(row["min_cluster_size"]),
        "min_samples": int(row["min_samples"]),
        "selection_method": str(row["selection_method"]),
    }


def build_mode_a_candidates(sweep_df: pd.DataFrame) -> list[dict[str, Any]]:
    required = {"pca", "umap", "min_cluster_size", "min_samples", "selection_method"}
    missing = required - set(sweep_df.columns)
    if missing:
        raise ValueError(f"sweep CSV missing required columns: {sorted(missing)}")

    dedup = sweep_df.drop_duplicates(subset=["pca", "umap", "min_cluster_size", "min_samples", "selection_method"])  # type: ignore[arg-type]
    return [normalize_candidate_row(row) for _, row in dedup.iterrows()]


def build_mode_b_candidates(args: argparse.Namespace) -> list[dict[str, Any]]:
    grid = list(
        itertools.product(
            args.pca_values,
            args.min_cluster_sizes,
            args.min_samples_values,
            args.selection_methods,
            args.umap_options,
        )
    )
    candidates: list[dict[str, Any]] = []
    for pca, min_cluster_size, min_samples, selection_method, umap_opt in grid:
        candidates.append(
            {
                "pca": float(pca),
                "umap": umap_opt == "on",
                "min_cluster_size": int(min_cluster_size),
                "min_samples": int(min_samples),
                "selection_method": str(selection_method),
            }
        )
    return candidates


def list_top_configs(df: pd.DataFrame, topn: int) -> None:
    display_cols = [
        "coarse_rank",
        "coarse_score",
        "noise_ratio",
        "n_clusters",
        "largest_cluster_fraction",
        "second_largest_cluster_fraction",
        "effective_num_clusters",
        "size_entropy",
        "mean_prob_non_noise",
        "pca",
        "umap",
        "min_cluster_size",
        "min_samples",
        "selection_method",
        "invalid",
        "invalid_reason",
    ]
    top_df = df.loc[:, display_cols].head(topn).copy()
    print(top_df.to_string(index=False))


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
    y_true = load_labels(args.labels, ids)

    mode = "A"
    candidates: list[dict[str, Any]]
    if args.sweep_csv is not None and args.sweep_csv.exists():
        LOGGER.info("Mode A: loading sweep CSV from %s", args.sweep_csv)
        sweep_csv = resolve_file_or_recursive_search(
            args.sweep_csv,
            patterns=["sweep_results.csv"],
            fallback_patterns=["*.csv"],
            label="sweep_csv",
        )
        sweep_df = pd.read_csv(sweep_csv)
        candidates = build_mode_a_candidates(sweep_df)
    else:
        mode = "B"
        LOGGER.info("Mode B: sweep CSV not found; running internal fallback grid search")
        candidates = build_mode_b_candidates(args)

    rows: list[dict[str, Any]] = []
    best_score = float("-inf")
    best_row: dict[str, Any] | None = None
    best_df: pd.DataFrame | None = None
    best_preprocess_summary: dict[str, Any] | None = None

    for cfg in tqdm(candidates, desc=f"coarse-sweep-mode-{mode}"):
        try:
            df, labels, probs, preprocess_summary = run_config(
                X,
                ids,
                normalize=args.normalize,
                metric=args.metric,
                pca=float(cfg["pca"]),
                use_umap=bool(cfg["umap"]),
                min_cluster_size=int(cfg["min_cluster_size"]),
                min_samples=int(cfg["min_samples"]),
                selection_method=str(cfg["selection_method"]),
                umap_n_components=args.umap_n_components,
                umap_n_neighbors=args.umap_n_neighbors,
                umap_min_dist=args.umap_min_dist,
                umap_metric=args.umap_metric,
                seed=args.seed,
            )
            coarse_metrics = compute_coarse_metrics(
                labels,
                probs,
                min_cluster_size=int(cfg["min_cluster_size"]),
                selection_method=str(cfg["selection_method"]),
            )
            coarse_score = compute_coarse_score(coarse_metrics)
            diag = compute_diagnostic_metrics(labels, y_true)
        except Exception:
            LOGGER.exception("Config failed: %s", cfg)
            df = pd.DataFrame({"specimen_id": ids, "cluster_id": np.full(len(ids), -1), "prob": np.zeros(len(ids))})
            preprocess_summary = {}
            coarse_metrics = {
                "noise_ratio": 1.0,
                "n_clusters": 0.0,
                "largest_cluster_fraction": 0.0,
                "second_largest_cluster_fraction": 0.0,
                "size_entropy": 0.0,
                "effective_num_clusters": 0.0,
                "mean_prob_non_noise": 0.0,
                "min_cluster_size": float(cfg["min_cluster_size"]),
                "selection_method": str(cfg["selection_method"]),
                "small_cluster_count": 0.0,
                "giant_cluster_penalty": 1.0,
                "fragmentation_penalty": 0.0,
                "invalid": True,
                "invalid_reason": "evaluation_failure",
            }
            coarse_score = float("-inf")
            diag = {"ari": float("nan"), "nmi": float("nan"), "purity": float("nan")}

        row = {
            "pca": float(cfg["pca"]),
            "umap": bool(cfg["umap"]),
            "min_cluster_size": int(cfg["min_cluster_size"]),
            "min_samples": int(cfg["min_samples"]),
            "selection_method": str(cfg["selection_method"]),
            **coarse_metrics,
            "coarse_score": coarse_score,
            **diag,
        }
        rows.append(row)

        if coarse_score > best_score:
            best_score = coarse_score
            best_row = row
            best_df = df
            best_preprocess_summary = preprocess_summary

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        raise RuntimeError("No candidate was evaluated")

    result_df = result_df.sort_values("coarse_score", ascending=False, na_position="last").reset_index(drop=True)
    result_df.insert(0, "coarse_rank", np.arange(1, len(result_df) + 1, dtype=int))

    required_columns = [
        "coarse_rank",
        "coarse_score",
        "noise_ratio",
        "n_clusters",
        "largest_cluster_fraction",
        "second_largest_cluster_fraction",
        "effective_num_clusters",
        "size_entropy",
        "mean_prob_non_noise",
        "pca",
        "umap",
        "min_cluster_size",
        "min_samples",
        "selection_method",
        "invalid",
        "invalid_reason",
        "ari",
        "nmi",
        "purity",
    ]
    additional_columns = [c for c in result_df.columns if c not in required_columns]
    result_df = result_df.loc[:, required_columns + additional_columns]

    coarse_csv = args.out / "coarse_sweep_results.csv"
    result_df.to_csv(coarse_csv, index=False)
    LOGGER.info("Saved coarse sweep results to %s", coarse_csv)

    if args.list:
        list_top_configs(result_df, args.topn)
        return

    if args.pick is not None:
        selected_df = result_df[result_df["coarse_rank"] == int(args.pick)]
        if selected_df.empty:
            raise ValueError(f"--pick={args.pick} does not exist. Use --list to inspect ranks.")
        selected = selected_df.iloc[0]
        selected_cfg = {
            "pca": float(selected["pca"]),
            "umap": bool(selected["umap"]),
            "min_cluster_size": int(selected["min_cluster_size"]),
            "min_samples": int(selected["min_samples"]),
            "selection_method": str(selected["selection_method"]),
        }
    else:
        if best_row is None:
            raise RuntimeError("No best row available")
        selected = result_df.iloc[0]
        selected_cfg = {
            "pca": float(selected["pca"]),
            "umap": bool(selected["umap"]),
            "min_cluster_size": int(selected["min_cluster_size"]),
            "min_samples": int(selected["min_samples"]),
            "selection_method": str(selected["selection_method"]),
        }

    # Re-run selected config to guarantee output consistency (required by spec)
    try:
        final_df, final_labels, _, final_preprocess_summary = run_config(
            X,
            ids,
            normalize=args.normalize,
            metric=args.metric,
            pca=selected_cfg["pca"],
            use_umap=selected_cfg["umap"],
            min_cluster_size=selected_cfg["min_cluster_size"],
            min_samples=selected_cfg["min_samples"],
            selection_method=selected_cfg["selection_method"],
            umap_n_components=args.umap_n_components,
            umap_n_neighbors=args.umap_n_neighbors,
            umap_min_dist=args.umap_min_dist,
            umap_metric=args.umap_metric,
            seed=args.seed,
        )
        final_diag = compute_diagnostic_metrics(final_labels, y_true)
    except Exception:
        LOGGER.exception("Failed to rerun selected config; fallback to tracked best outputs")
        if best_df is None:
            raise
        final_df = best_df
        final_labels = final_df["cluster_id"].to_numpy(dtype=int)
        final_preprocess_summary = best_preprocess_summary or {}
        final_diag = compute_diagnostic_metrics(final_labels, y_true)

    best_clusters_csv = args.out / "best_coarse_clusters.csv"
    final_df.to_csv(best_clusters_csv, index=False)
    LOGGER.info("Saved best coarse clusters to %s", best_clusters_csv)

    selected_rank = int(selected["coarse_rank"]) if "coarse_rank" in selected else -1
    config_payload = {
        "selected_by": "pick" if args.pick is not None else "best_coarse_score",
        "mode": mode,
        "selected_rank": selected_rank,
        "selected_config": selected_cfg,
        "selected_metrics": {
            "coarse_score": float(selected["coarse_score"]),
            "invalid": bool(selected["invalid"]),
            "noise_ratio": float(selected["noise_ratio"]),
            "n_clusters": int(float(selected["n_clusters"])),
            "largest_cluster_fraction": float(selected["largest_cluster_fraction"]),
            "second_largest_cluster_fraction": float(selected["second_largest_cluster_fraction"]),
            "size_entropy": float(selected["size_entropy"]),
            "effective_num_clusters": float(selected["effective_num_clusters"]),
            "mean_prob_non_noise": float(selected["mean_prob_non_noise"]),
        },
        "validation": final_diag,
        "diagnostic_note": "ARI/NMI/purity are diagnostic only and never used in coarse_score.",
        "coarse_mode_goal": "Select a small number of large and stable clusters, prioritizing coarse two-way structure.",
        "coarse_mode_priority": "Strongly prioritize second_largest_cluster_fraction and effective_num_clusters close to 2.",
        "coarse_score_note": {
            "hard_reject": {
                "n_clusters": "<=1",
                "noise_ratio": ">0.10",
                "largest_cluster_fraction": ">0.70",
                "second_largest_cluster_fraction": "<0.15",
                "effective_num_clusters": "<1.8",
            },
            "weights": {
                "second_largest_cluster_fraction": 1.5,
                "effective_num_clusters_near_2": 1.2,
                "size_entropy": 0.8,
                "one_minus_noise_ratio": 0.6,
                "mean_prob_non_noise": 0.4,
                "min_cluster_size_bonus": 0.3,
                "eom_bonus": 0.2,
                "giant_cluster_penalty": -1.2,
                "cluster_count_penalty": -0.3,
            },
            "formula_terms": {
                "giant_cluster_penalty": "max(0.0, (largest_cluster_fraction - 0.55) / 0.15)",
                "cluster_count_penalty": "max(0.0, abs(n_clusters - 2.0) - 0.5)",
            },
            "label_usage": "labels are used only for post-hoc diagnostics (ARI/NMI/purity), never for ranking/search.",
        },
        "settings": {
            "normalize": args.normalize,
            "metric": args.metric,
            "umap_metric": args.umap_metric,
            "umap_n_components": args.umap_n_components,
            "umap_n_neighbors": args.umap_n_neighbors,
            "umap_min_dist": args.umap_min_dist,
            "seed": args.seed,
        },
        "preprocess_summary": final_preprocess_summary,
    }

    best_yaml = args.out / "best_coarse_config.yaml"
    best_yaml.write_text(yaml.safe_dump(config_payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    LOGGER.info("Saved best coarse config to %s", best_yaml)

    summary = {
        "mode": mode,
        "n_candidates": int(len(result_df)),
        "selected_rank": selected_rank,
        "selected_config": selected_cfg,
        "selected_coarse_score": float(selected["coarse_score"]),
        "selected_invalid": bool(selected["invalid"]),
        "selected_noise_ratio": float(selected["noise_ratio"]),
        "selected_n_clusters": int(float(selected["n_clusters"])),
        "selected_largest_cluster_fraction": float(selected["largest_cluster_fraction"]),
        "selected_second_largest_cluster_fraction": float(selected["second_largest_cluster_fraction"]),
        "selected_size_entropy": float(selected["size_entropy"]),
        "selected_effective_num_clusters": float(selected["effective_num_clusters"]),
        "selected_mean_prob_non_noise": float(selected["mean_prob_non_noise"]),
        "validation": final_diag,
        "coarse_mode_goal": "Select a small number of large and stable clusters.",
        "coarse_mode_priority": "Prioritize second_largest_cluster_fraction and effective_num_clusters close to 2.",
        "label_usage": "labels are diagnostics-only and never used for coarse_score.",
    }
    summary_json = args.out / "summary.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Saved summary to %s", summary_json)


if __name__ == "__main__":
    main()
