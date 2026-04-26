from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

BRANCH_DETECTOR_ERROR = (
    "Your installed hdbscan package does not provide BranchDetector. "
    "Please upgrade hdbscan or use cluster_recursive_hdbscan instead."
)


def _safe_attr(obj: Any, name: str, n: int) -> np.ndarray:
    if not hasattr(obj, name):
        return np.full(n, np.nan)
    value = getattr(obj, name)
    if value is None:
        return np.full(n, np.nan)
    arr = np.asarray(value)
    if arr.ndim == 0:
        return np.full(n, arr.item())
    if arr.shape[0] != n:
        return np.full(n, np.nan)
    return arr


def _cluster_count_and_noise(labels: np.ndarray) -> tuple[int, float]:
    labels = labels.astype(float)
    n = int(labels.size)
    noise_ratio = float(np.sum(labels == -1) / max(1, n))
    valid = labels[labels != -1]
    n_clusters = int(np.unique(valid).size)
    return n_clusters, noise_ratio


def _fit_branch_detector(hdbscan_module: Any, clusterer: Any, detector_kwargs: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    attempts = [
        detector_kwargs,
        {k: v for k, v in detector_kwargs.items() if k != "label_sides_as_branches"},
        {
            k: v
            for k, v in detector_kwargs.items()
            if k not in {"label_sides_as_branches", "branch_detection_method"}
        },
        {k: v for k, v in detector_kwargs.items() if k == "min_cluster_size"},
        {},
    ]

    last_error: Exception | None = None
    for i, kwargs in enumerate(attempts):
        try:
            detector = hdbscan_module.BranchDetector(**kwargs).fit(clusterer)
            if i > 0:
                LOGGER.warning("BranchDetector constructor fallback was used. Active kwargs: %s", kwargs)
            return detector, kwargs
        except TypeError as exc:
            last_error = exc
            continue
    raise RuntimeError(f"Failed to initialize BranchDetector with compatible arguments: {last_error}")


def _save_branch_persistences(persistences: Any, out_path: Path) -> None:
    rows: list[dict[str, Any]] = []

    if persistences is None:
        pd.DataFrame(columns=["base_cluster_index", "branch_index", "branch_persistence"]).to_csv(out_path, index=False)
        return

    def _to_scalar(value: Any) -> Any:
        if value is None:
            return np.nan
        arr = np.asarray(value)
        if arr.ndim == 0:
            item = arr.item()
            if isinstance(item, np.generic):
                return item.item()
            return item
        return json.dumps(arr.tolist(), ensure_ascii=False)

    if isinstance(persistences, (list, tuple)):
        for base_cluster_index, item in enumerate(persistences):
            if item is None:
                rows.append(
                    {
                        "base_cluster_index": base_cluster_index,
                        "branch_index": np.nan,
                        "branch_persistence": np.nan,
                    }
                )
                continue

            item_arr = np.asarray(item, dtype=object)
            if item_arr.ndim == 0:
                rows.append(
                    {
                        "base_cluster_index": base_cluster_index,
                        "branch_index": 0,
                        "branch_persistence": _to_scalar(item_arr),
                    }
                )
            elif item_arr.ndim == 1:
                for branch_index, value in enumerate(item_arr):
                    rows.append(
                        {
                            "base_cluster_index": base_cluster_index,
                            "branch_index": branch_index,
                            "branch_persistence": _to_scalar(value),
                        }
                    )
            else:
                for branch_index, value in np.ndenumerate(item_arr):
                    rows.append(
                        {
                            "base_cluster_index": base_cluster_index,
                            "branch_index": ".".join(map(str, branch_index)),
                            "branch_persistence": _to_scalar(value),
                        }
                    )
    else:
        arr = np.asarray(persistences, dtype=object)
        if arr.ndim == 0:
            rows.append(
                {
                    "base_cluster_index": 0,
                    "branch_index": 0,
                    "branch_persistence": _to_scalar(arr),
                }
            )
        elif arr.ndim == 1:
            for branch_index, value in enumerate(arr):
                rows.append(
                    {
                        "base_cluster_index": np.nan,
                        "branch_index": branch_index,
                        "branch_persistence": _to_scalar(value),
                    }
                )
        else:
            for index, value in np.ndenumerate(arr):
                rows.append(
                    {
                        "base_cluster_index": index[0],
                        "branch_index": ".".join(map(str, index[1:])),
                        "branch_persistence": _to_scalar(value),
                    }
                )

    pd.DataFrame(rows).to_csv(out_path, index=False)


def run_branch_detector(
    x: np.ndarray,
    ids: list[str],
    out_dir: Path,
    *,
    metric: str,
    min_cluster_size: int,
    min_samples: int | None,
    selection_method: str,
    branch_min_cluster_size: int | None,
    branch_selection_method: str,
    branch_detection_method: str,
    label_sides_as_branches: bool,
    raw_config: dict[str, Any],
) -> None:
    if len(ids) != x.shape[0]:
        raise ValueError("ids length mismatch")

    import hdbscan

    if not hasattr(hdbscan, "BranchDetector"):
        raise RuntimeError(BRANCH_DETECTOR_ERROR)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=selection_method,
        branch_detection_data=True,
        prediction_data=True,
    ).fit(x)

    detector_kwargs = {
        "min_cluster_size": branch_min_cluster_size or min_cluster_size,
        "cluster_selection_method": branch_selection_method,
        "branch_detection_method": branch_detection_method,
        "label_sides_as_branches": label_sides_as_branches,
    }
    branch_detector, effective_kwargs = _fit_branch_detector(hdbscan, clusterer, detector_kwargs)

    n = x.shape[0]
    subgroup_labels = _safe_attr(branch_detector, "labels_", n)
    subgroup_probs = _safe_attr(branch_detector, "probabilities_", n)
    base_cluster_labels = _safe_attr(branch_detector, "cluster_labels_", n)
    base_cluster_probs = _safe_attr(branch_detector, "cluster_probabilities_", n)
    branch_labels = _safe_attr(branch_detector, "branch_labels_", n)
    branch_probs = _safe_attr(branch_detector, "branch_probabilities_", n)

    pd.DataFrame(
        {
            "specimen_id": ids,
            "cluster_id": subgroup_labels,
            "cluster": subgroup_labels,
            "prob": subgroup_probs,
        }
    ).to_csv(out_dir / "clusters.csv", index=False)

    pd.DataFrame(
        {
            "specimen_id": ids,
            "subgroup_label": subgroup_labels,
            "subgroup_probability": subgroup_probs,
            "base_cluster_label": base_cluster_labels,
            "base_cluster_probability": base_cluster_probs,
            "branch_label": branch_labels,
            "branch_probability": branch_probs,
        }
    ).to_csv(out_dir / "branch_labels.csv", index=False)

    persistences = getattr(branch_detector, "branch_persistences_", None)
    _save_branch_persistences(persistences, out_dir / "branch_persistences.csv")

    n_base_clusters, base_noise_ratio = _cluster_count_and_noise(base_cluster_labels.astype(float))
    n_subgroups, subgroup_noise_ratio = _cluster_count_and_noise(subgroup_labels.astype(float))
    n_branch_labels, branch_noise_ratio = _cluster_count_and_noise(branch_labels.astype(float))

    summary = {
        "n_samples": int(n),
        "n_base_clusters": n_base_clusters,
        "base_noise_ratio": base_noise_ratio,
        "n_subgroups": n_subgroups,
        "subgroup_noise_ratio": subgroup_noise_ratio,
        "n_branch_labels": n_branch_labels,
        "branch_noise_ratio": branch_noise_ratio,
        "branch_detection_method": branch_detection_method,
        "branch_selection_method": branch_selection_method,
        "config": raw_config,
        "effective_branch_detector_kwargs": effective_kwargs,
    }

    (out_dir / "branch_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    (out_dir / "config.json").write_text(
        json.dumps(raw_config, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
