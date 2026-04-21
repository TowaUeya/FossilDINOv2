from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import logging

from src.utils.io import load_ids, resolve_file_or_recursive_search
from src.utils.vision import l2_normalize


EPS = 1e-12
SIZE_FEATURE_CHOICES = [
    "log_size_scalar",
    "aspect_xy",
    "elongation_12",
    "flatness_ratio",
    "bbox_longest",
    "log_bbox_longest",
    "bbox_volume",
    "log_bbox_volume",
    "apparent_size_proxy",
    "apparent_size_volume_proxy",
]
LOGGER = logging.getLogger(__name__)


def load_embeddings(emb_path: Path, ids_path: Path, metric: str) -> tuple[np.ndarray, list[str]]:
    resolved_emb = resolve_file_or_recursive_search(
        emb_path,
        patterns=["embeddings.npy"],
        fallback_patterns=["*.npy"],
        label="embeddings",
    )
    resolved_ids = resolve_file_or_recursive_search(
        ids_path,
        patterns=["ids.txt"],
        fallback_patterns=["*.txt"],
        label="ids",
    )
    X = np.load(resolved_emb).astype(np.float32)
    ids = load_ids(resolved_ids)
    if X.shape[0] != len(ids):
        raise ValueError("ids and embeddings row mismatch")
    if metric == "cosine":
        X = l2_normalize(X)
    return X, ids


def build_size_lookup(prefilter_csv: Path | None) -> dict[str, dict[str, float | str]]:
    if prefilter_csv is None or not prefilter_csv.exists():
        return {}
    resolved_prefilter = resolve_file_or_recursive_search(
        prefilter_csv,
        patterns=["prefilter_metadata.csv"],
        fallback_patterns=["*.csv"],
        label="prefilter_csv",
    )
    df = pd.read_csv(resolved_prefilter)
    required = {"specimen_id", "pregroup_id"}
    if not required.issubset(set(df.columns)):
        return {}
    available_size_cols = [c for c in SIZE_FEATURE_CHOICES if c in df.columns]
    table: dict[str, dict[str, float | str]] = {}
    for row in df.itertuples(index=False):
        sid = str(getattr(row, "specimen_id"))
        pgroup = str(getattr(row, "pregroup_id"))
        record: dict[str, float | str] = {"pregroup_id": pgroup}
        for col in SIZE_FEATURE_CHOICES:
            v = np.nan
            if col in available_size_cols:
                try:
                    v = float(getattr(row, col))
                except (TypeError, ValueError):
                    v = np.nan
            if not np.isfinite(v):
                v = np.nan
            record[col] = v
        table[sid] = record
    return table


def normalize_size_distances(values: np.ndarray, mode: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).copy()
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.float32)

    arr[~finite] = 0.0
    mode = str(mode).strip().lower()
    if mode == "none":
        return arr

    valid = arr[finite]
    if mode == "zscore":
        scale = float(np.std(valid))
        if scale <= EPS:
            return arr
        return arr / scale

    if mode == "minmax":
        vmin = float(np.min(valid))
        vmax = float(np.max(valid))
        scale = vmax - vmin
        if scale <= EPS:
            return np.zeros_like(arr, dtype=np.float32)
        out = (arr - vmin) / scale
        out[out < 0] = 0.0
        return out

    # robust (default): divide by IQR
    q1 = float(np.quantile(valid, 0.25))
    q3 = float(np.quantile(valid, 0.75))
    iqr = q3 - q1
    if iqr <= EPS:
        med = float(np.median(valid))
        iqr = med if med > EPS else 1.0
    return arr / float(iqr)


def load_texture_lookup(texture_features_path: Path | None, ids_path: Path | None) -> dict[str, np.ndarray]:
    if texture_features_path is None or ids_path is None:
        return {}
    if not texture_features_path.exists() or not ids_path.exists():
        LOGGER.warning(
            "Texture fusion disabled: missing file(s): features=%s ids=%s",
            texture_features_path,
            ids_path,
        )
        return {}
    resolved_texture = resolve_file_or_recursive_search(
        texture_features_path,
        patterns=["texture_features.npy"],
        fallback_patterns=["*.npy"],
        label="texture_features",
    )
    resolved_texture_ids = resolve_file_or_recursive_search(
        ids_path,
        patterns=["ids.txt"],
        fallback_patterns=["*.txt"],
        label="texture_ids",
    )
    feats = np.load(resolved_texture).astype(np.float32)
    tids = load_ids(resolved_texture_ids)
    if feats.shape[0] != len(tids):
        LOGGER.warning(
            "Texture fusion disabled: row mismatch features=%d ids=%d (%s, %s)",
            int(feats.shape[0]),
            int(len(tids)),
            texture_features_path,
            ids_path,
        )
        return {}
    if feats.shape[0] == 0:
        LOGGER.warning("Texture fusion disabled: texture features are empty (%s)", texture_features_path)
        return {}
    feats = l2_normalize(feats)
    return {sid: feats[i] for i, sid in enumerate(tids)}


def pair_distance(a: np.ndarray, b: np.ndarray, metric: str) -> float:
    if metric == "cosine":
        aa = a / max(float(np.linalg.norm(a)), EPS)
        bb = b / max(float(np.linalg.norm(b)), EPS)
        return float(1.0 - np.clip(np.dot(aa, bb), -1.0, 1.0))
    return float(np.linalg.norm(a - b))


def get_size_feature_value(meta: dict[str, float | str] | None, size_feature: str) -> float:
    if meta is None:
        return np.nan
    if size_feature == "apparent_size_proxy":
        v = meta.get("log_bbox_longest", np.nan)
    elif size_feature == "apparent_size_volume_proxy":
        v = meta.get("log_bbox_volume", np.nan)
    else:
        v = meta.get(size_feature, np.nan)
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return np.nan
    return fv if np.isfinite(fv) else np.nan
