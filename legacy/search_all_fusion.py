"""旧実験コード。本研究の主張には使わない。"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.fusion_common import (
    SIZE_FEATURE_CHOICES,
    build_size_lookup,
    get_size_feature_value,
    load_embeddings,
    load_texture_lookup,
    normalize_size_distances,
)
from src.search_all import add_file_logger, output_csv_path
from src.utils.io import ensure_dir, set_seed, setup_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Shape-first retrieval with optional size/texture late fusion reranking.")
    p.add_argument("--emb", type=Path, required=True)
    p.add_argument("--ids", type=Path, required=True)
    p.add_argument("--prefilter_csv", type=Path, default=None)
    p.add_argument("--texture_features", type=Path, default=None)
    p.add_argument("--texture_ids", type=Path, default=None)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--prefilter_mode", choices=["off", "soft", "strict"], default="soft")
    p.add_argument("--expand_strategy", choices=["adjacent", "global"], default="adjacent")
    p.add_argument("--candidate_multiplier", type=float, default=2.0)
    p.add_argument("--lambda_size", type=float, default=0.05)
    p.add_argument("--lambda_tex", type=float, default=0.05)
    p.add_argument("--size_feature", type=str, choices=SIZE_FEATURE_CHOICES, default="log_size_scalar")
    p.add_argument(
        "--size_penalty_mode",
        type=str,
        choices=["plain_distance", "ratio_penalty", "margin_gate"],
        default="ratio_penalty",
    )
    p.add_argument("--size_ratio_threshold", type=float, default=1.5)
    p.add_argument("--size_gate_penalty", type=float, default=0.05)
    p.add_argument("--size_eps", type=float, default=1e-8)
    p.add_argument("--size_distance_norm", type=str, choices=["none", "zscore", "minmax", "robust"], default="robust")
    p.add_argument("--rerank_topk", type=int, default=50)
    p.add_argument("--metric", choices=["cosine", "l2"], default="cosine")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--merged_name", type=str, default="knn_all_fusion.csv")
    p.add_argument("--log_name", type=str, default="search_all_fusion.log")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _parse_group_tokens(group_id: str) -> tuple[int | None, str | None, str | None]:
    head = group_id.split("__", 1)[0]
    if not head.startswith("vol_q"):
        return None, None, None
    suffix = head.replace("vol_q", "")
    if not suffix.isdigit():
        return None, None, None

    rest = group_id.split("__", 1)[1] if "__" in group_id else None
    axis = None
    band = None
    if rest and "_" in rest:
        axis, band = rest.split("_", 1)
    return int(suffix), axis, band


def _adjacent_groups(target: str, all_groups: set[str]) -> list[str]:
    if target in {"-1", "nan", "None"}:
        return []
    q, axis, band = _parse_group_tokens(target)
    if q is None:
        return []
    if axis and band:
        flip = "high" if band == "low" else "low"
        cands = [f"vol_q{q}__{axis}_{flip}", f"vol_q{q - 1}__{axis}_{band}", f"vol_q{q + 1}__{axis}_{band}"]
    else:
        cands = [f"vol_q{q - 1}", f"vol_q{q + 1}"]
    return [g for g in cands if g in all_groups]


def _candidate_indices(
    q_idx: int,
    ids: list[str],
    size_table: dict[str, dict[str, float | str]],
    mode: str,
    topk: int,
    candidate_multiplier: float,
    expand_strategy: str,
) -> list[int]:
    if mode == "off" or not size_table:
        return [i for i in range(len(ids)) if i != q_idx]

    qid = ids[q_idx]
    q_meta = size_table.get(qid)
    if q_meta is None:
        return [i for i in range(len(ids)) if i != q_idx]

    q_group = str(q_meta["pregroup_id"])
    same = [i for i, sid in enumerate(ids) if i != q_idx and str(size_table.get(sid, {}).get("pregroup_id")) == q_group]
    if mode == "strict":
        return same if same else [i for i in range(len(ids)) if i != q_idx]

    min_needed = max(topk, int(np.ceil(topk * max(candidate_multiplier, 1.0))))
    if len(same) >= min_needed:
        return same

    seen = set(same)
    expanded = list(same)
    if expand_strategy == "adjacent":
        all_groups = {str(v.get("pregroup_id")) for v in size_table.values()}
        for ng in _adjacent_groups(q_group, all_groups):
            for i, sid in enumerate(ids):
                if i == q_idx or i in seen:
                    continue
                if str(size_table.get(sid, {}).get("pregroup_id")) == ng:
                    expanded.append(i)
                    seen.add(i)
                if len(expanded) >= min_needed:
                    return expanded

    for i in range(len(ids)):
        if i == q_idx or i in seen:
            continue
        expanded.append(i)
        if len(expanded) >= min_needed:
            break
    return expanded


def _tex_distance(query_tex: np.ndarray | None, cand_tex: np.ndarray | None, metric: str) -> float:
    if query_tex is None or cand_tex is None:
        return 0.0
    if metric == "cosine":
        return float(1.0 - np.clip(np.dot(query_tex, cand_tex), -1.0, 1.0))
    return float(np.linalg.norm(query_tex - cand_tex))


def _size_distance(
    query_id: str,
    cand_id: str,
    table: dict[str, dict[str, float | str]],
    size_feature: str,
    size_penalty_mode: str,
    size_ratio_threshold: float,
    size_gate_penalty: float,
    size_eps: float,
) -> float:
    qv = get_size_feature_value(table.get(query_id), size_feature)
    cv = get_size_feature_value(table.get(cand_id), size_feature)
    if not np.isfinite(qv) or not np.isfinite(cv):
        return 0.0
    q = float(qv)
    c = float(cv)
    eps = max(float(size_eps), 1e-12)
    mode = str(size_penalty_mode).strip().lower()
    log_like_features = {"log_size_scalar", "log_bbox_longest", "log_bbox_volume", "apparent_size_proxy", "apparent_size_volume_proxy"}
    if size_feature in log_like_features:
        q_ratio_base = float(np.power(10.0, q))
        c_ratio_base = float(np.power(10.0, c))
    else:
        q_ratio_base = abs(q)
        c_ratio_base = abs(c)

    if mode == "plain_distance":
        return float(abs(q - c))

    ratio_dist = float(abs(np.log(max(q_ratio_base, eps)) - np.log(max(c_ratio_base, eps))))
    if mode == "ratio_penalty":
        return ratio_dist

    if mode == "margin_gate":
        num = max(q_ratio_base, c_ratio_base)
        den = max(min(q_ratio_base, c_ratio_base), eps)
        ratio = float(num / den)
        gate = float(size_gate_penalty) if ratio > float(size_ratio_threshold) else 0.0
        return ratio_dist + gate
    return ratio_dist


def main() -> None:
    args = parse_args()
    setup_logging()
    set_seed(args.seed)
    ensure_dir(args.out)
    log_path = add_file_logger(args.out, args.log_name)

    metric = "cosine" if args.metric == "cosine" else "l2"
    X, ids = load_embeddings(args.emb, args.ids, metric=metric)
    size_table = build_size_lookup(args.prefilter_csv)

    texture_ids = args.texture_ids
    if texture_ids is None and args.texture_features is not None:
        default_texture_ids = args.texture_features.with_name("ids.txt")
        if default_texture_ids.exists():
            texture_ids = default_texture_ids
    tex_table = load_texture_lookup(args.texture_features, texture_ids)

    base_metric = "cosine" if metric == "cosine" else "euclidean"
    merged_rows: list[dict[str, float | str | int]] = []

    for q_idx, qid in enumerate(ids):
        candidate_idx = _candidate_indices(
            q_idx,
            ids,
            size_table,
            mode=args.prefilter_mode,
            topk=args.rerank_topk,
            candidate_multiplier=args.candidate_multiplier,
            expand_strategy=args.expand_strategy,
        )
        if not candidate_idx:
            continue

        cand_X = X[candidate_idx]
        k = min(max(int(args.rerank_topk), int(args.topk)), len(candidate_idx))
        local_nn = NearestNeighbors(metric=base_metric)
        local_nn.fit(cand_X)
        dists, idxs = local_nn.kneighbors(X[q_idx : q_idx + 1], n_neighbors=k)

        q_tex = tex_table.get(qid)
        candidate_rows = []
        for d_shape, local_idx in zip(dists[0], idxs[0]):
            cid = ids[candidate_idx[int(local_idx)]]
            if cid == qid:
                continue
            shape_dist = float(d_shape)
            size_dist = (
                _size_distance(
                    qid,
                    cid,
                    size_table,
                    size_feature=args.size_feature,
                    size_penalty_mode=args.size_penalty_mode,
                    size_ratio_threshold=args.size_ratio_threshold,
                    size_gate_penalty=args.size_gate_penalty,
                    size_eps=args.size_eps,
                )
                if size_table
                else 0.0
            )
            tex_dist = _tex_distance(q_tex, tex_table.get(cid), metric=metric) if tex_table else 0.0
            candidate_rows.append((cid, shape_dist, float(size_dist), float(tex_dist)))

        if candidate_rows:
            raw_size = np.array([r[2] for r in candidate_rows], dtype=np.float32)
            norm_size = normalize_size_distances(raw_size, mode=args.size_distance_norm)
        else:
            norm_size = np.zeros((0,), dtype=np.float32)

        rows = []
        for i, (cid, shape_dist, size_dist_raw, tex_dist) in enumerate(candidate_rows):
            size_dist = float(norm_size[i]) if len(norm_size) > i else 0.0
            final_dist = shape_dist + args.lambda_size * size_dist + args.lambda_tex * tex_dist
            rows.append(
                {
                    "query_id": qid,
                    "neighbor_id": cid,
                    "distance_shape": shape_dist,
                    "distance_size": size_dist,
                    "distance_size_raw": float(size_dist_raw),
                    "distance_tex": float(tex_dist),
                    "distance_final": float(final_dist),
                    "size_feature": args.size_feature,
                    "size_penalty_mode": args.size_penalty_mode,
                    "size_ratio_threshold": float(args.size_ratio_threshold),
                    "size_gate_penalty": float(args.size_gate_penalty),
                    "size_distance_norm": args.size_distance_norm,
                }
            )

        rows = sorted(rows, key=lambda x: x["distance_final"])[: args.topk]
        for rank, row in enumerate(rows, start=1):
            row["rank_final"] = rank

        merged_rows.extend(rows)
        out_csv = output_csv_path(args.out, qid)
        pd.DataFrame(rows).to_csv(out_csv, index=False)

    merged_csv = args.out / args.merged_name
    pd.DataFrame(merged_rows).to_csv(merged_csv, index=False)

    LOGGER.info("Saved fusion k-NN CSV: %s", merged_csv)
    LOGGER.info("prefilter enabled=%s", bool(size_table))
    LOGGER.info("texture enabled=%s", bool(tex_table))
    LOGGER.info("Execution log: %s", log_path)


if __name__ == "__main__":
    main()
