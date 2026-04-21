from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.prefilter_common import prefilter_lookup
from src.search_all import add_file_logger, output_csv_path
from src.utils.io import ensure_dir, load_ids, resolve_file_or_recursive_search, set_seed, setup_logging
from src.utils.vision import l2_normalize

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch k-NN search with optional prefilter routing (independent from shape embedding pipeline)."
    )
    parser.add_argument("--emb", type=Path, required=True)
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--prefilter_csv", type=Path, default=None)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--metric", choices=["cosine", "l2"], default="cosine")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prefilter_mode", choices=["off", "soft", "strict"], default="soft")
    parser.add_argument("--expand_strategy", choices=["adjacent", "global"], default="adjacent")
    parser.add_argument("--candidate_multiplier", type=float, default=2.0)
    parser.add_argument("--merged_name", type=str, default="knn_all.csv")
    parser.add_argument("--log_name", type=str, default="search_all_prefilter.log")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


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

    candidates: list[str] = []
    if axis and band:
        flip = "high" if band == "low" else "low"
        candidates.append(f"vol_q{q}__{axis}_{flip}")
        candidates.append(f"vol_q{q - 1}__{axis}_{band}")
        candidates.append(f"vol_q{q + 1}__{axis}_{band}")
    else:
        candidates.append(f"vol_q{q - 1}")
        candidates.append(f"vol_q{q + 1}")

    return [g for g in candidates if g in all_groups]


def _candidate_indices(
    q_idx: int,
    ids: list[str],
    table: dict[str, dict[str, float | int | str]],
    mode: str,
    topk: int,
    candidate_multiplier: float,
    expand_strategy: str,
) -> list[int]:
    if mode == "off" or not table:
        return [i for i in range(len(ids)) if i != q_idx]

    qid = ids[q_idx]
    q_meta = table.get(qid)
    if q_meta is None:
        return [i for i in range(len(ids)) if i != q_idx]

    q_group = str(q_meta["pregroup_id"])
    same = [i for i, sid in enumerate(ids) if i != q_idx and str(table.get(sid, {}).get("pregroup_id")) == q_group]

    if mode == "strict":
        return same if same else [i for i in range(len(ids)) if i != q_idx]

    min_needed = max(topk, int(np.ceil(topk * max(candidate_multiplier, 1.0))))
    if len(same) >= min_needed:
        return same

    seen = set(same)
    expanded = list(same)

    if expand_strategy == "adjacent":
        all_groups = {str(v.get("pregroup_id")) for v in table.values()}
        for ng in _adjacent_groups(q_group, all_groups):
            for i, sid in enumerate(ids):
                if i == q_idx or i in seen:
                    continue
                if str(table.get(sid, {}).get("pregroup_id")) == ng:
                    expanded.append(i)
                    seen.add(i)
                if len(expanded) >= min_needed:
                    return expanded

    for i, sid in enumerate(ids):
        if i == q_idx or i in seen:
            continue
        if sid in table:
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


def main() -> None:
    args = parse_args()
    setup_logging()
    set_seed(args.seed)
    ensure_dir(args.out)
    log_path = add_file_logger(args.out, args.log_name)

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
    X = np.load(emb_path)
    ids = load_ids(ids_path)
    if len(ids) != X.shape[0]:
        raise ValueError("ids and embedding rows mismatch")

    metric = "euclidean" if args.metric == "l2" else "cosine"
    X_work = l2_normalize(X.astype(np.float32)) if metric == "cosine" else X.astype(np.float32)

    table: dict[str, dict[str, float | int | str]] = {}
    if args.prefilter_csv is not None and args.prefilter_mode != "off":
        if not args.prefilter_csv.exists():
            LOGGER.warning("prefilter_csv does not exist: %s", args.prefilter_csv)
        else:
            prefilter_csv = resolve_file_or_recursive_search(
                args.prefilter_csv,
                patterns=["prefilter_metadata.csv"],
                fallback_patterns=["*.csv"],
                label="prefilter_csv",
            )
            pf = pd.read_csv(prefilter_csv)
            required_cols = {"specimen_id", "pregroup_id", "pregroup_prob"}
            if required_cols.issubset(set(pf.columns)):
                table = prefilter_lookup(pf)
            else:
                LOGGER.warning("prefilter_csv missing required columns %s. fallback to off.", required_cols)

    merged_rows = []
    total_candidates = 0
    full_candidates = max(len(ids) - 1, 1)

    for q_idx, query_id in enumerate(ids):
        candidate_idx = _candidate_indices(
            q_idx,
            ids,
            table,
            args.prefilter_mode,
            args.topk,
            args.candidate_multiplier,
            args.expand_strategy,
        )
        total_candidates += len(candidate_idx)

        rows = []
        if not candidate_idx:
            out_csv = output_csv_path(args.out, query_id)
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            continue

        cand_X = X_work[candidate_idx]
        cand_ids = [ids[i] for i in candidate_idx]

        nn = NearestNeighbors(metric=metric)
        nn.fit(cand_X)
        k = min(args.topk, len(candidate_idx))
        distances, indices = nn.kneighbors(X_work[q_idx : q_idx + 1], n_neighbors=k)

        for dist, nidx in zip(distances[0], indices[0]):
            rows.append({"query_id": query_id, "neighbor_id": cand_ids[int(nidx)], "distance": float(dist)})

        merged_rows.extend(rows)
        out_csv = output_csv_path(args.out, query_id)
        pd.DataFrame(rows).to_csv(out_csv, index=False)

    merged_csv = args.out / args.merged_name
    pd.DataFrame(merged_rows).to_csv(merged_csv, index=False)

    avg_ratio = float(total_candidates / (len(ids) * full_candidates)) if len(ids) else 1.0
    LOGGER.info("prefilter_mode=%s", args.prefilter_mode)
    LOGGER.info("expand_strategy=%s", args.expand_strategy)
    LOGGER.info("Average candidate ratio vs full search: %.4f", avg_ratio)
    LOGGER.info("Saved merged k-NN CSV: %s", merged_csv)
    LOGGER.info("Execution log: %s", log_path)


if __name__ == "__main__":
    main()
