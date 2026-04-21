"""旧実験コード。本研究の主張には使わない。"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.utils.io import ensure_dir, load_ids, resolve_file_or_recursive_search, set_seed, setup_logging
from src.utils.vision import l2_normalize

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="k-NN search over specimen embeddings")
    parser.add_argument("--emb", type=Path, required=True, help="Path to embeddings.npy")
    parser.add_argument("--ids", type=Path, required=True, help="Path to ids.txt")
    parser.add_argument("--query", type=str, required=True, help="Query specimen id")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--metric", choices=["cosine", "l2"], default="cosine")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_query_id(ids: list[str], query: str) -> tuple[int, str]:
    """Resolve query string to a unique specimen id in `ids`."""
    if query in ids:
        return ids.index(query), query

    basename_matches = [sid for sid in ids if sid.rsplit("/", 1)[-1] == query]
    if len(basename_matches) == 1:
        resolved = basename_matches[0]
        return ids.index(resolved), resolved

    suffix_matches = [sid for sid in ids if sid.endswith(f"_{query}")]
    if len(suffix_matches) == 1:
        resolved = suffix_matches[0]
        return ids.index(resolved), resolved

    if query.isdigit():
        padded = query.zfill(4)
        padded_matches = [sid for sid in ids if sid.endswith(f"_{padded}")]
        if len(padded_matches) == 1:
            resolved = padded_matches[0]
            return ids.index(resolved), resolved

        prefix_matches = [sid for sid in ids if sid.endswith(f"_{query}") or sid.endswith(f"_{padded}")]
        if not prefix_matches:
            prefix_matches = [sid for sid in ids if sid.rsplit("_", 1)[-1].startswith(query)]
        if len(prefix_matches) > 1:
            preview = ", ".join(prefix_matches[:5])
            raise ValueError(
                f"query id is ambiguous: {query}. candidates (first 5): {preview}. "
                "Please provide a full id like 'airplane/airplane_0001'."
            )

    raise ValueError(f"query id not found: {query}. Try a full id from --ids file.")


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
    X = np.load(emb_path)
    ids = load_ids(ids_path)
    if len(ids) != X.shape[0]:
        raise ValueError("ids and embedding rows mismatch")

    metric = "euclidean" if args.metric == "l2" else "cosine"
    if metric == "cosine":
        X_work = l2_normalize(X.astype(np.float32))
    else:
        X_work = X.astype(np.float32)

    q_idx, resolved_query = resolve_query_id(ids, args.query)

    nn = NearestNeighbors(metric=metric)
    nn.fit(X_work)

    k = min(args.topk + 1, len(ids))
    distances, indices = nn.kneighbors(X_work[q_idx : q_idx + 1], n_neighbors=k)

    rows = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == q_idx:
            continue
        rows.append({"query_id": resolved_query, "neighbor_id": ids[idx], "distance": float(dist)})
        if len(rows) >= args.topk:
            break

    df = pd.DataFrame(rows)
    out_csv = args.out / f"knn_{resolved_query.replace('/', '_')}.csv"
    df.to_csv(out_csv, index=False)

    LOGGER.info("Top-%d neighbors for %s", len(df), resolved_query)
    for _, row in df.iterrows():
        LOGGER.info("neighbor=%s distance=%.6f", row["neighbor_id"], row["distance"])
    LOGGER.info("Saved %s", out_csv)


if __name__ == "__main__":
    main()
