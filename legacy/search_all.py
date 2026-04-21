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
    parser = argparse.ArgumentParser(description="Batch k-NN search for all specimen embeddings")
    parser.add_argument("--emb", type=Path, required=True, help="Path to embeddings.npy")
    parser.add_argument("--ids", type=Path, required=True, help="Path to ids.txt")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--metric", choices=["cosine", "l2"], default="cosine")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for per-query CSV files")
    parser.add_argument(
        "--merged_name",
        type=str,
        default="knn_all.csv",
        help="File name for merged CSV (saved under --out)",
    )
    parser.add_argument("--log_name", type=str, default="search_all.log", help="Log file name under --out")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def add_file_logger(out_dir: Path, log_name: str) -> Path:
    log_path = out_dir / log_name
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    return log_path


def output_csv_path(root_out: Path, specimen_id: str) -> Path:
    sid_path = Path(specimen_id)
    out_dir = root_out / sid_path.parent
    ensure_dir(out_dir)
    return out_dir / f"knn_{sid_path.name}.csv"


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

    nn = NearestNeighbors(metric=metric)
    nn.fit(X_work)

    k = min(args.topk + 1, len(ids))
    distances, indices = nn.kneighbors(X_work, n_neighbors=k)
    merged_rows = []

    for q_idx, query_id in enumerate(ids):
        rows = []
        for dist, idx in zip(distances[q_idx], indices[q_idx]):
            if idx == q_idx:
                continue
            rows.append({"query_id": query_id, "neighbor_id": ids[idx], "distance": float(dist)})
            if len(rows) >= args.topk:
                break

        merged_rows.extend(rows)

        out_csv = output_csv_path(args.out, query_id)
        pd.DataFrame(rows).to_csv(out_csv, index=False)

    merged_csv = args.out / args.merged_name
    pd.DataFrame(merged_rows).to_csv(merged_csv, index=False)

    LOGGER.info("Saved per-query k-NN results for %d specimens", len(ids))
    LOGGER.info("Saved merged k-NN CSV: %s", merged_csv)
    LOGGER.info("Output directory: %s", args.out)
    LOGGER.info("Execution log: %s", log_path)


if __name__ == "__main__":
    main()
