from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.utils.io import ensure_dir, load_ids
from src.utils.vision import l2_normalize


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Report local neighbors in embedding space")
    p.add_argument("--emb", type=Path, required=True)
    p.add_argument("--ids", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--k", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.out)
    ensure_dir(args.out / "exemplar_neighbors")

    x = l2_normalize(np.load(args.emb))
    ids = load_ids(args.ids)

    knn = NearestNeighbors(n_neighbors=min(args.k + 1, len(ids)), metric="euclidean").fit(x)
    dist, idx = knn.kneighbors(x)

    rows = []
    for i, sid in enumerate(ids):
        for rank, (j, d) in enumerate(zip(idx[i][1:], dist[i][1:]), start=1):
            rows.append({"specimen_id": sid, "neighbor_rank": rank, "neighbor_id": ids[int(j)], "distance": float(d)})
    pd.DataFrame(rows).to_csv(args.out / "knn_results.csv", index=False)


if __name__ == "__main__":
    main()
