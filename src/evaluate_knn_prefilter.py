from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import ensure_dir


QUERY_CANDIDATES = ["query_id", "query", "qid", "query_specimen_id"]
NEIGHBOR_CANDIDATES = ["neighbor_id", "neighbor", "nid", "candidate_id", "retrieved_id"]
RANK_CANDIDATES = ["rank", "neighbor_rank", "position", "k", "idx"]
DISTANCE_CANDIDATES = ["distance", "dist", "l2_distance", "cosine_distance"]
SCORE_CANDIDATES = ["score", "similarity", "cosine_similarity", "sim"]
LABEL_ID_CANDIDATES = ["specimen_id", "specimen", "id"]
LABEL_NAME_CANDIDATES = ["Fossil category", "fossil_category", "label", "category"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate off/soft/strict k-NN results with Fossil_category labels and prefilter routing recall."
    )
    p.add_argument("--knn_dir", type=Path, required=True)
    p.add_argument("--labels_csv", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--prefilter_csv", type=Path, default=None)
    p.add_argument("--topk_values", type=str, default="1,5,10")
    p.add_argument("--merged_name", type=str, default="knn_all.csv")
    return p.parse_args()


def _normalize(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def _pick_col(df: pd.DataFrame, candidates: list[str], role: str) -> str:
    norm_to_orig = {_normalize(c): c for c in df.columns}
    for cand in candidates:
        found = norm_to_orig.get(_normalize(cand))
        if found is not None:
            return found
    raise ValueError(f"Required column for '{role}' not found. candidates={candidates}, actual={list(df.columns)}")


def _parse_topk_values(raw: str) -> list[int]:
    values: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        k = int(token)
        if k <= 0:
            raise ValueError(f"topk must be >0, got {k}")
        values.append(k)
    if not values:
        raise ValueError("--topk_values produced no valid integers")
    return sorted(set(values))


def _load_knn(knn_dir: Path, merged_name: str) -> pd.DataFrame:
    merged_path = knn_dir / merged_name
    if merged_path.exists():
        df = pd.read_csv(merged_path)
        if df.empty:
            raise ValueError(f"{merged_path} is empty")
        return df

    csv_paths = sorted(knn_dir.rglob("knn_*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No {merged_name} and no knn_*.csv found under {knn_dir}")

    frames: list[pd.DataFrame] = []
    for p in csv_paths:
        df = pd.read_csv(p)
        if not df.empty:
            frames.append(df)
    if not frames:
        raise ValueError(f"Found {len(csv_paths)} CSV files but all were empty")

    return pd.concat(frames, ignore_index=True)


def _standardize_knn(df: pd.DataFrame) -> pd.DataFrame:
    query_col = _pick_col(df, QUERY_CANDIDATES, "query")
    neighbor_col = _pick_col(df, NEIGHBOR_CANDIDATES, "neighbor")

    out = df.copy()
    out = out.rename(columns={query_col: "query_id", neighbor_col: "neighbor_id"})
    out["query_id"] = out["query_id"].astype(str)
    out["neighbor_id"] = out["neighbor_id"].astype(str)

    out["_row_order"] = np.arange(len(out), dtype=np.int64)

    rank_col = None
    for c in RANK_CANDIDATES:
        try:
            rank_col = _pick_col(out, [c], "rank")
            break
        except ValueError:
            continue

    if rank_col is not None:
        out["rank"] = pd.to_numeric(out[rank_col], errors="coerce")
        out = out.sort_values(["query_id", "rank", "_row_order"], ascending=[True, True, True])
    else:
        distance_col = None
        score_col = None
        for c in DISTANCE_CANDIDATES:
            try:
                distance_col = _pick_col(out, [c], "distance")
                break
            except ValueError:
                continue
        if distance_col is None:
            for c in SCORE_CANDIDATES:
                try:
                    score_col = _pick_col(out, [c], "score")
                    break
                except ValueError:
                    continue

        if distance_col is not None:
            out["_distance"] = pd.to_numeric(out[distance_col], errors="coerce")
            out = out.sort_values(["query_id", "_distance", "_row_order"], ascending=[True, True, True])
        elif score_col is not None:
            out["_score"] = pd.to_numeric(out[score_col], errors="coerce")
            out = out.sort_values(["query_id", "_score", "_row_order"], ascending=[True, False, True])
        else:
            out = out.sort_values(["query_id", "_row_order"], ascending=[True, True])

        out["rank"] = out.groupby("query_id").cumcount() + 1

    out["rank"] = pd.to_numeric(out["rank"], errors="coerce").fillna(np.inf)
    out = out.loc[np.isfinite(out["rank"])].copy()
    out["rank"] = out["rank"].astype(int)
    return out


def _load_labels(labels_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)
    specimen_col = _pick_col(df, LABEL_ID_CANDIDATES, "specimen_id")
    label_col = _pick_col(df, LABEL_NAME_CANDIDATES, "label")

    out = df[[specimen_col, label_col]].copy()
    out.columns = ["specimen_id", "label"]
    out["specimen_id"] = out["specimen_id"].astype(str)
    out["label"] = out["label"].astype(str)
    out = out.drop_duplicates(subset=["specimen_id"], keep="first")
    return out


def evaluate_knn(knn: pd.DataFrame, labels: pd.DataFrame, topk_values: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    label_map = labels.set_index("specimen_id")["label"].to_dict()

    eval_df = knn.copy()
    eval_df["query_label"] = eval_df["query_id"].map(label_map)
    eval_df["neighbor_label"] = eval_df["neighbor_id"].map(label_map)
    eval_df = eval_df.dropna(subset=["query_label", "neighbor_label"]).copy()

    all_queries = sorted(eval_df["query_id"].unique().tolist())
    qlabel_map = eval_df[["query_id", "query_label"]].drop_duplicates().set_index("query_id")["query_label"].to_dict()

    per_query_rows: list[dict[str, object]] = []
    per_label_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for k in topk_values:
        topk_df = eval_df.loc[eval_df["rank"] <= k].copy()
        topk_df["is_match"] = (topk_df["query_label"] == topk_df["neighbor_label"]).astype(np.int32)

        match_counts = topk_df.groupby("query_id")["is_match"].sum().to_dict()

        for qid in all_queries:
            match_count = int(match_counts.get(qid, 0))
            per_query_rows.append(
                {
                    "query_id": qid,
                    "query_label": qlabel_map.get(qid, ""),
                    "topk": int(k),
                    "match_count": match_count,
                    "match_rate": float(match_count / k),
                }
            )

        per_query_k = pd.DataFrame([r for r in per_query_rows if r["topk"] == int(k)])
        grouped = per_query_k.groupby("query_label", as_index=False)["match_rate"]
        label_summary = grouped.agg(
            n_queries="count",
            mean_match_rate="mean",
            std_match_rate="std",
            min_match_rate="min",
            q25_match_rate=lambda s: s.quantile(0.25),
            median_match_rate="median",
            q75_match_rate=lambda s: s.quantile(0.75),
            max_match_rate="max",
        )
        label_summary["std_match_rate"] = label_summary["std_match_rate"].fillna(0.0)
        label_summary = label_summary.rename(columns={"query_label": "label"})
        label_summary["topk"] = int(k)
        per_label_rows.extend(label_summary.to_dict(orient="records"))

        summary_rows.append(
            {
                "topk": int(k),
                "n_queries": int(len(per_query_k)),
                "mean_match_rate": float(per_query_k["match_rate"].mean()),
                "std_match_rate": float(per_query_k["match_rate"].std(ddof=1) if len(per_query_k) > 1 else 0.0),
            }
        )

    per_query_df = pd.DataFrame(per_query_rows).sort_values(["topk", "query_label", "query_id"])
    per_label_df = pd.DataFrame(per_label_rows).sort_values(["topk", "label"]) if per_label_rows else pd.DataFrame()
    summary = {
        "n_rows_after_label_join": int(len(eval_df)),
        "n_queries": int(len(all_queries)),
        "overall": summary_rows,
    }
    return per_query_df, per_label_df, summary


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


def evaluate_prefilter_recall(prefilter_csv: Path, labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    pf = pd.read_csv(prefilter_csv)
    specimen_col = _pick_col(pf, ["specimen_id"], "specimen_id")
    pregroup_col = _pick_col(pf, ["pregroup_id"], "pregroup_id")

    pf = pf[[specimen_col, pregroup_col]].copy()
    pf.columns = ["specimen_id", "pregroup_id"]
    pf["specimen_id"] = pf["specimen_id"].astype(str)
    pf["pregroup_id"] = pf["pregroup_id"].astype(str)

    merged = pf.merge(labels, on="specimen_id", how="inner")
    if merged.empty:
        raise ValueError("No rows remained after joining prefilter_csv and labels_csv on specimen_id")

    all_groups = set(merged["pregroup_id"].astype(str).unique().tolist())
    by_label: dict[str, list[tuple[str, str]]] = {}
    for row in merged.itertuples(index=False):
        by_label.setdefault(str(row.label), []).append((str(row.specimen_id), str(row.pregroup_id)))

    rows: list[dict[str, object]] = []
    for row in merged.itertuples(index=False):
        qid = str(row.specimen_id)
        qlabel = str(row.label)
        qgroup = str(row.pregroup_id)
        peers = [(sid, gid) for sid, gid in by_label.get(qlabel, []) if sid != qid]
        denom = len(peers)

        if denom == 0:
            same_group_recall = np.nan
            same_adjacent_recall = np.nan
        else:
            same_group_hits = sum(1 for _, gid in peers if gid == qgroup)
            adj = set(_adjacent_groups(qgroup, all_groups))
            allowed = adj | {qgroup}
            same_adjacent_hits = sum(1 for _, gid in peers if gid in allowed)
            same_group_recall = float(same_group_hits / denom)
            same_adjacent_recall = float(same_adjacent_hits / denom)

        rows.append(
            {
                "query_id": qid,
                "query_label": qlabel,
                "pregroup_id": qgroup,
                "n_same_label_total": int(denom),
                "same_group_recall": same_group_recall,
                "same_adjacent_group_recall": same_adjacent_recall,
            }
        )

    per_query = pd.DataFrame(rows).sort_values(["query_label", "query_id"])

    per_label = (
        per_query.groupby("query_label", as_index=False)
        .agg(
            n_queries=("query_id", "count"),
            mean_same_group_recall=("same_group_recall", "mean"),
            mean_same_adjacent_group_recall=("same_adjacent_group_recall", "mean"),
        )
        .rename(columns={"query_label": "label"})
        .sort_values("label")
    )

    summary = {
        "n_queries": int(len(per_query)),
        "mean_same_group_recall": float(per_query["same_group_recall"].mean()),
        "mean_same_adjacent_group_recall": float(per_query["same_adjacent_group_recall"].mean()),
    }

    return per_query, per_label, summary


def main() -> None:
    args = parse_args()
    ensure_dir(args.out)

    topk_values = _parse_topk_values(args.topk_values)
    knn_raw = _load_knn(args.knn_dir, args.merged_name)
    knn = _standardize_knn(knn_raw)
    labels = _load_labels(args.labels_csv)

    per_query, per_label, summary = evaluate_knn(knn, labels, topk_values)
    per_query.to_csv(args.out / "knn_eval_per_query.csv", index=False)
    per_label.to_csv(args.out / "knn_eval_per_label.csv", index=False)
    (args.out / "knn_eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.prefilter_csv is not None:
        pf_per_query, pf_per_label, pf_summary = evaluate_prefilter_recall(args.prefilter_csv, labels)
        pf_per_query.to_csv(args.out / "prefilter_recall_per_query.csv", index=False)
        pf_per_label.to_csv(args.out / "prefilter_recall_per_label.csv", index=False)
        (args.out / "prefilter_recall_summary.json").write_text(
            json.dumps(pf_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
