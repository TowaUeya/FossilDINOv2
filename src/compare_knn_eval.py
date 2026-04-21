from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare off/soft/strict k-NN evaluation outputs.")
    p.add_argument("--baseline_dir", type=Path, required=True, help="Directory of off-mode evaluation outputs")
    p.add_argument("--soft_dir", type=Path, required=True, help="Directory of soft-mode evaluation outputs")
    p.add_argument("--strict_dir", type=Path, required=True, help="Directory of strict-mode evaluation outputs")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def _load_summary_json(eval_dir: Path) -> pd.DataFrame:
    path = eval_dir / "knn_eval_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing summary JSON: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("overall", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Invalid or empty 'overall' in {path}")
    return pd.DataFrame(rows)


def _load_per_label(eval_dir: Path) -> pd.DataFrame:
    path = eval_dir / "knn_eval_per_label.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing per-label CSV: {path}")
    df = pd.read_csv(path)
    required = {"label", "topk", "mean_match_rate"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df


def main() -> None:
    args = parse_args()
    ensure_dir(args.out)

    mode_dirs = {
        "off": args.baseline_dir,
        "soft": args.soft_dir,
        "strict": args.strict_dir,
    }

    overall_frames: list[pd.DataFrame] = []
    per_label_frames: list[pd.DataFrame] = []

    for mode, eval_dir in mode_dirs.items():
        overall = _load_summary_json(eval_dir).copy()
        overall["mode"] = mode
        overall_frames.append(overall)

        per_label = _load_per_label(eval_dir)[["label", "topk", "mean_match_rate"]].copy()
        per_label = per_label.rename(columns={"mean_match_rate": f"mean_match_rate_{mode}"})
        per_label_frames.append(per_label)

    compare_overall = pd.concat(overall_frames, ignore_index=True)
    compare_overall = compare_overall[["mode", "topk", "n_queries", "mean_match_rate", "std_match_rate"]]
    compare_overall = compare_overall.sort_values(["topk", "mode"]).reset_index(drop=True)
    compare_overall.to_csv(args.out / "compare_overall.csv", index=False)

    merged = per_label_frames[0]
    for frame in per_label_frames[1:]:
        merged = merged.merge(frame, on=["label", "topk"], how="outer")

    compare_per_label = merged.sort_values(["topk", "label"]).reset_index(drop=True)
    compare_per_label.to_csv(args.out / "compare_per_label.csv", index=False)


if __name__ == "__main__":
    main()
