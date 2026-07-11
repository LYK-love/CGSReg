#!/usr/bin/env python3
"""Summarize TWISTER-replay fixed-seed real-ALE eval outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


PROJECT_LABELS = {
    "dreamer": "DreamerV3",
    "diamond": "DIAMOND",
    "twister": "TWISTER",
    "simulus": "Simulus",
    "storm": "STORM",
}


WEIGHT_LABELS = {
    "w0": "0",
    "w0p001": "0.001",
    "w0p003": "0.003",
    "w0p005": "0.005",
    "w0p01": "0.01",
    "w0p02": "0.02",
    "w0p05": "0.05",
    "w0p1": "0.1",
    "w1": "1.0",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/dataset_ablation_twister_replay/fixed_real_eval"),
        help="Directory containing one subdirectory per project.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/dataset_ablation_twister_replay/twexp_replay_fixed_real_eval_20seed_scores.csv"),
    )
    return parser.parse_args()


def infer_weight(policy: str) -> str:
    parts = policy.replace("-", "_").split("_")
    for part in reversed(parts):
        if part in WEIGHT_LABELS:
            return WEIGHT_LABELS[part]
    return ""


def rows_from_summary(project: str, path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "project": PROJECT_LABELS.get(project, project),
                    "project_key": project,
                    "w": infer_weight(row.get("policy", "")),
                    "policy": row.get("policy", ""),
                    "episodes": row.get("episodes", ""),
                    "score_mean": row.get("score_mean", ""),
                    "score_std_sample": row.get("score_std_sample", ""),
                    "score_min": row.get("score_min", ""),
                    "score_max": row.get("score_max", ""),
                    "length_mean": row.get("length_mean", ""),
                    "checkpoint": row.get("checkpoint", ""),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    all_rows: list[dict[str, str]] = []
    for project_dir in sorted(args.root.glob("*")):
        if not project_dir.is_dir():
            continue
        summary = project_dir / "pong_real_policy_eval_summary.csv"
        if not summary.exists():
            continue
        all_rows.extend(rows_from_summary(project_dir.name, summary))

    if not all_rows:
        raise FileNotFoundError(f"No fixed eval summaries found under {args.root}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "project",
        "project_key",
        "w",
        "policy",
        "episodes",
        "score_mean",
        "score_std_sample",
        "score_min",
        "score_max",
        "length_mean",
        "checkpoint",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
