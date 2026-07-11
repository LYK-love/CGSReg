#!/usr/bin/env python3
"""Evaluate a native TWISTER Pong policy in real ALE and write summary CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--twister-root", type=Path, default=Path.home() / "projects" / "twister")
    parser.add_argument("--checkpoint", default="checkpoints_epoch_50_step_100000.ckpt")
    parser.add_argument("--env-name", default="atari100k-pong")
    parser.add_argument("--run-name", default="atari100k")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--cuda-device", default="0")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((x - mean) ** 2 for x in values) / (len(values) - 1))


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    os.environ["env_name"] = args.env_name
    os.environ["run_name"] = args.run_name
    os.environ["override_config"] = json.dumps({"eval_episodes": args.episodes})

    twister_root = args.twister_root.expanduser().resolve()
    sys.path.insert(0, str(twister_root))
    os.chdir(twister_root)

    import functions  # noqa: PLC0415
    import main as twister_main  # noqa: PLC0415

    load_args = argparse.Namespace(
        config_file="configs/twister.py",
        checkpoint=args.checkpoint,
        cpu=False,
        load_last=False,
        show_dict=False,
        show_modules=False,
    )
    load_args.config = __import__("configs.twister", fromlist=["dummy"])
    model = functions.load_model(load_args)
    model.eval()

    rows = []
    for episode in range(args.episodes):
        output = model.play(verbose=False)
        score = float(output.score.detach().cpu().item() if hasattr(output.score, "detach") else output.score)
        steps = float(output.steps.detach().cpu().item() if hasattr(output.steps, "detach") else output.steps)
        row = {"episode": episode, "score": score, "length": steps}
        rows.append(row)
        print(f"episode {episode + 1:02d}/{args.episodes}: score={score:+.1f} length={steps:.0f}", flush=True)

    scores = [row["score"] for row in rows]
    lengths = [row["length"] for row in rows]
    summary = {
        "policy": "twister_dyna_style",
        "episodes": len(rows),
        "score_mean": sum(scores) / len(scores),
        "score_std_sample": sample_std(scores),
        "score_min": min(scores),
        "score_max": max(scores),
        "length_mean": sum(lengths) / len(lengths),
        "checkpoint": str((twister_root / "callbacks" / args.run_name / args.env_name / args.checkpoint).resolve()),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "pong_real_policy_eval_episodes.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "score", "length"])
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "pong_real_policy_eval_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    (args.output_dir / "pong_real_policy_eval.json").write_text(
        json.dumps({"summary": summary, "episodes": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary, flush=True)


if __name__ == "__main__":
    main()
