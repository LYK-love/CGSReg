#!/usr/bin/env python3
"""Quantify Pong rollout failures from CUTIE object tracks."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


OBJECTS = ("right_paddle", "left_paddle", "ball")
DISPLAY_WM = {
    "dreamer": "Dreamer",
    "diamond": "DIAMOND",
    "simulus": "Simulus",
    "storm": "STORM",
    "twister": "Twister",
}
DISPLAY_COND = {
    "exp_repro": "exp_repro",
    "w0": "w=0",
    "cgsreg_w0p01": "CGSReg",
    "cgsreg_w1p0": "CGSReg",
}
WM_ORDER = {name: i for i, name in enumerate(("dreamer", "diamond", "simulus", "twister", "storm"))}
COND_ORDER = {name: i for i, name in enumerate(("exp_repro", "w0", "cgsreg_w0p01", "cgsreg_w1p0"))}


def read_track_csv(path: Path) -> dict[str, dict[str, np.ndarray]]:
    rows_by_obj: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows_by_obj[row["object"]].append(row)

    out: dict[str, dict[str, np.ndarray]] = {}
    for obj in OBJECTS:
        rows = sorted(rows_by_obj[obj], key=lambda r: int(r["t"]))
        def f32(key: str) -> np.ndarray:
            vals = []
            for row in rows:
                val = row.get(key, "")
                vals.append(float(val) if val not in ("", None) else np.nan)
            return np.asarray(vals, dtype=np.float32)

        out[obj] = {
            "t": np.asarray([int(r["t"]) for r in rows], dtype=np.int32),
            "present": np.asarray([r["present"] == "True" for r in rows], dtype=bool),
            "area": f32("area"),
            "x": f32("x"),
            "y": f32("y"),
            "xmin": f32("xmin"),
            "xmax": f32("xmax"),
            "ymin": f32("ymin"),
            "ymax": f32("ymax"),
        }
    return out


def find_runs(mask: np.ndarray, value: bool = True) -> list[tuple[int, int]]:
    runs = []
    start = None
    for i, v in enumerate(mask):
        if bool(v) == value and start is None:
            start = i
        elif bool(v) != value and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def first_trailing_absence(
    present: np.ndarray,
    *,
    min_len: int,
) -> int | None:
    hits = np.flatnonzero(present)
    if not hits.size:
        return 0
    last_present = int(hits[-1])
    first_absent = last_present + 1
    if len(present) - first_absent >= min_len:
        return first_absent
    return None


def count_short_absent_gaps(
    present: np.ndarray,
    *,
    max_gap: int,
    min_context: int,
) -> int:
    hits = np.flatnonzero(present)
    if not hits.size:
        return 0
    lo, hi = int(hits[0]), int(hits[-1]) + 1
    count = 0
    for a, b in find_runs(~present[lo:hi], True):
        length = b - a
        abs_a, abs_b = lo + a, lo + b
        left_ok = present[max(0, abs_a - min_context):abs_a].any()
        right_ok = present[abs_b:min(len(present), abs_b + min_context)].any()
        if 1 <= length <= max_gap and left_ok and right_ok:
            count += 1
    return count


def count_spurious_mid_bounces(
    ball_x: np.ndarray,
    all_present: np.ndarray,
    *,
    mid_left: float = 30.0,
    mid_right: float = 130.0,
    min_speed: float = 0.25,
    min_separation: int = 4,
) -> int:
    valid = all_present & np.isfinite(ball_x)
    if valid.sum() < 4:
        return 0
    x = ball_x.copy()
    x[~valid] = np.nan
    last_event = -10_000
    count = 0
    for t in range(2, len(x) - 1):
        if not np.all(np.isfinite(x[t - 2:t + 2])):
            continue
        dx0 = x[t] - x[t - 1]
        dx1 = x[t + 1] - x[t]
        if abs(dx0) < min_speed or abs(dx1) < min_speed:
            continue
        if np.sign(dx0) == np.sign(dx1):
            continue
        if not (mid_left <= x[t] <= mid_right):
            continue
        if t - last_event < min_separation:
            continue
        count += 1
        last_event = t
    return count


def finite_std(x: np.ndarray, mask: np.ndarray) -> float:
    vals = x[mask & np.isfinite(x)]
    return float(np.std(vals)) if vals.size else float("nan")


def action_response_range(right_y: np.ndarray, present: np.ndarray, actions: list[int]) -> float:
    if not actions:
        return float("nan")
    n = min(len(actions), len(right_y) - 1)
    groups: dict[int, list[float]] = defaultdict(list)
    for t in range(n):
        if not present[t] or not present[t + 1]:
            continue
        dy = float(right_y[t + 1] - right_y[t])
        if math.isfinite(dy):
            groups[int(actions[t])].append(dy)
    means = [mean(v) for v in groups.values() if len(v) >= 8]
    if len(means) < 2:
        return float("nan")
    return float(max(means) - min(means))


def load_metadata(metadata_root: Path, wm: str, cond: str, episode: str) -> dict[str, Any]:
    path = metadata_root / wm / cond / f"{episode}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def analyze_episode(track_path: Path, metadata_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    sample_id = track_path.parent.name
    wm, cond, episode = sample_id.split("__", 2)
    tracks = read_track_csv(track_path)
    metadata = load_metadata(metadata_root, wm, cond, episode)

    present = {obj: tracks[obj]["present"] for obj in OBJECTS}
    frames = int(len(present["ball"]))
    all_present = present["right_paddle"] & present["left_paddle"] & present["ball"]

    first_losses = {
        obj: first_trailing_absence(
            present[obj],
            min_len=args.permanent_absence_frames,
        )
        for obj in OBJECTS
    }
    finite_losses = [x for x in first_losses.values() if x is not None]
    first_any_loss = min(finite_losses) if finite_losses else None
    survival_frames = first_any_loss if first_any_loss is not None else frames

    ball_flickers = count_short_absent_gaps(
        present["ball"],
        max_gap=args.flicker_max_gap,
        min_context=args.flicker_context,
    )
    left_flickers = count_short_absent_gaps(
        present["left_paddle"],
        max_gap=args.flicker_max_gap,
        min_context=args.flicker_context,
    )
    right_flickers = count_short_absent_gaps(
        present["right_paddle"],
        max_gap=args.flicker_max_gap,
        min_context=args.flicker_context,
    )
    spurious_bounces = count_spurious_mid_bounces(tracks["ball"]["x"], all_present)
    right_motion_std = finite_std(tracks["right_paddle"]["y"], present["right_paddle"])
    left_motion_std = finite_std(tracks["left_paddle"]["y"], present["left_paddle"])
    response_range = action_response_range(
        tracks["right_paddle"]["y"],
        present["right_paddle"],
        metadata.get("actions", []),
    )

    return {
        "sample_id": sample_id,
        "wm": wm,
        "wm_display": DISPLAY_WM.get(wm, wm),
        "condition": cond,
        "condition_display": DISPLAY_COND.get(cond, cond),
        "episode": episode,
        "frames": frames,
        "all_present_frames": int(all_present.sum()),
        "all_present_frac": float(all_present.mean()),
        "survival_frames_before_trailing_object_loss": int(survival_frames),
        "survival_frac_before_trailing_object_loss": float(survival_frames / frames),
        "any_permanent_loss": bool(first_any_loss is not None),
        "first_any_permanent_loss_t": "" if first_any_loss is None else int(first_any_loss),
        "ball_present_frac": float(present["ball"].mean()),
        "left_paddle_present_frac": float(present["left_paddle"].mean()),
        "right_paddle_present_frac": float(present["right_paddle"].mean()),
        "first_ball_permanent_loss_t": "" if first_losses["ball"] is None else int(first_losses["ball"]),
        "first_left_paddle_permanent_loss_t": "" if first_losses["left_paddle"] is None else int(first_losses["left_paddle"]),
        "first_right_paddle_permanent_loss_t": "" if first_losses["right_paddle"] is None else int(first_losses["right_paddle"]),
        "ball_flicker_events": int(ball_flickers),
        "left_paddle_flicker_events": int(left_flickers),
        "right_paddle_flicker_events": int(right_flickers),
        "spurious_midfield_x_bounce_events": int(spurious_bounces),
        "right_paddle_y_std": right_motion_std,
        "left_paddle_y_std": left_motion_std,
        "right_paddle_action_response_range": response_range,
        "policy_actions": len(metadata.get("actions", [])),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def ratio(num: float, den: float, scale: float = 1.0) -> float:
    return float(num / den * scale) if den else float("nan")


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["wm"], row["condition"])].append(row)

    out = []
    def key(item: tuple[tuple[str, str], list[dict[str, Any]]]) -> tuple[int, int, str, str]:
        (wm, cond), _ = item
        return (WM_ORDER.get(wm, 999), COND_ORDER.get(cond, 999), wm, cond)

    for (wm, cond), xs in sorted(groups.items(), key=key):
        frames = sum(int(x["frames"]) for x in xs)
        all_present_frames = sum(int(x["all_present_frames"]) for x in xs)
        ball_present_frames = sum(float(x["ball_present_frac"]) * int(x["frames"]) for x in xs)
        any_loss_eps = sum(1 for x in xs if x["any_permanent_loss"])
        ball_perm_loss_eps = sum(1 for x in xs if x["first_ball_permanent_loss_t"] != "")
        left_perm_loss_eps = sum(1 for x in xs if x["first_left_paddle_permanent_loss_t"] != "")
        right_perm_loss_eps = sum(1 for x in xs if x["first_right_paddle_permanent_loss_t"] != "")
        ball_flickers = sum(int(x["ball_flicker_events"]) for x in xs)
        paddle_flickers = sum(int(x["left_paddle_flicker_events"]) + int(x["right_paddle_flicker_events"]) for x in xs)
        spurious = sum(int(x["spurious_midfield_x_bounce_events"]) for x in xs)
        response_vals = [
            float(x["right_paddle_action_response_range"])
            for x in xs
            if math.isfinite(float(x["right_paddle_action_response_range"]))
        ]
        out.append({
            "wm": wm,
            "wm_display": DISPLAY_WM.get(wm, wm),
            "condition": cond,
            "condition_display": DISPLAY_COND.get(cond, cond),
            "episodes": len(xs),
            "mean_survival_frac": mean(float(x["survival_frac_before_trailing_object_loss"]) for x in xs),
            "mean_all_present_frac": mean(float(x["all_present_frac"]) for x in xs),
            "any_permanent_loss_eps": any_loss_eps,
            "ball_permanent_loss_eps": ball_perm_loss_eps,
            "left_paddle_permanent_loss_eps": left_perm_loss_eps,
            "right_paddle_permanent_loss_eps": right_perm_loss_eps,
            "paddle_permanent_loss_objects": left_perm_loss_eps + right_perm_loss_eps,
            "ball_flicker_per_100_ball_visible_frames": ratio(ball_flickers, ball_present_frames, 100.0),
            "paddle_flicker_per_100_paddle_visible_frames": ratio(
                paddle_flickers,
                sum((float(x["left_paddle_present_frac"]) + float(x["right_paddle_present_frac"])) * int(x["frames"]) for x in xs),
                100.0,
            ),
            "spurious_midfield_bounce_per_100_all_visible_frames": ratio(spurious, all_present_frames, 100.0),
            "mean_right_paddle_action_response_range": mean(response_vals) if response_vals else float("nan"),
            "total_frames": frames,
            "all_present_frames": all_present_frames,
            "ball_flicker_events": ball_flickers,
            "paddle_flicker_events": paddle_flickers,
            "spurious_midfield_x_bounce_events": spurious,
        })
    return out


def fmt_float(x: Any, ndigits: int = 3) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(v):
        return "NA"
    return f"{v:.{ndigits}f}"


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "WM",
        "ckpt",
        "survival",
        "visible",
        "ball perm ep",
        "L pad perm ep",
        "R pad perm ep",
        "ball flicker /100 vis",
        "paddle flicker /100 vis",
        "self-bounce /100 vis",
        "right paddle response",
    ]
    lines = [
        "# CUTIE Mask-Based Rollout Quantification",
        "",
        "Rates are normalized by observable frames. `survival` is the mean fraction",
        "of frames before any object enters a trailing absence of at least 32 frames.",
        "This keeps episodes where an object permanently disappears from looking",
        "artificially good just because later motion errors become unobservable.",
        "",
        "|" + "|".join(headers) + "|",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("|" + "|".join([
            row["wm_display"],
            row["condition_display"],
            fmt_float(row["mean_survival_frac"]),
            fmt_float(row["mean_all_present_frac"]),
            f'{row["ball_permanent_loss_eps"]}/{row["episodes"]}',
            f'{row["left_paddle_permanent_loss_eps"]}/{row["episodes"]}',
            f'{row["right_paddle_permanent_loss_eps"]}/{row["episodes"]}',
            fmt_float(row["ball_flicker_per_100_ball_visible_frames"]),
            fmt_float(row["paddle_flicker_per_100_paddle_visible_frames"]),
            fmt_float(row["spurious_midfield_bounce_per_100_all_visible_frames"]),
            fmt_float(row["mean_right_paddle_action_response_range"]),
        ]) + "|")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, default=Path("paper_rollout_eval_bundle"))
    parser.add_argument("--output-dir", type=Path, default=Path("paper_rollout_eval_bundle/results"))
    parser.add_argument("--permanent-absence-frames", type=int, default=32)
    parser.add_argument("--flicker-max-gap", type=int, default=8)
    parser.add_argument("--flicker-context", type=int, default=4)
    args = parser.parse_args()

    seg_root = args.bundle_root / "cutie_segmentations"
    metadata_root = args.bundle_root / "rollout_metadata"
    track_paths = sorted(seg_root.glob("*__*__*/cutie_tracks.csv"))
    if not track_paths:
        raise FileNotFoundError(f"No CUTIE tracks found under {seg_root}")

    episode_rows = [analyze_episode(path, metadata_root, args) for path in track_paths]
    summary_rows = summarize(episode_rows)

    write_csv(args.output_dir / "cutie_mask_quant_episode.csv", episode_rows)
    write_csv(args.output_dir / "cutie_mask_quant_summary.csv", summary_rows)
    write_markdown(args.output_dir / "cutie_mask_quant_summary.md", summary_rows)
    print(f"Wrote {len(episode_rows)} episode rows and {len(summary_rows)} summary rows")


if __name__ == "__main__":
    main()
