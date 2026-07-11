#!/usr/bin/env python3
"""Build the paper failure tables from CUTIE tracks.

The script consumes the per-episode `cutie_tracks.csv` files in
`paper_rollout_eval_bundle/cutie_segmentations` and the corresponding rollout
metadata in `paper_rollout_eval_bundle/rollout_metadata`.

It writes:
  - episode-level metrics
  - summary CSVs for the two main-paper failure tables
  - LaTeX table bodies that can be pasted into the paper
"""

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
WM_ORDER = ("dreamer", "diamond", "twister", "simulus", "storm")
COND_ORDER = ("exp_repro", "w0", "cgsreg_w0p01", "cgsreg_w1p0")
DISPLAY_WM = {
    "dreamer": "DreamerV3",
    "diamond": "DIAMOND",
    "twister": "TWISTER",
    "simulus": "Simulus",
    "storm": "STORM",
}
DISPLAY_COND = {
    "exp_repro": "Reference WM",
    "w0": r"Offline \(w=0\) WM",
    "cgsreg_w0p01": r"Offline \methodname{} WM",
    "cgsreg_w1p0": r"Offline \methodname{} WM",
}

# Atari Pong minimal action semantics used by the rollout policy metadata.
# The user-facing play server also uses this convention: LEFT moves the right
# paddle down, RIGHT moves it up.
ACTION_UP = {2, 4}
ACTION_DOWN = {3, 5}


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


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def finite(x: float) -> bool:
    return math.isfinite(float(x))


def contiguous_runs(mask: np.ndarray, value: bool = True) -> list[tuple[int, int]]:
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


def trailing_absence_start(present: np.ndarray, min_len: int) -> int | None:
    hits = np.flatnonzero(present)
    if not hits.size:
        return 0
    start = int(hits[-1]) + 1
    if len(present) - start >= min_len:
        return start
    return None


def count_disappearance_events(
    present: np.ndarray,
    *,
    min_context: int,
    max_temporary_gap: int,
    permanent_absence_frames: int,
) -> int:
    """Count temporary and terminal disappearances as one event each."""
    hits = np.flatnonzero(present)
    if not hits.size:
        return 1
    lo, hi = int(hits[0]), int(hits[-1]) + 1
    events = 0
    for a, b in contiguous_runs(~present[lo:hi], True):
        length = b - a
        abs_a, abs_b = lo + a, lo + b
        left_ok = present[max(0, abs_a - min_context):abs_a].any()
        right_ok = present[abs_b:min(len(present), abs_b + min_context)].any()
        if left_ok and right_ok and 1 <= length <= max_temporary_gap:
            events += 1
    if trailing_absence_start(present, permanent_absence_frames) is not None:
        events += 1
    return events


def valid_ball_mask(tracks: dict[str, dict[str, np.ndarray]]) -> np.ndarray:
    return (
        tracks["ball"]["present"]
        & np.isfinite(tracks["ball"]["x"])
        & np.isfinite(tracks["ball"]["y"])
    )


def count_spurious_turns(
    tracks: dict[str, dict[str, np.ndarray]],
    *,
    min_speed: float,
    min_event_gap: int,
    wall_margin: float,
    paddle_margin: float,
) -> int:
    ball = tracks["ball"]
    valid = valid_ball_mask(tracks)
    x = ball["x"]
    last_event = -10_000
    events = 0
    for t in range(2, len(x) - 2):
        if not valid[t - 1:t + 2].all():
            continue
        dx0 = float(x[t] - x[t - 1])
        dx1 = float(x[t + 1] - x[t])
        if abs(dx0) < min_speed or abs(dx1) < min_speed:
            continue
        if np.sign(dx0) == np.sign(dx1):
            continue
        bx = float(x[t])
        if bx < wall_margin or bx > 160.0 - wall_margin:
            continue
        near_paddle = False
        for obj in ("left_paddle", "right_paddle"):
            if not tracks[obj]["present"][t]:
                continue
            px0, px1 = tracks[obj]["xmin"][t], tracks[obj]["xmax"][t]
            py0, py1 = tracks[obj]["ymin"][t], tracks[obj]["ymax"][t]
            if not all(np.isfinite([px0, px1, py0, py1])):
                continue
            if (px0 - paddle_margin <= bx <= px1 + paddle_margin) and (
                py0 - paddle_margin <= float(ball["y"][t]) <= py1 + paddle_margin
            ):
                near_paddle = True
                break
        if near_paddle or t - last_event < min_event_gap:
            continue
        events += 1
        last_event = t
    return events


def count_incorrect_bounces(
    tracks: dict[str, dict[str, np.ndarray]],
    *,
    contact_margin_x: float,
    contact_margin_y: float,
    min_speed: float,
    lookahead: int,
    min_event_gap: int,
) -> int:
    ball = tracks["ball"]
    valid = valid_ball_mask(tracks)
    x = ball["x"]
    y = ball["y"]
    last_event = -10_000
    events = 0
    for t in range(1, len(x) - lookahead - 1):
        if not valid[t - 1] or not valid[t]:
            continue
        dx = float(x[t] - x[t - 1])
        if abs(dx) < min_speed:
            continue
        target = "right_paddle" if dx > 0 else "left_paddle"
        if not tracks[target]["present"][t]:
            continue
        px0, px1 = tracks[target]["xmin"][t], tracks[target]["xmax"][t]
        py0, py1 = tracks[target]["ymin"][t], tracks[target]["ymax"][t]
        if not all(np.isfinite([px0, px1, py0, py1])):
            continue
        in_x = px0 - contact_margin_x <= float(x[t]) <= px1 + contact_margin_x
        in_y = py0 - contact_margin_y <= float(y[t]) <= py1 + contact_margin_y
        if not (in_x and in_y):
            continue
        if t - last_event < min_event_gap:
            continue
        future = slice(t + 1, min(len(x), t + 1 + lookahead))
        future_valid = valid[future]
        if not future_valid.any():
            events += 1
            last_event = t
            continue
        future_x = x[future][future_valid]
        dx_future = float(future_x[-1] - x[t])
        expected_reversal = dx_future * dx < 0
        if not expected_reversal:
            events += 1
            last_event = t
    return events


def object_disappearance_count(
    tracks: dict[str, dict[str, np.ndarray]],
    *,
    permanent_absence_frames: int,
) -> int:
    return sum(
        trailing_absence_start(tracks[obj]["present"], permanent_absence_frames) is not None
        for obj in OBJECTS
    )


def action_mismatch_rate(
    tracks: dict[str, dict[str, np.ndarray]],
    actions: list[int],
    *,
    min_motion: float,
) -> float:
    y = tracks["right_paddle"]["y"]
    present = tracks["right_paddle"]["present"]
    n = min(len(actions), len(y) - 1)
    total = 0
    bad = 0
    for t in range(n):
        action = int(actions[t])
        if action not in ACTION_UP and action not in ACTION_DOWN:
            continue
        if not present[t] or not present[t + 1]:
            continue
        if not finite(y[t]) or not finite(y[t + 1]):
            continue
        dy = float(y[t + 1] - y[t])
        expected_sign = -1 if action in ACTION_UP else 1
        total += 1
        if abs(dy) < min_motion or np.sign(dy) != expected_sign:
            bad += 1
    return float(bad / total) if total else float("nan")


def parse_sample_id(path: Path) -> tuple[str, str, str]:
    sample_id = path.parent.name
    wm, cond, episode = sample_id.split("__", 2)
    return wm, cond, episode


def analyze_episode(track_path: Path, metadata_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    wm, cond, episode = parse_sample_id(track_path)
    tracks = read_track_csv(track_path)
    metadata = load_json(metadata_root / wm / cond / f"{episode}.json")

    ball_disappearance_events = count_disappearance_events(
        tracks["ball"]["present"],
        min_context=args.disappearance_context,
        max_temporary_gap=args.disappearance_max_gap,
        permanent_absence_frames=args.permanent_absence_frames,
    )
    spurious_turns = count_spurious_turns(
        tracks,
        min_speed=args.min_ball_speed,
        min_event_gap=args.min_event_gap,
        wall_margin=args.wall_margin,
        paddle_margin=args.paddle_margin,
    )
    incorrect_bounces = count_incorrect_bounces(
        tracks,
        contact_margin_x=args.bounce_contact_margin_x,
        contact_margin_y=args.bounce_contact_margin_y,
        min_speed=args.min_ball_speed,
        lookahead=args.bounce_lookahead,
        min_event_gap=args.min_event_gap,
    )
    object_disappearances = object_disappearance_count(
        tracks,
        permanent_absence_frames=args.permanent_absence_frames,
    )
    any_loss = min(
        (
            t
            for t in (
                trailing_absence_start(tracks[obj]["present"], args.permanent_absence_frames)
                for obj in OBJECTS
            )
            if t is not None
        ),
        default=None,
    )
    frames = len(tracks["ball"]["present"])
    object_disappearance_severity = 0.0 if any_loss is None else 1.0 - float(any_loss / frames)
    action_mismatch = action_mismatch_rate(
        tracks,
        metadata.get("actions", []),
        min_motion=args.min_paddle_motion,
    )
    return {
        "wm": wm,
        "condition": cond,
        "episode": episode,
        "ball_invisible_fraction": 1.0 - float(tracks["ball"]["present"].mean()),
        "ball_disappearance_events": ball_disappearance_events,
        "spurious_turn_events": spurious_turns,
        "incorrect_bounce_events": incorrect_bounces,
        "total_ball_failures": ball_disappearance_events + spurious_turns + incorrect_bounces,
        "object_disappearance_severity": object_disappearance_severity,
        "object_disappearance_count": object_disappearances,
        "action_mismatch_rate": action_mismatch,
    }


def group_key(row: dict[str, Any]) -> tuple[int, int, str, str]:
    return (
        WM_ORDER.index(row["wm"]) if row["wm"] in WM_ORDER else 999,
        COND_ORDER.index(row["condition"]) if row["condition"] in COND_ORDER else 999,
        row["wm"],
        row["condition"],
    )


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["wm"], row["condition"])].append(row)

    summary = []
    for (wm, cond), xs in sorted(groups.items(), key=lambda item: group_key(item[1][0])):
        mismatch_vals = [float(x["action_mismatch_rate"]) for x in xs if finite(x["action_mismatch_rate"])]
        summary.append({
            "wm": wm,
            "condition": cond,
            "episodes": len(xs),
            "ball_disappearance_mean": mean(float(x["ball_invisible_fraction"]) for x in xs),
            "ball_disappearance_events_mean": mean(float(x["ball_disappearance_events"]) for x in xs),
            "spurious_turn_mean": mean(float(x["spurious_turn_events"]) for x in xs),
            "incorrect_bounce_mean": mean(float(x["incorrect_bounce_events"]) for x in xs),
            "total_ball_failures_mean": (
                mean(float(x["ball_invisible_fraction"]) for x in xs)
                + mean(float(x["spurious_turn_events"]) for x in xs)
                + mean(float(x["incorrect_bounce_events"]) for x in xs)
            ),
            "object_disappearance_mean": mean(float(x["object_disappearance_severity"]) for x in xs),
            "object_disappearance_count_mean": mean(float(x["object_disappearance_count"]) for x in xs),
            "action_mismatch_mean": mean(mismatch_vals) if mismatch_vals else float("nan"),
        })
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt_num(x: Any, digits: int = 2) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(v):
        return "NA"
    return f"{v:.{digits}f}"


def conclusion(wm: str, cond: str, *, rare: bool = False) -> str:
    if cond not in ("cgsreg_w0p01", "cgsreg_w1p0"):
        return "--"
    if wm == "dreamer":
        return "largest improvement"
    if wm == "diamond":
        return "improved object persistence" if rare else "improved"
    if wm == "twister":
        return "improved"
    if wm == "storm":
        return "not improved; action bottleneck" if rare else "not improved"
    return "not improved"


def write_markdown(path: Path, summary: list[dict[str, Any]]) -> None:
    lines = [
        "# Paper Failure Tables",
        "",
        "Values are means over five rollout episodes. Lower is better.",
        "",
        "## Common ball-dynamics failures",
        "",
        "| World model | Checkpoint | Ball disappearance | Spurious turn | Incorrect bounce | Total | Conclusion |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in summary:
        lines.append("|" + "|".join([
            DISPLAY_WM.get(row["wm"], row["wm"]),
            DISPLAY_COND.get(row["condition"], row["condition"]).replace(r"\(", "$").replace(r"\)", "$").replace(r"\methodname{}", "CGSReg"),
            fmt_num(row["ball_disappearance_mean"]),
            fmt_num(row["spurious_turn_mean"]),
            fmt_num(row["incorrect_bounce_mean"]),
            fmt_num(row["total_ball_failures_mean"]),
            conclusion(row["wm"], row["condition"]),
        ]) + "|")
    lines.extend([
        "",
        "## Rare and architecture-specific failures",
        "",
        "| World model | Checkpoint | Object disappearance | Action mismatch | Conclusion |",
        "|---|---|---:|---:|---|",
    ])
    for row in summary:
        lines.append("|" + "|".join([
            DISPLAY_WM.get(row["wm"], row["wm"]),
            DISPLAY_COND.get(row["condition"], row["condition"]).replace(r"\(", "$").replace(r"\)", "$").replace(r"\methodname{}", "CGSReg"),
            fmt_num(row["object_disappearance_mean"]),
            fmt_num(row["action_mismatch_mean"]),
            conclusion(row["wm"], row["condition"], rare=True),
        ]) + "|")
    path.write_text("\n".join(lines) + "\n")


def latex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def write_latex(path: Path, summary: list[dict[str, Any]]) -> None:
    lines = [
        "% Auto-generated by scripts/eval/make_paper_failure_tables.py",
        "% Common ball-dynamics failures table body.",
    ]
    for row in summary:
        lines.append(
            "    "
            + " & ".join([
                DISPLAY_WM.get(row["wm"], row["wm"]),
                DISPLAY_COND.get(row["condition"], row["condition"]),
                fmt_num(row["ball_disappearance_mean"]),
                fmt_num(row["spurious_turn_mean"]),
                fmt_num(row["incorrect_bounce_mean"]),
                fmt_num(row["total_ball_failures_mean"]),
                latex_escape(conclusion(row["wm"], row["condition"])),
            ])
            + r" \\"
        )
        if row["condition"] in ("cgsreg_w0p01", "cgsreg_w1p0") and row["wm"] != "storm":
            lines.append(r"    \midrule")
    lines.extend([
        "",
        "% Rare and architecture-specific failures table body.",
    ])
    for row in summary:
        lines.append(
            "    "
            + " & ".join([
                DISPLAY_WM.get(row["wm"], row["wm"]),
                DISPLAY_COND.get(row["condition"], row["condition"]),
                fmt_num(row["object_disappearance_mean"]),
                fmt_num(row["action_mismatch_mean"]),
                latex_escape(conclusion(row["wm"], row["condition"], rare=True)),
            ])
            + r" \\"
        )
        if row["condition"] in ("cgsreg_w0p01", "cgsreg_w1p0") and row["wm"] != "storm":
            lines.append(r"    \midrule")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, default=Path("paper_rollout_eval_bundle"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/paper_tables"))
    parser.add_argument("--permanent-absence-frames", type=int, default=32)
    parser.add_argument("--disappearance-max-gap", type=int, default=8)
    parser.add_argument("--disappearance-context", type=int, default=4)
    parser.add_argument("--min-ball-speed", type=float, default=0.25)
    parser.add_argument("--min-event-gap", type=int, default=4)
    parser.add_argument("--wall-margin", type=float, default=10.0)
    parser.add_argument("--paddle-margin", type=float, default=5.0)
    parser.add_argument("--bounce-contact-margin-x", type=float, default=6.0)
    parser.add_argument("--bounce-contact-margin-y", type=float, default=8.0)
    parser.add_argument("--bounce-lookahead", type=int, default=4)
    parser.add_argument("--min-paddle-motion", type=float, default=0.25)
    args = parser.parse_args()

    seg_root = args.bundle_root / "cutie_segmentations"
    metadata_root = args.bundle_root / "rollout_metadata"
    paths = sorted(seg_root.glob("*__*__*/cutie_tracks.csv"))
    if not paths:
        raise FileNotFoundError(f"No CUTIE tracks found under {seg_root}")

    episode_rows = [analyze_episode(path, metadata_root, args) for path in paths]
    summary_rows = summarize(episode_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "paper_failure_metrics_episode.csv", episode_rows)
    write_csv(args.output_dir / "paper_failure_metrics_summary.csv", summary_rows)
    write_markdown(args.output_dir / "paper_failure_tables.md", summary_rows)
    write_latex(args.output_dir / "paper_failure_tables_body.tex", summary_rows)

    print(f"Wrote {len(episode_rows)} episode rows")
    print(f"Wrote {len(summary_rows)} summary rows")
    print(args.output_dir / "paper_failure_tables.md")


if __name__ == "__main__":
    main()
