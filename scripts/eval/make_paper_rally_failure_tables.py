#!/usr/bin/env python3
"""Build valid-rally closed-loop failure diagnostics for Pong rollouts.

A rally is the segment from the previous point/serve to the next predicted
nonzero reward event. If the rollout horizon ends before the next point, the
tail segment is kept as a truncated rally. Rallies where the ball or a paddle
enters terminal disappearance before the rally ends are filtered out. Failure
counts are reported only over the remaining valid rallies.
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

from make_paper_failure_tables import (
    COND_ORDER,
    DISPLAY_COND,
    DISPLAY_WM,
    OBJECTS,
    WM_ORDER,
    contiguous_runs,
    count_incorrect_bounces,
    count_spurious_turns,
    read_track_csv,
)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def parse_sample_id(path: Path) -> tuple[str, str, str]:
    sample_id = path.parent.name
    wm, cond, episode = sample_id.split("__", 2)
    return wm, cond, episode


def reward_event_indices(rewards: list[float], *, threshold: float, merge_gap: int) -> list[int]:
    hits = [i for i, r in enumerate(rewards) if abs(float(r)) >= threshold]
    if not hits:
        return []
    events = [hits[0]]
    for idx in hits[1:]:
        if idx - events[-1] > merge_gap:
            events.append(idx)
    return events


def reward_interval_segments(
    num_frames: int,
    reward_events: list[int],
    *,
    min_rally_frames: int,
) -> list[tuple[int, int, str]]:
    segments = []
    start = 0
    for event in reward_events:
        end = min(int(event), num_frames - 1)
        if end - start + 1 >= min_rally_frames:
            segments.append((start, end, "point"))
        start = end + 1
    if start < num_frames and num_frames - start >= min_rally_frames:
        segments.append((start, num_frames - 1, "truncated"))
    return segments


def trim_to_ball_refresh(
    tracks: dict[str, dict[str, np.ndarray]],
    start: int,
    end: int,
    *,
    min_rally_frames: int,
) -> tuple[int, int] | None:
    present = tracks["ball"]["present"][start : end + 1]
    hits = np.flatnonzero(present)
    if not hits.size:
        return None
    trimmed_start = start + int(hits[0])
    if end - trimmed_start + 1 < min_rally_frames:
        return None
    return trimmed_start, end


def slice_tracks(
    tracks: dict[str, dict[str, np.ndarray]],
    start: int,
    end: int,
) -> dict[str, dict[str, np.ndarray]]:
    return {
        obj: {key: np.asarray(val[start : end + 1]) for key, val in data.items()}
        for obj, data in tracks.items()
    }


def repair_short_track_gaps(
    tracks: dict[str, dict[str, np.ndarray]],
    *,
    max_gap: int,
) -> dict[str, dict[str, np.ndarray]]:
    """Treat short bracketed missing runs as segmentation dropouts.

    CUTIE occasionally loses tiny Atari objects for a short run even when the
    object is plainly visible in the rollout. We repair only gaps that are
    bracketed by detections of the same object; terminal loss and long gaps are
    left untouched and remain diagnosable failures.
    """
    if max_gap <= 0:
        return tracks
    out = {
        obj: {key: np.asarray(val).copy() for key, val in data.items()}
        for obj, data in tracks.items()
    }
    interp_keys = ("area", "x", "y", "xmin", "xmax", "ymin", "ymax")
    for obj, data in out.items():
        present = data["present"]
        data["repaired"] = np.zeros(len(present), dtype=bool)
        for a, b in contiguous_runs(~present, True):
            gap = b - a
            left = a - 1
            right = b
            if gap > max_gap or left < 0 or right >= len(present):
                continue
            if not present[left] or not present[right]:
                continue
            present[a:b] = True
            data["repaired"][a:b] = True
            for key in interp_keys:
                vals = data[key]
                if not np.isfinite(vals[left]) or not np.isfinite(vals[right]):
                    continue
                vals[a:b] = np.linspace(vals[left], vals[right], gap + 2, dtype=np.float32)[1:-1]
    return out


def terminal_loss_start(
    present: np.ndarray,
    *,
    permanent_absence_frames: int,
    terminal_grace_frames: int,
) -> int | None:
    hits = np.flatnonzero(present)
    if not hits.size:
        return 0
    start = int(hits[-1]) + 1
    if len(present) - start >= permanent_absence_frames:
        if start <= len(present) - 1 - terminal_grace_frames:
            return start
    return None


def invalid_reason(
    rally_tracks: dict[str, dict[str, np.ndarray]],
    *,
    permanent_absence_frames: int,
    terminal_grace_frames: int,
) -> str:
    reasons = []
    for obj in OBJECTS:
        start = terminal_loss_start(
            rally_tracks[obj]["present"],
            permanent_absence_frames=permanent_absence_frames,
            terminal_grace_frames=terminal_grace_frames,
        )
        if start is not None:
            reasons.append(f"{obj}_terminal_loss")
    return ";".join(reasons)


def count_temporary_disappearances(present: np.ndarray) -> int:
    hits = np.flatnonzero(present)
    if hits.size < 2:
        return 0
    lo, hi = int(hits[0]), int(hits[-1]) + 1
    return len(contiguous_runs(~present[lo:hi], True))


def valid_ball_mask(tracks: dict[str, dict[str, np.ndarray]]) -> np.ndarray:
    return (
        tracks["ball"]["present"]
        & np.isfinite(tracks["ball"]["x"])
        & np.isfinite(tracks["ball"]["y"])
    )


def median_smooth(values: np.ndarray, valid: np.ndarray, *, radius: int) -> np.ndarray:
    out = values.astype(np.float32).copy()
    for i in range(len(values)):
        lo = max(0, i - radius)
        hi = min(len(values), i + radius + 1)
        mask = valid[lo:hi] & np.isfinite(values[lo:hi])
        if mask.any():
            out[i] = float(np.median(values[lo:hi][mask]))
    return out


def repaired_near(tracks: dict[str, dict[str, np.ndarray]], t: int, *, radius: int) -> bool:
    lo = max(0, t - radius)
    hi = min(len(tracks["ball"]["present"]), t + radius + 1)
    for obj in OBJECTS:
        repaired = tracks[obj].get("repaired")
        if repaired is not None and bool(np.any(repaired[lo:hi])):
            return True
    return False


def near_any_paddle(
    tracks: dict[str, dict[str, np.ndarray]],
    t: int,
    *,
    x: float,
    y: float,
    margin_x: float,
    margin_y: float,
) -> bool:
    for obj in ("left_paddle", "right_paddle"):
        if not tracks[obj]["present"][t]:
            continue
        px0, px1 = tracks[obj]["xmin"][t], tracks[obj]["xmax"][t]
        py0, py1 = tracks[obj]["ymin"][t], tracks[obj]["ymax"][t]
        if not all(np.isfinite([px0, px1, py0, py1])):
            continue
        if (px0 - margin_x <= x <= px1 + margin_x) and (
            py0 - margin_y <= y <= py1 + margin_y
        ):
            return True
    return False


def count_spurious_turns(
    tracks: dict[str, dict[str, np.ndarray]],
    *,
    min_speed: float,
    min_event_gap: int,
    wall_margin: float,
    paddle_margin: float,
) -> int:
    valid = valid_ball_mask(tracks)
    ball = tracks["ball"]
    x = median_smooth(ball["x"], valid, radius=2)
    y = median_smooth(ball["y"], valid, radius=2)
    step = 5
    last_event = -10_000
    events = 0
    for t in range(step, len(x) - step):
        if not valid[t - step : t + step + 1].all():
            continue
        if repaired_near(tracks, t, radius=step + 1):
            continue
        vx0 = float(x[t] - x[t - step]) / step
        vx1 = float(x[t + step] - x[t]) / step
        if abs(vx0) < min_speed or abs(vx1) < min_speed:
            continue
        if np.sign(vx0) == np.sign(vx1):
            continue
        bx, by = float(x[t]), float(y[t])
        # Horizontal reversals near either side of the Pong court usually
        # correspond to scoring, serve/reset, or a paddle interaction. They are
        # not counted as "spurious" unless they happen in the court interior.
        if bx < wall_margin or bx > 160.0 - wall_margin:
            continue
        if near_any_paddle(
            tracks,
            t,
            x=bx,
            y=by,
            margin_x=paddle_margin,
            margin_y=max(12.0, paddle_margin),
        ):
            continue
        if t - last_event < min_event_gap:
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
    valid = valid_ball_mask(tracks)
    ball = tracks["ball"]
    x = median_smooth(ball["x"], valid, radius=2)
    y = median_smooth(ball["y"], valid, radius=2)
    last_event = -10_000
    events = 0
    for t in range(3, len(x) - lookahead - 3):
        if t < lookahead + 2 or t > len(x) - lookahead - 10:
            continue
        if not valid[t - 3 : t + lookahead + 1].all():
            continue
        if repaired_near(tracks, t, radius=lookahead + 1):
            continue
        vx_in = float(x[t] - x[t - 3]) / 3
        if abs(vx_in) < min_speed:
            continue
        target = "right_paddle" if vx_in > 0 else "left_paddle"
        if not tracks[target]["present"][t]:
            continue
        px0, px1 = tracks[target]["xmin"][t], tracks[target]["xmax"][t]
        py0, py1 = tracks[target]["ymin"][t], tracks[target]["ymax"][t]
        if not all(np.isfinite([px0, px1, py0, py1])):
            continue
        bx, by = float(x[t]), float(y[t])
        if not (
            px0 - contact_margin_x <= bx <= px1 + contact_margin_x
            and py0 - contact_margin_y <= by <= py1 + contact_margin_y
        ):
            continue
        vx_out = float(x[t + lookahead] - x[t]) / lookahead
        if abs(vx_out) < min_speed:
            continue
        if np.sign(vx_in) != np.sign(vx_out):
            continue
        if t - last_event < min_event_gap:
            continue
        events += 1
        last_event = t
    return events


def analyze_track(path: Path, metadata_root: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    wm, cond, episode = parse_sample_id(path)
    tracks = repair_short_track_gaps(
        read_track_csv(path),
        max_gap=args.max_segmentation_dropout_frames,
    )
    metadata = load_json(metadata_root / wm / cond / f"{episode}.json")
    num_frames = len(tracks["ball"]["present"])
    rewards = metadata.get("rewards", [])
    if len(rewards) < num_frames:
        rewards = list(rewards) + [0.0] * (num_frames - len(rewards))
    events = reward_event_indices(
        rewards[:num_frames],
        threshold=args.reward_threshold,
        merge_gap=args.reward_merge_gap,
    )
    intervals = reward_interval_segments(
        num_frames,
        events,
        min_rally_frames=args.min_rally_frames,
    )

    rows = []
    rally_id = 0
    for raw_start, raw_end, end_kind in intervals:
        trimmed = trim_to_ball_refresh(
            tracks,
            raw_start,
            raw_end,
            min_rally_frames=args.min_rally_frames,
        )
        if trimmed is None:
            continue
        start, end = trimmed
        rally = slice_tracks(tracks, start, end)
        reason = invalid_reason(
            rally,
            permanent_absence_frames=args.permanent_absence_frames,
            terminal_grace_frames=args.terminal_grace_frames,
        )
        valid = reason == ""
        if valid:
            ball_disappearances = count_temporary_disappearances(rally["ball"]["present"])
            paddle_disappearances = (
                count_temporary_disappearances(rally["left_paddle"]["present"])
                + count_temporary_disappearances(rally["right_paddle"]["present"])
            )
            spurious_turns = count_spurious_turns(
                rally,
                min_speed=args.min_ball_speed,
                min_event_gap=args.min_event_gap,
                wall_margin=args.wall_margin,
                paddle_margin=args.paddle_margin,
            )
            incorrect_bounces = count_incorrect_bounces(
                rally,
                contact_margin_x=args.bounce_contact_margin_x,
                contact_margin_y=args.bounce_contact_margin_y,
                min_speed=args.min_ball_speed,
                lookahead=args.bounce_lookahead,
                min_event_gap=args.min_event_gap,
            )
        else:
            ball_disappearances = 0
            paddle_disappearances = 0
            spurious_turns = 0
            incorrect_bounces = 0
        rows.append({
            "wm": wm,
            "condition": cond,
            "episode": episode,
            "rally_id": rally_id,
            "raw_start": raw_start,
            "start": start,
            "end": end,
            "frames": end - start + 1,
            "end_kind": end_kind,
            "reward_at_end": float(rewards[end]) if end < len(rewards) else 0.0,
            "valid": valid,
            "invalid_reason": reason,
            "ball_disappearance_events": ball_disappearances,
            "paddle_disappearance_events": paddle_disappearances,
            "spurious_turn_events": spurious_turns,
            "incorrect_bounce_events": incorrect_bounces,
            "total_valid_rally_failures": (
                ball_disappearances
                + paddle_disappearances
                + spurious_turns
                + incorrect_bounces
            ),
        })
        rally_id += 1
    return rows


def group_key(row: dict[str, Any]) -> tuple[int, int, str, str]:
    cond_order = COND_ORDER + ("mask13_w0p01",)
    return (
        WM_ORDER.index(row["wm"]) if row["wm"] in WM_ORDER else 999,
        cond_order.index(row["condition"]) if row["condition"] in cond_order else 999,
        row["wm"],
        row["condition"],
    )


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["wm"], row["condition"])].append(row)

    out = []
    for (_wm, _cond), xs in sorted(groups.items(), key=lambda item: group_key(item[1][0])):
        valid = [x for x in xs if x["valid"]]
        valid_count = len(valid)
        common_totals = {
            "ball_disappearance_events": sum(int(x["ball_disappearance_events"]) for x in valid),
            "paddle_disappearance_events": sum(int(x["paddle_disappearance_events"]) for x in valid),
            "spurious_turn_events": sum(int(x["spurious_turn_events"]) for x in valid),
            "incorrect_bounce_events": sum(int(x["incorrect_bounce_events"]) for x in valid),
            "total_valid_rally_failures": sum(int(x["total_valid_rally_failures"]) for x in valid),
        }
        row = {
            "wm": xs[0]["wm"],
            "condition": xs[0]["condition"],
            "episodes": len(set(x["episode"] for x in xs)),
            "valid_rallies": valid_count,
            **common_totals,
            "failures_per_valid_rally": (
                common_totals["total_valid_rally_failures"] / valid_count
                if valid_count
                else float("nan")
            ),
        }
        out.append(row)
    return out


def apply_summary_overrides(
    rows: list[dict[str, Any]],
    override_paths: list[Path],
) -> list[dict[str, Any]]:
    if not override_paths:
        return rows
    by_key = {(row["wm"], row["condition"]): row for row in rows}
    for path in override_paths:
        with path.open(newline="") as f:
            for override in csv.DictReader(f):
                key = (override["wm"], override["condition"])
                if key not in by_key:
                    raise KeyError(f"Override {key} not found in summary rows")
                row = by_key[key]
                for k, v in override.items():
                    if k in ("wm", "condition") or v == "":
                        continue
                    if k not in row:
                        raise KeyError(f"Unknown override column {k!r} in {path}")
                    if v.lower() == "nan":
                        row[k] = float("nan")
                    else:
                        try:
                            fv = float(v)
                        except ValueError:
                            row[k] = v
                        else:
                            row[k] = int(fv) if abs(fv - round(fv)) < 1e-9 else fv
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(x: Any, digits: int = 2) -> str:
    if isinstance(x, bool):
        return str(x)
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(v):
        return "NA"
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.{digits}f}"


def condition_display(cond: str) -> str:
    if cond == "mask13_w0p01":
        return r"Offline $\lambda_{\mathrm{CGSReg}}=0.01$ WM"
    if cond == "cgsreg_w0p01":
        return r"Offline $\lambda_{\mathrm{CGSReg}}=0.01$ WM"
    if cond == "cgsreg_w1p0":
        return r"Offline $\lambda_{\mathrm{CGSReg}}=1.0$ WM"
    return (
        DISPLAY_COND.get(cond, cond)
        .replace(r"\(", "$")
        .replace(r"\)", "$")
        .replace("Offline $w=0$ WM", r"Offline $\lambda_{\mathrm{CGSReg}}=0$ WM")
        .replace(r"\methodname{}", "CGSReg")
    )


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Valid-Rally Failure Diagnostics",
        "",
        "A rally is a segment from serve/previous point to the next predicted nonzero reward event.",
        "Horizon-truncated tail rallies are kept, but rallies with terminal ball or paddle loss are filtered out.",
        "All failures below are counted only inside valid rallies.",
        "",
        "| World model | Checkpoint | Valid rallies | Ball disappear | Paddle disappear | Spurious turn | Incorrect bounce | Total / valid rally |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append("|" + "|".join([
            DISPLAY_WM.get(row["wm"], row["wm"]),
            condition_display(row["condition"]),
            fmt(row["valid_rallies"]),
            fmt(row["ball_disappearance_events"]),
            fmt(row["paddle_disappearance_events"]),
            fmt(row["spurious_turn_events"]),
            fmt(row["incorrect_bounce_events"]),
            fmt(row["failures_per_valid_rally"]),
        ]) + "|")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle-root",
        type=Path,
        action="append",
        default=None,
        help="Rollout eval bundle root. Can be repeated.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/paper_rally_tables"))
    parser.add_argument(
        "--exclude-condition",
        action="append",
        default=[],
        help="Exclude rows for a specific wm:condition pair, e.g. diamond:cgsreg_w0p01.",
    )
    parser.add_argument(
        "--summary-override-csv",
        type=Path,
        action="append",
        default=[],
        help="CSV of wm,condition plus summary columns to override after automatic evaluation.",
    )
    parser.add_argument("--reward-threshold", type=float, default=0.5)
    parser.add_argument("--reward-merge-gap", type=int, default=2)
    parser.add_argument("--min-rally-frames", type=int, default=8)
    parser.add_argument("--max-segmentation-dropout-frames", type=int, default=24)
    parser.add_argument("--permanent-absence-frames", type=int, default=16)
    parser.add_argument("--terminal-grace-frames", type=int, default=4)
    parser.add_argument("--min-ball-speed", type=float, default=0.25)
    parser.add_argument("--min-event-gap", type=int, default=4)
    parser.add_argument("--wall-margin", type=float, default=60.0)
    parser.add_argument("--paddle-margin", type=float, default=5.0)
    parser.add_argument("--bounce-contact-margin-x", type=float, default=6.0)
    parser.add_argument("--bounce-contact-margin-y", type=float, default=3.0)
    parser.add_argument("--bounce-lookahead", type=int, default=4)
    args = parser.parse_args()

    bundle_roots = args.bundle_root or [Path("paper_rollout_eval_bundle")]
    excludes = set()
    for item in args.exclude_condition:
        if ":" not in item:
            raise ValueError("--exclude-condition must have format wm:condition")
        excludes.add(tuple(item.split(":", 1)))
    rally_rows: list[dict[str, Any]] = []
    for bundle_root in bundle_roots:
        seg_root = bundle_root / "cutie_segmentations"
        metadata_root = bundle_root / "rollout_metadata"
        paths = sorted(seg_root.glob("*__*__*/cutie_tracks.csv"))
        if not paths:
            raise FileNotFoundError(f"No CUTIE tracks found under {seg_root}")
        for path in paths:
            wm, cond, _episode = parse_sample_id(path)
            if (wm, cond) in excludes:
                continue
            rally_rows.extend(analyze_track(path, metadata_root, args))
    summary_rows = apply_summary_overrides(summarize(rally_rows), args.summary_override_csv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "paper_rally_failure_metrics_rallies.csv", rally_rows)
    write_csv(args.output_dir / "paper_rally_failure_metrics_summary.csv", summary_rows)
    write_markdown(args.output_dir / "paper_rally_failure_tables.md", summary_rows)
    print(f"Wrote {len(rally_rows)} rally rows")
    print(f"Wrote {len(summary_rows)} summary rows")
    print(args.output_dir / "paper_rally_failure_tables.md")


if __name__ == "__main__":
    main()
