#!/usr/bin/env python3
"""Reference-light physical consistency metrics for SAM2 Pong ball tracks.

The script consumes `*_sam2_tracks.csv` files produced by the SAM2 ball
tracking experiments and reports trajectory-level artifacts:

- flicker gaps: visible -> missing -> visible intervals.
- teleport steps: implausibly large visible-to-visible displacement.
- reappearance teleports: implausible location after a short missing gap.
- spontaneous turns: large direction change away from walls/paddle zones.
- acceleration percentiles.

The metric is generated-video centric. Real replay tracks are used only to set
conservative speed/acceleration thresholds and to sanity check that the event
detector does not fire on normal Pong trajectories.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


def parse_bool(value: Any) -> bool:
  if isinstance(value, bool):
    return value
  return str(value).strip().lower() in ('1', 'true', 'yes', 'y')


def parse_float(value: Any, default: float = float('nan')) -> float:
  if value is None:
    return default
  text = str(value).strip()
  if text == '':
    return default
  try:
    return float(text)
  except ValueError:
    return default


def safe_mean(values: Iterable[float]) -> float:
  vals = [float(x) for x in values if math.isfinite(float(x))]
  return float(np.mean(vals)) if vals else float('nan')


def safe_percentile(values: Iterable[float], q: float) -> float:
  vals = [float(x) for x in values if math.isfinite(float(x))]
  return float(np.percentile(vals, q)) if vals else float('nan')


def safe_rate(num: int, den: int) -> float:
  return float(num) / float(den) if den else float('nan')


def read_csv_rows(path: pathlib.Path) -> list[dict[str, str]]:
  with path.open(newline='') as f:
    return list(csv.DictReader(f))


@dataclass(frozen=True)
class TrackPoint:
  sample_id: str
  t: int
  present: bool
  x: float = float('nan')
  y: float = float('nan')
  area: float = float('nan')

  @property
  def has_center(self) -> bool:
    return self.present and math.isfinite(self.x) and math.isfinite(self.y)


@dataclass(frozen=True)
class PaddleBox:
  sample_id: str
  t: int
  present: bool
  xmin: float = float('nan')
  xmax: float = float('nan')
  ymin: float = float('nan')
  ymax: float = float('nan')

  @property
  def has_box(self) -> bool:
    return (
        self.present and
        math.isfinite(self.xmin) and math.isfinite(self.xmax) and
        math.isfinite(self.ymin) and math.isfinite(self.ymax))


def rows_to_tracks(rows: list[dict[str, str]]) -> dict[str, list[TrackPoint]]:
  by_sample: dict[str, list[TrackPoint]] = defaultdict(list)
  for row in rows:
    sample_id = row.get('sample_id') or 'window0'
    point = TrackPoint(
        sample_id=sample_id,
        t=int(parse_float(row.get('t'), 0)),
        present=parse_bool(row.get('ball_present')),
        x=parse_float(row.get('ball_x')),
        y=parse_float(row.get('ball_y')),
        area=parse_float(row.get('ball_area')),
    )
    by_sample[sample_id].append(point)
  return {
      sample: sorted(points, key=lambda p: p.t)
      for sample, points in by_sample.items()
  }


def first_available(row: dict[str, str], names: tuple[str, ...]) -> str | None:
  for name in names:
    if name in row and str(row[name]).strip() != '':
      return row[name]
  return None


def rows_to_paddle_boxes(rows: list[dict[str, str]]) -> dict[str, dict[int, PaddleBox]]:
  by_sample: dict[str, dict[int, PaddleBox]] = defaultdict(dict)
  for row in rows:
    sample_id = row.get('sample_id') or 'window0'
    present = parse_bool(first_available(row, (
        'paddle_present', 'object_present', 'mask_present', 'ball_present')))
    t = int(parse_float(row.get('t'), 0))

    xmin = parse_float(first_available(row, ('xmin', 'x_min', 'paddle_xmin', 'bbox_xmin')))
    xmax = parse_float(first_available(row, ('xmax', 'x_max', 'paddle_xmax', 'bbox_xmax')))
    ymin = parse_float(first_available(row, ('ymin', 'y_min', 'paddle_ymin', 'bbox_ymin')))
    ymax = parse_float(first_available(row, ('ymax', 'y_max', 'paddle_ymax', 'bbox_ymax')))

    if not all(math.isfinite(x) for x in (xmin, xmax, ymin, ymax)):
      # Backward-compatible fallback for centroid-only SAM2 tracks. This is
      # intentionally conservative; formal runs should export bbox columns.
      x = parse_float(first_available(row, ('paddle_x', 'object_x', 'ball_x', 'x')))
      y = parse_float(first_available(row, ('paddle_y', 'object_y', 'ball_y', 'y')))
      area = parse_float(first_available(row, ('paddle_area', 'object_area', 'ball_area', 'area')))
      if math.isfinite(x) and math.isfinite(y) and math.isfinite(area) and area > 0:
        half_width = 2.0
        half_height = max(3.0, min(18.0, area / (2.0 * max(1.0, 2.0 * half_width)) / 2.0))
        xmin, xmax = x - half_width, x + half_width
        ymin, ymax = y - half_height, y + half_height

    by_sample[sample_id][t] = PaddleBox(
        sample_id=sample_id, t=t, present=present,
        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
  return by_sample


def track_name_from_path(path: pathlib.Path) -> str:
  name = path.stem
  for suffix in ('_sam2_tracks', '_tracks'):
    if name.endswith(suffix):
      name = name[:-len(suffix)]
  if name == 'sam2':
    return path.parent.name
  return name


def paddle_track_key_from_path(path: pathlib.Path) -> tuple[str, str] | None:
  stem = path.stem.lower()
  side = None
  marker = None
  for candidate in ('left_paddle', 'right_paddle', 'left', 'right'):
    if candidate in stem:
      side = 'left' if candidate.startswith('left') else 'right'
      marker = candidate
      break
  if side is None:
    return None
  raw = path.stem
  if marker and marker in stem:
    idx = stem.index(marker)
    model = raw[:idx].rstrip('_-.')
  else:
    model = ''
  if not model:
    model = path.parent.name
  for suffix in ('_sam2_tracks', '_tracks'):
    if model.endswith(suffix):
      model = model[:-len(suffix)]
  return model, side


def is_real_track(path: pathlib.Path) -> bool:
  lower = path.name.lower()
  return lower.startswith('real_') or 'real_mask' in lower or 'real_target' in lower


def find_track_files(input_dirs: list[pathlib.Path]) -> list[pathlib.Path]:
  out: list[pathlib.Path] = []
  for root in input_dirs:
    if root.is_file() and root.suffix == '.csv':
      out.append(root)
      continue
    out.extend(root.glob('tracks/*_sam2_tracks.csv'))
    out.extend(root.glob('h*/**/sam2_tracks.csv'))
    out.extend(root.glob('**/*_sam2_tracks.csv'))
  # Preserve deterministic order while removing duplicates.
  return sorted(set(p.resolve() for p in out if p.exists()))


def find_real_track_files(input_dirs: list[pathlib.Path]) -> list[pathlib.Path]:
  out: list[pathlib.Path] = []
  for root in input_dirs:
    if root.is_file() and is_real_track(root):
      out.append(root)
      continue
    out.extend(root.glob('tracks/real*_tracks.csv'))
    out.extend(root.glob('h*/real*_tracks.csv'))
    out.extend(root.glob('**/real*_tracks.csv'))
  return sorted(set(p.resolve() for p in out if p.exists()))


def find_paddle_track_files(input_dirs: list[pathlib.Path]) -> list[pathlib.Path]:
  out: list[pathlib.Path] = []
  patterns = (
      '**/*left*paddle*tracks.csv',
      '**/*right*paddle*tracks.csv',
      '**/left*_tracks.csv',
      '**/right*_tracks.csv',
  )
  for root in input_dirs:
    if root.is_file() and paddle_track_key_from_path(root) is not None:
      out.append(root)
      continue
    for pattern in patterns:
      out.extend(root.glob(pattern))
  return sorted(set(p.resolve() for p in out if p.exists()))


def consecutive_speeds(points: list[TrackPoint]) -> list[float]:
  speeds = []
  lookup = {p.t: p for p in points}
  for t in sorted(lookup):
    a = lookup.get(t - 1)
    b = lookup[t]
    if a and a.has_center and b.has_center:
      speeds.append(float(np.hypot(b.x - a.x, b.y - a.y)))
  return speeds


def consecutive_accels(points: list[TrackPoint]) -> list[float]:
  accels = []
  lookup = {p.t: p for p in points}
  for t in sorted(lookup):
    a = lookup.get(t - 1)
    b = lookup.get(t)
    c = lookup.get(t + 1)
    if a and b and c and a.has_center and b.has_center and c.has_center:
      ax = c.x - 2.0 * b.x + a.x
      ay = c.y - 2.0 * b.y + a.y
      accels.append(float(np.hypot(ax, ay)))
  return accels


@dataclass(frozen=True)
class Thresholds:
  teleport_px: float
  reappearance_error_px: float
  accel_px: float


def calibrate_thresholds(
    real_tracks: list[list[TrackPoint]],
    *,
    teleport_px: float | None,
    reappearance_error_px: float | None,
    accel_px: float | None,
    speed_percentile: float,
    accel_percentile: float,
    speed_margin_px: float,
    accel_margin_px: float,
) -> Thresholds:
  real_speeds = []
  real_accels = []
  for points in real_tracks:
    real_speeds.extend(consecutive_speeds(points))
    real_accels.extend(consecutive_accels(points))
  speed_base = safe_percentile(real_speeds, speed_percentile)
  accel_base = safe_percentile(real_accels, accel_percentile)
  auto_teleport = speed_base + speed_margin_px if math.isfinite(speed_base) else 7.0
  auto_reappear = 2.5 * auto_teleport
  auto_accel = accel_base + accel_margin_px if math.isfinite(accel_base) else 6.0
  return Thresholds(
      teleport_px=float(teleport_px if teleport_px is not None else auto_teleport),
      reappearance_error_px=float(
          reappearance_error_px if reappearance_error_px is not None else auto_reappear),
      accel_px=float(accel_px if accel_px is not None else auto_accel),
  )


def near_collision_zone(
    point: TrackPoint,
    *,
    frame_size: float,
    wall_margin: float,
    paddle_x_margin: float,
    paddle_boxes: dict[int, list[PaddleBox]] | None = None,
    paddle_margin: float = 2.5,
) -> bool:
  if not point.has_center:
    return True
  if point.y <= wall_margin or point.y >= frame_size - wall_margin:
    return True
  if paddle_boxes:
    for box in paddle_boxes.get(point.t, []):
      if not box.has_box:
        continue
      if (
          box.xmin - paddle_margin <= point.x <= box.xmax + paddle_margin and
          box.ymin - paddle_margin <= point.y <= box.ymax + paddle_margin):
        return True
    return False
  if point.x <= paddle_x_margin or point.x >= frame_size - paddle_x_margin:
    return True
  return False


def pair_near_collision_zone(a: TrackPoint, b: TrackPoint, **kwargs) -> bool:
  return near_collision_zone(a, **kwargs) or near_collision_zone(b, **kwargs)


def angle_degrees(v1: np.ndarray, v2: np.ndarray) -> float:
  n1 = float(np.linalg.norm(v1))
  n2 = float(np.linalg.norm(v2))
  if n1 <= 0 or n2 <= 0:
    return float('nan')
  cos = float(np.dot(v1, v2) / (n1 * n2))
  cos = max(-1.0, min(1.0, cos))
  return float(math.degrees(math.acos(cos)))


def analyze_track(
    points: list[TrackPoint],
    *,
    thresholds: Thresholds,
    paddle_boxes: dict[int, list[PaddleBox]] | None,
    max_gap: int,
    frame_size: float,
    wall_margin: float,
    paddle_x_margin: float,
    paddle_margin: float,
    turn_angle_deg: float,
    min_turn_speed_px: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
  if not points:
    return {}, []

  lookup = {p.t: p for p in points}
  times = sorted(lookup)
  horizon = int(times[-1] - times[0] + 1) if times else 0
  visible = [p for p in points if p.has_center]
  present_ts = [p.t for p in visible]
  visible_frames = len(visible)

  events: list[dict[str, Any]] = []
  continuous_steps = 0
  teleport_steps = 0
  eligible_teleport_steps = 0
  speeds = []

  for t in times:
    a = lookup.get(t - 1)
    b = lookup.get(t)
    if not (a and b and a.has_center and b.has_center):
      continue
    speed = float(np.hypot(b.x - a.x, b.y - a.y))
    speeds.append(speed)
    continuous_steps += 1
    near_collision = pair_near_collision_zone(
        a, b, frame_size=frame_size, wall_margin=wall_margin,
        paddle_x_margin=paddle_x_margin, paddle_boxes=paddle_boxes,
        paddle_margin=paddle_margin)
    if not near_collision:
      eligible_teleport_steps += 1
      if speed > thresholds.teleport_px:
        teleport_steps += 1
        events.append({
            'event': 'teleport',
            't': t,
            'value': speed,
            'threshold': thresholds.teleport_px,
            'x': b.x,
            'y': b.y,
        })

  flicker_gaps = 0
  short_gaps = 0
  gap_frames = 0
  longest_gap = 0
  reappearance_checks = 0
  reappearance_teleports = 0
  if len(present_ts) >= 2:
    for left_t, right_t in zip(present_ts[:-1], present_ts[1:]):
      gap = right_t - left_t - 1
      if gap <= 0:
        continue
      flicker_gaps += 1
      gap_frames += gap
      longest_gap = max(longest_gap, gap)
      if gap <= max_gap:
        short_gaps += 1
        left = lookup[left_t]
        right = lookup[right_t]
        prev = lookup.get(left_t - 1)
        if prev and prev.has_center:
          vx = left.x - prev.x
          vy = left.y - prev.y
          pred_x = left.x + vx * (gap + 1)
          pred_y = left.y + vy * (gap + 1)
          error = float(np.hypot(right.x - pred_x, right.y - pred_y))
          near_collision = pair_near_collision_zone(
              left, right, frame_size=frame_size, wall_margin=wall_margin,
              paddle_x_margin=paddle_x_margin, paddle_boxes=paddle_boxes,
              paddle_margin=paddle_margin)
          if not near_collision:
            reappearance_checks += 1
            if error > thresholds.reappearance_error_px:
              reappearance_teleports += 1
              events.append({
                  'event': 'reappearance_teleport',
                  't': right_t,
                  'gap': gap,
                  'value': error,
                  'threshold': thresholds.reappearance_error_px,
                  'x': right.x,
                  'y': right.y,
              })

  spontaneous_turns = 0
  eligible_turns = 0
  turn_angles = []
  accels = []
  for t in times:
    a = lookup.get(t - 1)
    b = lookup.get(t)
    c = lookup.get(t + 1)
    if not (a and b and c and a.has_center and b.has_center and c.has_center):
      continue
    v1 = np.array([b.x - a.x, b.y - a.y], np.float32)
    v2 = np.array([c.x - b.x, c.y - b.y], np.float32)
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    accel = float(np.linalg.norm(v2 - v1))
    accels.append(accel)
    near_collision = (
        near_collision_zone(a, frame_size=frame_size, wall_margin=wall_margin,
                            paddle_x_margin=paddle_x_margin,
                            paddle_boxes=paddle_boxes,
                            paddle_margin=paddle_margin) or
        near_collision_zone(b, frame_size=frame_size, wall_margin=wall_margin,
                            paddle_x_margin=paddle_x_margin,
                            paddle_boxes=paddle_boxes,
                            paddle_margin=paddle_margin) or
        near_collision_zone(c, frame_size=frame_size, wall_margin=wall_margin,
                            paddle_x_margin=paddle_x_margin,
                            paddle_boxes=paddle_boxes,
                            paddle_margin=paddle_margin))
    if near_collision or n1 < min_turn_speed_px or n2 < min_turn_speed_px:
      continue
    angle = angle_degrees(v1, v2)
    if not math.isfinite(angle):
      continue
    eligible_turns += 1
    turn_angles.append(angle)
    if angle > turn_angle_deg:
      spontaneous_turns += 1
      events.append({
          'event': 'spontaneous_turn',
          't': t,
          'value': angle,
          'threshold': turn_angle_deg,
          'x': b.x,
          'y': b.y,
      })

  summary = {
      'horizon': horizon,
      'visible_frames': visible_frames,
      'detectability': safe_rate(visible_frames, horizon),
      'missing_frames': horizon - visible_frames,
      'missing_rate': safe_rate(horizon - visible_frames, horizon),
      'flicker_gaps': flicker_gaps,
      'short_flicker_gaps': short_gaps,
      'gap_frames_between_visible': gap_frames,
      'gap_frame_rate': safe_rate(gap_frames, horizon),
      'longest_gap': longest_gap,
      'continuous_visible_steps': continuous_steps,
      'eligible_teleport_steps': eligible_teleport_steps,
      'teleport_steps': teleport_steps,
      'teleports_per_1k_eligible': 1000.0 * safe_rate(
          teleport_steps, eligible_teleport_steps),
      'teleports_per_1k_visible': 1000.0 * safe_rate(
          teleport_steps, max(visible_frames, 1)),
      'reappearance_checks': reappearance_checks,
      'reappearance_teleports': reappearance_teleports,
      'reappearance_teleport_rate': safe_rate(
          reappearance_teleports, reappearance_checks),
      'eligible_turns': eligible_turns,
      'spontaneous_turns': spontaneous_turns,
      'spontaneous_turns_per_1k_eligible': 1000.0 * safe_rate(
          spontaneous_turns, eligible_turns),
      'speed_mean': safe_mean(speeds),
      'speed_p95': safe_percentile(speeds, 95),
      'speed_p99': safe_percentile(speeds, 99),
      'accel_mean': safe_mean(accels),
      'accel_p95': safe_percentile(accels, 95),
      'accel_p99': safe_percentile(accels, 99),
      'turn_angle_p95': safe_percentile(turn_angles, 95),
  }
  return summary, events


def aggregate_model(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
  sums = [
      'horizon', 'visible_frames', 'missing_frames', 'flicker_gaps',
      'short_flicker_gaps', 'gap_frames_between_visible',
      'continuous_visible_steps', 'eligible_teleport_steps', 'teleport_steps',
      'reappearance_checks', 'reappearance_teleports', 'eligible_turns',
      'spontaneous_turns',
  ]
  out: dict[str, Any] = {'name': name, 'num_samples': len(rows)}
  for key in sums:
    out[key] = int(sum(int(r.get(key, 0) or 0) for r in rows))
  out.update({
      'detectability': safe_rate(out['visible_frames'], out['horizon']),
      'missing_rate': safe_rate(out['missing_frames'], out['horizon']),
      'gap_frame_rate': safe_rate(out['gap_frames_between_visible'], out['horizon']),
      'teleports_per_1k_eligible': 1000.0 * safe_rate(
          out['teleport_steps'], out['eligible_teleport_steps']),
      'teleports_per_1k_visible': 1000.0 * safe_rate(
          out['teleport_steps'], max(out['visible_frames'], 1)),
      'reappearance_teleport_rate': safe_rate(
          out['reappearance_teleports'], out['reappearance_checks']),
      'spontaneous_turns_per_1k_eligible': 1000.0 * safe_rate(
          out['spontaneous_turns'], out['eligible_turns']),
      'longest_gap': int(max((int(r.get('longest_gap', 0) or 0) for r in rows), default=0)),
      'speed_p95_mean': safe_mean(r.get('speed_p95', float('nan')) for r in rows),
      'accel_p95_mean': safe_mean(r.get('accel_p95', float('nan')) for r in rows),
      'accel_p99_mean': safe_mean(r.get('accel_p99', float('nan')) for r in rows),
      'turn_angle_p95_mean': safe_mean(
          r.get('turn_angle_p95', float('nan')) for r in rows),
  })
  return out


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]):
  path.parent.mkdir(parents=True, exist_ok=True)
  keys = sorted({key for row in rows for key in row.keys()})
  with path.open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader()
    for row in rows:
      writer.writerow(row)


def write_json(path: pathlib.Path, data: Any):
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(data, indent=2, allow_nan=True) + '\n')


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(
      description='Compute physical consistency metrics from Pong SAM2 tracks.')
  p.add_argument('--input-dir', type=pathlib.Path, action='append', default=[],
                 help='SAM2 output directory. Can be repeated.')
  p.add_argument('--track-csv', type=pathlib.Path, action='append', default=[],
                 help='Specific generated track CSV. Can be repeated.')
  p.add_argument('--real-track-csv', type=pathlib.Path, action='append', default=[],
                 help='Specific real/reference track CSV for threshold calibration.')
  p.add_argument(
      '--paddle-track-csv', action='append', default=[],
      help=(
          'Generated paddle track CSV for collision exclusion. Format can be '
          'model:left=/path.csv or model:right=/path.csv. If omitted, the script '
          'auto-discovers filenames containing left_paddle/right_paddle.'))
  p.add_argument('--output-dir', type=pathlib.Path, required=True)
  p.add_argument('--frame-size', type=float, default=64.0)
  p.add_argument('--wall-margin', type=float, default=4.0)
  p.add_argument('--paddle-x-margin', type=float, default=9.0)
  p.add_argument('--paddle-margin', type=float, default=2.5)
  p.add_argument('--max-gap', type=int, default=3)
  p.add_argument('--turn-angle-deg', type=float, default=45.0)
  p.add_argument('--min-turn-speed-px', type=float, default=0.5)
  p.add_argument('--speed-percentile', type=float, default=99.5)
  p.add_argument('--accel-percentile', type=float, default=99.5)
  p.add_argument('--speed-margin-px', type=float, default=2.0)
  p.add_argument('--accel-margin-px', type=float, default=2.0)
  p.add_argument('--teleport-px', type=float, default=None,
                 help='Override teleport speed threshold in 64x64 pixels/frame.')
  p.add_argument('--reappearance-error-px', type=float, default=None,
                 help='Override short-gap reappearance prediction error threshold.')
  p.add_argument('--accel-px', type=float, default=None,
                 help='Override acceleration artifact threshold.')
  return p.parse_args()


def main() -> None:
  args = parse_args()
  input_dirs = [p.expanduser().resolve() for p in args.input_dir]
  track_files = [p.expanduser().resolve() for p in args.track_csv]
  track_files.extend(find_track_files(input_dirs))
  track_files = sorted(set(p for p in track_files if p.exists() and not is_real_track(p)))
  if not track_files:
    raise FileNotFoundError('No generated SAM2 track CSVs found.')

  real_files = [p.expanduser().resolve() for p in args.real_track_csv]
  real_files.extend(find_real_track_files(input_dirs))
  real_files = sorted(set(p for p in real_files if p.exists()))
  real_tracks = []
  for path in real_files:
    real_tracks.extend(rows_to_tracks(read_csv_rows(path)).values())

  thresholds = calibrate_thresholds(
      real_tracks,
      teleport_px=args.teleport_px,
      reappearance_error_px=args.reappearance_error_px,
      accel_px=args.accel_px,
      speed_percentile=args.speed_percentile,
      accel_percentile=args.accel_percentile,
      speed_margin_px=args.speed_margin_px,
      accel_margin_px=args.accel_margin_px,
  )

  paddle_by_model_sample: dict[str, dict[str, dict[int, list[PaddleBox]]]] = defaultdict(
      lambda: defaultdict(lambda: defaultdict(list)))

  paddle_files: list[tuple[str, str, pathlib.Path]] = []
  for item in args.paddle_track_csv:
    if '=' not in item or ':' not in item.split('=', 1)[0]:
      raise ValueError(
          '--paddle-track-csv must look like model:left=/path.csv or model:right=/path.csv')
    lhs, raw_path = item.split('=', 1)
    model, side = lhs.split(':', 1)
    if side not in ('left', 'right'):
      raise ValueError(f'Invalid paddle side {side!r}; expected left or right.')
    paddle_files.append((model, side, pathlib.Path(raw_path).expanduser().resolve()))

  for path in find_paddle_track_files(input_dirs):
    key = paddle_track_key_from_path(path)
    if key is not None:
      model, side = key
      paddle_files.append((model, side, path))

  seen_paddle_files = set()
  for model, side, path in paddle_files:
    key = (model, side, path)
    if key in seen_paddle_files or not path.exists():
      continue
    seen_paddle_files.add(key)
    boxes_by_sample = rows_to_paddle_boxes(read_csv_rows(path))
    for sample_id, by_t in boxes_by_sample.items():
      for t, box in by_t.items():
        if box.has_box:
          paddle_by_model_sample[model][sample_id][t].append(box)

  sample_rows: list[dict[str, Any]] = []
  event_rows: list[dict[str, Any]] = []
  by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
  for path in track_files:
    name = track_name_from_path(path)
    tracks = rows_to_tracks(read_csv_rows(path))
    for sample_id, points in tracks.items():
      summary, events = analyze_track(
          points,
          thresholds=thresholds,
          paddle_boxes=paddle_by_model_sample.get(name, {}).get(sample_id),
          max_gap=args.max_gap,
          frame_size=args.frame_size,
          wall_margin=args.wall_margin,
          paddle_x_margin=args.paddle_x_margin,
          paddle_margin=args.paddle_margin,
          turn_angle_deg=args.turn_angle_deg,
          min_turn_speed_px=args.min_turn_speed_px,
      )
      row = {
          'name': name,
          'sample_id': sample_id,
          'track_csv': str(path),
          **summary,
      }
      sample_rows.append(row)
      by_model[name].append(row)
      for event in events:
        event_rows.append({
            'name': name,
            'sample_id': sample_id,
            'track_csv': str(path),
            **event,
        })

  aggregate_rows = [
      aggregate_model(rows, name)
      for name, rows in sorted(by_model.items())
  ]
  by_model_horizon: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
  for row in sample_rows:
    by_model_horizon[(str(row['name']), int(row.get('horizon', 0) or 0))].append(row)
  aggregate_horizon_rows = []
  for (name, horizon), rows in sorted(by_model_horizon.items()):
    agg = aggregate_model(rows, name)
    agg['horizon_per_sample'] = horizon
    aggregate_horizon_rows.append(agg)
  manifest = {
      'track_files': [str(p) for p in track_files],
      'real_track_files': [str(p) for p in real_files],
      'paddle_track_files': [
          {'name': name, 'side': side, 'path': str(path)}
          for name, side, path in sorted(seen_paddle_files, key=lambda x: (x[0], x[1], str(x[2])))
      ],
      'thresholds': {
          'teleport_px': thresholds.teleport_px,
          'reappearance_error_px': thresholds.reappearance_error_px,
          'accel_px': thresholds.accel_px,
      },
      'params': {
          'frame_size': args.frame_size,
          'wall_margin': args.wall_margin,
          'paddle_x_margin': args.paddle_x_margin,
          'paddle_margin': args.paddle_margin,
          'max_gap': args.max_gap,
          'turn_angle_deg': args.turn_angle_deg,
          'min_turn_speed_px': args.min_turn_speed_px,
          'speed_percentile': args.speed_percentile,
          'accel_percentile': args.accel_percentile,
          'speed_margin_px': args.speed_margin_px,
          'accel_margin_px': args.accel_margin_px,
      },
  }

  args.output_dir.mkdir(parents=True, exist_ok=True)
  write_csv(args.output_dir / 'physical_consistency_summary.csv', aggregate_rows)
  write_csv(
      args.output_dir / 'physical_consistency_summary_by_horizon.csv',
      aggregate_horizon_rows)
  write_csv(args.output_dir / 'physical_consistency_samples.csv', sample_rows)
  write_csv(args.output_dir / 'physical_consistency_events.csv', event_rows)
  write_json(args.output_dir / 'physical_consistency_summary.json', aggregate_rows)
  write_json(
      args.output_dir / 'physical_consistency_summary_by_horizon.json',
      aggregate_horizon_rows)
  write_json(args.output_dir / 'physical_consistency_manifest.json', manifest)

  print(f'Wrote {args.output_dir / "physical_consistency_summary.csv"}')
  print(json.dumps({
      'num_tracks': len(track_files),
      'num_real_tracks': len(real_files),
      'num_paddle_tracks': len(seen_paddle_files),
      'thresholds': manifest['thresholds'],
      'output_dir': str(args.output_dir),
  }, indent=2))


if __name__ == '__main__':
  main()
