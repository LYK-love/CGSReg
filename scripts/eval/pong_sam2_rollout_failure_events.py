#!/usr/bin/env python3
"""Detect closed-loop Pong rollout failure events from SAM2 ball tracks.

The script consumes rollout artifacts produced by the frontend-free play API
recorders. Each rollout `.npz` is expected to contain `frames` with one
bootstrap frame followed by action-indexed generated frames.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import re
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROJECTS_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))


EVENTS = ('ball_disappears', 'missed_player_bounce', 'spurious_x_bounce')
COLLISION_SIDES = ('left', 'right')
COLLISION_EVENTS = ('paddle_collision_expected_success', 'paddle_collision_expected_failure')


MANUAL_DREAMER_COUNTS = {
    ('ep00_seed0', 'size200m_repro'): {'ball_disappears': 3, 'missed_player_bounce': 0, 'spurious_x_bounce': 0},
    ('ep00_seed0', 'size200m_w0'): {'ball_disappears': 1, 'missed_player_bounce': 1, 'spurious_x_bounce': 0},
    ('ep00_seed0', 'size200m_w001'): {'ball_disappears': 0, 'missed_player_bounce': 0, 'spurious_x_bounce': 0},
    ('ep00_seed0', 'size400m_w0'): {'ball_disappears': 0, 'missed_player_bounce': 0, 'spurious_x_bounce': 2},
    ('ep00_seed0', 'size400m_w001'): {'ball_disappears': 0, 'missed_player_bounce': 0, 'spurious_x_bounce': 0},
    ('ep01_seed1', 'size200m_repro'): {'ball_disappears': 0, 'missed_player_bounce': 5, 'spurious_x_bounce': 0},
    ('ep01_seed1', 'size200m_w0'): {'ball_disappears': 0, 'missed_player_bounce': 3, 'spurious_x_bounce': 0},
    ('ep01_seed1', 'size200m_w001'): {'ball_disappears': 0, 'missed_player_bounce': 0, 'spurious_x_bounce': 0},
    ('ep01_seed1', 'size400m_w0'): {'ball_disappears': 0, 'missed_player_bounce': 2, 'spurious_x_bounce': 0},
    ('ep01_seed1', 'size400m_w001'): {'ball_disappears': 0, 'missed_player_bounce': 0, 'spurious_x_bounce': 0},
    ('ep02_seed2', 'size200m_repro'): {'ball_disappears': 4, 'missed_player_bounce': 0, 'spurious_x_bounce': 0},
    ('ep02_seed2', 'size200m_w0'): {'ball_disappears': 1, 'missed_player_bounce': 0, 'spurious_x_bounce': 0},
    ('ep02_seed2', 'size200m_w001'): {'ball_disappears': 0, 'missed_player_bounce': 1, 'spurious_x_bounce': 0},
    ('ep02_seed2', 'size400m_w0'): {'ball_disappears': 0, 'missed_player_bounce': 1, 'spurious_x_bounce': 0},
    ('ep02_seed2', 'size400m_w001'): {'ball_disappears': 0, 'missed_player_bounce': 0, 'spurious_x_bounce': 0},
    ('ep03_seed3', 'size200m_repro'): {'ball_disappears': 0, 'missed_player_bounce': 5, 'spurious_x_bounce': 0},
    ('ep03_seed3', 'size200m_w0'): {'ball_disappears': 0, 'missed_player_bounce': 1, 'spurious_x_bounce': 0},
    ('ep03_seed3', 'size200m_w001'): {'ball_disappears': 0, 'missed_player_bounce': 0, 'spurious_x_bounce': 0},
    ('ep03_seed3', 'size400m_w0'): {'ball_disappears': 0, 'missed_player_bounce': 3, 'spurious_x_bounce': 0},
    ('ep03_seed3', 'size400m_w001'): {'ball_disappears': 0, 'missed_player_bounce': 0, 'spurious_x_bounce': 0},
    ('ep04_seed4', 'size200m_repro'): {'ball_disappears': 0, 'missed_player_bounce': 5, 'spurious_x_bounce': 0},
    ('ep04_seed4', 'size200m_w0'): {'ball_disappears': 0, 'missed_player_bounce': 0, 'spurious_x_bounce': 0},
    ('ep04_seed4', 'size200m_w001'): {'ball_disappears': 0, 'missed_player_bounce': 0, 'spurious_x_bounce': 0},
    ('ep04_seed4', 'size400m_w0'): {'ball_disappears': 0, 'missed_player_bounce': 0, 'spurious_x_bounce': 1},
    ('ep04_seed4', 'size400m_w001'): {'ball_disappears': 0, 'missed_player_bounce': 0, 'spurious_x_bounce': 0},
}


@dataclass(frozen=True)
class RolloutSpec:
  source: str
  root: pathlib.Path
  episode: str
  model: str
  path: pathlib.Path


@dataclass(frozen=True)
class SegmentSpec:
  name: str
  frames64: np.ndarray


@dataclass
class PaddleBox:
  present: bool = False
  xmin: float = math.nan
  xmax: float = math.nan
  ymin: float = math.nan
  ymax: float = math.nan


def ensure_dir(path: pathlib.Path):
  path.mkdir(parents=True, exist_ok=True)


def write_json(path: pathlib.Path, data: Any):
  ensure_dir(path.parent)
  path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]):
  ensure_dir(path.parent)
  keys = sorted({key for row in rows for key in row})
  with path.open('w', newline='') as f:
    if not keys:
      f.write('')
      return
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)


def resize_video(frames: np.ndarray, size: int) -> np.ndarray:
  frames = np.asarray(frames)
  if frames.ndim != 4 or frames.shape[-1] != 3:
    raise ValueError(f'Expected frames shape (T,H,W,3), got {frames.shape}')
  if frames.shape[1] == size and frames.shape[2] == size:
    return frames.astype(np.uint8)
  out = []
  for frame in frames:
    image = Image.fromarray(frame.astype(np.uint8))
    image = image.resize((int(size), int(size)), resample=Image.NEAREST)
    out.append(np.asarray(image, dtype=np.uint8))
  return np.stack(out, 0)


def save_video(path: pathlib.Path, frames: np.ndarray, fps: int):
  import imageio.v3 as iio

  ensure_dir(path.parent)
  iio.imwrite(path, np.asarray(frames, np.uint8), fps=int(fps))


def render_overlay(frames: np.ndarray, mask: np.ndarray, color=(255, 64, 128), alpha=0.45):
  frames = np.asarray(frames, np.uint8)
  mask = np.asarray(mask) > 0
  out = frames.copy().astype(np.float32)
  color_arr = np.asarray(color, np.float32)
  out[mask] = (1.0 - alpha) * out[mask] + alpha * color_arr
  return np.clip(out, 0, 255).astype(np.uint8)


def run_sam2(spec: SegmentSpec, prompt: dict[str, Any], args):
  sys.path.insert(0, str(args.sam2_client_root))
  try:
    from sam2_client import chunked_sam2_segmentation
  except ImportError as exc:
    raise RuntimeError(f'Could not import sam2_client from {args.sam2_client_root}') from exc

  video_arr = resize_video(spec.frames64, args.sam2_size)
  masks = chunked_sam2_segmentation(
      video_arr=video_arr,
      guiding_video_arr=video_arr,
      prompts=[prompt],
      backend_endpoint=args.backend_endpoint,
      selected_obj_ids=(1,),
      verbose_mode=bool(args.verbose),
      max_frames_per_chunk=int(args.max_frames_per_chunk),
  )
  return np.asarray(masks[1], np.uint8)


def add_velocity(rows: list[dict[str, Any]]):
  prev = None
  for row in rows:
    if prev and row.get('ball_present') and prev.get('ball_present'):
      vx = float(row['ball_x']) - float(prev['ball_x'])
      vy = float(row['ball_y']) - float(prev['ball_y'])
      row['ball_vx'] = vx
      row['ball_vy'] = vy
      row['ball_vx_sign'] = int(np.sign(vx))
      row['ball_vy_sign'] = int(np.sign(vy))
    else:
      row['ball_vx'] = float('nan')
      row['ball_vy'] = float('nan')
      row['ball_vx_sign'] = 0
      row['ball_vy_sign'] = 0
    prev = row


def tracks_from_masks(masks: np.ndarray, downscale: float, sample_id: str):
  rows = []
  for t, mask in enumerate(np.asarray(masks)):
    ys, xs = np.nonzero(mask > 0)
    row = {
        'sample_id': sample_id,
        't': int(t),
        'ball_present': bool(xs.size),
        'ball_area': int(xs.size),
    }
    if xs.size:
      row['ball_x'] = float(xs.mean() / downscale)
      row['ball_y'] = float(ys.mean() / downscale)
      row['ball_x_sam2'] = float(xs.mean())
      row['ball_y_sam2'] = float(ys.mean())
    rows.append(row)
  add_velocity(rows)
  return rows


def component_boxes(mask: np.ndarray) -> list[dict[str, float]]:
  mask = np.asarray(mask, bool)
  seen = np.zeros(mask.shape, bool)
  boxes = []
  height, width = mask.shape
  for y0, x0 in zip(*np.nonzero(mask & ~seen)):
    stack = [(int(y0), int(x0))]
    seen[y0, x0] = True
    xs = []
    ys = []
    while stack:
      y, x = stack.pop()
      xs.append(x)
      ys.append(y)
      for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
        if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
          seen[ny, nx] = True
          stack.append((ny, nx))
    arr_x = np.asarray(xs)
    arr_y = np.asarray(ys)
    boxes.append({
        'area': float(arr_x.size),
        'xmin': float(arr_x.min()),
        'xmax': float(arr_x.max()),
        'ymin': float(arr_y.min()),
        'ymax': float(arr_y.max()),
        'x': float(arr_x.mean()),
        'y': float(arr_y.mean()),
    })
  return boxes


def detect_pong_ball_center(frame: np.ndarray):
  frame = np.asarray(frame, np.uint8)
  gray = frame.astype(np.float32).mean(axis=-1)
  candidates = gray > 80.0
  candidates[:12, :] = False
  candidates[57:, :] = False
  comps = []
  for comp in component_boxes(candidates):
    width = comp['xmax'] - comp['xmin'] + 1
    height = comp['ymax'] - comp['ymin'] + 1
    patch = frame[int(comp['ymin']):int(comp['ymax']) + 1, int(comp['xmin']):int(comp['xmax']) + 1]
    patch_mask = candidates[int(comp['ymin']):int(comp['ymax']) + 1, int(comp['xmin']):int(comp['xmax']) + 1]
    mean_rgb = patch[patch_mask].mean(axis=0)
    green_bias = float(mean_rgb[1] - max(mean_rgb[0], mean_rgb[2]))
    if 2 <= comp['area'] <= 30 and 2 <= width <= 7 and 2 <= height <= 7:
      if 10 < comp['x'] < 54 and 12 < comp['y'] < 57 and green_bias < 70.0:
        comps.append(comp)
  if not comps:
    return None

  def score(comp):
    center_bias = abs(comp['x'] - 32.0) * 0.05 + abs(comp['y'] - 34.0) * 0.02
    side_penalty = 6.0 if comp['x'] < 10 or comp['x'] > 54 else 0.0
    area_penalty = abs(comp['area'] - 12.0) * 0.08
    return center_bias + side_penalty + area_penalty

  best = min(comps, key=score)
  return best['x'], best['y']


def prompt_from_center(center, frame_idx: int, source_size: int, sam2_size: int):
  x64, y64 = center
  scale = sam2_size / float(source_size)
  x = (x64 + 0.5) * scale
  y = (y64 + 0.5) * scale
  return {
      'frame_idx': int(frame_idx),
      'obj_id': 1,
      'labels': [1],
      'points': [[float(x / sam2_size), float(y / sam2_size)]],
      'point_px_source': [float(x64), float(y64)],
      'point_px_sam2': [float(x), float(y)],
      'source': 'wm_heuristic',
  }


def detect_paddle(frame64: np.ndarray, side: str) -> PaddleBox:
  frame = np.asarray(frame64, np.uint8)
  playfield = frame[12:58]
  colors, counts = np.unique(playfield.reshape(-1, 3), axis=0, return_counts=True)
  background = colors[int(np.argmax(counts))].astype(np.float32)
  color_dist = np.linalg.norm(frame.astype(np.float32) - background, axis=-1)
  mask = color_dist > 25.0
  mask[:12, :] = False
  mask[58:, :] = False
  if side == 'left':
    mask[:, 16:] = False
  elif side == 'right':
    mask[:, :48] = False
  else:
    raise ValueError(f'Invalid paddle side {side!r}')

  column_has_pixels = mask.any(axis=0)
  candidates = []
  x = 0
  while x < len(column_has_pixels):
    if not column_has_pixels[x]:
      x += 1
      continue
    start = x
    while x < len(column_has_pixels) and column_has_pixels[x]:
      x += 1
    end = x
    width = end - start
    if not (1 <= width <= 6):
      continue
    rows = mask[:, start:end].any(axis=1)
    y = 0
    while y < len(rows):
      if not rows[y]:
        y += 1
        continue
      y_start = y
      while y < len(rows) and rows[y]:
        y += 1
      y_end = y
      height = y_end - y_start
      area = int(mask[y_start:y_end, start:end].sum())
      if 6 <= height <= 24 and area >= 8:
        candidates.append({
            'area': float(area),
            'xmin': float(start),
            'xmax': float(end - 1),
            'ymin': float(y_start),
            'ymax': float(y_end - 1),
        })
  if candidates:
    best = max(candidates, key=lambda box: (box['ymax'] - box['ymin'] + 1.0, box['area']))
    return PaddleBox(True, best['xmin'], best['xmax'], best['ymin'], best['ymax'])
  return PaddleBox(False)


def discover_rollouts(roots: list[pathlib.Path]) -> list[RolloutSpec]:
  specs = []
  for root in roots:
    root = root.expanduser().resolve()
    source = root.name
    for path in sorted(root.glob('ep*_seed*/rollouts/*.npz')):
      specs.append(RolloutSpec(source, root, path.parent.parent.name, path.stem, path))
  return specs


def load_frames(path: pathlib.Path, max_frames: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  with np.load(path, allow_pickle=False) as data:
    frames = np.asarray(data['frames'], np.uint8)
    rewards = np.asarray(data.get('rewards', np.zeros(max(frames.shape[0] - 1, 0))), np.float32)
    actions = np.asarray(data.get('actions', np.zeros(max(frames.shape[0] - 1, 0))), np.int64)
  frames = frames[1:]
  if max_frames is not None:
    frames = frames[:max_frames]
    rewards = rewards[:max_frames]
    actions = actions[:max_frames]
  return frames, rewards, actions


def to_game64(frames: np.ndarray) -> np.ndarray:
  frames = np.asarray(frames, np.uint8)
  if frames.shape[1] == 64 and frames.shape[2] == 64:
    return frames
  return resize_video(frames, 64)


def make_prompt(frames64: np.ndarray, prompt_search_frames: int, sam2_size: int) -> dict[str, Any] | None:
  for t in range(min(int(prompt_search_frames), len(frames64))):
    center = detect_pong_ball_center(frames64[t])
    if center is not None:
      return prompt_from_center(center, t, frames64.shape[1], sam2_size)
  return None


def rows_to_arrays(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  present = np.asarray([bool(row.get('ball_present')) for row in rows], bool)
  xs = np.asarray([float(row.get('ball_x', np.nan)) for row in rows], np.float32)
  ys = np.asarray([float(row.get('ball_y', np.nan)) for row in rows], np.float32)
  areas = np.asarray([float(row.get('ball_area', 0.0)) for row in rows], np.float32)
  return present, xs, ys, areas


def heuristic_rows(frames64: np.ndarray, sample_id: str) -> list[dict[str, Any]]:
  rows = []
  for t, frame in enumerate(frames64):
    center = detect_pong_ball_center(frame)
    row = {
        'sample_id': sample_id,
        't': int(t),
        'ball_present': center is not None,
        'ball_area': 0,
    }
    if center is not None:
      row['ball_x'], row['ball_y'] = center
    rows.append(row)
  add_velocity(rows)
  return rows


def hybridize_tracks(
    sam2_rows: list[dict[str, Any]],
    heuristic: list[dict[str, Any]],
    args,
) -> list[dict[str, Any]]:
  out = []
  for sam2, heur in zip(sam2_rows, heuristic):
    row = dict(sam2)
    row['sam2_ball_present'] = bool(sam2.get('ball_present'))
    row['sam2_ball_x'] = sam2.get('ball_x', '')
    row['sam2_ball_y'] = sam2.get('ball_y', '')
    row['heuristic_ball_present'] = bool(heur.get('ball_present'))
    row['heuristic_ball_x'] = heur.get('ball_x', '')
    row['heuristic_ball_y'] = heur.get('ball_y', '')
    row['ball_source'] = 'sam2' if row['sam2_ball_present'] else 'missing'

    use_heuristic = False
    if bool(heur.get('ball_present')):
      if not row['sam2_ball_present']:
        use_heuristic = True
      elif args.prefer_heuristic_when_disagree:
        dx = float(sam2['ball_x']) - float(heur['ball_x'])
        dy = float(sam2['ball_y']) - float(heur['ball_y'])
        if math.hypot(dx, dy) > args.max_sam2_heuristic_dist:
          use_heuristic = True

    if args.track_source == 'heuristic':
      use_heuristic = bool(heur.get('ball_present'))
    elif args.track_source == 'sam2':
      use_heuristic = False

    if use_heuristic:
      row['ball_present'] = True
      row['ball_x'] = float(heur['ball_x'])
      row['ball_y'] = float(heur['ball_y'])
      row['ball_area'] = row.get('ball_area', 0)
      row['ball_source'] = 'heuristic'
    out.append(row)
  add_velocity(out)
  return out


def velocity(xs: np.ndarray, present: np.ndarray, window: int) -> np.ndarray:
  out = np.full(xs.shape, np.nan, np.float32)
  for t in range(int(window), len(xs)):
    if present[t] and present[t - int(window)]:
      out[t] = (xs[t] - xs[t - int(window)]) / float(window)
  return out


def ball_near_paddle(
    x: float,
    y: float,
    box: PaddleBox,
    side: str,
    x_margin: float,
    y_margin: float,
) -> bool:
  if not box.present or not np.isfinite(x) or not np.isfinite(y):
    return False
  if not (box.ymin - y_margin <= y <= box.ymax + y_margin):
    return False
  if side == 'right':
    return box.xmin - x_margin <= x <= box.xmax + x_margin
  return box.xmin - x_margin <= x <= box.xmax + x_margin


def near_any_paddle(
    t: int,
    x: float,
    y: float,
    paddles: dict[str, list[PaddleBox]],
    x_margin: float,
    y_margin: float,
) -> bool:
  for side in ('left', 'right'):
    boxes = paddles.get(side, [])
    if t < len(boxes) and ball_near_paddle(x, y, boxes[t], side, x_margin, y_margin):
      return True
  return False


def event_row(spec: RolloutSpec, event: str, t: int, **kwargs) -> dict[str, Any]:
  row = {
      'source': spec.source,
      'episode': spec.episode,
      'model': spec.model,
      'event': event,
      't': int(t),
  }
  row.update(kwargs)
  return row


def detect_ball_disappears(
    spec: RolloutSpec,
    present: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    rewards: np.ndarray,
    args,
) -> list[dict[str, Any]]:
  events = []
  t = 0
  while t < len(present):
    if present[t]:
      t += 1
      continue
    start = t
    while t < len(present) and not present[t]:
      t += 1
    end = t
    prev = start - 1
    nxt = end
    if prev < 0 or nxt >= len(present) or not present[prev] or not present[nxt]:
      continue
    gap_len = end - start
    if gap_len < args.min_disappear_gap:
      continue
    if (xs[prev] <= args.boundary_margin or xs[prev] >= 64.0 - args.boundary_margin or
        xs[nxt] <= args.boundary_margin or xs[nxt] >= 64.0 - args.boundary_margin):
      continue
    reward_window = rewards[max(0, prev - 1):min(len(rewards), nxt + 2)]
    if reward_window.size and np.nanmax(np.abs(reward_window)) >= args.reward_event_threshold:
      continue
    events.append(event_row(
        spec, 'ball_disappears', start, x=float(xs[prev]), y=float(ys[prev]),
        gap_len=int(gap_len), reason='visible_missing_visible_inside_field'))
  return events


def detect_missed_player_bounce(
    spec: RolloutSpec,
    present: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    vx: np.ndarray,
    paddles: dict[str, list[PaddleBox]],
    args,
) -> list[dict[str, Any]]:
  events = []
  side = args.player_side
  cooldown_until = -1
  expect_sign = -1 if side == 'right' else 1
  toward_sign = 1 if side == 'right' else -1
  boxes = paddles[side]
  for t in range(1, len(xs)):
    if t <= cooldown_until or not present[t] or not np.isfinite(vx[t]):
      continue
    if np.sign(vx[t]) != toward_sign or abs(float(vx[t])) < args.min_vx:
      continue
    if t >= len(boxes) or not ball_near_paddle(
        float(xs[t]), float(ys[t]), boxes[t], side,
        args.paddle_x_margin, args.paddle_y_margin):
      continue

    look = range(t + 1, min(len(xs), t + 1 + args.bounce_lookahead))
    visible = [k for k in look if present[k] and np.isfinite(vx[k])]
    if not visible:
      continue
    reversed_steps = [k for k in visible if np.sign(vx[k]) == expect_sign and abs(float(vx[k])) >= args.min_vx]
    if reversed_steps:
      cooldown_until = max(cooldown_until, reversed_steps[0] + args.event_cooldown)
      continue
    events.append(event_row(
        spec, 'missed_player_bounce', t, x=float(xs[t]), y=float(ys[t]),
        vx_before=float(vx[t]), vx_after=float(vx[visible[-1]]),
        player_side=side, reason='entered_player_paddle_zone_without_x_reversal'))
    cooldown_until = t + args.event_cooldown
  return events


def detect_expected_paddle_collisions(
    spec: RolloutSpec,
    present: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    vx: np.ndarray,
    paddles: dict[str, list[PaddleBox]],
    args,
) -> list[dict[str, Any]]:
  """Count paddle collision opportunities and whether x velocity reverses.

  One opportunity is counted when the visible ball is moving toward a paddle and
  enters that paddle's collision zone. It is successful if vx reverses within a
  short lookahead window; otherwise it is an expected collision failure.
  """
  events = []
  cooldown_until = {side: -1 for side in COLLISION_SIDES}
  for t in range(1, len(xs)):
    if not present[t] or not np.isfinite(vx[t]) or abs(float(vx[t])) < args.min_vx:
      continue
    for side in COLLISION_SIDES:
      if t <= cooldown_until[side]:
        continue
      toward_sign = -1 if side == 'left' else 1
      reversed_sign = 1 if side == 'left' else -1
      if np.sign(vx[t]) != toward_sign:
        continue
      boxes = paddles.get(side, [])
      if t >= len(boxes) or not ball_near_paddle(
          float(xs[t]), float(ys[t]), boxes[t], side,
          args.paddle_x_margin, args.paddle_y_margin):
        continue

      look = range(t + 1, min(len(xs), t + 1 + args.bounce_lookahead))
      visible = [k for k in look if present[k] and np.isfinite(vx[k])]
      if not visible:
        continue
      reversed_steps = [
          k for k in visible
          if np.sign(vx[k]) == reversed_sign and abs(float(vx[k])) >= args.min_vx]
      if reversed_steps:
        reversal_t = int(reversed_steps[0])
        events.append(event_row(
            spec, 'paddle_collision_expected_success', t,
            side=side, x=float(xs[t]), y=float(ys[t]),
            vx_before=float(vx[t]), vx_after=float(vx[reversal_t]),
            reversal_t=reversal_t,
            reason='entered_paddle_zone_and_x_velocity_reversed'))
        cooldown_until[side] = reversal_t + args.event_cooldown
      else:
        events.append(event_row(
            spec, 'paddle_collision_expected_failure', t,
            side=side, x=float(xs[t]), y=float(ys[t]),
            vx_before=float(vx[t]), vx_after=float(vx[visible[-1]]),
            reason='entered_paddle_zone_without_x_velocity_reversal'))
        cooldown_until[side] = t + args.event_cooldown
  return events


def detect_spurious_x_bounce(
    spec: RolloutSpec,
    present: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    vx: np.ndarray,
    sources: list[str],
    paddles: dict[str, list[PaddleBox]],
    args,
) -> list[dict[str, Any]]:
  events = []
  cooldown_until = -1
  for t in range(2, len(xs)):
    if t <= cooldown_until or not present[t] or not present[t - 1]:
      continue
    before = vx[t - 1]
    after = vx[t]
    if args.spurious_requires_sam2 and (
        t >= len(sources) or t - 1 >= len(sources) or
        sources[t] != 'sam2' or sources[t - 1] != 'sam2'):
      continue
    if not np.isfinite(before) or not np.isfinite(after):
      continue
    if abs(float(before)) < args.min_vx or abs(float(after)) < args.min_vx:
      continue
    if np.sign(before) == np.sign(after):
      continue
    x = float(xs[t])
    y = float(ys[t])
    if x <= args.boundary_margin or x >= 64.0 - args.boundary_margin:
      continue
    if x <= args.spurious_side_margin or x >= 64.0 - args.spurious_side_margin:
      continue
    if near_any_paddle(t, x, y, paddles, args.paddle_x_margin, args.paddle_y_margin):
      continue
    events.append(event_row(
        spec, 'spurious_x_bounce', t, x=x, y=y, vx_before=float(before),
        vx_after=float(after), reason='x_velocity_reversed_away_from_paddles'))
    cooldown_until = t + args.event_cooldown
  return events


def render_event_overlay(frames64: np.ndarray, masks: np.ndarray, events: list[dict[str, Any]], args) -> np.ndarray:
  frames = resize_video(frames64, args.sam2_size)
  overlay = render_overlay(frames, masks)
  colors = {
      'ball_disappears': np.asarray([255, 48, 48], np.uint8),
      'missed_player_bounce': np.asarray([255, 220, 32], np.uint8),
      'spurious_x_bounce': np.asarray([160, 64, 255], np.uint8),
      'paddle_collision_expected_success': np.asarray([64, 220, 96], np.uint8),
      'paddle_collision_expected_failure': np.asarray([255, 160, 32], np.uint8),
  }
  scale = args.sam2_size / 64.0
  for event in events:
    t = int(event['t'])
    if not (0 <= t < len(overlay)):
      continue
    color = colors.get(str(event['event']), np.asarray([255, 255, 255], np.uint8))
    x = int(round(float(event.get('x', 32.0)) * scale))
    y = int(round(float(event.get('y', 32.0)) * scale))
    x = int(np.clip(x, 0, args.sam2_size - 1))
    y = int(np.clip(y, 0, args.sam2_size - 1))
    overlay[t, max(0, y - 6):min(args.sam2_size, y + 7), max(0, x - 1):min(args.sam2_size, x + 2)] = color
    overlay[t, max(0, y - 1):min(args.sam2_size, y + 2), max(0, x - 6):min(args.sam2_size, x + 7)] = color
    overlay[t, :, :3] = color
  return overlay


def safe_name(spec: RolloutSpec) -> str:
  return f'{spec.source}__{spec.episode}__{spec.model}'


def analyze_rollout(spec: RolloutSpec, args) -> dict[str, Any]:
  out_dir = args.output_dir / 'rollouts' / safe_name(spec)
  ensure_dir(out_dir)
  events_path = out_dir / 'events.csv'
  mask_path = out_dir / 'sam2_ball.npz'
  track_path = out_dir / 'sam2_tracks.csv'

  frames_raw, rewards, actions = load_frames(spec.path, args.max_frames)
  frames64 = to_game64(frames_raw)
  prompt = make_prompt(frames64, args.prompt_search_frames, args.sam2_size)
  if args.track_source == 'heuristic':
    masks = np.zeros((len(frames64), args.sam2_size, args.sam2_size), np.uint8)
    prompt_meta = {'source': 'heuristic_only', 'reason': 'track_source_heuristic'}
  elif prompt is None:
    masks = np.zeros((len(frames64), args.sam2_size, args.sam2_size), np.uint8)
    prompt_meta = {'source': 'none', 'reason': 'no_generated_ball_prompt'}
  elif mask_path.exists() and not args.force_sam2:
    with np.load(mask_path, allow_pickle=False) as data:
      masks = np.asarray(data['mask1'], np.uint8)
    prompt_meta = prompt
  else:
    print(f'Running SAM2: {spec.source}/{spec.episode}/{spec.model}', flush=True)
    sam2_args = argparse.Namespace(**vars(args))
    sam2_args.real_frames64 = frames64
    masks = run_sam2(SegmentSpec(spec.model, frames64), prompt, sam2_args)
    np.savez_compressed(mask_path, mask1=masks)
    prompt_meta = prompt

  if masks.shape[0] != len(frames64):
    masks = masks[:len(frames64)]
  sam2_rows = tracks_from_masks(masks, downscale=args.sam2_size / 64.0, sample_id=safe_name(spec))
  heur_rows = heuristic_rows(frames64, sample_id=safe_name(spec))
  rows = hybridize_tracks(sam2_rows, heur_rows, args)
  write_csv(track_path, rows)
  present, xs, ys, areas = rows_to_arrays(rows)
  vx = velocity(xs, present, args.velocity_window)
  paddles = {
      'left': [detect_paddle(frame, 'left') for frame in frames64],
      'right': [detect_paddle(frame, 'right') for frame in frames64],
  }

  events = []
  events.extend(detect_ball_disappears(spec, present, xs, ys, rewards, args))
  events.extend(detect_expected_paddle_collisions(spec, present, xs, ys, vx, paddles, args))
  events.extend(detect_missed_player_bounce(spec, present, xs, ys, vx, paddles, args))
  sources = [str(row.get('ball_source', '')) for row in rows]
  events.extend(detect_spurious_x_bounce(spec, present, xs, ys, vx, sources, paddles, args))
  events.sort(key=lambda row: (int(row['t']), str(row['event'])))
  write_csv(events_path, events)

  if args.save_overlays:
    overlay = render_event_overlay(frames64, masks, events, args)
    save_video(out_dir / 'sam2_event_overlay.mp4', overlay, args.fps)

  counts = {event: 0 for event in EVENTS}
  collision_counts = {
      'expected_paddle_collisions': 0,
      'successful_paddle_collisions': 0,
      'failed_expected_paddle_collisions': 0,
  }
  for side in COLLISION_SIDES:
    collision_counts[f'{side}_expected_paddle_collisions'] = 0
    collision_counts[f'{side}_successful_paddle_collisions'] = 0
    collision_counts[f'{side}_failed_expected_paddle_collisions'] = 0
  for row in events:
    event = str(row['event'])
    if event in counts:
      counts[event] += 1
    if event in COLLISION_EVENTS:
      side = str(row.get('side', ''))
      collision_counts['expected_paddle_collisions'] += 1
      if side in COLLISION_SIDES:
        collision_counts[f'{side}_expected_paddle_collisions'] += 1
      if event == 'paddle_collision_expected_success':
        collision_counts['successful_paddle_collisions'] += 1
        if side in COLLISION_SIDES:
          collision_counts[f'{side}_successful_paddle_collisions'] += 1
      elif event == 'paddle_collision_expected_failure':
        collision_counts['failed_expected_paddle_collisions'] += 1
        if side in COLLISION_SIDES:
          collision_counts[f'{side}_failed_expected_paddle_collisions'] += 1
  expected = int(collision_counts['expected_paddle_collisions'])
  failed = int(collision_counts['failed_expected_paddle_collisions'])
  collision_counts['failed_expected_collision_rate'] = (
      float(failed / expected) if expected else float('nan'))
  for side in COLLISION_SIDES:
    side_expected = int(collision_counts[f'{side}_expected_paddle_collisions'])
    side_failed = int(collision_counts[f'{side}_failed_expected_paddle_collisions'])
    collision_counts[f'{side}_failed_expected_collision_rate'] = (
        float(side_failed / side_expected) if side_expected else float('nan'))
  summary = {
      'source': spec.source,
      'episode': spec.episode,
      'model': spec.model,
      'path': str(spec.path),
      'frames': int(len(frames64)),
      'actions': int(len(actions)),
      'prompt': prompt_meta,
      'sam2_present_frames': int(present.sum()),
      'sam2_raw_present_frames': int(sum(bool(row.get('sam2_ball_present')) for row in rows)),
      'sam2_raw_present_rate': float(np.mean([bool(row.get('sam2_ball_present')) for row in rows])) if rows else float('nan'),
      'heuristic_present_frames': int(sum(bool(row.get('heuristic_ball_present')) for row in rows)),
      'hybrid_present_frames': int(present.sum()),
      'sam2_present_rate': float(present.mean()) if len(present) else float('nan'),
      'hybrid_present_rate': float(present.mean()) if len(present) else float('nan'),
      'mean_ball_area': float(np.nanmean(areas[present])) if present.any() else float('nan'),
      'output_dir': str(out_dir),
  }
  summary.update(counts)
  summary.update(collision_counts)
  write_json(out_dir / 'summary.json', summary)
  return summary


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
  grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
  for row in rows:
    grouped.setdefault((str(row['source']), str(row['model'])), []).append(row)
  out = []
  for (source, model), items in sorted(grouped.items()):
    agg = {
        'source': source,
        'model': model,
        'episodes': len({str(row['episode']) for row in items}),
        'sam2_present_rate_mean': float(np.nanmean([float(row['sam2_present_rate']) for row in items])),
    }
    for event in EVENTS:
      agg[event] = int(sum(int(row.get(event, 0)) for row in items))
    for key in (
        'expected_paddle_collisions',
        'successful_paddle_collisions',
        'failed_expected_paddle_collisions',
        'left_expected_paddle_collisions',
        'left_successful_paddle_collisions',
        'left_failed_expected_paddle_collisions',
        'right_expected_paddle_collisions',
        'right_successful_paddle_collisions',
        'right_failed_expected_paddle_collisions',
    ):
      agg[key] = int(sum(int(row.get(key, 0)) for row in items))
    expected = int(agg['expected_paddle_collisions'])
    failed = int(agg['failed_expected_paddle_collisions'])
    agg['failed_expected_collision_rate'] = float(failed / expected) if expected else float('nan')
    for side in COLLISION_SIDES:
      side_expected = int(agg[f'{side}_expected_paddle_collisions'])
      side_failed = int(agg[f'{side}_failed_expected_paddle_collisions'])
      agg[f'{side}_failed_expected_collision_rate'] = (
          float(side_failed / side_expected) if side_expected else float('nan'))
    out.append(agg)
  return out


def manual_comparison(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
  by_key = {(str(row['episode']), str(row['model'])): row for row in rows}
  out = []
  for key, manual in sorted(MANUAL_DREAMER_COUNTS.items()):
    if key not in by_key:
      continue
    auto = by_key[key]
    row = {'episode': key[0], 'model': key[1]}
    for event in EVENTS:
      row[f'manual_{event}'] = int(manual[event])
      row[f'sam2_{event}'] = int(auto.get(event, 0))
      row[f'delta_{event}'] = int(auto.get(event, 0)) - int(manual[event])
    out.append(row)
  return out


def parse_roots(values: list[str]) -> list[pathlib.Path]:
  roots = []
  for value in values:
    for part in value.split(','):
      part = part.strip()
      if part:
        roots.append(pathlib.Path(part))
  return roots


def parse_args(argv: list[str] | None = None):
  parser = argparse.ArgumentParser(
      description='Run SAM2-based closed-loop Pong rollout failure analysis.')
  parser.add_argument(
      '--input-root', action='append',
      default=[],
      help='Rollout artifact root. Can be repeated or comma-separated.')
  parser.add_argument('--output-dir', type=pathlib.Path,
                      default=ROOT / 'eval_outputs/closed_loop_rollout_failure_sam2')
  parser.add_argument('--models', default='',
                      help='Optional comma-separated model name regex filters.')
  parser.add_argument('--episodes', default='',
                      help='Optional comma-separated episode regex filters.')
  parser.add_argument('--limit-rollouts', type=int, default=0)
  parser.add_argument('--max-frames', type=int, default=0,
                      help='Analyze only the first N action-indexed frames.')
  parser.add_argument('--player-side', choices=('left', 'right'), default='right')
  parser.add_argument('--prompt-search-frames', type=int, default=96)
  parser.add_argument('--sam2-size', type=int, default=256)
  parser.add_argument('--sam2-client-root', type=pathlib.Path,
                      default=PROJECTS_ROOT / 'NovelWorldModel/sam2_client')
  parser.add_argument('--backend-endpoint', default='http://localhost:7263')
  parser.add_argument('--guiding-source', choices=('wm', 'real'), default='wm')
  parser.add_argument('--max-frames-per-chunk', type=int, default=1024)
  parser.add_argument('--force-sam2', action='store_true')
  parser.add_argument('--save-overlays', action='store_true')
  parser.add_argument('--fps', type=int, default=30)
  parser.add_argument('--verbose', action='store_true')
  parser.add_argument('--track-source', choices=('sam2', 'hybrid', 'heuristic'), default='hybrid',
                      help='Track used by event detectors. Hybrid uses SAM2 and fills confirmed missing frames with the Pong ball heuristic.')
  parser.add_argument('--prefer-heuristic-when-disagree', action='store_true', default=True)
  parser.add_argument('--max-sam2-heuristic-dist', type=float, default=6.0)
  parser.add_argument('--velocity-window', type=int, default=2)
  parser.add_argument('--min-vx', type=float, default=0.45)
  parser.add_argument('--boundary-margin', type=float, default=10.0)
  parser.add_argument('--spurious-side-margin', type=float, default=14.0)
  parser.add_argument('--paddle-x-margin', type=float, default=4.0)
  parser.add_argument('--paddle-y-margin', type=float, default=3.0)
  parser.add_argument('--bounce-lookahead', type=int, default=4)
  parser.add_argument('--event-cooldown', type=int, default=8)
  parser.add_argument('--min-disappear-gap', type=int, default=16)
  parser.add_argument('--spurious-requires-sam2', action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument('--reward-event-threshold', type=float, default=0.5)
  args = parser.parse_args(argv)
  input_roots = args.input_root or [
      str(ROOT / 'artifacts/dreamer_all5_real_policy_h512_rollouts'),
      str(ROOT / 'artifacts/torch_selected_wms_real_policy_h512_rollouts'),
  ]
  args.input_root = parse_roots(input_roots)
  args.max_frames = int(args.max_frames) or None
  return args


def matches(value: str, patterns: str) -> bool:
  if not patterns:
    return True
  return any(re.search(pattern.strip(), value) for pattern in patterns.split(',') if pattern.strip())


def main(argv: list[str] | None = None):
  args = parse_args(argv)
  ensure_dir(args.output_dir)
  specs = [
      spec for spec in discover_rollouts(args.input_root)
      if matches(spec.model, args.models) and matches(spec.episode, args.episodes)
  ]
  if args.limit_rollouts:
    specs = specs[:int(args.limit_rollouts)]
  if not specs:
    raise FileNotFoundError('No rollout .npz files matched the requested filters.')

  write_json(args.output_dir / 'run_config.json', {
      key: str(value) if isinstance(value, pathlib.Path) else
      [str(x) for x in value] if isinstance(value, list) else value
      for key, value in vars(args).items()
  })

  rows = []
  for index, spec in enumerate(specs, 1):
    print(f'[{index}/{len(specs)}] {spec.source}/{spec.episode}/{spec.model}', flush=True)
    rows.append(analyze_rollout(spec, args))

  write_csv(args.output_dir / 'rollout_summaries.csv', rows)
  agg = aggregate(rows)
  write_csv(args.output_dir / 'aggregate_by_model.csv', agg)
  cmp_rows = manual_comparison(rows)
  if cmp_rows:
    write_csv(args.output_dir / 'manual_dreamer_comparison.csv', cmp_rows)
  print(f'Wrote SAM2 rollout failure analysis to {args.output_dir}', flush=True)


if __name__ == '__main__':
  main()
