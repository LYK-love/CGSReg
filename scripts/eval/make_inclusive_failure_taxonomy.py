#!/usr/bin/env python3
"""Build an inclusive closed-loop failure taxonomy table for Pong WM rollouts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/eval'))

from pong_sam2_rollout_failure_events import (  # noqa: E402
    detect_paddle,
    detect_pong_ball_center,
    resize_video,
)


CONDITION_LABELS = {
    'exp_repro': 'Exp repro',
    'w0': 'w=0',
    'w_recommended': 'CGSReg',
}
CONDITION_ORDER = ('exp_repro', 'w0', 'w_recommended')


def ensure_dir(path: pathlib.Path):
  path.mkdir(parents=True, exist_ok=True)


def read_json(path: pathlib.Path) -> Any:
  return json.loads(path.read_text())


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
  with path.open(newline='') as f:
    return list(csv.DictReader(f))


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]):
  ensure_dir(path.parent)
  keys = sorted({key for row in rows for key in row})
  with path.open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)


def rel_or_abs(path: str) -> pathlib.Path:
  value = pathlib.Path(path).expanduser()
  return value if value.is_absolute() else ROOT / value


def safe_float(value: Any, default: float = float('nan')) -> float:
  try:
    return float(value)
  except (TypeError, ValueError):
    return default


def visible_gap_count(present: np.ndarray, *, min_len: int = 1) -> int:
  count = 0
  t = 0
  while t < len(present):
    if present[t]:
      t += 1
      continue
    start = t
    while t < len(present) and not present[t]:
      t += 1
    end = t
    if start > 0 and end < len(present) and present[start - 1] and present[end]:
      if end - start >= min_len:
        count += 1
  return count


def center(box) -> float:
  if not box.present:
    return float('nan')
  return 0.5 * (float(box.ymin) + float(box.ymax))


def action_response_score(actions: np.ndarray, right_y: np.ndarray, present: np.ndarray) -> float:
  """Heuristic controllability proxy for the right paddle.

  Atari Pong action ids 2 and 5 dominate these policies. Instead of assuming
  their semantic names, measure whether the right paddle has clearly different
  vertical motion under the two action ids.
  """
  dy = right_y[1:] - right_y[:-1]
  valid = present[1:] & present[:-1] & np.isfinite(dy)
  if len(actions) > len(valid):
    actions = actions[:len(valid)]
  elif len(actions) < len(valid):
    valid = valid[:len(actions)]
    dy = dy[:len(actions)]
  vals = []
  for action in (2, 5):
    mask = valid & (actions == action)
    if np.count_nonzero(mask) < 10:
      return float('nan')
    vals.append(float(np.nanmean(dy[mask])))
  return abs(vals[1] - vals[0])


def load_rollout(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
  with np.load(path, allow_pickle=False) as data:
    frames = np.asarray(data['frames'], np.uint8)
    actions = np.asarray(data.get('actions', []), np.int64)
  return frames, actions


def rollout_stats(path: pathlib.Path) -> dict[str, float]:
  frames_raw, actions = load_rollout(path)
  frames64 = resize_video(frames_raw, 64)
  ball_present = []
  left_present = []
  right_present = []
  right_y = []
  for frame in frames64:
    ball_present.append(detect_pong_ball_center(frame) is not None)
    left = detect_paddle(frame, 'left')
    right = detect_paddle(frame, 'right')
    left_present.append(left.present)
    right_present.append(right.present)
    right_y.append(center(right))
  ball_present_arr = np.asarray(ball_present, bool)
  left_present_arr = np.asarray(left_present, bool)
  right_present_arr = np.asarray(right_present, bool)
  right_y_arr = np.asarray(right_y, np.float32)
  return {
      'ball_track_present_rate': float(np.mean(ball_present_arr)) if len(ball_present_arr) else float('nan'),
      'ball_flicker_gaps': float(visible_gap_count(ball_present_arr, min_len=1)),
      'left_paddle_present_rate': float(np.mean(left_present_arr)) if len(left_present_arr) else float('nan'),
      'right_paddle_present_rate': float(np.mean(right_present_arr)) if len(right_present_arr) else float('nan'),
      'left_paddle_gaps': float(visible_gap_count(left_present_arr, min_len=1)),
      'right_paddle_gaps': float(visible_gap_count(right_present_arr, min_len=1)),
      'paddle_action_response': action_response_score(actions, right_y_arr, right_present_arr),
  }


def target_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
  rows = []
  for target in manifest['targets']:
    for cond in target['conditions']:
      rows.append({
          'project': target['project'],
          'condition': cond['condition'],
          'condition_label': CONDITION_LABELS.get(cond['condition'], cond['condition']),
          'weight': cond.get('weight', '0' if cond['condition'] == 'w0' else ''),
          'recommended_weight': target.get('recommended_weight', ''),
          'root': cond['root'],
          'model': cond['model'],
      })
  order = {name: idx for idx, name in enumerate(CONDITION_ORDER)}
  return sorted(rows, key=lambda r: (r['project'], order.get(r['condition'], 99)))


def mean(values: list[float]) -> float:
  finite = [float(x) for x in values if math.isfinite(float(x))]
  return float(np.mean(finite)) if finite else float('nan')


def dominant_modes(row: dict[str, Any]) -> str:
  modes = []
  if safe_float(row['ball_flicker_gaps_per_ep']) >= 2.0:
    modes.append('ball flicker/missing')
  if safe_float(row['spurious_x_bounce_per_ep']) >= 1.0:
    modes.append('spurious ball bounce')
  if min(safe_float(row['left_paddle_present_rate']), safe_float(row['right_paddle_present_rate'])) < 0.75:
    modes.append('paddle disappearance')
  if safe_float(row['paddle_action_response']) < 0.10:
    modes.append('weak paddle action response')
  if not modes:
    modes.append('none detected by heuristics')
  return '; '.join(modes)


def make_rows(manifest: dict[str, Any], failure_csv: pathlib.Path | None) -> list[dict[str, Any]]:
  failure_by_model = {}
  if failure_csv and failure_csv.exists():
    failure_by_model = {row['model']: row for row in read_csv(failure_csv)}
  rows = []
  for meta in target_rows(manifest):
    root = rel_or_abs(meta['root'])
    paths = sorted(root.glob(f'ep*_seed*/rollouts/{meta["model"]}.npz'))
    per_episode = [rollout_stats(path) for path in paths]
    failure = failure_by_model.get(meta['model'], {})
    episodes = len(paths)
    row = {
        **meta,
        'episodes': episodes,
        'ball_track_present_rate': mean([r['ball_track_present_rate'] for r in per_episode]),
        'ball_flicker_gaps_per_ep': mean([r['ball_flicker_gaps'] for r in per_episode]),
        'long_ball_disappears_per_ep': safe_float(failure.get('ball_disappears'), 0.0) / episodes if episodes else float('nan'),
        'spurious_x_bounce_per_ep': safe_float(failure.get('spurious_x_bounce'), 0.0) / episodes if episodes else float('nan'),
        'left_paddle_present_rate': mean([r['left_paddle_present_rate'] for r in per_episode]),
        'right_paddle_present_rate': mean([r['right_paddle_present_rate'] for r in per_episode]),
        'paddle_gaps_per_ep': mean([r['left_paddle_gaps'] + r['right_paddle_gaps'] for r in per_episode]),
        'paddle_action_response': mean([r['paddle_action_response'] for r in per_episode]),
    }
    row['dominant_detected_failure_modes'] = dominant_modes(row)
    rows.append(row)
  return rows


def format_cell(value: Any) -> str:
  if isinstance(value, float):
    if not math.isfinite(value):
      return ''
    return f'{value:.2f}'
  return str(value)


def write_markdown(path: pathlib.Path, rows: list[dict[str, Any]]):
  ensure_dir(path.parent)
  cols = [
      ('WM', 'project'),
      ('Ckpt', 'condition_label'),
      ('Ball flicker / ep ↓', 'ball_flicker_gaps_per_ep'),
      ('Spurious bounce / ep ↓', 'spurious_x_bounce_per_ep'),
      ('Left paddle present ↑', 'left_paddle_present_rate'),
      ('Right paddle present ↑', 'right_paddle_present_rate'),
      ('Paddle gaps / ep ↓', 'paddle_gaps_per_ep'),
      ('Action response ↑', 'paddle_action_response'),
      ('Detected modes', 'dominant_detected_failure_modes'),
  ]
  lines = [
      '| ' + ' | '.join(name for name, _ in cols) + ' |',
      '| ' + ' | '.join('---' for _ in cols) + ' |',
  ]
  for row in rows:
    lines.append('| ' + ' | '.join(format_cell(row[key]) for _, key in cols) + ' |')
  path.write_text('\n'.join(lines) + '\n')


def main(argv: list[str] | None = None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--manifest', type=pathlib.Path,
                      default=ROOT / 'experiments/paper_wm_ckpt_matrix/target_rollouts.json')
  parser.add_argument('--failure-aggregate', type=pathlib.Path,
                      default=ROOT / 'eval_outputs/paper_wm_ckpt_matrix_failure_heuristic/aggregate_by_model.csv')
  parser.add_argument('--output-dir', type=pathlib.Path,
                      default=ROOT / 'results/paper_wm_ckpt_matrix')
  args = parser.parse_args(argv)
  rows = make_rows(read_json(args.manifest), args.failure_aggregate)
  write_csv(args.output_dir / 'inclusive_failure_taxonomy.csv', rows)
  write_markdown(args.output_dir / 'inclusive_failure_taxonomy.md', rows)
  print(f'Wrote inclusive failure taxonomy to {args.output_dir}')


if __name__ == '__main__':
  main()
