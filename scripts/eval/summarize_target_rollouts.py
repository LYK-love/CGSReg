#!/usr/bin/env python3
"""Summarize the paper target WM checkpoint rollout matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[2]


def ensure_dir(path: pathlib.Path):
  path.mkdir(parents=True, exist_ok=True)


def read_json(path: pathlib.Path) -> Any:
  return json.loads(path.read_text())


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


def rel_or_abs(path: str) -> pathlib.Path:
  value = pathlib.Path(path).expanduser()
  return value if value.is_absolute() else ROOT / value


def safe_float(value: Any, default: float = float('nan')) -> float:
  try:
    return float(value)
  except (TypeError, ValueError):
    return default


def load_rollout(path: pathlib.Path) -> dict[str, Any]:
  with np.load(path, allow_pickle=False) as data:
    rewards = np.asarray(data.get('rewards', []), np.float32)
    actions = np.asarray(data.get('actions', []), np.int64)
    done = np.asarray(data.get('done', []), bool)
    trunc = np.asarray(data.get('trunc', []), bool)
    frames = np.asarray(data.get('frames', []))
  return {
      'return': float(np.nansum(rewards)),
      'reward_events': int(np.count_nonzero(np.abs(rewards) >= 0.5)),
      'positive_rewards': int(np.count_nonzero(rewards > 0.5)),
      'negative_rewards': int(np.count_nonzero(rewards < -0.5)),
      'actions': int(actions.shape[0]),
      'frames': int(frames.shape[0]) if hasattr(frames, 'shape') and frames.ndim else 0,
      'done_any': bool(done.any()) if done.size else False,
      'trunc_any': bool(trunc.any()) if trunc.size else False,
  }


def discover_condition_rollouts(root: pathlib.Path, model: str) -> list[pathlib.Path]:
  return sorted(root.glob(f'ep*_seed*/rollouts/{model}.npz'))


def summarize_values(values: list[float]) -> dict[str, Any]:
  finite = [float(x) for x in values if math.isfinite(float(x))]
  if not finite:
    return {
        'episodes': 0,
        'return_mean': float('nan'),
        'return_std': float('nan'),
        'return_min': float('nan'),
        'return_max': float('nan'),
        'return_values': '',
    }
  return {
      'episodes': len(finite),
      'return_mean': float(np.mean(finite)),
      'return_std': float(np.std(finite, ddof=0)),
      'return_min': float(np.min(finite)),
      'return_max': float(np.max(finite)),
      'return_values': ' '.join(str(int(x)) if float(x).is_integer() else f'{x:.3g}' for x in finite),
  }


def make_rows(manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
  episode_rows: list[dict[str, Any]] = []
  summary_rows: list[dict[str, Any]] = []
  for target in manifest['targets']:
    project = target['project']
    recommended_weight = target.get('recommended_weight', '')
    for cond in target['conditions']:
      root = rel_or_abs(cond['root'])
      model = cond['model']
      paths = discover_condition_rollouts(root, model) if root.exists() else []
      returns = []
      status = cond.get('status', 'available')
      for path in paths:
        rec = load_rollout(path)
        returns.append(float(rec['return']))
        episode_rows.append({
            'project': project,
            'condition': cond['condition'],
            'weight': cond.get('weight', '0' if cond['condition'] == 'w0' else ''),
            'recommended_weight': recommended_weight,
            'model': model,
            'root': str(root.relative_to(ROOT) if root.is_relative_to(ROOT) else root),
            'episode': path.parent.parent.name,
            'path': str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
            **rec,
        })
      summary = {
          'project': project,
          'condition': cond['condition'],
          'weight': cond.get('weight', '0' if cond['condition'] == 'w0' else ''),
          'recommended_weight': recommended_weight,
          'model': model,
          'root': str(root.relative_to(ROOT) if root.exists() and root.is_relative_to(ROOT) else root),
          'status': 'available' if paths else status,
          'expected_episodes': int(manifest.get('protocol', {}).get('episodes', 5)),
          'missing_episodes': max(0, int(manifest.get('protocol', {}).get('episodes', 5)) - len(paths)),
          'notes': target.get('notes', ''),
      }
      summary.update(summarize_values(returns))
      summary_rows.append(summary)
  return episode_rows, summary_rows


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
  with path.open(newline='') as f:
    return list(csv.DictReader(f))


def join_failure_metrics(
    summary_rows: list[dict[str, Any]],
    failure_csv: pathlib.Path | None,
) -> list[dict[str, Any]]:
  if not failure_csv or not failure_csv.exists():
    return summary_rows
  by_model = {row['model']: row for row in read_csv(failure_csv)}
  metric_names = (
      'sam2_present_rate_mean',
      'ball_disappears',
      'missed_player_bounce',
      'spurious_x_bounce',
      'expected_paddle_collisions',
      'successful_paddle_collisions',
      'failed_expected_paddle_collisions',
      'failed_expected_collision_rate',
      'left_expected_paddle_collisions',
      'left_failed_expected_paddle_collisions',
      'left_failed_expected_collision_rate',
      'right_expected_paddle_collisions',
      'right_failed_expected_paddle_collisions',
      'right_failed_expected_collision_rate',
  )
  out = []
  for row in summary_rows:
    joined = dict(row)
    failure = by_model.get(str(row['model']))
    for name in metric_names:
      joined[f'failure_{name}'] = failure.get(name, '') if failure else ''
    out.append(joined)
  return out


def main(argv: list[str] | None = None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--manifest', type=pathlib.Path,
                      default=ROOT / 'experiments/paper_wm_ckpt_matrix/target_rollouts.json')
  parser.add_argument('--output-dir', type=pathlib.Path,
                      default=ROOT / 'results/paper_wm_ckpt_matrix')
  parser.add_argument('--failure-aggregate', type=pathlib.Path, default=None,
                      help='Optional aggregate_by_model.csv from pong_sam2_rollout_failure_events.py.')
  args = parser.parse_args(argv)

  manifest = read_json(args.manifest)
  episode_rows, summary_rows = make_rows(manifest)
  joined_rows = join_failure_metrics(summary_rows, args.failure_aggregate)
  ensure_dir(args.output_dir)
  write_csv(args.output_dir / 'target_rollout_episodes.csv', episode_rows)
  write_csv(args.output_dir / 'target_rollout_summary.csv', summary_rows)
  write_csv(args.output_dir / 'target_rollout_failure_summary.csv', joined_rows)
  write_json(args.output_dir / 'target_rollout_manifest.json', manifest)
  print(f'Wrote {args.output_dir}')


if __name__ == '__main__':
  main()
