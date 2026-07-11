#!/usr/bin/env python3
"""Create compact paper artifacts for the target WM rollout matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = pathlib.Path(__file__).resolve().parents[2]
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


def target_lookup(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
  out = {}
  for target in manifest['targets']:
    for cond in target['conditions']:
      out[cond['model']] = {
          'project': target['project'],
          'condition': cond['condition'],
          'weight': cond.get('weight', '0' if cond['condition'] == 'w0' else ''),
          'recommended_weight': target.get('recommended_weight', ''),
          'root': cond['root'],
          'model': cond['model'],
      }
  return out


def make_quant_rows(manifest: dict[str, Any], failure_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
  lookup = target_lookup(manifest)
  rows = []
  for failure in failure_rows:
    meta = lookup.get(failure['model'])
    if not meta:
      continue
    episodes = int(safe_float(failure.get('episodes'), 0))
    ball = int(safe_float(failure.get('ball_disappears'), 0))
    spurious = int(safe_float(failure.get('spurious_x_bounce'), 0))
    missed = int(safe_float(failure.get('missed_player_bounce'), 0))
    total = ball + spurious + missed
    rows.append({
        'project': meta['project'],
        'condition': meta['condition'],
        'condition_label': CONDITION_LABELS.get(meta['condition'], meta['condition']),
        'weight': meta['weight'],
        'model': meta['model'],
        'episodes': episodes,
        'ball_disappears': ball,
        'spurious_x_bounce': spurious,
        'missed_player_bounce': missed,
        'total_failures': total,
        'ball_disappears_per_ep': ball / episodes if episodes else float('nan'),
        'spurious_x_bounce_per_ep': spurious / episodes if episodes else float('nan'),
        'total_failures_per_ep': total / episodes if episodes else float('nan'),
        'track_present_rate': safe_float(failure.get('sam2_present_rate_mean')),
    })
  order = {name: idx for idx, name in enumerate(CONDITION_ORDER)}
  return sorted(rows, key=lambda r: (r['project'], order.get(r['condition'], 99)))


def format_cell(value: Any) -> str:
  if isinstance(value, float):
    if not math.isfinite(value):
      return ''
    return f'{value:.2f}'
  return str(value)


def write_markdown_table(path: pathlib.Path, rows: list[dict[str, Any]]):
  ensure_dir(path.parent)
  cols = [
      ('WM', 'project'),
      ('Ckpt', 'condition_label'),
      ('Weight', 'weight'),
      ('Ball disappear / ep ↓', 'ball_disappears_per_ep'),
      ('Spurious bounce / ep ↓', 'spurious_x_bounce_per_ep'),
      ('Total failure / ep ↓', 'total_failures_per_ep'),
      ('Track present ↑', 'track_present_rate'),
  ]
  lines = [
      '| ' + ' | '.join(name for name, _ in cols) + ' |',
      '| ' + ' | '.join('---' for _ in cols) + ' |',
  ]
  for row in rows:
    lines.append('| ' + ' | '.join(format_cell(row[key]) for _, key in cols) + ' |')
  path.write_text('\n'.join(lines) + '\n')


def load_rollout_frame(root: pathlib.Path, model: str, episode: str, t: int, size: int) -> Image.Image:
  path = root / episode / 'rollouts' / f'{model}.npz'
  with np.load(path, allow_pickle=False) as data:
    frames = np.asarray(data['frames'], np.uint8)
  idx = min(max(int(t), 0), len(frames) - 1)
  image = Image.fromarray(frames[idx])
  if image.size != (size, size):
    image = image.resize((size, size), resample=Image.Resampling.NEAREST)
  return image


def find_condition(manifest: dict[str, Any], project: str, condition: str) -> dict[str, Any]:
  for target in manifest['targets']:
    if target['project'] != project:
      continue
    for cond in target['conditions']:
      if cond['condition'] == condition:
        return cond
  raise KeyError((project, condition))


def make_panel(
    manifest: dict[str, Any],
    projects: list[str],
    episode: str,
    times: list[int],
    output: pathlib.Path,
    *,
    frame_size: int,
):
  label_h = 26
  project_w = 82
  cell_w = frame_size
  cell_h = frame_size
  header_h = label_h * 2
  width = project_w + len(CONDITION_ORDER) * len(times) * cell_w
  height = header_h + len(projects) * (cell_h + label_h)
  canvas = Image.new('RGB', (width, height), (255, 255, 255))
  draw = ImageDraw.Draw(canvas)
  font = ImageFont.load_default()

  x = project_w
  for cond in CONDITION_ORDER:
    span = len(times) * cell_w
    draw.text((x + 4, 6), CONDITION_LABELS[cond], fill=(0, 0, 0), font=font)
    for i, t in enumerate(times):
      draw.text((x + i * cell_w + 4, label_h + 4), f't={t}', fill=(60, 60, 60), font=font)
    x += span

  y = header_h
  for project in projects:
    draw.text((4, y + 6), project, fill=(0, 0, 0), font=font)
    x = project_w
    for cond_name in CONDITION_ORDER:
      cond = find_condition(manifest, project, cond_name)
      root = rel_or_abs(cond['root'])
      for t in times:
        frame = load_rollout_frame(root, cond['model'], episode, t, frame_size)
        canvas.paste(frame, (x, y + label_h))
        x += cell_w
    y += cell_h + label_h

  draw.rectangle((0, 0, width - 1, height - 1), outline=(180, 180, 180))
  ensure_dir(output.parent)
  canvas.save(output)


def main(argv: list[str] | None = None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--manifest', type=pathlib.Path,
                      default=ROOT / 'experiments/paper_wm_ckpt_matrix/target_rollouts.json')
  parser.add_argument('--failure-aggregate', type=pathlib.Path,
                      default=ROOT / 'eval_outputs/paper_wm_ckpt_matrix_failure_heuristic/aggregate_by_model.csv')
  parser.add_argument('--output-dir', type=pathlib.Path,
                      default=ROOT / 'results/paper_wm_ckpt_matrix')
  parser.add_argument('--panel-projects', default='dreamer,diamond,twister')
  parser.add_argument('--panel-episode', default='ep00_seed0')
  parser.add_argument('--panel-times', default='0,64,128,256,512')
  parser.add_argument('--frame-size', type=int, default=96)
  args = parser.parse_args(argv)

  manifest = read_json(args.manifest)
  failure_rows = read_csv(args.failure_aggregate)
  quant_rows = make_quant_rows(manifest, failure_rows)
  write_csv(args.output_dir / 'closed_loop_failure_quant.csv', quant_rows)
  write_markdown_table(args.output_dir / 'closed_loop_failure_quant.md', quant_rows)
  make_panel(
      manifest,
      [x.strip() for x in args.panel_projects.split(',') if x.strip()],
      args.panel_episode,
      [int(x) for x in args.panel_times.split(',') if x.strip()],
      args.output_dir / 'rollout_frame_panel.png',
      frame_size=int(args.frame_size),
  )
  print(f'Wrote paper rollout artifacts to {args.output_dir}')


if __name__ == '__main__':
  main()
