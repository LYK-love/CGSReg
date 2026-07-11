#!/usr/bin/env python3
"""Evaluate external Pong pixel WMs with the generated-video SAM2 ball metric."""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import subprocess
import sys
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROJECTS_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from scripts.eval import pong_wm_metrics as wm_metrics  # noqa: E402
from scripts.eval.pong_sam2_ball_experiment import (  # noqa: E402
    DEFAULT_EVAL_REPLAY,
    SegmentSpec,
    ensure_dir,
    make_prompt,
    real_tracks_from_masks,
    render_overlay,
    resize_video,
    run_sam2,
    save_frame_dir,
    save_video,
    summarize_tracks,
    tracks_from_masks,
    valid_ball_prompt,
    write_csv,
    write_json,
)


@dataclass(frozen=True)
class ExternalWMSpec:
  name: str
  project: str
  project_root: pathlib.Path
  python: pathlib.Path
  checkpoint: pathlib.Path
  initial_source: str = 'real'
  warmup_dataset: pathlib.Path | None = None
  bootstrap_dataset: pathlib.Path | None = None
  config_path: pathlib.Path | None = None
  wm_env_name: str = ''
  sample_mode: str = 'probs'


def env_path(name: str, fallback: str | pathlib.Path) -> pathlib.Path:
  return pathlib.Path(os.environ.get(name, fallback))


DEFAULT_SPECS = {
    'diamond': ExternalWMSpec(
        name='diamond_repro',
        project='diamond',
        project_root=env_path('DIAMOND_ROOT', PROJECTS_ROOT / 'diamond'),
        python=env_path('DIAMOND_PYTHON', sys.executable),
        checkpoint=env_path('DIAMOND_PONG_CKPT', PROJECTS_ROOT / 'diamond-assets/checkpoints/Pong.pt'),
        initial_source='dataset',
        warmup_dataset=env_path('DIAMOND_WARMUP_DATASET', PROJECTS_ROOT / 'diamond-assets/datasets/pong/test'),
    ),
    'simulus': ExternalWMSpec(
        name='simulus_repro',
        project='simulus',
        project_root=env_path('SIMULUS_ROOT', PROJECTS_ROOT / 'Simulus'),
        python=env_path('SIMULUS_PYTHON', sys.executable),
        checkpoint=env_path('SIMULUS_PONG_CKPT', PROJECTS_ROOT / 'Simulus/checkpoints/Pong.pt'),
        initial_source='real',
    ),
    'twister': ExternalWMSpec(
        name='twister_repro',
        project='twister',
        project_root=env_path('TWISTER_ROOT', PROJECTS_ROOT / 'twister'),
        python=env_path('TWISTER_PYTHON', sys.executable),
        checkpoint=env_path(
            'TWISTER_PONG_CKPT',
            PROJECTS_ROOT / 'twister/callbacks/atari100k/atari100k-pong/'
            'checkpoints_epoch_50_step_100000.ckpt'),
        initial_source='real',
    ),
    'storm': ExternalWMSpec(
        name='storm_repro',
        project='storm',
        project_root=env_path('STORM_ROOT', PROJECTS_ROOT / 'STORM'),
        python=env_path('STORM_PYTHON', sys.executable),
    checkpoint=env_path(
        'STORM_PONG_CKPT',
        PROJECTS_ROOT / 'oc-storm/runs/pong_atari100k_reproduction/'
        'logdir/Pong-STORM-base/ckpt/latest_agent.pth'),
    initial_source='real',
    config_path=ROOT / 'scripts/eval/configs/STORM_repro_hidden256.yaml',
    wm_env_name='PongNoFrameskip-v4',
  ),
}


def parse_horizons(value: str) -> list[int]:
  horizons = sorted({int(x) for x in value.split(',') if x.strip()})
  if not horizons or any(x <= 0 for x in horizons):
    raise ValueError(f'Invalid --horizons {value!r}')
  return horizons


def parse_project_override(value: str) -> tuple[str, dict[str, str]]:
  if ':' not in value:
    raise ValueError('Project overrides must look like project:key=value,key=value')
  project, rest = value.split(':', 1)
  updates = {}
  for part in rest.split(','):
    if not part:
      continue
    if '=' not in part:
      raise ValueError(f'Invalid override part {part!r}')
    key, raw = part.split('=', 1)
    updates[key] = raw
  return project, updates


def with_overrides(specs: dict[str, ExternalWMSpec], overrides: list[str]):
  out = dict(specs)
  for item in overrides:
    project, updates = parse_project_override(item)
    if project not in out:
      raise ValueError(f'Unknown project override {project!r}.')
    spec = out[project]
    kwargs: dict[str, Any] = {}
    for key, raw in updates.items():
      if key in {'project_root', 'python', 'checkpoint', 'warmup_dataset',
                 'bootstrap_dataset', 'config_path'}:
        kwargs[key] = pathlib.Path(raw)
      elif key in {'name', 'initial_source', 'wm_env_name', 'sample_mode'}:
        kwargs[key] = raw
      else:
        raise ValueError(f'Unknown override key {key!r}.')
    out[project] = replace(spec, **kwargs)
  return out


def load_window(args, max_horizon: int):
  chunks = wm_metrics.load_replay_chunks(args.eval_replay_dir, args.limit_chunks)
  chunk_id = int(args.chunk_id)
  start = int(args.start)
  if chunk_id < 0 or chunk_id >= len(chunks):
    raise ValueError(f'--chunk-id {chunk_id} outside [0, {len(chunks)})')
  chunk_path, chunk = chunks[chunk_id]
  if not wm_metrics.valid_window(chunk, start, args.context, max_horizon):
    raise ValueError(
        f'Invalid replay window: chunk_id={chunk_id}, start={start}, '
        f'context={args.context}, horizon={max_horizon}')
  frames = np.asarray(chunk['image'][start + 1:start + max_horizon + 1], np.uint8)
  masks = np.asarray(chunk['mask1'][start + 1:start + max_horizon + 1])
  actions = [int(x) for x in chunk['action'][start:start + max_horizon]]
  if valid_ball_prompt(masks[int(args.prompt_t)]) is None:
    raise ValueError(f'Prompt frame {args.prompt_t} has no real mask1 ball.')
  return chunk_path, chunk, frames, masks, actions


def component_centers(binary: np.ndarray):
  binary = np.asarray(binary, bool)
  seen = np.zeros(binary.shape, bool)
  height, width = binary.shape
  centers = []
  for y0, x0 in zip(*np.nonzero(binary & ~seen)):
    stack = [(int(y0), int(x0))]
    seen[y0, x0] = True
    xs = []
    ys = []
    while stack:
      y, x = stack.pop()
      xs.append(x)
      ys.append(y)
      for ny in (y - 1, y, y + 1):
        for nx in (x - 1, x, x + 1):
          if ny < 0 or nx < 0 or ny >= height or nx >= width:
            continue
          if seen[ny, nx] or not binary[ny, nx]:
            continue
          seen[ny, nx] = True
          stack.append((ny, nx))
    xs_arr = np.asarray(xs)
    ys_arr = np.asarray(ys)
    centers.append({
        'area': int(xs_arr.size),
        'x': float(xs_arr.mean()),
        'y': float(ys_arr.mean()),
        'xmin': int(xs_arr.min()),
        'xmax': int(xs_arr.max()),
        'ymin': int(ys_arr.min()),
        'ymax': int(ys_arr.max()),
    })
  return centers


def detect_pong_ball_center(frame: np.ndarray):
  frame = np.asarray(frame, np.uint8)
  gray = frame.astype(np.float32).mean(axis=-1)
  # Ignore score/top border and side paddles; the ball is a compact bright blob.
  candidates = gray > 80.0
  candidates[:12, :] = False
  candidates[57:, :] = False
  comps = []
  for comp in component_centers(candidates):
    width = comp['xmax'] - comp['xmin'] + 1
    height = comp['ymax'] - comp['ymin'] + 1
    patch = frame[comp['ymin']:comp['ymax'] + 1, comp['xmin']:comp['xmax'] + 1]
    patch_mask = candidates[comp['ymin']:comp['ymax'] + 1, comp['xmin']:comp['xmax'] + 1]
    mean_rgb = patch[patch_mask].mean(axis=0)
    green_bias = float(mean_rgb[1] - max(mean_rgb[0], mean_rgb[2]))
    if 2 <= comp['area'] <= 30 and 2 <= width <= 7 and 2 <= height <= 7:
      if 10 < comp['x'] < 54 and 12 < comp['y'] < 57 and green_bias < 70.0:
        comps.append(comp)
  if not comps:
    return None
  # Prefer the most ball-like small component away from paddle columns.
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


def make_external_prompt(frames64: np.ndarray, real_masks64: np.ndarray, args):
  if args.prompt_source == 'real':
    prompt = make_prompt(
        real_masks64[int(args.prompt_t)],
        frame_idx=int(args.prompt_t),
        source_size=real_masks64.shape[1],
        sam2_size=args.sam2_size)
    prompt['source'] = 'real_mask1'
    return prompt
  if args.prompt_source in {'wm-heuristic', 'auto'}:
    max_t = min(len(frames64), int(args.prompt_search_frames))
    for t in range(max_t):
      center = detect_pong_ball_center(frames64[t])
      if center is not None:
        return prompt_from_center(center, t, frames64.shape[1], args.sam2_size)
    if args.no_prompt_policy == 'real-fallback':
      prompt = make_prompt(
          real_masks64[int(args.prompt_t)],
          frame_idx=int(args.prompt_t),
          source_size=real_masks64.shape[1],
          sam2_size=args.sam2_size)
      prompt['source'] = 'real_mask1_fallback'
      return prompt
    if args.no_prompt_policy == 'zero':
      return None
  raise RuntimeError(
      'Could not produce a generated-frame ball prompt; retry with '
      '--prompt-source real or inspect the exported reset/video frames.')


def run_worker(spec: ExternalWMSpec, actions: list[int], args, max_horizon: int):
  rollout_path = args.output_dir / 'rollouts' / f'{spec.name}_h{max_horizon}.npz'
  if rollout_path.exists() and not args.force_rollout:
    print(f'Using existing rollout {rollout_path}', flush=True)
    return rollout_path

  cmd = [
      str(spec.python.expanduser().resolve()),
      str((ROOT / 'scripts/eval/external_wm_rollout_worker.py').resolve()),
      '--project', spec.project,
      '--project-root', str(spec.project_root.expanduser().resolve()),
      '--checkpoint', str(spec.checkpoint.expanduser().resolve()),
      '--output', str(rollout_path.resolve()),
      '--actions-json', json.dumps(actions),
      '--env-name', args.env_name,
      '--seed', str(args.seed),
      '--device', args.torch_device,
      '--horizon', str(max_horizon),
      '--wm-env-horizon', str(max_horizon + 1),
      '--wm-initial-source', spec.initial_source,
      '--reward-threshold', str(args.reward_threshold),
  ]
  if spec.warmup_dataset:
    cmd.extend(['--warmup-dataset', str(spec.warmup_dataset.expanduser().resolve())])
  if spec.bootstrap_dataset:
    cmd.extend(['--wm-bootstrap-dataset', str(spec.bootstrap_dataset.expanduser().resolve())])
  if spec.config_path:
    cmd.extend(['--config-path', str(spec.config_path.expanduser().resolve())])
  if spec.wm_env_name:
    cmd.extend(['--wm-env-name', spec.wm_env_name])
  if spec.sample_mode:
    cmd.extend(['--sample-mode', spec.sample_mode])
  if args.respect_terminal:
    cmd.append('--respect-terminal')

  ensure_dir(rollout_path.parent)
  log_path = args.output_dir / 'logs' / f'{spec.name}_rollout.log'
  ensure_dir(log_path.parent)
  env = os.environ.copy()
  env['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)
  if args.pytorch_cuda_alloc_conf:
    env['PYTORCH_CUDA_ALLOC_CONF'] = args.pytorch_cuda_alloc_conf
  else:
    env.pop('PYTORCH_CUDA_ALLOC_CONF', None)
  env.setdefault('OMP_NUM_THREADS', '1')
  print(f'Rolling out {spec.name} on CUDA_VISIBLE_DEVICES={args.cuda_device}', flush=True)
  with log_path.open('w') as log:
    proc = subprocess.run(
        cmd,
        cwd=str(spec.project_root.expanduser().resolve()),
        env=env,
        text=True,
        stdout=log,
        stderr=subprocess.STDOUT,
        check=False,
    )
  if proc.returncode != 0:
    raise RuntimeError(f'{spec.name} rollout failed with code {proc.returncode}; see {log_path}')
  return rollout_path


def load_rollout(path: pathlib.Path):
  with np.load(path, allow_pickle=False) as data:
    meta = json.loads(str(data['metadata']))
    return {
        'frames': np.asarray(data['frames'], np.uint8),
        'reset_frame': np.asarray(data['reset_frame'], np.uint8),
        'rewards': np.asarray(data['rewards'], np.float32),
        'dones': np.asarray(data['dones'], bool),
        'truncs': np.asarray(data['truncs'], bool),
        'actions': np.asarray(data['actions'], np.int64),
        'metadata': meta,
    }


def write_aggregate_csv(path: pathlib.Path, rows: list[dict[str, Any]]):
  ensure_dir(path.parent)
  keys = [
      'horizon', 'name', 'project', 'checkpoint', 'prompt_source',
      'guiding_source', 'wm_initial_source', 'start_aligned',
      'sam2_present_frames', 'real_present_frames', 'present_rate',
      'missing_rate', 'first_present_t', 'last_present_t',
      'ball_l2_rmse', 'ball_l2_mean', 'ball_vx_sign_accuracy',
      'ball_vy_sign_accuracy', 'output_dir',
  ]
  extra = sorted({key for row in rows for key in row if key not in keys})
  with path.open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[*keys, *extra])
    writer.writeheader()
    writer.writerows(rows)


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--project', action='append', choices=sorted(DEFAULT_SPECS),
                      help='Project(s) to evaluate. Defaults to all four.')
  parser.add_argument('--override', action='append', default=[],
                      help='Override a spec: project:key=value,key=value')
  parser.add_argument('--eval-replay-dir', type=pathlib.Path, default=DEFAULT_EVAL_REPLAY)
  parser.add_argument('--output-dir', type=pathlib.Path,
                      default=pathlib.Path('eval_outputs/pong_external_wm_sam2_ball_repro'))
  parser.add_argument('--env-name', default='PongNoFrameskip-v4')
  parser.add_argument('--context', type=int, default=5)
  parser.add_argument('--horizons', default='48')
  parser.add_argument('--chunk-id', type=int, default=0)
  parser.add_argument('--start', type=int, default=8)
  parser.add_argument('--prompt-t', type=int, default=0)
  parser.add_argument('--prompt-source', choices=['wm-heuristic', 'real', 'auto'],
                      default='auto')
  parser.add_argument('--no-prompt-policy', choices=['zero', 'error', 'real-fallback'],
                      default='zero')
  parser.add_argument('--prompt-search-frames', type=int, default=12)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--limit-chunks', type=int, default=None)
  parser.add_argument('--fps', type=int, default=15)
  parser.add_argument('--sam2-size', type=int, default=256)
  parser.add_argument('--sam2-client-root', type=pathlib.Path,
                      default=PROJECTS_ROOT / 'NovelWorldModel/sam2_client')
  parser.add_argument('--backend-endpoint', default='http://209.137.198.192:7263')
  parser.add_argument('--guiding-source', choices=['real', 'wm'], default='wm')
  parser.add_argument('--max-frames-per-chunk', type=int, default=256)
  parser.add_argument('--divergence-px', type=float, default=8.0)
  parser.add_argument('--cuda-device', default='3')
  parser.add_argument('--torch-device', default='cuda')
  parser.add_argument('--reward-threshold', type=float, default=0.5)
  parser.add_argument('--pytorch-cuda-alloc-conf', default='')
  parser.add_argument('--respect-terminal', action='store_true')
  parser.add_argument('--force-rollout', action='store_true')
  parser.add_argument('--skip-sam2', action='store_true')
  parser.add_argument('--verbose', action='store_true')
  args = parser.parse_args(argv)

  specs = with_overrides(DEFAULT_SPECS, args.override)
  projects = args.project or sorted(specs)
  horizons = parse_horizons(args.horizons)
  max_horizon = max(horizons)
  ensure_dir(args.output_dir)

  chunk_path, _chunk, real_frames64, real_masks64, actions = load_window(args, max_horizon)
  real_tracks_by_horizon = {
      h: real_tracks_from_masks(real_masks64[:h], sample_id='window0')
      for h in horizons
  }
  real_video = resize_video(real_frames64, args.sam2_size)
  save_video(args.output_dir / 'real_target_hmax.mp4', real_video, args.fps)
  save_frame_dir(args.output_dir / 'frames' / 'real_target_hmax', real_video)

  manifest = {
      'eval_replay_dir': str(args.eval_replay_dir),
      'chunk': str(chunk_path),
      'chunk_id': int(args.chunk_id),
      'start': int(args.start),
      'context': int(args.context),
      'horizons': horizons,
      'prompt_t': int(args.prompt_t),
      'prompt_source': args.prompt_source,
      'guiding_source': args.guiding_source,
      'actions': actions,
      'external_start_aligned_with_replay': False,
      'alignment_note': (
          'External project play APIs are reset through their own real/dataset/prior '
          'initialization, not through the exact Dreamer replay hidden context. '
          'Coverage/detectability is therefore the main cross-project signal; '
          'ball_l2_rmse is diagnostic unless a project-specific exact context injector is used.'
      ),
      'projects': {
          key: {
              'name': specs[key].name,
              'project_root': str(specs[key].project_root),
              'python': str(specs[key].python),
              'checkpoint': str(specs[key].checkpoint),
              'initial_source': specs[key].initial_source,
          }
          for key in projects
      },
  }
  write_json(args.output_dir / 'manifest.json', manifest)

  aggregate_rows = []
  by_horizon_rows: dict[int, dict[str, list[dict[str, Any]]]] = {
      h: {} for h in horizons
  }
  for project in projects:
    spec = specs[project]
    rollout_path = run_worker(spec, actions, args, max_horizon)
    rollout = load_rollout(rollout_path)
    frames64 = rollout['frames']
    save_frame_dir(args.output_dir / 'frames' / spec.name, resize_video(frames64, args.sam2_size))
    save_video(args.output_dir / 'videos' / f'{spec.name}.mp4',
               resize_video(frames64, args.sam2_size), args.fps)
    write_json(args.output_dir / 'rollouts' / f'{spec.name}.json', {
        'name': spec.name,
        'project': spec.project,
        'checkpoint': str(spec.checkpoint),
        'metadata': rollout['metadata'],
        'actions': rollout['actions'].tolist(),
        'rewards': rollout['rewards'].tolist(),
        'dones': rollout['dones'].astype(bool).tolist(),
        'truncs': rollout['truncs'].astype(bool).tolist(),
    })

    if args.skip_sam2:
      continue

    for horizon in horizons:
      horizon_dir = args.output_dir / f'h{horizon}' / spec.name
      frames_h = frames64[:horizon]
      args.real_frames64 = real_frames64[:horizon]
      prompt = make_external_prompt(frames_h, real_masks64[:horizon], args)
      write_json(horizon_dir / 'prompt.json', [] if prompt is None else [prompt])
      save_video(horizon_dir / 'video.mp4', resize_video(frames_h, args.sam2_size), args.fps)
      save_frame_dir(horizon_dir / 'frames', resize_video(frames_h, args.sam2_size))
      if prompt is None:
        print(f'No generated ball prompt: {spec.name} h={horizon}; writing zero mask.', flush=True)
        mask = np.zeros((horizon, args.sam2_size, args.sam2_size), dtype=np.uint8)
      else:
        sam2_args = argparse.Namespace(**vars(args))
        sam2_args.real_frames64 = real_frames64[:horizon]
        seg_spec = SegmentSpec(spec.name, frames_h, horizon_dir / 'video.mp4', horizon_dir / 'frames')
        print(f'Running SAM2: {spec.name} h={horizon}', flush=True)
        mask = run_sam2(seg_spec, prompt, sam2_args)
      np.savez_compressed(horizon_dir / 'sam2_ball.npz', mask1=mask)
      wm_rows = tracks_from_masks(
          mask, downscale=args.sam2_size / 64.0, sample_id='window0')
      by_horizon_rows[horizon][spec.name] = wm_rows
      write_csv(horizon_dir / 'sam2_tracks.csv', wm_rows)
      overlay = render_overlay(resize_video(frames_h, args.sam2_size), mask)
      save_video(horizon_dir / 'sam2_ball_overlay.mp4', overlay, args.fps)

      summary = summarize_tracks(
          real_tracks_by_horizon[horizon], {spec.name: wm_rows}, args.divergence_px)
      metrics = dict(summary[spec.name])
      metrics.update({
          'horizon': int(horizon),
          'name': spec.name,
          'project': spec.project,
          'checkpoint': str(spec.checkpoint),
          'prompt_source': 'none_generated_prompt_not_found' if prompt is None else prompt.get('source', args.prompt_source),
          'guiding_source': args.guiding_source,
          'wm_initial_source': spec.initial_source,
          'start_aligned': False,
          'output_dir': str(horizon_dir),
      })
      present = float(metrics.get('sam2_present_frames', 0))
      metrics['present_rate'] = present / float(horizon)
      present_ts = [int(row['t']) for row in wm_rows if row.get('ball_present')]
      metrics['first_present_t'] = min(present_ts) if present_ts else None
      metrics['last_present_t'] = max(present_ts) if present_ts else None
      aggregate_rows.append(metrics)
      write_json(horizon_dir / 'sam2_ball_summary.json', metrics)

  if args.skip_sam2:
    print(f'Generated external WM videos under {args.output_dir}; skipped SAM2.')
    return

  write_aggregate_csv(args.output_dir / 'external_sam2_ball_summary.csv', aggregate_rows)
  write_json(args.output_dir / 'external_sam2_ball_summary.json', aggregate_rows)
  for horizon, rows in by_horizon_rows.items():
    if rows:
      write_csv(args.output_dir / f'h{horizon}' / 'real_mask_tracks.csv',
                real_tracks_by_horizon[horizon])
      write_json(
          args.output_dir / f'h{horizon}' / 'sam2_ball_summary.json',
          summarize_tracks(real_tracks_by_horizon[horizon], rows, args.divergence_px))
  print(f'Wrote external WM SAM2 ball outputs to {args.output_dir}')


if __name__ == '__main__':
  main()
