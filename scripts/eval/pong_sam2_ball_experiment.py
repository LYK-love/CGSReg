#!/usr/bin/env python3
"""Generate short-horizon Dreamer WM Pong videos and segment the ball with SAM2.

The experiment uses one real replay window as shared context/actions, rolls out
multiple frozen Dreamer WMs from that context, and sends each generated video to
a SAM2 backend with the same real-frame ball prompt.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import sys
from dataclasses import dataclass
from typing import Any

import imageio.v3 as iio
import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROJECTS_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from scripts.eval import pong_wm_metrics as wm_metrics  # noqa: E402
from rollout_pong_helpers import build_agent_and_load, find_config, update_config_allow_new


DEFAULT_RUNS_ROOT = pathlib.Path(os.environ.get(
    'DREAMERV3_RUNS_ROOT', PROJECTS_ROOT / 'dreamerv3-runs'))
DEFAULT_EVAL_REPLAY = (
    DEFAULT_RUNS_ROOT / 'datasets/pong_wm_reg_sweep/eval_replay')
DEFAULT_WMS = [
    (
        'size200m_repro',
        'pong_atari100k_reproduction/logdir/size200m/ckpt/latest',
    ),
    (
        'size400m_w0',
        'pong_wm_reg_sweep/logdir/'
        'pong_wm_reg_size400m_mask1_spatial_0_temporal_1/ckpt/latest',
    ),
    (
        'size400m_w0p01',
        'pong_wm_reg_sweep/logdir/'
        'pong_wm_reg_size400m_mask1_spatial_0p01_temporal_1/ckpt/latest',
    ),
]


@dataclass(frozen=True)
class SegmentSpec:
  name: str
  frames64: np.ndarray
  video_path: pathlib.Path
  frame_dir: pathlib.Path


def parse_name_path(value: str) -> tuple[str, pathlib.Path]:
  return wm_metrics.parse_name_path(value)


def parse_policy_path(value: str) -> tuple[str, pathlib.Path]:
  if '=' not in value:
    raise ValueError(f'Expected name=/path, got {value!r}')
  name, path = value.split('=', 1)
  return name, pathlib.Path(path).expanduser().resolve()


def _load_policy_config(policy_ckpt: pathlib.Path, task: str, jax_platform: str):
  import elements
  from dreamerv3.interactive import dreamer_adapter

  try:
    config_path = find_config(policy_ckpt)
    config = dreamer_adapter.load_config(config_path, jax_platform)
  except FileNotFoundError:
    # Converted standalone policy npz files can live outside the original run
    # tree. They use the standard Pong pixel policy structure, so default config
    # is enough to build the agent before loading converted weights.
    config_path = None
    config = wm_metrics.load_default_config(task, jax_platform)
  if bool(getattr(config.agent, 'pure_wm', False)):
    config = config.update(agent=config.agent.update(pure_wm=False))
  config = config.update(task=task)
  # Keep the policy env long enough so closed-loop rollouts do not hit unexpected
  # replay-size limits from the env wrapper.
  if task.startswith(('atari_', 'atari100k_')):
    repeat = int(config.env.atari100k.repeat)
    length = 27000 * repeat
    config = update_config_allow_new(elements, config, **{'env.atari100k.length': length})
  return config_path, config


def _load_policy_agent(policy_ckpt: pathlib.Path, task: str, jax_platform: str):
  import elements
  from dreamerv3 import main as dreamer_main

  config_path, config = _load_policy_config(policy_ckpt, task, jax_platform)
  agent = build_agent_and_load(dreamer_main, elements, config, policy_ckpt)
  return config_path, config, agent


def _policy_obs(agent, obs):
  keys = set(agent.obs_space.keys())
  formatted = {}
  for key in keys:
    if key not in obs:
      continue
    value = np.asarray(obs[key])
    if value.ndim <= 3:
      value = np.expand_dims(value, axis=0)
    formatted[key] = value
  return formatted


def _policy_step(agent, carry, obs):
  carry, action, _ = agent.policy(carry, _policy_obs(agent, obs), mode='eval')
  action_value = np.asarray(action['action']).reshape(-1)[0]
  return carry, int(action_value)


def _set_prev_action(policy_state, action: int):
  if not isinstance(policy_state, tuple):
    return policy_state
  target = np.asarray(action).reshape(1)
  return (*policy_state[:-1], target)


def _bootstrap_policy_from_replay(agent, chunk, start: int, context: int):
  first = int(start - context)
  policy_state = agent.init_policy(1)
  obs = wm_metrics.obs_at(chunk, first)
  for t in range(first, start):
    policy_state, _ = _policy_step(agent, policy_state, obs)
    policy_state = _set_prev_action(policy_state, int(chunk['action'][t]))
    obs = wm_metrics.obs_at(chunk, t + 1)

  policy_state, first_action = _policy_step(agent, policy_state, obs)
  return policy_state, first_action


def rollout_with_policy(agent, runner, chunk, start: int, context: int, horizon: int):
  state = wm_metrics.bootstrap_state_from_replay(runner, chunk, start, context)
  policy_state, action = _bootstrap_policy_from_replay(agent, chunk, start, context)
  frames = []
  pred_rewards = []
  pred_mode_rewards = []
  pred_cont = []
  pred_actions = []

  target_rewards = np.asarray(chunk['reward'][start + 1:start + horizon + 1], np.float32)
  for _ in range(int(horizon)):
    pred_actions.append(int(action))
    state, result = runner.imagine_step(state, action)
    pred_rewards.append(float(result.info.get('reward_expected', result.reward)))
    pred_mode_rewards.append(float(result.info.get('reward_mode', result.reward)))
    pred_cont.append(float(result.info.get('cont_prob', np.nan)))
    if 'image' in result.obs:
      frames.append(np.asarray(result.obs['image']))
    else:
      frames.append(np.zeros((64, 64, 3), dtype=np.uint8))
    if bool(result.trunc) or bool(result.done):
      if len(pred_actions) >= int(horizon):
        break
    policy_state, action = _policy_step(agent, policy_state, result.obs)
  if len(pred_actions) < int(horizon):
    pred_actions.extend([0] * (int(horizon) - len(pred_actions)))
    pred_rewards.extend([0.0] * (int(horizon) - len(pred_rewards)))
    pred_mode_rewards.extend([0.0] * (int(horizon) - len(pred_mode_rewards)))
    pred_cont.extend([np.nan] * (int(horizon) - len(pred_cont)))

  return {
      'pred_frames': np.asarray(frames, np.uint8),
      'pred_reward': np.asarray(pred_rewards, np.float32),
      'pred_reward_mode': np.asarray(pred_mode_rewards, np.float32),
      'pred_cont_prob': np.asarray(pred_cont, np.float32),
      'target_reward': target_rewards,
      'policy_actions': np.asarray(pred_actions, np.int32),
  }


def ensure_dir(path: pathlib.Path):
  path.mkdir(parents=True, exist_ok=True)


def write_json(path: pathlib.Path, data: Any):
  ensure_dir(path.parent)
  path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]):
  ensure_dir(path.parent)
  keys = sorted({key for row in rows for key in row})
  with path.open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)


def default_wm_specs(runs_root: pathlib.Path):
  return [
      wm_metrics.WMSpec(name, (runs_root / rel).expanduser().resolve())
      for name, rel in DEFAULT_WMS
  ]


def resize_video(frames: np.ndarray, size: int) -> np.ndarray:
  frames = np.asarray(frames)
  if frames.ndim != 4 or frames.shape[-1] != 3:
    raise ValueError(f'Expected frames shape (T,H,W,3), got {frames.shape}')
  if size <= 0 or frames.shape[1] == size and frames.shape[2] == size:
    return frames.astype(np.uint8)
  out = []
  for frame in frames:
    img = Image.fromarray(frame.astype(np.uint8))
    img = img.resize((size, size), resample=Image.NEAREST)
    out.append(np.asarray(img, dtype=np.uint8))
  return np.stack(out, 0)


def save_video(path: pathlib.Path, frames: np.ndarray, fps: int):
  ensure_dir(path.parent)
  iio.imwrite(path, np.asarray(frames, np.uint8), fps=int(fps))


def save_frame_dir(directory: pathlib.Path, frames: np.ndarray):
  ensure_dir(directory)
  for index, frame in enumerate(np.asarray(frames, np.uint8)):
    Image.fromarray(frame).save(directory / f'{index:04d}.png')


def valid_ball_prompt(mask: np.ndarray):
  ys, xs = np.nonzero(np.asarray(mask) > 0)
  if xs.size == 0:
    return None
  return float(xs.mean()), float(ys.mean())


def choose_window(chunks, context: int, horizon: int, seed: int):
  candidates = wm_metrics.sample_windows(
      chunks, samples=2048, context=context, horizon=horizon, seed=seed)
  for chunk_id, start in candidates:
    _, chunk = chunks[chunk_id]
    if 'mask1' not in chunk:
      continue
    masks = chunk['mask1'][start + 1:start + horizon + 1]
    for t, mask in enumerate(masks):
      if valid_ball_prompt(mask) is not None:
        return chunk_id, start, t
  raise RuntimeError(
      'No valid short-horizon window with a non-empty mask1 ball prompt.')


def resolve_window(args, chunks):
  explicit = args.chunk_id is not None or args.start is not None
  if not explicit:
    chunk_id, start, prompt_t = choose_window(
        chunks, args.context, args.horizon, args.seed)
    return chunk_id, start, prompt_t, 'seed'
  if args.chunk_id is None or args.start is None:
    raise ValueError('--chunk-id and --start must be provided together.')
  chunk_id = int(args.chunk_id)
  start = int(args.start)
  if chunk_id < 0 or chunk_id >= len(chunks):
    raise ValueError(f'--chunk-id {chunk_id} is outside [0, {len(chunks)})')
  _, chunk = chunks[chunk_id]
  if not wm_metrics.valid_window(chunk, start, args.context, args.horizon):
    raise ValueError(
        f'Explicit window is invalid: chunk_id={chunk_id}, start={start}, '
        f'context={args.context}, horizon={args.horizon}')
  masks = np.asarray(chunk['mask1'][start + 1:start + args.horizon + 1])
  if args.prompt_t is None:
    for prompt_t, mask in enumerate(masks):
      if valid_ball_prompt(mask) is not None:
        return chunk_id, start, prompt_t, 'explicit'
    raise ValueError('Explicit window has no non-empty mask1 prompt frame.')
  prompt_t = int(args.prompt_t)
  if prompt_t < 0 or prompt_t >= args.horizon:
    raise ValueError(f'--prompt-t {prompt_t} is outside [0, {args.horizon})')
  if valid_ball_prompt(masks[prompt_t]) is None:
    raise ValueError(f'Explicit --prompt-t {prompt_t} has an empty mask1.')
  return chunk_id, start, prompt_t, 'explicit'


def make_prompt(mask: np.ndarray, frame_idx: int, source_size: int, sam2_size: int):
  center = valid_ball_prompt(mask)
  if center is None:
    raise ValueError('Cannot prompt SAM2 from an empty ball mask.')
  x64, y64 = center
  scale = sam2_size / float(source_size)
  x = (x64 + 0.5) * scale
  y = (y64 + 0.5) * scale
  # The local SAM2 client expects normalized point coordinates when bypassing
  # the path loader that normally performs normalization.
  return {
      'frame_idx': int(frame_idx),
      'obj_id': 1,
      'labels': [1],
      'points': [[float(x / sam2_size), float(y / sam2_size)]],
      'point_px_source': [float(x64), float(y64)],
      'point_px_sam2': [float(x), float(y)],
  }


def real_tracks_from_masks(masks: np.ndarray, sample_id: str = 'window0'):
  rows = []
  for t, mask in enumerate(masks):
    center = valid_ball_prompt(mask)
    row = {
        'sample_id': sample_id,
        't': int(t),
        'ball_present': center is not None,
        'ball_area': int(np.asarray(mask > 0).sum()),
      }
    if center is not None:
      row['ball_x'], row['ball_y'] = center
    rows.append(row)
  add_velocity(rows)
  return rows


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


def summarize_tracks(real_rows, by_wm_rows, divergence_px: float):
  real_by_t = {int(row['t']): row for row in real_rows}
  summary = {}
  for name, rows in by_wm_rows.items():
    squared_dists = []
    dists = []
    horizon = len(rows)
    detected = 0
    direction_pairs = []
    if horizon == 0:
      summary[name] = {
          'ball_detectability': float('nan'),
          'ball_center_mse': float('nan'),
          'ball_l2_mean': float('nan'),
          'ball_l2_rmse': float('nan'),
          'real_present_frames': 0,
          'sam2_present_frames': 0,
          'object_divergence_px': float(divergence_px),
          'first_divergence_t': None,
      }
      continue
    for row in rows:
      t = int(row['t'])
      real = real_by_t.get(t, {})
      if not row.get('ball_present'):
        continue
      detected += 1
      if not real.get('ball_present'):
        continue
      dist = float(np.hypot(
          float(row['ball_x']) - float(real['ball_x']),
          float(row['ball_y']) - float(real['ball_y'])))
      dists.append(dist)
      squared_dists.append(dist * dist)
      if np.isfinite(row.get('ball_vx', np.nan)) and np.isfinite(real.get('ball_vx', np.nan)):
        direction_pairs.append((
            int(row.get('ball_vx_sign', 0)) == int(real.get('ball_vx_sign', 0)),
            int(row.get('ball_vy_sign', 0)) == int(real.get('ball_vy_sign', 0)),
        ))
    arr = np.asarray(dists, np.float64)
    se_arr = np.asarray(squared_dists, np.float64)
    detectability = detected / float(horizon)
    mse = float(np.nanmean(se_arr)) if se_arr.size else float('nan')
    summary[name] = {
        'ball_detectability': detectability,
        'ball_center_mse': mse,
        'ball_l2_mean': float(np.nanmean(arr)) if arr.size else float('nan'),
        'ball_l2_rmse': float(np.sqrt(mse)) if np.isfinite(mse) else float('nan'),
        'real_present_frames': int(sum(
            1 for r in rows if real_by_t.get(int(r['t']), {}).get('ball_present'))),
        'sam2_present_frames': int(detected),
        'object_divergence_px': float(divergence_px),
    }
    if direction_pairs:
      dirs = np.asarray(direction_pairs, bool)
      summary[name]['ball_vx_sign_accuracy'] = float(dirs[:, 0].mean())
      summary[name]['ball_vy_sign_accuracy'] = float(dirs[:, 1].mean())
  return summary


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
    raise RuntimeError(
        'Could not import sam2_client. Install it with '
        '`conda run -n dreamer pip install -e '
        f'{args.sam2_client_root}` or install its dependencies.') from exc

  video_arr = resize_video(spec.frames64, args.sam2_size)
  if args.guiding_source == 'wm':
    guiding_arr = video_arr
  else:
    guiding_arr = resize_video(args.real_frames64, args.sam2_size)
  masks = chunked_sam2_segmentation(
      video_arr=video_arr,
      guiding_video_arr=guiding_arr,
      prompts=[prompt],
      backend_endpoint=args.backend_endpoint,
      selected_obj_ids=(1,),
      verbose_mode=bool(args.verbose),
      max_frames_per_chunk=int(args.max_frames_per_chunk),
  )
  return np.asarray(masks[1], np.uint8)


def generate_rollouts(args):
  chunks = wm_metrics.load_replay_chunks(args.eval_replay_dir, args.limit_chunks)
  chunk_id, start, prompt_t, window_source = resolve_window(args, chunks)
  chunk_path, chunk = chunks[chunk_id]
  real_frames64 = np.asarray(chunk['image'][start + 1:start + args.horizon + 1], np.uint8)
  real_masks64 = np.asarray(chunk['mask1'][start + 1:start + args.horizon + 1])
  actions = [int(x) for x in chunk['action'][start:start + args.horizon]]

  ensure_dir(args.output_dir)
  prompt = make_prompt(
      real_masks64[prompt_t],
      frame_idx=prompt_t,
      source_size=real_frames64.shape[1],
      sam2_size=args.sam2_size)
  manifest = {
      'chunk': str(chunk_path),
      'chunk_id': int(chunk_id),
      'start': int(start),
      'window_source': window_source,
      'context': int(args.context),
      'horizon': int(args.horizon),
      'prompt_t': int(prompt_t),
      'sam2_size': int(args.sam2_size),
      'guiding_source': args.guiding_source,
      'prompt': prompt,
      'actions': actions,
  }
  write_json(args.output_dir / 'manifest.json', manifest)
  write_json(args.output_dir / 'prompt.json', [prompt])
  args.real_frames64 = real_frames64

  real_video = resize_video(real_frames64, args.sam2_size)
  save_video(args.output_dir / 'real_target.mp4', real_video, args.fps)
  save_frame_dir(args.output_dir / 'frames' / 'real_target', real_video)
  real_tracks = real_tracks_from_masks(real_masks64)
  write_csv(args.output_dir / 'tracks' / 'real_mask_tracks.csv', real_tracks)

  wm_specs = (
      [wm_metrics.WMSpec(*parse_name_path(x)) for x in args.wm]
      if args.wm else default_wm_specs(args.runs_root))
  policy_agent = None
  policy_name = None
  policy_ckpt = None
  if args.policy:
    if len(args.policy) > 1:
      raise ValueError('Only one --policy is supported per run.')
    policy_name, policy_ckpt = parse_policy_path(args.policy[0])
    policy_task = args.policy_task or args.task
    _, _, policy_agent = _load_policy_agent(
        policy_ckpt, policy_task, args.jax_platform)

  generated = []
  for spec in wm_specs:
    print(f'Loading WM {spec.name}: {spec.checkpoint}', flush=True)
    runner = wm_metrics.load_wm_runner(spec, args.task, args.jax_platform)
    if policy_agent is not None:
      rec = rollout_with_policy(
          policy_agent, runner, chunk, start, args.context, args.horizon)
    else:
      rec = wm_metrics.rollout_prediction(
          runner, chunk, start, args.context, args.horizon)
    frames64 = np.asarray(rec['pred_frames'], np.uint8)
    video = resize_video(frames64, args.sam2_size)
    video_path = args.output_dir / 'videos' / f'{spec.name}.mp4'
    frame_dir = args.output_dir / 'frames' / spec.name
    save_video(video_path, video, args.fps)
    save_frame_dir(frame_dir, video)
    write_json(args.output_dir / 'rollouts' / f'{spec.name}.json', {
        'wm': spec.name,
        'checkpoint': str(spec.checkpoint),
        'pred_reward': rec['pred_reward'].tolist(),
        'pred_reward_mode': rec['pred_reward_mode'].tolist(),
        'target_reward': rec['target_reward'].tolist(),
        'policy': policy_name,
        'policy_ckpt': str(policy_ckpt) if policy_ckpt is not None else None,
        'policy_task': args.policy_task or args.task if args.policy else None,
        'policy_actions': rec.get('policy_actions', []).tolist()
        if 'policy_actions' in rec else None,
    })
    generated.append(SegmentSpec(spec.name, frames64, video_path, frame_dir))
    del runner
  return real_tracks, generated, prompt


def main(argv=None):
  parser = argparse.ArgumentParser(
      description='Generate three Dreamer WM Pong videos and segment ball with SAM2.')
  parser.add_argument('--wm', action='append', default=[],
                      help='Optional name=/path/to/checkpoint. Defaults to the three paper WMs.')
  parser.add_argument('--runs-root', type=pathlib.Path, default=DEFAULT_RUNS_ROOT)
  parser.add_argument('--eval-replay-dir', type=pathlib.Path, default=DEFAULT_EVAL_REPLAY)
  parser.add_argument('--output-dir', type=pathlib.Path,
                      default=pathlib.Path('eval_outputs/pong_sam2_ball_concept_short_horizon'))
  parser.add_argument('--task', default='atari100k_pong')
  parser.add_argument('--jax-platform', default='cuda')
  parser.add_argument('--context', type=int, default=5)
  parser.add_argument('--horizon', type=int, default=32)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--chunk-id', type=int, default=None,
                      help='Explicit replay chunk index to evaluate.')
  parser.add_argument('--start', type=int, default=None,
                      help='Explicit replay start index. Requires --chunk-id.')
  parser.add_argument('--prompt-t', type=int, default=None,
                      help='Prompt frame offset within the rollout window.')
  parser.add_argument('--limit-chunks', type=int, default=None)
  parser.add_argument('--fps', type=int, default=15)
  parser.add_argument('--sam2-size', type=int, default=256)
  parser.add_argument('--sam2-client-root', type=pathlib.Path,
                      default=PROJECTS_ROOT / 'NovelWorldModel/sam2_client')
  parser.add_argument('--backend-endpoint', default='http://localhost:7263')
  parser.add_argument('--guiding-source', choices=['real', 'wm'], default='real')
  parser.add_argument('--policy', action='append', default=[],
                      help='Optional policy=name=/path/to/policy.ckpt (.npz or ckpt dir).')
  parser.add_argument('--policy-task', default=None,
                      help='Task to use for policy rollout. Defaults to --task.')
  parser.add_argument('--max-frames-per-chunk', type=int, default=256)
  parser.add_argument('--divergence-px', type=float, default=8.0)
  parser.add_argument('--skip-sam2', action='store_true',
                      help='Only generate WM videos/prompts; do not call SAM2.')
  parser.add_argument('--verbose', action='store_true')
  args = parser.parse_args(argv)

  real_tracks, generated, prompt = generate_rollouts(args)
  if args.skip_sam2:
    print(f'Generated videos and prompt under {args.output_dir}; skipped SAM2.')
    return

  by_wm_rows = {}
  for spec in generated:
    print(f'Running SAM2 for {spec.name}: {spec.video_path}', flush=True)
    mask = run_sam2(spec, prompt, args)
    mask_dir = args.output_dir / 'masks'
    ensure_dir(mask_dir)
    np.savez_compressed(mask_dir / f'{spec.name}_sam2_ball.npz', mask1=mask)
    wm_rows = tracks_from_masks(
        mask, downscale=args.sam2_size / 64.0, sample_id='window0')
    by_wm_rows[spec.name] = wm_rows
    write_csv(args.output_dir / 'tracks' / f'{spec.name}_sam2_tracks.csv', wm_rows)
    overlay = render_overlay(resize_video(spec.frames64, args.sam2_size), mask)
    save_video(args.output_dir / 'segmented_videos' / f'{spec.name}_sam2_ball.mp4',
               overlay, args.fps)

  summary = summarize_tracks(real_tracks, by_wm_rows, args.divergence_px)
  write_json(args.output_dir / 'sam2_ball_summary.json', summary)
  rows = [{'name': name, **metrics} for name, metrics in summary.items()]
  write_csv(args.output_dir / 'sam2_ball_summary.csv', rows)
  print(f'Wrote SAM2 ball experiment outputs to {args.output_dir}')


if __name__ == '__main__':
  main()
