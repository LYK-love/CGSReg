#!/usr/bin/env python3
"""Control-relevant Pong world-model evaluation.

The script provides three paper-facing metric groups:

1. k-step reward/return and score-event prediction from replay windows.
2. imagined-return vs real-return correlation for policy checkpoints.
3. object-track metrics from real RAM tracks and WM/SAM2 tracks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import pathlib
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DREAMER_ROOT = pathlib.Path(
    os.environ.get('WM_EVAL_DREAMER_ROOT', ROOT)).expanduser().resolve()
if str(DREAMER_ROOT) not in sys.path:
  sys.path.insert(1, str(DREAMER_ROOT))

import elements  # noqa: E402
import ruamel.yaml as yaml  # noqa: E402

from dreamerv3 import main as dreamer_main  # noqa: E402
from dreamerv3.interactive import dreamer_adapter  # noqa: E402
from embodied.jax import agent as jax_agent  # noqa: E402
from embodied.jax import internal as jax_internal  # noqa: E402


PONG_DIMS = dreamer_adapter.PONG_DIMS
SIGNED_PONG_DIMS = dreamer_adapter.SIGNED_PONG_DIMS
MASK_SPECS = (
    ('mask1', 'ball_mse', 'ball'),
    ('mask2', 'left_paddle_mse', 'left_paddle'),
    ('mask3', 'right_paddle_mse', 'right_paddle'),
)
_SUPPORTED_JAX_KEYS = (
    set(jax_internal.setup.__code__.co_varnames) |
    set(jax_agent.Options.__dataclass_fields__))


def load_config_compat(config_path, jax_platform):
  """Load old checkpoint configs while dropping stale JAX setup keys."""
  config_path = pathlib.Path(config_path).expanduser().resolve()
  data = yaml.YAML(typ='safe').load(config_path.read_text())
  default_path = DREAMER_ROOT / 'dreamerv3' / 'configs.yaml'
  defaults = yaml.YAML(typ='safe').load(default_path.read_text())['defaults']
  data = dreamer_adapter._fill_missing(data, defaults)
  data.setdefault('jax', {})
  data['jax'] = {
      key: value for key, value in data['jax'].items()
      if key in _SUPPORTED_JAX_KEYS}
  data['jax']['platform'] = jax_platform
  data['jax']['precompile'] = False
  data['jax']['profiler'] = False
  data['jax']['prealloc'] = False
  data['jax']['transfer_guard'] = False
  return elements.Config(data)


dreamer_adapter.load_config = load_config_compat


def load_checkpoint_compat(path, loadfns):
  """Load current directory checkpoints or older single-file Dreamer ckpts."""
  path = pathlib.Path(path).expanduser().resolve()
  if path.is_file() and path.suffix == '.ckpt':
    with path.open('rb') as f:
      data = pickle.load(f)
    agent_data = data.get('agent', data)
    params = normalize_legacy_params(agent_data)
    for name, loadfn in loadfns.items():
      if name != 'agent':
        raise KeyError(name)
      loadfn({'params': params, 'counters': {}})
    return
  return _ORIGINAL_CHECKPOINT_LOAD(path, loadfns)


def normalize_legacy_params(agent_data):
  params = {}
  prefix_map = {
      'agent/enc/': 'enc/',
      'agent/dyn/': 'dyn/',
      'agent/dec/': 'dec/',
      'agent/rew/': 'rew/',
      'agent/con/': 'con/',
      'agent/actor/': 'pol/',
      'agent/critic/': 'val/',
      'agent/slowcritic/': 'slowval/',
  }
  for key, value in agent_data.items():
    for old, new in prefix_map.items():
      if key.startswith(old):
        params[normalize_legacy_key(new + key[len(old):])] = value
        break
    else:
      if key.startswith(('enc/', 'dyn/', 'dec/', 'rew/', 'con/', 'pol/')):
        params[normalize_legacy_key(key)] = value
  return params


def normalize_legacy_key(key: str) -> str:
  key = re.sub(r'^enc/conv([0-9]+)/(bias|kernel)$', r'enc/cnn\1/\2', key)
  key = re.sub(r'^enc/conv([0-9]+)/norm/scale[0-9]+$', r'enc/cnn\1norm/scale', key)
  key = re.sub(r'^(rew|con)/h([0-9]+)/(bias|kernel)$', r'\1/mlp/linear\2/\3', key)
  key = re.sub(r'^(rew|con)/h([0-9]+)/norm/scale[0-9]+$', r'\1/mlp/norm\2/scale', key)
  key = re.sub(r'^rew/dist/out/(bias|kernel)$', r'rew/head/logits/\1', key)
  key = re.sub(r'^con/dist/out/(bias|kernel)$', r'con/head/logit/\1', key)
  key = re.sub(r'^pol/action/out/(bias|kernel)$', r'pol/action/logits/\1', key)
  return key


_ORIGINAL_CHECKPOINT_LOAD = elements.checkpoint.load
elements.checkpoint.load = load_checkpoint_compat


@dataclass(frozen=True)
class WMSpec:
  name: str
  checkpoint: pathlib.Path


@dataclass(frozen=True)
class PolicySpec:
  name: str
  checkpoint: pathlib.Path


def parse_name_path(value: str) -> tuple[str, pathlib.Path]:
  if '=' in value:
    name, path = value.split('=', 1)
  else:
    path = value
    name = pathlib.Path(path).parent.name or pathlib.Path(path).stem
  return name, pathlib.Path(path).expanduser().resolve()


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


def load_default_config(task: str, jax_platform: str):
  config_path = DREAMER_ROOT / 'dreamerv3' / 'configs.yaml'
  configs = yaml.YAML(typ='safe').load(config_path.read_text())
  config = elements.Config(configs['defaults'])
  updates = {
      'task': task,
      'jax.platform': jax_platform,
      'jax.precompile': False,
      'jax.profiler': False,
      'jax.prealloc': False,
      'agent.pure_wm': False,
      'agent.freeze_wm': False,
      'agent.env_rl': False,
  }
  if 'jax.transfer_guard' in config.flat:
    updates['jax.transfer_guard'] = False
  return config.update(updates)


def load_wm_runner(spec: WMSpec, task: str, jax_platform: str):
  config = load_default_config(task, jax_platform)
  return dreamer_adapter.load_model_runner(config, str(spec.checkpoint))


def load_policy_runner(spec: PolicySpec, task: str, jax_platform: str):
  config = load_default_config(task, jax_platform)
  return dreamer_adapter.load_policy_runner(config, str(spec.checkpoint), jax_platform)


def load_replay_chunks(replay_dir: pathlib.Path, limit_chunks: int | None):
  paths = sorted(replay_dir.expanduser().glob('*.npz'))
  if limit_chunks:
    paths = paths[-int(limit_chunks):]
  if not paths:
    raise FileNotFoundError(f'No replay chunks found in {replay_dir}')
  chunks = []
  for path in paths:
    with np.load(path) as data:
      chunk = {key: data[key] for key in data.files}
    chunks.append((path, chunk))
  return chunks


def obs_at(chunk: dict[str, np.ndarray], index: int):
  keys = (
      'image', 'ram', 'reward', 'is_first', 'is_last', 'is_terminal',
      'mask1', 'mask2', 'mask3')
  obs = {key: chunk[key][index] for key in keys if key in chunk}
  if 'ram' not in obs and all(key in chunk for key in ('mask1', 'mask2', 'mask3')):
    obs['ram'] = synthesize_pong_ram_from_masks(chunk, index)
  return obs


def _mask_center(mask: np.ndarray):
  ys, xs = np.nonzero(np.asarray(mask) > 0)
  if xs.size == 0:
    return None
  return float(xs.mean()), float(ys.mean())


def _encode_signed_byte(value: float):
  value = int(np.clip(round(value), -128, 127))
  return np.uint8(value % 256)


def synthesize_pong_ram_from_masks(chunk: dict[str, np.ndarray], index: int):
  """Approximate the RAM fields needed by old RAM-conditioned Pong WMs.

  Some held-out mask replays intentionally do not contain ALE RAM. Older
  baseline checkpoints still have a RAM encoder branch, so the open-loop visual
  comparison needs a compatible vector input for the shared context frames.
  The values below are derived only from mask centroids and fill the Pong RAM
  dimensions used elsewhere in this script; all unrelated bytes remain zero.
  """
  ram = np.zeros((128,), dtype=np.uint8)
  centers = {
      'ball': _mask_center(chunk['mask1'][index]),
      'left': _mask_center(chunk['mask2'][index]),
      'right': _mask_center(chunk['mask3'][index]),
  }
  if centers['ball'] is not None:
    x, y = centers['ball']
    ram[PONG_DIMS['ball_x']] = np.uint8(np.clip(round(x * 2.5 + 50.0), 0, 255))
    ram[PONG_DIMS['ball_y']] = np.uint8(np.clip(round(y * 3.0 + 24.0), 0, 255))
  if centers['left'] is not None:
    _, y = centers['left']
    ram[PONG_DIMS['paddle_cpu']] = np.uint8(np.clip(round(y * 3.0 + 12.0), 0, 255))
  if centers['right'] is not None:
    _, y = centers['right']
    ram[PONG_DIMS['paddle_player_y']] = np.uint8(np.clip(round(y * 3.0 + 24.0), 0, 255))
    ram[PONG_DIMS['paddle_player_y_next']] = ram[PONG_DIMS['paddle_player_y']]
  if index > 0:
    prev = _mask_center(chunk['mask1'][index - 1])
    curr = centers['ball']
    if prev is not None and curr is not None:
      ram[PONG_DIMS['v_x']] = _encode_signed_byte((curr[0] - prev[0]) * 2.5)
      ram[PONG_DIMS['v_y']] = _encode_signed_byte((curr[1] - prev[1]) * 3.0)
  return ram


def valid_window(chunk: dict[str, np.ndarray], start: int, context: int, horizon: int):
  length = len(chunk['reward'])
  if start - context < 0 or start + horizon >= length:
    return False
  # Do not evaluate across episode boundaries; the reset dynamics are a
  # different problem from open-loop Pong prediction.
  lo = start - context + 1
  hi = start + horizon + 1
  if 'is_first' in chunk and bool(np.any(chunk['is_first'][lo:hi])):
    return False
  if 'is_last' in chunk and bool(np.any(chunk['is_last'][start:hi - 1])):
    return False
  return True


def sample_windows(
    chunks: list[tuple[pathlib.Path, dict[str, np.ndarray]]],
    samples: int,
    context: int,
    horizon: int,
    seed: int):
  candidates = []
  for chunk_id, (_, chunk) in enumerate(chunks):
    for start in range(context, len(chunk['reward']) - horizon):
      if valid_window(chunk, start, context, horizon):
        candidates.append((chunk_id, start))
  if not candidates:
    raise RuntimeError(
        f'No valid replay windows for context={context}, horizon={horizon}.')
  rng = random.Random(seed)
  rng.shuffle(candidates)
  return candidates[:min(samples, len(candidates))]


def bootstrap_state_from_replay(runner, chunk: dict[str, np.ndarray], start: int, context: int):
  first = start - context
  state = runner.init_real_state(obs_at(chunk, first))
  for t in range(first, start):
    state = runner.advance_real_state(
        state, int(chunk['action'][t]), obs_at(chunk, t + 1))
  return runner.real_to_world_state(state)


def quantize_reward(values, threshold: float):
  values = np.asarray(values, dtype=np.float32)
  return np.where(values >= threshold, 1.0, np.where(values <= -threshold, -1.0, 0.0))


def rollout_prediction(runner, chunk: dict[str, np.ndarray], start: int, context: int, horizon: int):
  state = bootstrap_state_from_replay(runner, chunk, start, context)
  pred_rewards = []
  pred_mode_rewards = []
  pred_cont = []
  pred_frames = []
  for offset in range(horizon):
    action = int(chunk['action'][start + offset])
    state, result = runner.imagine_step(state, action)
    pred_rewards.append(float(result.info.get('reward_expected', result.reward)))
    pred_mode_rewards.append(float(result.info.get('reward_mode', result.reward)))
    pred_cont.append(float(result.info.get('cont_prob', np.nan)))
    if 'image' in result.obs:
      pred_frames.append(np.asarray(result.obs['image']))
  target_rewards = np.asarray(chunk['reward'][start + 1:start + horizon + 1], np.float32)
  return {
      'pred_reward': np.asarray(pred_rewards, np.float32),
      'pred_reward_mode': np.asarray(pred_mode_rewards, np.float32),
      'pred_cont_prob': np.asarray(pred_cont, np.float32),
      'target_reward': target_rewards,
      'pred_frames': pred_frames,
  }


def safe_mean(xs):
  xs = np.asarray(xs, dtype=np.float64)
  xs = xs[np.isfinite(xs)]
  return float(xs.mean()) if xs.size else float('nan')


def binary_stats(pred: np.ndarray, target: np.ndarray):
  pred = np.asarray(pred, bool)
  target = np.asarray(target, bool)
  tp = int(np.logical_and(pred, target).sum())
  fp = int(np.logical_and(pred, ~target).sum())
  fn = int(np.logical_and(~pred, target).sum())
  tn = int(np.logical_and(~pred, ~target).sum())
  precision = tp / (tp + fp) if tp + fp else float('nan')
  recall = tp / (tp + fn) if tp + fn else float('nan')
  f1 = 2 * precision * recall / (precision + recall) if precision + recall else float('nan')
  acc = (tp + tn) / max(tp + fp + fn + tn, 1)
  return {
      'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
      'accuracy': float(acc),
      'precision': float(precision),
      'recall': float(recall),
      'f1': float(f1),
  }


def first_event_timing(pred_events: np.ndarray, target_events: np.ndarray):
  signed = []
  absolute = []
  missed_target = 0
  false_alarm = 0
  for pred, target in zip(pred_events, target_events):
    pred_idx = np.flatnonzero(pred)
    target_idx = np.flatnonzero(target)
    if target_idx.size and pred_idx.size:
      delta = int(pred_idx[0] - target_idx[0])
      signed.append(delta)
      absolute.append(abs(delta))
    elif target_idx.size:
      missed_target += 1
    elif pred_idx.size:
      false_alarm += 1
  return {
      'first_event_timing_signed_mean': safe_mean(signed),
      'first_event_timing_abs_mean': safe_mean(absolute),
      'first_event_timing_pairs': int(len(signed)),
      'first_event_missed_target_windows': int(missed_target),
      'first_event_false_alarm_windows': int(false_alarm),
  }


def reward_divergence(pred_q: np.ndarray, target_q: np.ndarray, horizons: list[int]):
  divs = []
  for pred, target in zip(pred_q, target_q):
    mismatch = np.flatnonzero(pred != target)
    divs.append(int(mismatch[0] + 1) if mismatch.size else horizons[-1] + 1)
  out = {'reward_divergence_horizon_mean': safe_mean(divs)}
  for h in horizons:
    out[f'reward_not_diverged_rate@{h}'] = float(np.mean(np.asarray(divs) > h))
  return out


def summarize_reward_predictions(records: list[dict[str, np.ndarray]], horizons: list[int], threshold: float):
  pred = np.stack([r['pred_reward'] for r in records])
  pred_mode = np.stack([r['pred_reward_mode'] for r in records])
  target = np.stack([r['target_reward'] for r in records])
  pred_q = quantize_reward(pred, threshold)
  pred_mode_q = quantize_reward(pred_mode, threshold)
  target_q = quantize_reward(target, threshold)

  metrics = {
      'samples': int(len(records)),
      'max_horizon': int(pred.shape[1]),
      'reward_threshold': float(threshold),
  }
  for h in horizons:
    pe = pred[:, h - 1]
    te = target[:, h - 1]
    metrics[f'reward_mae@{h}'] = float(np.mean(np.abs(pe - te)))
    metrics[f'reward_rmse@{h}'] = float(np.sqrt(np.mean(np.square(pe - te))))
    metrics[f'reward_sign_accuracy@{h}'] = float(np.mean(pred_q[:, h - 1] == target_q[:, h - 1]))
    pred_return = pred[:, :h].sum(axis=1)
    target_return = target[:, :h].sum(axis=1)
    metrics[f'return_mae@{h}'] = float(np.mean(np.abs(pred_return - target_return)))
    metrics[f'return_rmse@{h}'] = float(np.sqrt(np.mean(np.square(pred_return - target_return))))
    metrics[f'return_sign_accuracy@{h}'] = float(
        np.mean(np.sign(pred_return) == np.sign(target_return)))

  metrics.update({
      f'score_event_expected/{k}': v
      for k, v in binary_stats(np.abs(pred_q) > 0, np.abs(target_q) > 0).items()
  })
  metrics.update({
      f'score_event_mode/{k}': v
      for k, v in binary_stats(np.abs(pred_mode_q) > 0, np.abs(target_q) > 0).items()
  })
  sign_target_events = np.abs(target_q) > 0
  sign_correct = (pred_q == target_q) & sign_target_events
  metrics['score_event_expected/sign_accuracy_on_target_events'] = float(
      sign_correct.sum() / max(sign_target_events.sum(), 1))
  metrics.update({
      f'score_event_expected/{k}': v
      for k, v in first_event_timing(np.abs(pred_q) > 0, np.abs(target_q) > 0).items()
  })
  metrics.update(reward_divergence(pred_q, target_q, horizons))
  return metrics


def frames_to_metric_float(frames: np.ndarray, pixel_scale: str):
  arr = np.asarray(frames, dtype=np.float32)
  finite = arr[np.isfinite(arr)]
  if not finite.size:
    return arr
  if pixel_scale == 'minus-one-one':
    if finite.min() < -0.05:
      return arr
    if finite.max() <= 1.5:
      return arr * 2.0 - 1.0
    return arr / 127.5 - 1.0
  if pixel_scale == 'zero-one':
    if finite.min() < -0.05:
      return (arr + 1.0) / 2.0
    if finite.max() <= 1.5:
      return arr
    return arr / 255.0
  if pixel_scale == 'uint8':
    if finite.min() < -0.05:
      return (arr + 1.0) * 127.5
    if finite.max() <= 1.5:
      return arr * 255.0
    return arr
  raise ValueError(f'Unknown pixel scale {pixel_scale!r}.')


def _masked_mse_values(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    horizon: int):
  mask = np.asarray(mask[:horizon]) > 0
  se = np.square(pred[:horizon] - target[:horizon])
  values = []
  for t in range(int(horizon)):
    active = mask[t]
    if not active.any():
      continue
    values.append(float(se[t][active].mean()))
  return values


def visual_prediction_record(
    rec: dict[str, Any],
    chunk: dict[str, np.ndarray],
    start: int,
    max_horizon: int,
    horizons: list[int],
    pixel_scale: str):
  pred_u8 = np.asarray(rec['pred_frames'])
  if pred_u8.shape[0] < max_horizon:
    raise ValueError(
        f'World model produced {pred_u8.shape[0]} frames, expected at least {max_horizon}.')
  target_u8 = np.asarray(chunk['image'][start + 1:start + max_horizon + 1])
  if target_u8.shape[0] != max_horizon:
    raise ValueError(
        f'Replay target has {target_u8.shape[0]} frames, expected {max_horizon}.')
  if pred_u8.shape[1:] != target_u8.shape[1:]:
    raise ValueError(
        f'Predicted frame shape {pred_u8.shape[1:]} does not match target '
        f'shape {target_u8.shape[1:]}.')

  missing = [key for key, _, _ in MASK_SPECS if key not in chunk]
  if missing:
    raise KeyError(f'Replay chunk is missing required object masks: {missing}')

  pred = frames_to_metric_float(pred_u8[:max_horizon], pixel_scale)
  target = frames_to_metric_float(target_u8, pixel_scale)
  by_horizon = {}
  for horizon in horizons:
    h = int(horizon)
    metrics = {
        'global_mse': float(np.square(pred[:h] - target[:h]).mean()),
    }
    for mask_key, metric_key, _ in MASK_SPECS:
      values = _masked_mse_values(
          pred, target, np.asarray(chunk[mask_key][start + 1:start + max_horizon + 1]), h)
      metrics[metric_key] = safe_mean(values)
      metrics[f'{metric_key}_frames'] = int(len(values))
    by_horizon[h] = metrics
  return by_horizon


def summarize_visual_predictions(records: list[dict[int, dict[str, float]]], horizons: list[int]):
  summary = {
      'samples': int(len(records)),
      'horizons': [int(x) for x in horizons],
  }
  metric_keys = ['global_mse', *(metric_key for _, metric_key, _ in MASK_SPECS)]
  for horizon in horizons:
    h = int(horizon)
    for metric_key in metric_keys:
      values = [record[h][metric_key] for record in records]
      summary[f'{metric_key}@{h}'] = safe_mean(values)
    for _, metric_key, _ in MASK_SPECS:
      summary[f'{metric_key}_frames@{h}'] = int(sum(
          record[h][f'{metric_key}_frames'] for record in records))
  return summary


def write_visual_flat_metrics(path: pathlib.Path, by_name: dict[str, dict[str, Any]], horizons: list[int]):
  keys = ['samples']
  for horizon in horizons:
    for key in ('global_mse', 'ball_mse', 'left_paddle_mse', 'right_paddle_mse'):
      keys.append(f'{key}@{int(horizon)}')
    for key in ('ball_mse_frames', 'left_paddle_mse_frames', 'right_paddle_mse_frames'):
      keys.append(f'{key}@{int(horizon)}')
  extra = sorted({key for metrics in by_name.values() for key in metrics if key not in keys and key != 'horizons'})
  with path.open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['name', *keys, *extra])
    writer.writeheader()
    for name, metrics in by_name.items():
      row = {'name': name}
      row.update({key: metrics.get(key) for key in [*keys, *extra]})
      writer.writerow(row)


def command_visual_prediction(args):
  wm_specs = [WMSpec(*parse_name_path(x)) for x in args.wm]
  chunks = load_replay_chunks(args.eval_replay_dir, args.limit_chunks)
  horizons = sorted({int(x) for x in args.horizons.split(',') if x})
  max_horizon = max(horizons)
  windows = sample_windows(chunks, args.samples, args.context, max_horizon, args.seed)

  out = {
      'eval_replay_dir': str(args.eval_replay_dir),
      'context': int(args.context),
      'horizons': horizons,
      'num_windows': int(len(windows)),
      'pixel_scale': args.pixel_scale,
      'protocol': (
          'Replay-aligned Dreamer visual prediction: bootstrap from real '
          'context frames, then feed the recorded future action sequence open-loop.'),
      'mask_keys': {
          object_name: mask_key for mask_key, _, object_name in MASK_SPECS
      },
      'world_models': {},
  }
  detail_rows = []
  ensure_dir(args.output_dir)
  for spec in wm_specs:
    print(f'Loading WM {spec.name}: {spec.checkpoint}')
    runner = load_wm_runner(spec, args.task, args.jax_platform)
    records = []
    for idx, (chunk_id, start) in enumerate(windows):
      chunk_path, chunk = chunks[chunk_id]
      rec = rollout_prediction(runner, chunk, start, args.context, max_horizon)
      visual = visual_prediction_record(
          rec, chunk, start, max_horizon, horizons, args.pixel_scale)
      records.append(visual)
      for horizon, metrics in visual.items():
        detail_rows.append({
            'name': spec.name,
            'window': int(idx),
            'chunk': str(chunk_path),
            'chunk_id': int(chunk_id),
            'start': int(start),
            'horizon': int(horizon),
            **metrics,
        })
      if args.export_wm_frames and idx < args.export_wm_frames:
        frame_dir = args.output_dir / 'wm_frames' / spec.name / f'window_{idx:05d}'
        ensure_dir(frame_dir)
        save_frames(frame_dir, rec['pred_frames'])
        target_dir = args.output_dir / 'target_frames' / f'window_{idx:05d}'
        ensure_dir(target_dir)
        save_frames(target_dir, chunk['image'][start + 1:start + max_horizon + 1])
        write_json(frame_dir / 'manifest.json', {
            'wm': spec.name,
            'window': idx,
            'chunk': str(chunk_path),
            'start': int(start),
            'context': int(args.context),
            'horizon': int(max_horizon),
            'actions': [int(x) for x in chunk['action'][start:start + max_horizon]],
        })
    out['world_models'][spec.name] = summarize_visual_predictions(records, horizons)
    del runner

  write_json(args.output_dir / 'visual_prediction_metrics.json', out)
  write_visual_flat_metrics(
      args.output_dir / 'visual_prediction_metrics.csv',
      out['world_models'],
      horizons)
  write_csv(args.output_dir / 'visual_prediction_window_metrics.csv', detail_rows)
  print(f'Wrote {args.output_dir / "visual_prediction_metrics.json"}')


def command_rollout_prediction(args):
  wm_specs = [WMSpec(*parse_name_path(x)) for x in args.wm]
  chunks = load_replay_chunks(args.eval_replay_dir, args.limit_chunks)
  horizons = sorted({int(x) for x in args.horizons.split(',') if x})
  max_horizon = max(horizons)
  windows = sample_windows(chunks, args.samples, args.context, max_horizon, args.seed)

  out = {
      'eval_replay_dir': str(args.eval_replay_dir),
      'context': int(args.context),
      'horizons': horizons,
      'num_windows': int(len(windows)),
      'world_models': {},
  }
  ensure_dir(args.output_dir)
  for spec in wm_specs:
    print(f'Loading WM {spec.name}: {spec.checkpoint}')
    runner = load_wm_runner(spec, args.task, args.jax_platform)
    records = []
    for idx, (chunk_id, start) in enumerate(windows):
      _, chunk = chunks[chunk_id]
      rec = rollout_prediction(runner, chunk, start, args.context, max_horizon)
      records.append(rec)
      if args.export_wm_frames and idx < args.export_wm_frames:
        frame_dir = args.output_dir / 'wm_frames' / spec.name / f'window_{idx:05d}'
        ensure_dir(frame_dir)
        save_frames(frame_dir, rec['pred_frames'])
        manifest = {
            'wm': spec.name,
            'window': idx,
            'chunk': str(chunks[chunk_id][0]),
            'start': int(start),
            'actions': [int(x) for x in chunk['action'][start:start + max_horizon]],
            'target_reward': rec['target_reward'].tolist(),
            'pred_reward': rec['pred_reward'].tolist(),
        }
        write_json(frame_dir / 'manifest.json', manifest)
    out['world_models'][spec.name] = summarize_reward_predictions(
        records, horizons, args.reward_threshold)
    del runner

  write_json(args.output_dir / 'rollout_prediction_metrics.json', out)
  write_flat_metrics(args.output_dir / 'rollout_prediction_metrics.csv', out['world_models'])
  print(f'Wrote {args.output_dir / "rollout_prediction_metrics.json"}')


def save_frames(directory: pathlib.Path, frames: list[np.ndarray]):
  from PIL import Image
  for idx, frame in enumerate(frames):
    img = np.asarray(frame)
    if img.ndim == 3 and img.shape[-1] == 1:
      img = np.repeat(img, 3, -1)
    Image.fromarray(img).save(directory / f'{idx:04d}.png')


def write_flat_metrics(path: pathlib.Path, by_name: dict[str, dict[str, Any]]):
  keys = sorted({key for metrics in by_name.values() for key in metrics})
  with path.open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['name', *keys])
    writer.writeheader()
    for name, metrics in by_name.items():
      writer.writerow({'name': name, **metrics})


def run_episode_with_policy(env, policy_runner, max_steps: int):
  obs, _ = env.reset()
  policy_state = policy_runner.init_real_state(obs)
  total = 0.0
  length = 0
  while length < max_steps:
    action = policy_runner.policy_action(policy_state.feat)
    result = env.step(action)
    total += float(result.reward)
    length += 1
    if result.done or result.trunc:
      break
    policy_state = policy_runner.advance_real_state(policy_state, action, result.obs)
  return total, length


def command_policy_correlation(args):
  wm_specs = [WMSpec(*parse_name_path(x)) for x in args.wm]
  policy_specs = [PolicySpec(*parse_name_path(x)) for x in args.policy]
  if len(policy_specs) < 2:
    raise ValueError('policy-correlation needs at least two --policy checkpoints.')

  config = load_default_config(args.task, args.jax_platform)
  real_env = dreamer_adapter.DreamerRealEnv(dreamer_main.make_env(config, 0))
  policy_runners = {
      spec.name: load_policy_runner(spec, args.task, args.jax_platform)
      for spec in policy_specs
  }
  real_returns = {}
  real_lengths = {}
  for spec in policy_specs:
    returns, lengths = [], []
    runner = policy_runners[spec.name]
    for _ in range(args.episodes):
      ret, length = run_episode_with_policy(real_env, runner, args.max_steps)
      returns.append(ret)
      lengths.append(length)
    real_returns[spec.name] = float(np.mean(returns))
    real_lengths[spec.name] = float(np.mean(lengths))

  wm_returns = {}
  for wm_spec in wm_specs:
    wm_runner = load_wm_runner(wm_spec, args.task, args.jax_platform)
    wm_env = dreamer_adapter.DreamerWorldModelEnv(wm_runner, real_env, args.max_steps)
    wm_returns[wm_spec.name] = {}
    for policy_spec in policy_specs:
      returns = []
      policy_runner = policy_runners[policy_spec.name]
      for _ in range(args.episodes):
        ret, _ = run_episode_with_policy(wm_env, policy_runner, args.max_steps)
        returns.append(ret)
      wm_returns[wm_spec.name][policy_spec.name] = float(np.mean(returns))

  out = {
      'task': args.task,
      'episodes': int(args.episodes),
      'max_steps': int(args.max_steps),
      'real_return': real_returns,
      'real_length': real_lengths,
      'world_models': {},
  }
  names = [p.name for p in policy_specs]
  real_vec = np.asarray([real_returns[name] for name in names], np.float64)
  for wm_name, by_policy in wm_returns.items():
    wm_vec = np.asarray([by_policy[name] for name in names], np.float64)
    out['world_models'][wm_name] = {
        'wm_return': by_policy,
        'pearson': pearson(wm_vec, real_vec),
        'spearman': pearson(rankdata(wm_vec), rankdata(real_vec)),
      }
  write_json(args.output_dir / 'policy_return_correlation.json', out)
  print(f'Wrote {args.output_dir / "policy_return_correlation.json"}')
  real_env.close()


def pearson(x: np.ndarray, y: np.ndarray):
  x = np.asarray(x, np.float64)
  y = np.asarray(y, np.float64)
  if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
    return float('nan')
  return float(np.corrcoef(x, y)[0, 1])


def rankdata(x: np.ndarray):
  order = np.argsort(x)
  ranks = np.empty_like(order, dtype=np.float64)
  ranks[order] = np.arange(len(x), dtype=np.float64)
  return ranks


def load_tracks(path: pathlib.Path):
  path = path.expanduser().resolve()
  if path.suffix == '.jsonl':
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
  elif path.suffix == '.json':
    data = json.loads(path.read_text())
    rows = data['tracks'] if isinstance(data, dict) and 'tracks' in data else data
  else:
    with path.open(newline='') as f:
      rows = list(csv.DictReader(f))
  by_key = {}
  for row in rows:
    sample = str(row.get('sample_id', row.get('window', row.get('episode', 0))))
    t = int(row.get('t', row.get('step', 0)))
    by_key[(sample, t)] = {k: to_float(v) for k, v in row.items()}
  return by_key


def to_float(value):
  try:
    return float(value)
  except (TypeError, ValueError):
    return value


def as_bool(value):
  if isinstance(value, str):
    return value.strip().lower() in ('1', 'true', 't', 'yes', 'y')
  return bool(value)


def command_object_metrics(args):
  real = load_tracks(args.real_tracks)
  wm = load_tracks(args.wm_tracks)
  keys = sorted(set(real) & set(wm))
  if not keys:
    raise RuntimeError('No overlapping (sample_id, t) rows between track files.')
  fields = [x.strip() for x in args.fields.split(',') if x.strip()]
  metrics = {'matched_points': int(len(keys))}
  for field in fields:
    pairs = [
        (wm[key].get(field), real[key].get(field))
        for key in keys
        if isinstance(wm[key].get(field), (int, float)) and isinstance(real[key].get(field), (int, float))
    ]
    if not pairs:
      continue
    pred = np.asarray([x for x, _ in pairs], np.float64)
    target = np.asarray([y for _, y in pairs], np.float64)
    metrics[f'{field}_mae'] = float(np.mean(np.abs(pred - target)))
    metrics[f'{field}_rmse'] = float(np.sqrt(np.mean(np.square(pred - target))))
    if field.endswith('_sign'):
      metrics[f'{field}_accuracy'] = float(np.mean(pred == target))

  for event in ('wall_bounce', 'paddle_collision', 'miss_event'):
    pairs = [
        (as_bool(wm[key].get(event)), as_bool(real[key].get(event)))
        for key in keys
        if event in wm[key] and event in real[key]
    ]
    if pairs:
      metrics.update({
          f'{event}/{k}': v
          for k, v in binary_stats(
              np.asarray([x for x, _ in pairs]),
              np.asarray([y for _, y in pairs])).items()
      })

  if 'ball_x' in fields and 'ball_y' in fields:
    metrics.update(object_divergence(real, wm, keys, args.divergence_px))

  write_json(args.output_dir / 'object_track_metrics.json', metrics)
  print(f'Wrote {args.output_dir / "object_track_metrics.json"}')


def object_divergence(real, wm, keys, threshold):
  grouped = defaultdict(list)
  for sample, t in keys:
    grouped[sample].append(t)
  divs = []
  for sample, ts in grouped.items():
    for t in sorted(ts):
      key = (sample, t)
      if not all(k in real[key] and k in wm[key] for k in ('ball_x', 'ball_y')):
        continue
      dist = math.hypot(
          float(wm[key]['ball_x']) - float(real[key]['ball_x']),
          float(wm[key]['ball_y']) - float(real[key]['ball_y']))
      if dist > threshold:
        divs.append(int(t))
        break
  return {
      'object_divergence_px': float(threshold),
      'object_divergence_horizon_mean': safe_mean(divs),
      'object_diverged_windows': int(len(divs)),
  }


def command_ram_tracks(args):
  chunks = load_replay_chunks(args.replay_dir, args.limit_chunks)
  rows = []
  sample_id = 0
  for path, chunk in chunks:
    if 'ram' not in chunk:
      raise KeyError(f'{path} has no ram field. Recreate replay with --include-ram.')
    chunk_rows = []
    for t, ram in enumerate(chunk['ram']):
      row = pong_objects_from_ram(ram)
      row.update({
          'sample_id': str(sample_id),
          'chunk': str(path),
          't': int(t),
          'reward': float(chunk['reward'][t]) if 'reward' in chunk else 0.0,
      })
      chunk_rows.append(row)
    annotate_pong_events(chunk_rows)
    rows.extend(chunk_rows)
    sample_id += 1
  ensure_dir(args.output_dir)
  write_tracks_csv(args.output_dir / 'real_ram_tracks.csv', rows)
  print(f'Wrote {args.output_dir / "real_ram_tracks.csv"}')


def decode_signed_ram_byte(raw: int):
  raw = int(raw) % 256
  return raw - 256 if raw >= 128 else raw


def pong_objects_from_ram(ram):
  ram = np.asarray(ram)
  vx = decode_signed_ram_byte(ram[PONG_DIMS['v_x']])
  vy = decode_signed_ram_byte(ram[PONG_DIMS['v_y']])
  return {
      'ball_x': int(ram[PONG_DIMS['ball_x']]),
      'ball_y': int(ram[PONG_DIMS['ball_y']]),
      'ball_vx': int(vx),
      'ball_vy': int(vy),
      'ball_vx_sign': int(np.sign(vx)),
      'ball_vy_sign': int(np.sign(vy)),
      'player_paddle_y': int(ram[PONG_DIMS['paddle_player_y']]),
      'opponent_paddle_y': int(ram[PONG_DIMS['paddle_cpu']]),
  }


def annotate_pong_events(rows):
  prev = None
  for row in rows:
    row['miss_event'] = bool(abs(float(row.get('reward', 0.0))) > 0)
    row['wall_bounce'] = False
    row['paddle_collision'] = False
    if prev is not None:
      row['wall_bounce'] = (
          int(row['ball_vy_sign']) != int(prev['ball_vy_sign']) and
          int(row['ball_vy_sign']) != 0 and int(prev['ball_vy_sign']) != 0)
      vx_flip = (
          int(row['ball_vx_sign']) != int(prev['ball_vx_sign']) and
          int(row['ball_vx_sign']) != 0 and int(prev['ball_vx_sign']) != 0)
      near_player = abs(float(row['ball_y']) - float(row['player_paddle_y'])) <= 16
      near_opponent = abs(float(row['ball_y']) - float(row['opponent_paddle_y'])) <= 16
      row['paddle_collision'] = bool(vx_flip and (near_player or near_opponent))
    prev = row


def write_tracks_csv(path: pathlib.Path, rows: list[dict[str, Any]]):
  keys = sorted({key for row in rows for key in row})
  with path.open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)


def build_parser():
  parser = argparse.ArgumentParser()
  sub = parser.add_subparsers(dest='cmd', required=True)

  p = sub.add_parser('rollout-prediction')
  p.add_argument('--wm', action='append', required=True, help='name=/path/to/checkpoint')
  p.add_argument('--eval-replay-dir', type=pathlib.Path, required=True)
  p.add_argument('--output-dir', type=pathlib.Path, required=True)
  p.add_argument('--task', default='atari100k_pong')
  p.add_argument('--jax-platform', default='gpu')
  p.add_argument('--horizons', default='1,5,15,30,50')
  p.add_argument('--context', type=int, default=5)
  p.add_argument('--samples', type=int, default=512)
  p.add_argument('--limit-chunks', type=int, default=None)
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--reward-threshold', type=float, default=0.5)
  p.add_argument('--export-wm-frames', type=int, default=0)
  p.set_defaults(func=command_rollout_prediction)

  p = sub.add_parser('visual-prediction')
  p.add_argument('--wm', action='append', required=True, help='name=/path/to/checkpoint')
  p.add_argument('--eval-replay-dir', type=pathlib.Path, required=True)
  p.add_argument('--output-dir', type=pathlib.Path, required=True)
  p.add_argument('--task', default='atari100k_pong')
  p.add_argument('--jax-platform', default='gpu')
  p.add_argument('--horizons', default='12,32,48')
  p.add_argument('--context', type=int, default=5)
  p.add_argument('--samples', type=int, default=512)
  p.add_argument('--limit-chunks', type=int, default=None)
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--pixel-scale', choices=['minus-one-one', 'zero-one', 'uint8'],
                 default='minus-one-one',
                 help='Pixel scale used before MSE. minus-one-one matches DIAMOND-style reports.')
  p.add_argument('--export-wm-frames', type=int, default=0)
  p.set_defaults(func=command_visual_prediction)

  p = sub.add_parser('policy-correlation')
  p.add_argument('--wm', action='append', required=True, help='name=/path/to/checkpoint')
  p.add_argument('--policy', action='append', required=True, help='name=/path/to/policy_checkpoint')
  p.add_argument('--output-dir', type=pathlib.Path, required=True)
  p.add_argument('--task', default='atari100k_pong')
  p.add_argument('--jax-platform', default='gpu')
  p.add_argument('--episodes', type=int, default=5)
  p.add_argument('--max-steps', type=int, default=27000)
  p.set_defaults(func=command_policy_correlation)

  p = sub.add_parser('ram-tracks')
  p.add_argument('--replay-dir', type=pathlib.Path, required=True)
  p.add_argument('--output-dir', type=pathlib.Path, required=True)
  p.add_argument('--limit-chunks', type=int, default=None)
  p.set_defaults(func=command_ram_tracks)

  p = sub.add_parser('object-metrics')
  p.add_argument('--real-tracks', type=pathlib.Path, required=True)
  p.add_argument('--wm-tracks', type=pathlib.Path, required=True)
  p.add_argument('--output-dir', type=pathlib.Path, required=True)
  p.add_argument('--fields', default='ball_x,ball_y,ball_vx_sign,ball_vy_sign,player_paddle_y,opponent_paddle_y')
  p.add_argument('--divergence-px', type=float, default=8.0)
  p.set_defaults(func=command_object_metrics)
  return parser


def main(argv=None):
  args = build_parser().parse_args(argv)
  ensure_dir(args.output_dir)
  args.func(args)


if __name__ == '__main__':
  main()
