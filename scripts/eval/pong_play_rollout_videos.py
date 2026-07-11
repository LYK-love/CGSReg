#!/usr/bin/env python3
"""Headless Dreamer wm-play rollouts for Pong video inspection."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import sys
from dataclasses import dataclass
from typing import Any

import imageio.v3 as iio
import numpy as np
from PIL import Image


ROOT = pathlib.Path(__file__).resolve().parents[2]


def default_dreamer_root() -> pathlib.Path:
  env_value = os.environ.get('WM_EVAL_DREAMER_ROOT') or os.environ.get('DREAMER_ROOT')
  if env_value:
    return pathlib.Path(env_value).expanduser().resolve()
  sibling = ROOT.parent / 'dreamerv3-reborn'
  if sibling.exists():
    return sibling.resolve()
  return pathlib.Path('dreamerv3-reborn').resolve()


DREAMER_ROOT = default_dreamer_root()
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(DREAMER_ROOT))
sys.path.insert(2, str(DREAMER_ROOT.parent))


@dataclass(frozen=True)
class WMSpec:
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


def resize_frame(frame: np.ndarray, size: int) -> np.ndarray:
  frame = np.asarray(frame)
  while frame.ndim > 3:
    frame = frame[0]
  if frame.ndim == 2:
    frame = np.repeat(frame[..., None], 3, axis=-1)
  if frame.shape[-1] == 1:
    frame = np.repeat(frame, 3, axis=-1)
  if frame.shape[-1] > 3:
    frame = frame[..., :3]
  if frame.dtype != np.uint8:
    if frame.size and np.nanmax(frame) <= 1.5:
      frame = frame * 255.0
    frame = np.clip(frame, 0, 255).astype(np.uint8)
  if size > 0 and (frame.shape[0] != size or frame.shape[1] != size):
    image = Image.fromarray(frame)
    frame = np.asarray(image.resize((size, size), resample=Image.NEAREST), np.uint8)
  return frame


def save_video(path: pathlib.Path, frames: np.ndarray, fps: int):
  ensure_dir(path.parent)
  iio.imwrite(path, np.asarray(frames, np.uint8), fps=int(fps))


def policy_action(policy_runner, policy_state) -> int:
  if callable(getattr(policy_runner, 'act_image', None)):
    return int(policy_runner.act_image(policy_state.obs['image']))
  if getattr(policy_runner, 'policy_observation_only', False):
    return int(policy_runner.policy_action(policy_state.obs))
  return int(policy_runner.policy_action(policy_state.feat))


def rollout_one(
    env,
    policy_runner,
    bootstrap_obs: dict[str, Any],
    horizon: int,
    size: int,
):
  obs = env.bootstrap_from_observation(bootstrap_obs)
  policy_state = policy_runner.init_real_state(obs)

  frames = [resize_frame(obs['image'], size)]
  actions = []
  rewards = []
  done = []
  trunc = []
  infos = []

  for _ in range(int(horizon)):
    action = policy_action(policy_runner, policy_state)
    result = env.step(action)
    actions.append(int(action))
    rewards.append(float(result.reward))
    done.append(bool(result.done))
    trunc.append(bool(result.trunc))
    infos.append(dict(result.info or {}))
    frames.append(resize_frame(result.obs['image'], size))
    policy_state = policy_runner.advance_real_state(
        policy_state, action, result.obs)
  return {
      'frames': np.asarray(frames, np.uint8),
      'actions': np.asarray(actions, np.int32),
      'rewards': np.asarray(rewards, np.float32),
      'done': np.asarray(done, bool),
      'trunc': np.asarray(trunc, bool),
      'infos': infos,
  }


def main(argv=None):
  parser = argparse.ArgumentParser(
      description='Record headless Dreamer wm-play rollouts without web UI.')
  parser.add_argument('--config', type=pathlib.Path, required=True)
  parser.add_argument('--wm', action='append', required=True,
                      help='name=/path/to/wm_checkpoint. Repeatable.')
  parser.add_argument('--policy', required=True,
                      help='name=/path/to/policy_checkpoint.')
  parser.add_argument('--policy-format', choices=['dreamer', 'diamond_actor', 'sb3_atari'], default='dreamer',
                      help='Controller policy checkpoint format.')
  parser.add_argument('--diamond-root', default='',
                      help='DIAMOND repository root for --policy-format diamond_actor.')
  parser.add_argument('--bootstrap-dataset', type=pathlib.Path, required=True)
  parser.add_argument('--output-dir', type=pathlib.Path, required=True)
  parser.add_argument('--episodes', type=int, default=3)
  parser.add_argument('--horizon', type=int, default=512)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--task', default='atari100k_pong')
  parser.add_argument('--jax-platform', choices=['cpu', 'cuda'], default='cuda')
  parser.add_argument('--fps', type=int, default=15)
  parser.add_argument('--size', type=int, default=256)
  args = parser.parse_args(argv)

  if args.horizon <= 0:
    raise ValueError('--horizon must be positive.')
  if args.episodes <= 0:
    raise ValueError('--episodes must be positive.')

  from dreamerv3 import main as dreamer_main
  from dreamerv3.interactive import dreamer_adapter
  from dreamerv3.wm_env import _BootstrapObsDataset

  config = dreamer_adapter.load_config(args.config, args.jax_platform)
  config = config.update(task=args.task)
  real_env = dreamer_adapter.DreamerRealEnv(dreamer_main.make_env(config, 0))
  dataset = _BootstrapObsDataset(args.bootstrap_dataset)
  wm_specs = [WMSpec(*parse_name_path(x)) for x in args.wm]
  policy_name, policy_ckpt = parse_name_path(args.policy)

  ensure_dir(args.output_dir)
  write_json(args.output_dir / 'manifest.json', {
      'config': str(args.config.expanduser().resolve()),
      'dreamer_root': str(DREAMER_ROOT),
      'bootstrap_dataset': str(args.bootstrap_dataset.expanduser().resolve()),
      'bootstrap_source': 'dataset_shared_observation',
      'policy': policy_name,
      'policy_ckpt': str(policy_ckpt),
      'policy_format': args.policy_format,
      'world_models': [
          {'name': spec.name, 'checkpoint': str(spec.checkpoint)}
          for spec in wm_specs],
      'episodes': int(args.episodes),
      'horizon': int(args.horizon),
      'video_frames_per_rollout': int(args.horizon) + 1,
      'seed': int(args.seed),
      'task': args.task,
      'jax_platform': args.jax_platform,
      'fps': int(args.fps),
      'size': int(args.size),
      'play_adapter': 'dreamerv3.interactive.dreamer_adapter',
  })

  runners = {
      spec.name: dreamer_adapter.load_model_runner(config, str(spec.checkpoint))
      for spec in wm_specs
  }
  if args.policy_format == 'dreamer':
    policy_runner = dreamer_adapter.load_policy_runner(
        config, str(policy_ckpt), args.jax_platform)
  elif args.policy_format == 'diamond_actor':
    from diamond_public_policy import load_diamond_public_policy
    import torch
    from play_rollout_common import obs_to_policy_tensor

    class _ExternalDreamerPolicy:
      policy_observation_only = True

      def __init__(self, policy):
        self.policy = policy

      def reseed(self, seed: int):
        torch.manual_seed(int(seed))
        self.policy.reset()

      def init_real_state(self, obs):
        class _State:
          pass
        state = _State()
        state.obs = obs
        return state

      def advance_real_state(self, policy_state, action, obs):
        policy_state.obs = obs
        return policy_state

      def act_image(self, obs_image):
        tensor = obs_to_policy_tensor({'image': obs_image}, 'cuda' if args.jax_platform == 'cuda' else 'cpu')
        action, _value, _info = self.policy.act(tensor)
        return int(action.reshape(-1)[0].detach().cpu().item())

    policy_runner = _ExternalDreamerPolicy(load_diamond_public_policy(
        policy_ckpt,
        num_actions=6,
        device='cuda' if args.jax_platform == 'cuda' else 'cpu',
        deterministic=True,
        diamond_root=(args.diamond_root or None)))
  elif args.policy_format == 'sb3_atari':
    import torch
    from play_rollout_common import obs_to_policy_tensor
    from sb3_atari_policy import load_sb3_atari_policy

    class _ExternalDreamerPolicy:
      policy_observation_only = True

      def __init__(self, policy):
        self.policy = policy

      def reseed(self, seed: int):
        torch.manual_seed(int(seed))
        self.policy.reset()

      def init_real_state(self, obs):
        class _State:
          pass
        state = _State()
        state.obs = obs
        return state

      def advance_real_state(self, policy_state, action, obs):
        policy_state.obs = obs
        return policy_state

      def act_image(self, obs_image):
        tensor = obs_to_policy_tensor({'image': obs_image}, 'cuda' if args.jax_platform == 'cuda' else 'cpu')
        action, _value, _info = self.policy.act(tensor)
        return int(action.reshape(-1)[0].detach().cpu().item())

    policy_runner = _ExternalDreamerPolicy(load_sb3_atari_policy(
        policy_ckpt,
        device='cuda' if args.jax_platform == 'cuda' else 'cpu',
        deterministic=True))
  else:
    raise ValueError(f'Unsupported policy format: {args.policy_format}')

  try:
    for episode in range(int(args.episodes)):
      episode_seed = int(args.seed) + episode
      random.seed(episode_seed)
      np.random.seed(episode_seed)
      bootstrap_obs = dataset.sample()
      episode_dir = args.output_dir / f'ep{episode:02d}_seed{episode_seed}'
      ensure_dir(episode_dir)
      Image.fromarray(resize_frame(bootstrap_obs['image'], args.size)).save(
          episode_dir / 'bootstrap_dataset_obs.png')
      episode_metadata = {
          'episode': int(episode),
          'seed': int(episode_seed),
          'source': 'dataset',
          'image_shape': list(np.asarray(bootstrap_obs['image']).shape),
          'bootstrap_source': 'dataset_shared_observation',
      }
      write_json(episode_dir / 'bootstrap_dataset_obs.json', episode_metadata)
      write_json(episode_dir / 'manifest.json', episode_metadata)

      for spec in wm_specs:
        runner = runners[spec.name]
        runner.reseed(episode_seed)
        policy_runner.reseed(episode_seed)
        env = dreamer_adapter.DreamerWorldModelEnv(
            runner,
            real_env,
            int(args.horizon),
            initial_source='real')
        rec = rollout_one(
            env,
            policy_runner,
            bootstrap_obs,
            int(args.horizon),
            int(args.size))
        video_path = episode_dir / 'videos' / f'{spec.name}.mp4'
        save_video(video_path, rec['frames'], args.fps)
        rollout_npz_path = episode_dir / 'rollouts' / f'{spec.name}.npz'
        ensure_dir(rollout_npz_path.parent)
        np.savez_compressed(
            rollout_npz_path,
            frames=rec['frames'],
            actions=rec['actions'],
            rewards=rec['rewards'],
            done=rec['done'],
            trunc=rec['trunc'])
        write_json(episode_dir / 'rollouts' / f'{spec.name}.json', {
            'episode': int(episode),
            'seed': int(episode_seed),
            'wm': spec.name,
            'checkpoint': str(spec.checkpoint),
            'policy': policy_name,
            'policy_ckpt': str(policy_ckpt),
            'policy_format': args.policy_format,
            'bootstrap_source': 'dataset_shared_observation',
            'horizon': int(args.horizon),
            'video': str(video_path),
            'num_frames': int(rec['frames'].shape[0]),
            'num_actions': int(rec['actions'].shape[0]),
            'return': float(np.sum(rec['rewards'])),
            'actions': rec['actions'].tolist(),
            'rewards': rec['rewards'].tolist(),
            'done': rec['done'].tolist(),
            'trunc': rec['trunc'].tolist(),
            'infos': rec['infos'],
        })
        print(f'Wrote {video_path}', flush=True)
  finally:
    real_env.close()


if __name__ == '__main__':
  main()
