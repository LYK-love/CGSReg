#!/usr/bin/env python3
"""Headless TWISTER WM rollouts controlled by a native TWISTER policy checkpoint."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import imageio.v2 as iio
import numpy as np
import torch


ROOT = pathlib.Path(__file__).resolve().parents[2]
PROJECTS_ROOT = ROOT.parent


DEFAULT_WMS = [
    (
        'twister_no_ac_cpc_w0',
        'runs/pong_offline_regu_no_ac_cpc_sweep/logdir/'
        'twister_pong_offline_no_ac_cpc_static_diamond_w0_model=base/'
        'checkpoints/checkpoints_100000.ckpt',
    ),
    (
        'twister_ac_cpc_w0p003',
        'runs/pong_offline_regu_sweep/logdir/'
        'twister_pong_offline_regu_model=base_mask1_spatial_0p003/'
        'checkpoints/latest.ckpt',
    ),
    (
        'twister_ac_cpc_w0p005',
        'runs/pong_offline_regu_sweep/logdir/'
        'twister_pong_offline_regu_model=base_mask1_spatial_0p005/'
        'checkpoints/latest.ckpt',
    ),
    (
        'twister_ac_cpc_w0p02',
        'runs/pong_offline_regu_sweep/logdir/'
        'twister_pong_offline_regu_model=base_mask1_spatial_0p02/'
        'checkpoints/latest.ckpt',
    ),
    (
        'twister_ac_cpc_w0p05',
        'runs/pong_offline_regu_sweep/logdir/'
        'twister_pong_offline_regu_model=base_mask1_spatial_0p05/'
        'checkpoints/latest.ckpt',
    ),
]


def ensure_dir(path: pathlib.Path):
  path.mkdir(parents=True, exist_ok=True)


def write_json(path: pathlib.Path, data):
  ensure_dir(path.parent)
  path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def save_video(path: pathlib.Path, frames: np.ndarray, fps: int):
  ensure_dir(path.parent)
  try:
    iio.mimsave(path, list(frames), fps=int(fps), macro_block_size=1)
  except Exception:
    import imageio.v3 as iio_v3

    iio_v3.imwrite(path, frames, fps=int(fps), codec='libx264')


def parse_wm(value: str, twister_root: pathlib.Path) -> tuple[str, pathlib.Path]:
  if '=' in value:
    name, raw_path = value.split('=', 1)
  else:
    raw_path = value
    name = pathlib.Path(raw_path).parent.parent.name
  path = pathlib.Path(raw_path).expanduser()
  if not path.is_absolute():
    path = twister_root / path
  return name, path.resolve()


def add_twister_paths(twister_root: pathlib.Path):
  twister_root = twister_root.expanduser().resolve()
  if not twister_root.exists():
    raise FileNotFoundError(f'TWISTER root does not exist: {twister_root}')
  for path in (twister_root, PROJECTS_ROOT / 'wm-play-common/src'):
    if path.exists():
      value = str(path.resolve())
      if value not in sys.path:
        sys.path.insert(0, value)


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--twister-root', type=pathlib.Path,
                      default=PROJECTS_ROOT / 'twister')
  parser.add_argument('--wm', action='append',
                      help='Repeatable name=/path/to/twister.ckpt. If omitted, uses the current fine sweep.')
  parser.add_argument('--policy-checkpoint', type=pathlib.Path,
                      default=PROJECTS_ROOT / 'twister/callbacks/atari100k/atari100k-pong/'
                      'checkpoints_epoch_50_step_100000.ckpt')
  parser.add_argument('--policy-name', default='twister_dyna_repro')
  parser.add_argument('--bootstrap-dataset', type=pathlib.Path,
                      default=ROOT / 'artifacts/bootstrap_datasets/diamond_pong_test_limit20/twister/test')
  parser.add_argument('--output-dir', type=pathlib.Path,
                      default=ROOT / 'artifacts/twister_fine_w_sweep_dyna_policy_h512_rollouts')
  parser.add_argument('--env-id', default='PongNoFrameskip-v4')
  parser.add_argument('--episodes', type=int, default=5)
  parser.add_argument('--horizon', type=int, default=512)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--device', default='cuda')
  parser.add_argument('--fps', type=int, default=15)
  parser.add_argument('--wm-initial-source', default='dataset',
                      choices=('real', 'prior', 'dataset'))
  parser.add_argument('--wm-respect-terminal', action=argparse.BooleanOptionalAction, default=False)
  args = parser.parse_args(argv)

  twister_root = args.twister_root.expanduser().resolve()
  add_twister_paths(twister_root)

  from interactive.twister_adapter import build_twister_session

  if args.episodes <= 0:
    raise ValueError('--episodes must be positive.')
  if args.horizon <= 0:
    raise ValueError('--horizon must be positive.')

  policy_ckpt = args.policy_checkpoint.expanduser()
  if not policy_ckpt.is_absolute():
    policy_ckpt = twister_root / policy_ckpt
  policy_ckpt = policy_ckpt.resolve()

  wm_values = args.wm or [f'{name}={path}' for name, path in DEFAULT_WMS]
  wm_specs = [parse_wm(value, twister_root) for value in wm_values]
  bootstrap_dataset = args.bootstrap_dataset.expanduser().resolve()
  output_dir = args.output_dir.expanduser().resolve()
  ensure_dir(output_dir)

  write_json(output_dir / 'manifest.json', {
      'env_id': args.env_id,
      'policy': args.policy_name,
      'policy_ckpt': str(policy_ckpt),
      'bootstrap_dataset': str(bootstrap_dataset),
      'wm_initial_source': args.wm_initial_source,
      'wm_respect_terminal': bool(args.wm_respect_terminal),
      'episodes': int(args.episodes),
      'horizon': int(args.horizon),
      'fps': int(args.fps),
      'world_models': [{'name': name, 'checkpoint': str(path)} for name, path in wm_specs],
  })

  for episode in range(int(args.episodes)):
    episode_seed = int(args.seed) + episode
    episode_dir = output_dir / f'ep{episode:02d}_seed{episode_seed}'
    write_json(episode_dir / 'manifest.json', {
        'episode': int(episode),
        'seed': int(episode_seed),
        'bootstrap_source': args.wm_initial_source,
    })

    for wm_name, wm_ckpt in wm_specs:
      session = build_twister_session(
          env_name=args.env_id,
          seed=episode_seed,
          checkpoint_args=[str(wm_ckpt)],
          wm_name_args=[wm_name],
          policy_checkpoint_args=[str(policy_ckpt)],
          policy_name_args=[args.policy_name],
          additional_policy_controller=True,
          device=args.device,
          wm_horizon=int(args.horizon) + 1,
          wm_respect_terminal=bool(args.wm_respect_terminal),
          wm_initial_source=args.wm_initial_source,
          wm_bootstrap_dataset=str(bootstrap_dataset) if args.wm_initial_source == 'dataset' else None,
      )
      try:
        session.current_backend_index = 1
        session.policy_slot_index = 0
        session.controller = args.policy_name
        session.reset()

        frames = [np.asarray(session.current_obs, dtype=np.uint8)]
        actions = []
        rewards = []
        done = []
        trunc = []
        infos = []
        for _ in range(int(args.horizon)):
          action = session.choose_action(0)
          result = session.step(action)
          frames.append(np.asarray(result.obs, dtype=np.uint8))
          actions.append(int(action))
          rewards.append(float(result.reward))
          done.append(bool(result.done))
          trunc.append(bool(result.trunc))
          infos.append(dict(result.info or {}))

        frames_arr = np.asarray(frames, np.uint8)
        video_path = episode_dir / 'videos' / f'{wm_name}.mp4'
        save_video(video_path, frames_arr, args.fps)

        rollout_path = episode_dir / 'rollouts' / f'{wm_name}.npz'
        ensure_dir(rollout_path.parent)
        np.savez_compressed(
            rollout_path,
            frames=frames_arr,
            actions=np.asarray(actions, np.int64),
            rewards=np.asarray(rewards, np.float32),
            done=np.asarray(done, bool),
            trunc=np.asarray(trunc, bool),
        )
        write_json(episode_dir / 'rollouts' / f'{wm_name}.json', {
            'episode': int(episode),
            'seed': int(episode_seed),
            'project': 'twister',
            'wm': wm_name,
            'checkpoint': str(wm_ckpt),
            'policy': args.policy_name,
            'policy_ckpt': str(policy_ckpt),
            'horizon': int(args.horizon),
            'video': str(video_path),
            'num_frames': int(frames_arr.shape[0]),
            'num_actions': len(actions),
            'return': float(np.sum(rewards)),
            'actions': actions,
            'rewards': rewards,
            'done': done,
            'trunc': trunc,
            'infos': infos,
        })
        print(f'Wrote {video_path}', flush=True)
      finally:
        close = getattr(session, 'close', None)
        if callable(close):
          close()
        del session
        if torch.cuda.is_available():
          torch.cuda.empty_cache()


if __name__ == '__main__':
  main()
