#!/usr/bin/env python3
"""Headless Torch WM policy rollouts for Pong video inspection."""

from __future__ import annotations

import argparse
import os
import pathlib
import random

import numpy as np

from play_rollout_common import (
    PixelRLContext,
    ensure_dir,
    first_bool,
    first_float,
    import_make_torch_wm_env,
    load_pixel_rl_policy,
    obs_to_policy_tensor,
    obs_to_uint8_frame,
    parse_extra,
    parse_project_wm,
    resolve_policy_checkpoint,
    save_video,
    write_json,
    action_tensor,
    require_torch,
)


def run_policy_rollout(env, policy, *, horizon: int, device: str, size: int):
  torch = require_torch()
  obs, _info = env.reset()
  policy.reset()
  frames = [obs_to_uint8_frame(obs, size)]
  actions = []
  rewards = []
  done = []
  trunc = []
  infos = []
  with torch.no_grad():
    for _ in range(int(horizon)):
      image = obs_to_policy_tensor(obs, device)
      action, value, _ = policy.act(image)
      action_int = int(action.reshape(-1)[0].detach().cpu().item())
      next_obs, reward, done_value, trunc_value, info = env.step(action_tensor(action_int, device))
      done_flag = first_bool(done_value)
      trunc_flag = first_bool(trunc_value)
      final_obs = info.get('final_observation') if isinstance(info, dict) else None
      render_obs = final_obs if final_obs is not None and (done_flag or trunc_flag) else next_obs
      frames.append(obs_to_uint8_frame(render_obs, size))
      actions.append(action_int)
      rewards.append(first_float(reward))
      done.append(done_flag)
      trunc.append(trunc_flag)
      infos.append({'value': float(value.reshape(-1)[0].detach().cpu().item())})
      obs = next_obs
  return {
      'frames': np.asarray(frames, np.uint8),
      'actions': np.asarray(actions, np.int64),
      'rewards': np.asarray(rewards, np.float32),
      'done': np.asarray(done, bool),
      'trunc': np.asarray(trunc, bool),
      'infos': infos,
  }


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--wm', action='append', required=True,
                      help='Repeatable project:name=/path/to/checkpoint, e.g. twister:twister50=/path/ckpt.')
  parser.add_argument('--policy', required=True,
                      help='name=/path/to/rl-in-pixel-env policy checkpoint or run dir.')
  parser.add_argument('--policy-format', choices=['pixel_rl', 'diamond_actor', 'sb3_atari'], default='pixel_rl',
                      help='Controller policy checkpoint format.')
  parser.add_argument('--diamond-root', default='',
                      help='DIAMOND repository root for --policy-format diamond_actor.')
  parser.add_argument('--bootstrap-dataset', default='')
  parser.add_argument('--output-dir', type=pathlib.Path, required=True)
  parser.add_argument('--env-id', default='PongNoFrameskip-v4')
  parser.add_argument('--episodes', type=int, default=3)
  parser.add_argument('--horizon', type=int, default=512)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--device', default='cuda')
  parser.add_argument('--fps', type=int, default=15)
  parser.add_argument('--size', type=int, default=256)
  parser.add_argument('--wm-initial-source', default='dataset')
  parser.add_argument('--wm-respect-terminal', action='store_true')
  parser.add_argument('--reward-threshold', type=float, default=0.5)
  parser.add_argument('--deterministic-policy', action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument('--extra', action='append', default=[],
                      help='Global project adapter extra key=value. Repeatable.')
  args = parser.parse_args(argv)

  torch = require_torch()

  if args.horizon <= 0:
    raise ValueError('--horizon must be positive.')
  if args.episodes <= 0:
    raise ValueError('--episodes must be positive.')

  policy_name, policy_path = args.policy.split('=', 1) if '=' in args.policy else ('policy', args.policy)
  policy_ckpt = resolve_policy_checkpoint(policy_path)
  wm_specs = [parse_project_wm(value) for value in args.wm]
  ensure_dir(args.output_dir)

  write_json(args.output_dir / 'manifest.json', {
      'env_id': args.env_id,
      'policy': policy_name,
      'policy_ckpt': str(policy_ckpt),
      'policy_format': args.policy_format,
      'bootstrap_dataset': str(pathlib.Path(args.bootstrap_dataset).expanduser()) if args.bootstrap_dataset else '',
      'wm_initial_source': args.wm_initial_source,
      'episodes': int(args.episodes),
      'horizon': int(args.horizon),
      'fps': int(args.fps),
      'size': int(args.size),
      'world_models': [
          {'project': spec.project, 'name': spec.name, 'checkpoint': str(spec.checkpoint)}
          for spec in wm_specs],
  })

  for episode in range(int(args.episodes)):
    episode_seed = int(args.seed) + episode
    random.seed(episode_seed)
    np.random.seed(episode_seed)
    torch.manual_seed(episode_seed)
    episode_dir = args.output_dir / f'ep{episode:02d}_seed{episode_seed}'
    write_json(episode_dir / 'manifest.json', {
        'episode': int(episode),
        'seed': int(episode_seed),
        'bootstrap_source': args.wm_initial_source,
    })

    for spec in wm_specs:
      project_root, make_torch_wm_env = import_make_torch_wm_env(spec.project, spec.project_root)
      extra = parse_extra(list(args.extra))
      extra['wm_initial_source'] = args.wm_initial_source
      extra['video_max_frames'] = int(args.horizon) + 1
      if args.bootstrap_dataset:
        extra['wm_bootstrap_dataset'] = args.bootstrap_dataset
        extra['bootstrap_dataset'] = args.bootstrap_dataset
        extra['warmup_dataset'] = args.bootstrap_dataset
      component_mode = bool(extra.get('denoiser_ckpt')) and bool(extra.get('rew_end_model_ckpt'))
      ctx = PixelRLContext(
          env_name=args.env_id,
          seed=episode_seed,
          num_envs=1,
          device=args.device,
          wm_checkpoint='' if component_mode else str(spec.checkpoint),
          wm_horizon=int(args.horizon) + 1,
          wm_reward_quantize_threshold=float(args.reward_threshold),
          wm_respect_terminal=bool(args.wm_respect_terminal),
          extra=extra,
      )
      cwd = pathlib.Path.cwd()
      os.chdir(project_root)
      env = make_torch_wm_env(ctx)
      try:
        num_actions = int(getattr(env, 'num_actions', 6))
        if args.policy_format == 'pixel_rl':
          policy = load_pixel_rl_policy(
              policy_ckpt,
              num_actions=num_actions,
              device=args.device,
              deterministic=bool(args.deterministic_policy))
        elif args.policy_format == 'diamond_actor':
          from diamond_public_policy import load_diamond_public_policy

          policy = load_diamond_public_policy(
              policy_ckpt,
              num_actions=num_actions,
              device=args.device,
              deterministic=bool(args.deterministic_policy),
              diamond_root=(args.diamond_root or None))
        elif args.policy_format == 'sb3_atari':
          from sb3_atari_policy import load_sb3_atari_policy

          policy = load_sb3_atari_policy(
              policy_ckpt,
              device=args.device,
              deterministic=bool(args.deterministic_policy))
        else:
          raise ValueError(f'Unsupported policy format: {args.policy_format}')
        rec = run_policy_rollout(env, policy, horizon=args.horizon, device=args.device, size=args.size)
      finally:
        close = getattr(env, 'close', None)
        if callable(close):
          close()
        os.chdir(cwd)

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
          'project': spec.project,
          'wm': spec.name,
          'checkpoint': str(spec.checkpoint),
          'policy': policy_name,
          'policy_ckpt': str(policy_ckpt),
          'policy_format': args.policy_format,
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


if __name__ == '__main__':
  main()
