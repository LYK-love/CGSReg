#!/usr/bin/env python3
"""Roll out one external Torch pixel world model under a fixed action sequence."""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from play_rollout_common import (
    PROJECT_PATHS,
    PixelRLContext,
    action_tensor,
    first_bool,
    first_float,
    import_make_torch_wm_env,
    obs_to_uint8_frame,
    parse_extra,
    prepare_checkpoint,
    require_torch,
)


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--project', required=True, choices=sorted(PROJECT_PATHS))
  parser.add_argument('--project-root', type=pathlib.Path, default=None)
  parser.add_argument('--checkpoint', required=True)
  parser.add_argument('--output', type=pathlib.Path, required=True)
  parser.add_argument('--actions-json', required=True)
  parser.add_argument('--env-name', default='PongNoFrameskip-v4')
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--device', default='cuda')
  parser.add_argument('--horizon', type=int, required=True)
  parser.add_argument('--wm-env-horizon', type=int, default=None)
  parser.add_argument('--wm-initial-source', default='real')
  parser.add_argument('--wm-bootstrap-dataset', default='')
  parser.add_argument('--warmup-dataset', default='')
  parser.add_argument('--warmup-steps', type=int, default=5000)
  parser.add_argument('--config-path', default='')
  parser.add_argument('--wm-env-name', default='')
  parser.add_argument('--sample-mode', default='probs')
  parser.add_argument('--reward-threshold', type=float, default=0.5)
  parser.add_argument('--respect-terminal', action='store_true')
  parser.add_argument('--extra', action='append', default=[])
  args = parser.parse_args(argv)

  torch = require_torch()
  actions = [int(x) for x in json.loads(args.actions_json)]
  if len(actions) < args.horizon:
    raise ValueError(f'Need at least {args.horizon} actions, got {len(actions)}.')

  project_root, make_torch_wm_env = import_make_torch_wm_env(
      args.project,
      args.project_root,
      clear_project_modules=False)

  extra = parse_extra(args.extra)
  extra['wm_initial_source'] = args.wm_initial_source
  if args.wm_bootstrap_dataset:
    extra['wm_bootstrap_dataset'] = args.wm_bootstrap_dataset
    extra['bootstrap_dataset'] = args.wm_bootstrap_dataset
  if args.warmup_dataset:
    extra['warmup_dataset'] = args.warmup_dataset
  if args.warmup_steps:
    extra['warmup_steps'] = int(args.warmup_steps)
  if args.config_path:
    extra['config_path'] = args.config_path
  if args.wm_env_name:
    extra['wm_env_name'] = args.wm_env_name
  if args.sample_mode:
    extra['sample_mode'] = args.sample_mode

  checkpoint_path = prepare_checkpoint(args.project, args.checkpoint, args.output)

  ctx = PixelRLContext(
      env_name=args.env_name,
      seed=int(args.seed),
      num_envs=1,
      device=args.device,
      wm_checkpoint=str(checkpoint_path),
      wm_horizon=int(args.wm_env_horizon or args.horizon + 1),
      wm_reward_quantize_threshold=float(args.reward_threshold),
      wm_respect_terminal=bool(args.respect_terminal),
      extra=extra,
  )

  frames = []
  rewards = []
  dones = []
  truncs = []
  env = make_torch_wm_env(ctx)
  try:
    obs, _info = env.reset()
    reset_frame = obs_to_uint8_frame(obs)
    for action in actions[:args.horizon]:
      obs, rew, done, trunc, info = env.step(action_tensor(action, args.device))
      done_flag = first_bool(done)
      trunc_flag = first_bool(trunc)
      final_obs = info.get('final_observation') if isinstance(info, dict) else None
      frame = (
          obs_to_uint8_frame(final_obs)
          if final_obs is not None and (done_flag or trunc_flag)
          else obs_to_uint8_frame(obs)
      )
      frames.append(frame)
      rewards.append(first_float(rew))
      dones.append(done_flag)
      truncs.append(trunc_flag)
  finally:
    close = getattr(env, 'close', None)
    if callable(close):
      close()

  args.output.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
      args.output,
      frames=np.asarray(frames, dtype=np.uint8),
      reset_frame=np.asarray(reset_frame, dtype=np.uint8),
      rewards=np.asarray(rewards, dtype=np.float32),
      dones=np.asarray(dones, dtype=bool),
      truncs=np.asarray(truncs, dtype=bool),
      actions=np.asarray(actions[:args.horizon], dtype=np.int64),
      metadata=np.asarray(json.dumps({
          'project': args.project,
          'project_root': str(project_root),
          'checkpoint': str(pathlib.Path(args.checkpoint).expanduser().resolve()),
          'effective_checkpoint': str(checkpoint_path),
          'env_name': args.env_name,
          'seed': int(args.seed),
          'wm_initial_source': args.wm_initial_source,
          'wm_env_horizon': int(ctx.wm_horizon),
          'device': args.device,
          'extra': extra,
      })),
  )
  print(f'Wrote {args.output}', flush=True)


if __name__ == '__main__':
  main()
