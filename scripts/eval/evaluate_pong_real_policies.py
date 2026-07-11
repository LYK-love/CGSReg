#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import statistics
import sys
import time


def _parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--policy', action='append', required=True,
                      help='name=/path/to/policy/ckpt/latest')
  parser.add_argument('--episodes', type=int, default=5)
  parser.add_argument('--cuda-device', default='0')
  parser.add_argument('--jax-platform', default='cuda')
  parser.add_argument('--max-steps-per-episode', type=int, default=30000)
  parser.add_argument('--output-dir', type=pathlib.Path, required=True)
  parser.add_argument('--autostart', action=argparse.BooleanOptionalAction,
                      default=True)
  parser.add_argument(
      '--seed-start', type=int, default=None,
      help='Use deterministic reset seeds seed_start..seed_start+episodes-1.')
  parser.add_argument(
      '--reset-seeds', default=None,
      help='Comma-separated deterministic reset seeds. Overrides --seed-start.')
  return parser.parse_args()


def _split_name_path(value):
  if '=' not in value:
    raise ValueError(f'Expected name=/path, got {value!r}')
  name, path = value.split('=', 1)
  return name, pathlib.Path(path).expanduser().resolve()


def _setup_runtime(args):
  os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)
  os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
  os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')


def _repo_root():
  root = pathlib.Path(__file__).resolve().parents[2]
  sys.path.insert(0, str(root))
  sys.path.insert(1, str(root.parent))
  sys.path.insert(2, str(root / 'notebooks'))
  return root


def _load_config(policy_ckpt, jax_platform, max_steps_per_episode):
  import elements
  from dreamerv3.interactive import dreamer_adapter

  from rollout_pong_helpers import find_config, update_config_allow_new

  config_path = find_config(policy_ckpt)
  config = dreamer_adapter.load_config(config_path, jax_platform)
  if bool(getattr(config.agent, 'pure_wm', False)):
    config = config.update(agent=config.agent.update(pure_wm=False))
  config = config.update(task='atari100k_pong')
  length = int(max_steps_per_episode) * int(config.env.atari100k.repeat)
  config = update_config_allow_new(
      elements, config, **{'env.atari100k.length': length})
  return config_path, config


def _reset_seeds(args):
  if args.reset_seeds:
    seeds = [int(x) for x in args.reset_seeds.split(',') if x.strip()]
    if len(seeds) < int(args.episodes):
      raise ValueError(
          f'Need at least {args.episodes} reset seeds, got {len(seeds)}.')
    return seeds
  if args.seed_start is None:
    return None
  return [int(args.seed_start) + i for i in range(int(args.episodes))]


class FixedResetSeed:

  def __init__(self, env, seeds):
    self.env = env
    self.seeds = list(seeds)
    self.index = 0

  @property
  def obs_space(self):
    return self.env.obs_space

  @property
  def act_space(self):
    return self.env.act_space

  def step(self, action):
    if bool(action.get('reset', False)):
      if self.index >= len(self.seeds):
        raise RuntimeError(
            f'Ran out of deterministic reset seeds after {self.index} resets.')
      self._reseed(self.seeds[self.index])
      self.index += 1
    return self.env.step(action)

  def close(self):
    close = getattr(self.env, 'close', None)
    if close:
      close()

  def _reseed(self, seed):
    import numpy as np

    ale_seed = int(seed % (2 ** 31))
    for candidate in self._env_chain():
      if hasattr(candidate, 'rng'):
        candidate.rng = np.random.default_rng(seed)
      ale = getattr(candidate, 'ale', None)
      if ale is not None and hasattr(ale, 'setInt'):
        ale.setInt(b'random_seed', ale_seed)

  def _env_chain(self):
    env = self.env
    seen = set()
    while env is not None and id(env) not in seen:
      seen.add(id(env))
      yield env
      env = getattr(env, 'env', None)


def _make_env(dreamer_main, config, autostart, reset_seeds):
  overrides = {}
  if autostart:
    overrides['autostart'] = True
  env = dreamer_main.make_env(config, 0, **overrides)
  if reset_seeds is not None:
    env = FixedResetSeed(env, reset_seeds)
  elif config.pixel_rl.random_reset_seed:
    env = dreamer_main.RandomResetSeed(env)
  return env


def _evaluate_policy(name, policy_ckpt, args):
  import elements
  import embodied
  import numpy as np

  from dreamerv3 import main as dreamer_main
  from rollout_pong_helpers import build_agent_and_load

  config_path, config = _load_config(
      policy_ckpt, args.jax_platform, args.max_steps_per_episode)
  agent = build_agent_and_load(dreamer_main, elements, config, policy_ckpt)
  reset_seeds = _reset_seeds(args)
  rows = []
  current = {'score': 0.0, 'agent_score': 0.0, 'opponent_score': 0.0, 'length': 0}
  step_count = 0
  start = time.perf_counter()

  def on_step(tran, worker):
    nonlocal current, step_count
    del worker
    step_count += 1
    if bool(tran['is_first']):
      current = {'score': 0.0, 'agent_score': 0.0, 'opponent_score': 0.0, 'length': 0}
    reward = float(tran['reward'])
    current['score'] += reward
    if reward > 0:
      current['agent_score'] += reward
    elif reward < 0:
      current['opponent_score'] += -reward
    current['length'] += 1
    if bool(tran['is_last']):
      row = {
          'policy': name,
          'episode': len(rows) + 1,
          'score': current['score'],
          'agent_score': current['agent_score'],
          'opponent_score': current['opponent_score'],
          'length': current['length'],
          'terminal': bool(tran['is_terminal']),
      }
      rows.append(row)
      print(
          f"{name} episode {len(rows):02d}/{args.episodes}: "
          f"score={row['score']:+.1f} "
          f"agent={row['agent_score']:.0f} opponent={row['opponent_score']:.0f} "
          f"length={row['length']} terminal={row['terminal']}",
          flush=True)

  def eval_policy(carry, obs):
    return agent.policy(carry, obs, mode='eval')

  def make_env():
    return _make_env(dreamer_main, config, args.autostart, reset_seeds)

  driver = embodied.Driver([make_env], parallel=False)
  driver.on_step(on_step)
  driver.reset(agent.init_policy)
  # The embodied driver emits the final transition after the configured
  # per-episode ALE horizon, so non-terminal truncated episodes contain one
  # more transition than max_steps_per_episode.
  max_total_steps = int(args.episodes) * (int(args.max_steps_per_episode) + 1)
  try:
    while len(rows) < int(args.episodes) and step_count < max_total_steps:
      driver(eval_policy, steps=1)
  finally:
    driver.close()
  elapsed = time.perf_counter() - start
  if len(rows) < int(args.episodes):
    raise RuntimeError(
        f'{name}: collected only {len(rows)}/{args.episodes} episodes '
        f'within {max_total_steps} steps.')
  scores = [row['score'] for row in rows]
  lengths = [row['length'] for row in rows]
  return {
      'name': name,
      'checkpoint': str(policy_ckpt),
      'config': str(config_path),
      'episodes': rows,
      'summary': {
          'episodes': len(rows),
          'score_mean': statistics.fmean(scores),
          'score_std_pop': statistics.pstdev(scores),
          'score_std_sample': statistics.stdev(scores) if len(scores) > 1 else 0.0,
          'score_min': min(scores),
          'score_max': max(scores),
          'length_mean': statistics.fmean(lengths),
          'elapsed_seconds': elapsed,
      },
  }


def main():
  args = _parse_args()
  _setup_runtime(args)
  _repo_root()
  args.output_dir.mkdir(parents=True, exist_ok=True)
  specs = [_split_name_path(x) for x in args.policy]
  results = []
  for name, path in specs:
    print(f'Evaluating {name}: {path}', flush=True)
    results.append(_evaluate_policy(name, path, args))

  payload = {
      'episodes_per_policy': int(args.episodes),
      'max_steps_per_episode': int(args.max_steps_per_episode),
      'autostart': bool(args.autostart),
      'reset_seeds': _reset_seeds(args),
      'policies': results,
  }
  json_path = args.output_dir / 'pong_real_policy_eval.json'
  json_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')

  csv_path = args.output_dir / 'pong_real_policy_eval_episodes.csv'
  with csv_path.open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(
        f, fieldnames=['policy', 'episode', 'score', 'agent_score',
                       'opponent_score', 'length', 'terminal'])
    writer.writeheader()
    for result in results:
      writer.writerows(result['episodes'])

  summary_path = args.output_dir / 'pong_real_policy_eval_summary.csv'
  with summary_path.open('w', newline='', encoding='utf-8') as f:
    fieldnames = ['policy', 'episodes', 'score_mean', 'score_std_sample',
                  'score_min', 'score_max', 'length_mean', 'checkpoint']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for result in results:
      row = {'policy': result['name'], 'checkpoint': result['checkpoint']}
      row.update({k: result['summary'][k] for k in fieldnames if k in result['summary']})
      writer.writerow(row)

  print(f'Wrote {json_path}', flush=True)
  print(f'Wrote {csv_path}', flush=True)
  print(f'Wrote {summary_path}', flush=True)


if __name__ == '__main__':
  main()
