#!/usr/bin/env python3
"""Evaluate Torch rl-in-pixel-env Pong policies in the real TWISTER Atari env."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import statistics
import sys
import time

from play_rollout_common import (
    PixelRLContext,
    add_project_paths,
    action_tensor,
    load_pixel_rl_policy,
    obs_to_policy_tensor,
    resolve_policy_checkpoint,
    require_torch,
)


ROOT = pathlib.Path(__file__).resolve().parents[2]
PROJECTS_ROOT = ROOT.parent


def parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--policy", action="append", required=True,
                      help="Repeatable name=/path/to/policy.pt or run dir.")
  parser.add_argument("--output-dir", type=pathlib.Path, required=True)
  parser.add_argument("--twister-root", type=pathlib.Path,
                      default=PROJECTS_ROOT / "twister")
  parser.add_argument("--env-id", default="PongNoFrameskip-v4")
  parser.add_argument("--episodes", type=int, default=5)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--reset-seeds", default="",
                      help="Comma-separated deterministic reset seeds. "
                           "Defaults to seed..seed+episodes-1.")
  parser.add_argument("--device", default="cuda")
  parser.add_argument("--max-steps-per-episode", type=int, default=30000)
  parser.add_argument("--deterministic-policy",
                      action=argparse.BooleanOptionalAction, default=True)
  return parser.parse_args(argv)


def reset_seeds(args):
  if args.reset_seeds:
    seeds = [int(x) for x in args.reset_seeds.split(",") if x.strip()]
    if len(seeds) < int(args.episodes):
      raise ValueError(
          f"Need at least {args.episodes} reset seeds, got {len(seeds)}.")
    return seeds[:int(args.episodes)]
  return [int(args.seed) + i for i in range(int(args.episodes))]


def split_name_path(value: str):
  if "=" not in value:
    raise ValueError(f"Expected name=/path, got {value!r}.")
  name, raw_path = value.split("=", 1)
  return name, resolve_policy_checkpoint(raw_path)


def write_json(path: pathlib.Path, data):
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def close_env(env):
  close = getattr(env, "close", None)
  if callable(close):
    close()


def evaluate_policy(name, checkpoint, make_env, args):
  seeds = reset_seeds(args)
  env = make_env(seeds[0])
  policy = load_pixel_rl_policy(
      checkpoint,
      num_actions=int(getattr(env, "num_actions", 6)),
      device=args.device,
      deterministic=bool(args.deterministic_policy),
  )
  rows = []
  start = time.perf_counter()
  torch = require_torch()
  try:
    with torch.no_grad():
      for episode, seed in enumerate(seeds):
        if episode:
          close_env(env)
          env = make_env(seed)
        policy.reset()
        obs, _ = env.reset()
        score = agent_score = opponent_score = 0.0
        length = 0
        done = trunc = False
        while not (done or trunc) and length < int(args.max_steps_per_episode):
          image = obs_to_policy_tensor(obs, args.device)
          action, _value, _state = policy.act(image)
          action_int = int(action.reshape(-1)[0].detach().cpu().item())
          obs, reward, done_value, trunc_value, _info = env.step(
              action_tensor(action_int, args.device))
          reward = float(reward.reshape(-1)[0].detach().cpu().item())
          done = bool(done_value.reshape(-1)[0].detach().cpu().item())
          trunc = bool(trunc_value.reshape(-1)[0].detach().cpu().item())
          score += reward
          agent_score += max(reward, 0.0)
          opponent_score += max(-reward, 0.0)
          length += 1
        row = {
            "policy": name,
            "episode": int(episode),
            "seed": int(seed),
            "score": float(score),
            "agent_score": float(agent_score),
            "opponent_score": float(opponent_score),
            "length": int(length),
            "done": bool(done),
            "trunc": bool(trunc),
        }
        rows.append(row)
        print(
            f"{name} ep{episode:02d} seed={seed}: score={score:+.1f} "
            f"agent={agent_score:.0f} opponent={opponent_score:.0f} "
            f"length={length} done={done} trunc={trunc}",
            flush=True,
        )
  finally:
    close_env(env)
  elapsed = time.perf_counter() - start
  scores = [row["score"] for row in rows]
  lengths = [row["length"] for row in rows]
  return {
      "name": name,
      "checkpoint": str(checkpoint),
      "episodes": rows,
      "summary": {
          "episodes": len(rows),
          "score_mean": statistics.fmean(scores),
          "score_std_pop": statistics.pstdev(scores),
          "score_std_sample": statistics.stdev(scores) if len(scores) > 1 else 0.0,
          "score_min": min(scores),
          "score_max": max(scores),
          "length_mean": statistics.fmean(lengths),
          "elapsed_seconds": elapsed,
      },
  }


def main(argv=None):
  args = parse_args(argv)
  if int(args.episodes) <= 0:
    raise ValueError("--episodes must be positive.")
  if int(args.max_steps_per_episode) <= 0:
    raise ValueError("--max-steps-per-episode must be positive.")

  twister_root = args.twister_root.expanduser().resolve()
  add_project_paths("twister", twister_root)
  if str(twister_root) not in sys.path:
    sys.path.insert(0, str(twister_root))
  from pixel_rl.adapter import make_torch_real_env

  args.output_dir.mkdir(parents=True, exist_ok=True)
  specs = [split_name_path(value) for value in args.policy]
  ctx = PixelRLContext(
      env_name=args.env_id,
      seed=int(args.seed),
      num_envs=1,
      device=args.device,
      wm_checkpoint="",
      wm_horizon=0,
  )
  def make_env(seed):
    ctx.seed = int(seed)
    return make_torch_real_env(ctx)

  results = [evaluate_policy(name, checkpoint, make_env, args)
             for name, checkpoint in specs]

  payload = {
      "env_id": args.env_id,
      "episodes_per_policy": int(args.episodes),
      "seed": int(args.seed),
      "reset_seeds": reset_seeds(args),
      "max_steps_per_episode": int(args.max_steps_per_episode),
      "deterministic_policy": bool(args.deterministic_policy),
      "policies": results,
  }
  write_json(args.output_dir / "pong_real_policy_eval.json", payload)

  with (args.output_dir / "pong_real_policy_eval_episodes.csv").open(
      "w", newline="", encoding="utf-8") as f:
    fieldnames = [
        "policy", "episode", "seed", "score", "agent_score",
        "opponent_score", "length", "done", "trunc"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for result in results:
      writer.writerows(result["episodes"])

  with (args.output_dir / "pong_real_policy_eval_summary.csv").open(
      "w", newline="", encoding="utf-8") as f:
    fieldnames = [
        "policy", "episodes", "score_mean", "score_std_sample",
        "score_min", "score_max", "length_mean", "checkpoint"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for result in results:
      row = {"policy": result["name"], "checkpoint": result["checkpoint"]}
      row.update({k: result["summary"][k] for k in fieldnames
                  if k in result["summary"]})
      writer.writerow(row)

  print(f"Wrote {args.output_dir / 'pong_real_policy_eval.json'}", flush=True)
  print(f"Wrote {args.output_dir / 'pong_real_policy_eval_episodes.csv'}", flush=True)
  print(f"Wrote {args.output_dir / 'pong_real_policy_eval_summary.csv'}", flush=True)


if __name__ == "__main__":
  main()
