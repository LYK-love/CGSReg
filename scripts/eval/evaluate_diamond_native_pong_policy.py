#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import statistics
import sys
import time

from diamond_public_policy import load_diamond_public_policy
from evaluate_torch_pong_real_policies import close_env, reset_seeds, write_json
from play_rollout_common import PixelRLContext, action_tensor, add_project_paths, obs_to_policy_tensor


ROOT = pathlib.Path(__file__).resolve().parents[2]
PROJECTS_ROOT = ROOT.parent


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
  parser.add_argument("--label", default="diamond_dyna_style")
  parser.add_argument("--output-dir", type=pathlib.Path, required=True)
  parser.add_argument("--twister-root", type=pathlib.Path, default=PROJECTS_ROOT / "twister")
  parser.add_argument("--env-id", default="PongNoFrameskip-v4")
  parser.add_argument("--episodes", type=int, default=20)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--reset-seeds", default="")
  parser.add_argument("--device", default="cuda")
  parser.add_argument("--max-steps-per-episode", type=int, default=30000)
  return parser.parse_args()


def main():
  args = parse_args()
  add_project_paths("twister", args.twister_root.expanduser().resolve())
  if str(args.twister_root.expanduser().resolve()) not in sys.path:
    sys.path.insert(0, str(args.twister_root.expanduser().resolve()))
  from pixel_rl.adapter import make_torch_real_env

  args.output_dir.mkdir(parents=True, exist_ok=True)
  seeds = reset_seeds(args)
  ctx = PixelRLContext(
      env_name=args.env_id,
      seed=int(seeds[0]),
      num_envs=1,
      device=args.device,
      wm_checkpoint="",
      wm_horizon=0,
  )

  def make_env(seed):
    ctx.seed = int(seed)
    return make_torch_real_env(ctx)

  env = make_env(seeds[0])
  policy = load_diamond_public_policy(
      args.checkpoint,
      num_actions=int(getattr(env, "num_actions", 6)),
      device=args.device,
      deterministic=True,
  )
  rows = []
  start = time.perf_counter()
  try:
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
        obs, reward, done_value, trunc_value, _info = env.step(action_tensor(action_int, args.device))
        reward = float(reward.reshape(-1)[0].detach().cpu().item())
        done = bool(done_value.reshape(-1)[0].detach().cpu().item())
        trunc = bool(trunc_value.reshape(-1)[0].detach().cpu().item())
        score += reward
        agent_score += max(reward, 0.0)
        opponent_score += max(-reward, 0.0)
        length += 1
      row = {
          "policy": args.label,
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
          f"{args.label} ep{episode:02d} seed={seed}: score={score:+.1f} "
          f"agent={agent_score:.0f} opponent={opponent_score:.0f} "
          f"length={length} done={done} trunc={trunc}",
          flush=True,
      )
  finally:
    close_env(env)

  scores = [row["score"] for row in rows]
  lengths = [row["length"] for row in rows]
  summary = {
      "policy": args.label,
      "episodes": len(rows),
      "score_mean": statistics.fmean(scores),
      "score_std_sample": statistics.stdev(scores) if len(scores) > 1 else 0.0,
      "score_min": min(scores),
      "score_max": max(scores),
      "length_mean": statistics.fmean(lengths),
      "checkpoint": str(args.checkpoint.expanduser().resolve()),
  }
  payload = {
      "env_id": args.env_id,
      "episodes_per_policy": int(args.episodes),
      "seed": int(args.seed),
      "reset_seeds": seeds,
      "max_steps_per_episode": int(args.max_steps_per_episode),
      "deterministic_policy": True,
      "elapsed_seconds": time.perf_counter() - start,
      "policies": [{"name": args.label, "checkpoint": summary["checkpoint"], "episodes": rows, "summary": summary}],
  }
  write_json(args.output_dir / "pong_real_policy_eval.json", payload)

  with (args.output_dir / "pong_real_policy_eval_episodes.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

  with (args.output_dir / "pong_real_policy_eval_summary.csv").open("w", newline="", encoding="utf-8") as f:
    fieldnames = ["policy", "episodes", "score_mean", "score_std_sample", "score_min", "score_max", "length_mean", "checkpoint"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow(summary)

  print(f"Wrote {args.output_dir / 'pong_real_policy_eval_summary.csv'}", flush=True)


if __name__ == "__main__":
  main()
