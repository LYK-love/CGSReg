#!/usr/bin/env python3
"""Generate ALE-conditioned DIAMOND rollout videos for distribution comparison.

For each episode, this script first runs the policy in the real ALE Pong
emulator and records the first DIAMOND conditioning window. Every DIAMOND WM
checkpoint is then reset from exactly that fixed emulator context. The ALE
reference video and all WM videos start at the final context frame, so frame 0
is aligned across the reference and generated rollouts. The output bundle stores
the real emulator baseline as `diamond/ale_emulator` alongside the DIAMOND WM
checkpoint conditions, and runs CUTIE segmentation by default.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import random
import subprocess
import sys
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "eval"
if str(SCRIPTS) not in sys.path:
  sys.path.insert(0, str(SCRIPTS))

from play_rollout_common import (  # noqa: E402
    PixelRLContext,
    action_tensor,
    add_project_paths,
    clear_pixel_rl_modules,
    ensure_dir,
    first_bool,
    first_float,
    load_pixel_rl_policy,
    obs_to_policy_tensor,
    obs_to_uint8_frame,
    parse_extra,
    resolve_policy_checkpoint,
    save_video,
    write_json,
    require_torch,
)


DEFAULT_POLICY = (
    ROOT.parent
    / "rl-in-pixel-env/runs/backend=real_ac20k_envs64_backup15_cuda3/"
    / "pixel_rl_ckpt/latest.pt"
)

DEFAULT_COMPONENT_AGENT = pathlib.Path(
    "/data/luyukuan/projects/diamond-assets/checkpoints/reproduce_pong/agent_versions/agent_epoch_01000.pt"
)

DEFAULT_DIAMOND_WMS = {
    "exp_repro": (
        "diamond_repro",
        DEFAULT_COMPONENT_AGENT,
        "",
    ),
    "w0": (
        "diamond_w0p0",
        pathlib.Path(
            "/data/luyukuan/projects/diamond/outputs/"
            "pong_wm_offline_base_denoiser_only/offline_base_w0/"
            "checkpoints/agent_versions/agent_epoch_01000.pt"
        ),
        "0",
    ),
    "w_recommended": (
        "diamond_mask13_w0p01",
        pathlib.Path(
            "/data/luyukuan/projects/diamond/outputs/"
            "pong_wm_offline_base_denoiser_only_fixed_sr_mask1_mask3/"
            "pong_wm_offline_size13m_b32_fixedsr_mask1_mask3_spatial_0p01_denoiser_only/"
            "checkpoints/agent_versions/agent_epoch_01000.pt"
        ),
        "0.01",
    ),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output-dir", type=pathlib.Path,
                      default=ROOT / "artifacts/diamond_ale_aligned_rollout_eval_bundle")
  parser.add_argument("--diamond-root", type=pathlib.Path,
                      default=ROOT.parent / "diamond")
  parser.add_argument("--env-id", default="PongNoFrameskip-v4")
  parser.add_argument("--policy", default=f"real_env_ac20k_backup15={DEFAULT_POLICY}",
                      help="name=/path/to/pixel_rl policy checkpoint or run dir.")
  parser.add_argument("--episodes", type=int, default=5)
  parser.add_argument("--horizon", type=int, default=512)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--reset-seeds", default="",
                      help="Comma-separated reset seeds. Defaults to seed..seed+episodes-1.")
  parser.add_argument("--device", default="cuda")
  parser.add_argument("--fps", type=int, default=15)
  parser.add_argument("--size", type=int, default=256)
  parser.add_argument("--reward-threshold", type=float, default=0.5)
  parser.add_argument("--deterministic-policy", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument("--run-cutie", action=argparse.BooleanOptionalAction, default=True,
                      help="Run CUTIE segmentation for the completed bundle. Enabled by default.")
  parser.add_argument("--cutie-output-dir", type=pathlib.Path, default=None,
                      help="CUTIE output directory. Defaults to output-dir/cutie_segmentations.")
  parser.add_argument("--cutie-geometry", choices=("square-to-atari", "native"), default="square-to-atari")
  parser.add_argument("--cutie-save-masks", action="store_true",
                      help="Also save cutie_masks.npz for each bundle video.")
  parser.add_argument("--extra", action="append", default=[],
                      help="Extra DIAMOND/ALE adapter key=value. Repeatable.")
  parser.add_argument("--wm", action="append", default=[],
                      help="Optional condition:name=/path/to/checkpoint. If omitted, uses the three default DIAMOND ckpts.")
  return parser.parse_args(argv)


def reset_seeds(args: argparse.Namespace) -> list[int]:
  if args.reset_seeds:
    seeds = [int(x) for x in args.reset_seeds.split(",") if x.strip()]
    if len(seeds) < int(args.episodes):
      raise ValueError(f"Need at least {args.episodes} reset seeds, got {len(seeds)}.")
    return seeds[: int(args.episodes)]
  return [int(args.seed) + i for i in range(int(args.episodes))]


def parse_policy(value: str) -> tuple[str, pathlib.Path]:
  if "=" in value:
    name, raw = value.split("=", 1)
  else:
    name, raw = "policy", value
  return name, resolve_policy_checkpoint(raw)


def parse_wm(value: str) -> tuple[str, str, pathlib.Path, str]:
  if ":" not in value or "=" not in value:
    raise ValueError("--wm must look like condition:name=/path/to/checkpoint")
  condition, rest = value.split(":", 1)
  name, raw = rest.split("=", 1)
  return condition, name, pathlib.Path(raw).expanduser().resolve(), ""


def wm_specs(args: argparse.Namespace) -> list[tuple[str, str, pathlib.Path, str]]:
  if args.wm:
    return [parse_wm(value) for value in args.wm]
  return [
      (condition, name, pathlib.Path(path).expanduser().resolve(), weight)
      for condition, (name, path, weight) in DEFAULT_DIAMOND_WMS.items()
  ]


def close_env(env: Any) -> None:
  close = getattr(env, "close", None)
  if callable(close):
    close()


def make_diamond_adapter(diamond_root: pathlib.Path):
  add_project_paths("diamond", diamond_root)
  clear_pixel_rl_modules()
  return __import__("pixel_rl.adapter", fromlist=[
      "PixelEnvMonitor",
      "make_torch_real_env",
      "_load_cfg",
      "_make_real_env",
      "_load_wm_agent",
  ])


def make_policy(policy_ckpt: pathlib.Path, *, num_actions: int, device: str, deterministic: bool):
  return load_pixel_rl_policy(
      policy_ckpt,
      num_actions=int(num_actions),
      device=device,
      deterministic=bool(deterministic),
  )


def collect_real_context_and_reference(
    env,
    policy,
    *,
    context_len: int,
    horizon: int,
    device: str,
    size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
  torch = require_torch()
  if int(context_len) < 1:
    raise ValueError("context_len must be at least 1.")

  obs, info = env.reset()
  policy.reset()
  all_obs = [obs]
  frames = [obs_to_uint8_frame(obs, size)]
  actions, rewards, done, trunc = [], [], [], []
  infos = [{"reset_info": str(info)}]
  total_steps = int(context_len) - 1 + int(horizon)

  with torch.no_grad():
    for _ in range(total_steps):
      image = obs_to_policy_tensor(obs, device)
      action, value, _ = policy.act(image)
      action_int = int(action.reshape(-1)[0].detach().cpu().item())
      next_obs, reward, done_value, trunc_value, step_info = env.step(action_tensor(action_int, device))
      done_flag = first_bool(done_value)
      trunc_flag = first_bool(trunc_value)
      final_obs = step_info.get("final_observation") if isinstance(step_info, dict) else None
      render_obs = final_obs if final_obs is not None and (done_flag or trunc_flag) else next_obs
      all_obs.append(render_obs)
      frames.append(obs_to_uint8_frame(render_obs, size))
      actions.append(action_int)
      rewards.append(first_float(reward))
      done.append(done_flag)
      trunc.append(trunc_flag)
      infos.append({"value": float(value.reshape(-1)[0].detach().cpu().item())})
      obs = next_obs

  context = {
      "obs": all_obs[: int(context_len)],
      "actions": np.asarray(actions[: int(context_len)], np.int64),
      "rewards": np.asarray(rewards[: int(context_len)], np.float32),
      "done": np.asarray(done[: int(context_len)], bool),
      "trunc": np.asarray(trunc[: int(context_len)], bool),
  }
  ref_start = int(context_len) - 1
  reference = {
      "frames": np.asarray(frames[ref_start : ref_start + int(horizon) + 1], np.uint8),
      "actions": np.asarray(actions[ref_start : ref_start + int(horizon)], np.int64),
      "rewards": np.asarray(rewards[ref_start : ref_start + int(horizon)], np.float32),
      "done": np.asarray(done[ref_start : ref_start + int(horizon)], bool),
      "trunc": np.asarray(trunc[ref_start : ref_start + int(horizon)], bool),
      "infos": infos[ref_start : ref_start + int(horizon) + 1],
  }
  return context, reference


def context_to_episode(context: dict[str, Any]):
  torch = require_torch()
  from data import Episode

  frames = [obs_to_uint8_frame(obs, size=64) for obs in context["obs"]]
  obs = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float().div(255.0).mul(2.0).sub(1.0)
  n = int(obs.shape[0])

  def pad(values, dtype, fill):
    arr = np.asarray(values)
    if len(arr) < n:
      arr = np.pad(arr, (0, n - len(arr)), constant_values=fill)
    return torch.as_tensor(arr[:n], dtype=dtype)

  return Episode(
      obs=obs,
      act=pad(context["actions"], torch.long, 0),
      rew=pad(context["rewards"], torch.float32, 0.0),
      end=pad(context["done"], torch.uint8, False),
      trunc=pad(context["trunc"], torch.uint8, False),
      info={},
  )


class FixedContextDataset:
  def __init__(self, episode):
    self.episode = episode
    self.num_episodes = 1
    self.lengths = np.asarray([len(episode)], dtype=np.int32)
    self.num_steps = int(len(episode))
    self.atari_game = None

  def __len__(self):
    return int(len(self.episode))

  def __getitem__(self, segment_id):
    from data import make_segment

    return make_segment(self.episode, segment_id, should_pad=False, atari_game=self.atari_game)


class FixedContextBatchSampler:
  batch_size = 1

  def __init__(self, context_len: int):
    self.context_len = int(context_len)

  def __iter__(self):
    from data import SegmentId

    while True:
      yield [SegmentId(episode_id=0, start=0, stop=self.context_len)]


def make_fixed_context_loader(context: dict[str, Any], *, context_len: int):
  torch = require_torch()
  from data import collate_segments_to_batch

  dataset = FixedContextDataset(context_to_episode(context))
  sampler = FixedContextBatchSampler(context_len)
  return torch.utils.data.DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_segments_to_batch)


def make_diamond_wm_env_from_context(
    adapter,
    *,
    checkpoint: pathlib.Path,
    context: dict[str, Any],
    context_len: int,
    env_id: str,
    horizon: int,
    device: str,
    reward_threshold: float,
):
  torch = require_torch()
  import torch as torch_module
  from hydra.utils import instantiate
  from agent import Agent
  from envs import WorldModelEnv

  cfg = adapter._load_cfg(env_id, int(horizon) + 1)
  real_env = adapter._make_real_env(cfg, torch.device(device), 1)
  try:
    num_actions = int(real_env.num_actions)
  finally:
    close_env(real_env)

  shell = Agent(instantiate(cfg.agent, num_actions=num_actions)).to(torch.device(device)).eval()
  try:
    raw_ckpt = torch_module.load(checkpoint, map_location="cpu", weights_only=False)
  except TypeError:
    raw_ckpt = torch_module.load(checkpoint, map_location="cpu")
  denoiser_only = isinstance(raw_ckpt, dict) and "rew_end_model" not in raw_ckpt
  extra = {}
  wm_checkpoint = str(checkpoint)
  if denoiser_only:
    extra = {
        "denoiser_ckpt": str(checkpoint),
        "rew_end_model_ckpt": str(DEFAULT_COMPONENT_AGENT),
        "actor_critic_ckpt": str(DEFAULT_COMPONENT_AGENT),
    }
    wm_checkpoint = ""

  ctx = PixelRLContext(
      env_name=env_id,
      seed=0,
      num_envs=1,
      device=device,
      wm_checkpoint=wm_checkpoint,
      wm_horizon=int(horizon) + 1,
      wm_reward_quantize_threshold=float(reward_threshold),
      wm_respect_terminal=False,
      extra=extra,
  )
  adapter._load_wm_agent(shell, ctx)
  wm_env_cfg = instantiate(cfg.world_model_env, num_batches_to_preload=1)
  wm_env_cfg.horizon = int(horizon) + 1
  loader = make_fixed_context_loader(context, context_len=int(context_len))
  env = WorldModelEnv(shell.denoiser, shell.rew_end_model, loader, wm_env_cfg)
  env.num_actions = num_actions
  return adapter.PixelEnvMonitor(env, capture_video=True, video_max_frames=int(horizon) + 1)


def run_policy_rollout(env, policy, *, horizon: int, device: str, size: int):
  torch = require_torch()
  obs, info = env.reset()
  policy.reset()
  frames = [obs_to_uint8_frame(obs, size)]
  actions, rewards, done, trunc = [], [], [], []
  infos = [{"reset_info": str(info)}]
  with torch.no_grad():
    for _ in range(int(horizon)):
      image = obs_to_policy_tensor(obs, device)
      action, value, _ = policy.act(image)
      action_int = int(action.reshape(-1)[0].detach().cpu().item())
      next_obs, reward, done_value, trunc_value, step_info = env.step(action_tensor(action_int, device))
      done_flag = first_bool(done_value)
      trunc_flag = first_bool(trunc_value)
      final_obs = step_info.get("final_observation") if isinstance(step_info, dict) else None
      render_obs = final_obs if final_obs is not None and (done_flag or trunc_flag) else next_obs
      frames.append(obs_to_uint8_frame(render_obs, size))
      actions.append(action_int)
      rewards.append(first_float(reward))
      done.append(done_flag)
      trunc.append(trunc_flag)
      infos.append({"value": float(value.reshape(-1)[0].detach().cpu().item())})
      obs = next_obs
  return {
      "frames": np.asarray(frames, np.uint8),
      "actions": np.asarray(actions, np.int64),
      "rewards": np.asarray(rewards, np.float32),
      "done": np.asarray(done, bool),
      "trunc": np.asarray(trunc, bool),
      "infos": infos,
  }


def write_rollout(
    *,
    output_dir: pathlib.Path,
    project: str,
    condition: str,
    episode_name: str,
    model: str,
    rec: dict[str, Any],
    fps: int,
    metadata: dict[str, Any],
) -> tuple[pathlib.Path, pathlib.Path]:
  video_path = output_dir / "videos" / project / condition / f"{episode_name}.mp4"
  meta_path = output_dir / "rollout_metadata" / project / condition / f"{episode_name}.json"
  save_video(video_path, rec["frames"], fps)
  write_json(meta_path, {
      **metadata,
      "video": str(video_path),
      "num_frames": int(rec["frames"].shape[0]),
      "num_actions": int(rec["actions"].shape[0]),
      "return": float(np.sum(rec["rewards"])),
      "actions": rec["actions"].tolist(),
      "rewards": rec["rewards"].tolist(),
      "done": rec["done"].tolist(),
      "trunc": rec["trunc"].tolist(),
      "infos": rec["infos"],
      "bundle_project": project,
      "bundle_condition": condition,
      "bundle_model": model,
  })
  return video_path, meta_path


def run_cutie_for_bundle(args: argparse.Namespace, output_dir: pathlib.Path) -> None:
  cutie_output_dir = (
      args.cutie_output_dir.expanduser().resolve()
      if args.cutie_output_dir is not None
      else output_dir / "cutie_segmentations"
  )
  cmd = [
      sys.executable,
      str(SCRIPTS / "run_oc_storm_cutie_pong_tracks.py"),
      "--bundle-root",
      str(output_dir),
      "--output-dir",
      str(cutie_output_dir),
      "--geometry",
      str(args.cutie_geometry),
      "--fps",
      str(int(args.fps)),
      "--skip-existing",
  ]
  if args.cutie_save_masks:
    cmd.append("--save-masks")
  print("Running CUTIE:", " ".join(cmd), flush=True)
  subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> None:
  args = parse_args(argv)
  if int(args.episodes) <= 0:
    raise ValueError("--episodes must be positive.")
  if int(args.horizon) <= 0:
    raise ValueError("--horizon must be positive.")

  output_dir = args.output_dir.expanduser().resolve()
  ensure_dir(output_dir)
  policy_name, policy_ckpt = parse_policy(args.policy)
  specs = wm_specs(args)
  for _condition, _name, ckpt, _weight in specs:
    if not ckpt.exists():
      raise FileNotFoundError(f"WM checkpoint does not exist: {ckpt}")

  diamond_root = args.diamond_root.expanduser().resolve()
  adapter = make_diamond_adapter(diamond_root)
  torch = require_torch()
  seeds = reset_seeds(args)
  extra_base = parse_extra(list(args.extra))
  cfg = adapter._load_cfg(args.env_id, int(args.horizon) + 1)
  context_len = int(cfg.agent.denoiser.inner_model.num_steps_conditioning)

  write_json(output_dir / "config" / "experiment.json", {
      "description": "DIAMOND WM rollouts compared to matched ALE Pong emulator rollouts.",
      "alignment": (
          "Each WM is initialized from the first num_steps_conditioning frames "
          "of the matched ALE emulator episode. Videos start at the final context frame."
      ),
      "env_id": args.env_id,
      "policy": policy_name,
      "policy_ckpt": str(policy_ckpt),
      "episodes": int(args.episodes),
      "reset_seeds": seeds,
      "horizon": int(args.horizon),
      "fps": int(args.fps),
      "size": int(args.size),
      "context_len": int(context_len),
      "wm_initial_source": "matched_ale_context",
      "real_emulator_condition": "ale_emulator",
      "cutie_enabled": bool(args.run_cutie),
      "cutie_output_dir": str(
          args.cutie_output_dir.expanduser().resolve()
          if args.cutie_output_dir is not None
          else output_dir / "cutie_segmentations"
      ),
      "world_models": [
          {"project": "diamond", "condition": condition, "name": name, "checkpoint": str(ckpt), "weight": weight}
          for condition, name, ckpt, weight in specs
      ],
  })

  entries: list[dict[str, Any]] = []
  for episode_idx, episode_seed in enumerate(seeds):
    episode_name = f"ep{episode_idx:02d}_seed{episode_seed}"
    random.seed(episode_seed)
    np.random.seed(episode_seed)
    torch.manual_seed(episode_seed)

    ctx = PixelRLContext(
        env_name=args.env_id,
        seed=int(episode_seed),
        num_envs=1,
        device=args.device,
        wm_checkpoint="",
        wm_horizon=0,
        extra=extra_base.copy(),
    )
    env = adapter.make_torch_real_env(ctx)
    try:
      policy = make_policy(
          policy_ckpt,
          num_actions=int(getattr(env, "num_actions", 6)),
          device=args.device,
          deterministic=bool(args.deterministic_policy),
      )
      context, real_rec = collect_real_context_and_reference(
          env,
          policy,
          context_len=context_len,
          horizon=args.horizon,
          device=args.device,
          size=args.size,
      )
    finally:
      close_env(env)

    real_project = "diamond"
    real_condition = "ale_emulator"
    real_video = output_dir / "videos" / real_project / real_condition / f"{episode_name}.mp4"
    if not (args.skip_existing and real_video.exists()):
      video, meta = write_rollout(
          output_dir=output_dir,
          project=real_project,
          condition=real_condition,
          episode_name=episode_name,
          model="ale_pong",
          rec=real_rec,
          fps=int(args.fps),
          metadata={
              "episode": int(episode_idx),
              "seed": int(episode_seed),
              "project": real_project,
              "wm": "ale_pong",
              "checkpoint": "",
              "policy": policy_name,
              "policy_ckpt": str(policy_ckpt),
              "horizon": int(args.horizon),
              "alignment_seed": int(episode_seed),
              "context_len": int(context_len),
              "video_starts_at_context_frame": int(context_len) - 1,
              "real_emulator": True,
              "reference": True,
          },
      )
      print(f"Wrote {video}", flush=True)
    else:
      meta = output_dir / "rollout_metadata" / real_project / real_condition / f"{episode_name}.json"
      video = real_video
      print(f"Keeping existing {video}", flush=True)
    entries.append({
        "project": real_project,
        "condition": real_condition,
        "weight": "",
        "episode": episode_name,
        "model": "ale_pong",
        "video": str(video),
        "metadata": str(meta),
    })

    for condition, model_name, checkpoint, weight in specs:
      wm_video = output_dir / "videos" / "diamond" / condition / f"{episode_name}.mp4"
      if not (args.skip_existing and wm_video.exists()):
        random.seed(episode_seed)
        np.random.seed(episode_seed)
        torch.manual_seed(episode_seed)
        cwd = pathlib.Path.cwd()
        os.chdir(diamond_root)
        env = make_diamond_wm_env_from_context(
            adapter,
            checkpoint=checkpoint,
            context=context,
            context_len=context_len,
            env_id=args.env_id,
            horizon=args.horizon,
            device=args.device,
            reward_threshold=args.reward_threshold,
        )
        try:
          policy = make_policy(
              policy_ckpt,
              num_actions=int(getattr(env, "num_actions", 6)),
              device=args.device,
              deterministic=bool(args.deterministic_policy),
          )
          rec = run_policy_rollout(env, policy, horizon=args.horizon, device=args.device, size=args.size)
        finally:
          close_env(env)
          os.chdir(cwd)
        video, meta = write_rollout(
            output_dir=output_dir,
            project="diamond",
            condition=condition,
            episode_name=episode_name,
            model=model_name,
            rec=rec,
            fps=int(args.fps),
            metadata={
                "episode": int(episode_idx),
                "seed": int(episode_seed),
                "project": "diamond",
                "wm": model_name,
                "checkpoint": str(checkpoint),
                "policy": policy_name,
                "policy_ckpt": str(policy_ckpt),
                "horizon": int(args.horizon),
                "alignment_seed": int(episode_seed),
                "aligned_reference_project": real_project,
                "aligned_reference_condition": real_condition,
                "wm_initial_source": "matched_ale_context",
                "context_len": int(context_len),
                "video_starts_at_context_frame": int(context_len) - 1,
                "reference": False,
            },
        )
        print(f"Wrote {video}", flush=True)
      else:
        meta = output_dir / "rollout_metadata" / "diamond" / condition / f"{episode_name}.json"
        video = wm_video
        print(f"Keeping existing {video}", flush=True)
      entries.append({
          "project": "diamond",
          "condition": condition,
          "weight": weight,
          "episode": episode_name,
          "model": model_name,
          "video": str(video),
          "metadata": str(meta),
      })

  write_json(output_dir / "bundle_manifest.json", {
      "source": "scripts/eval/diamond_ale_aligned_rollout_experiment.py",
      "copy": False,
      "episodes": int(args.episodes),
      "real_emulator_project": "diamond",
      "real_emulator_condition": "ale_emulator",
      "cutie_enabled": bool(args.run_cutie),
      "entries": entries,
      "missing": [],
  })
  (output_dir / "README.md").write_text(
      "# DIAMOND ALE-Conditioned Rollout Evaluation Bundle\n\n"
      "This bundle contains one real ALE Pong emulator rollout plus three DIAMOND "
      "WM checkpoint rollouts per episode, all under project `diamond`. The WM "
      "rollouts are initialized from the matched emulator context, and the videos "
      "start at the final context frame so trajectories are aligned at frame 0.\n\n"
      "Conditions:\n\n"
      "- `ale_emulator`: real ALE baseline used as the reference distribution.\n"
      "- `exp_repro`, `w0`, `w_recommended`: DIAMOND WM checkpoints initialized "
      "from the matched `ale_emulator` context.\n\n"
      "CUTIE segmentation is generated by this script by default and written to "
      "`cutie_segmentations/`. Pass `--no-run-cutie` to build only videos and "
      "metadata.\n",
      encoding="utf-8",
  )
  print(f"Wrote bundle manifest with {len(entries)} entries: {output_dir / 'bundle_manifest.json'}", flush=True)
  if args.run_cutie:
    run_cutie_for_bundle(args, output_dir)


if __name__ == "__main__":
  main()
