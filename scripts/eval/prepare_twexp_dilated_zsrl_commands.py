#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


HOME = Path.home()
ROOT = HOME / "projects" / "CGSReg"
COMMAND_DIR = ROOT / "experiments" / "dataset_ablation_twister_replay" / "commands"
WEIGHTS = {
    "w0": ("0", "0.0"),
    "w0p01": ("0p01", "0.01"),
    "w0p1": ("0p1", "0.1"),
    "w1": ("1", "1.0"),
}


def latest_child(path: Path) -> Path:
    children = [p for p in path.iterdir() if p.is_dir()]
    if not children:
        raise FileNotFoundError(f"No checkpoint directories under {path}")
    return max(children, key=lambda p: p.stat().st_mtime)


def latest_simulus_ckpt(run_name: str) -> Path:
    base = HOME / "projects" / "simulus" / "outputs" / "twexp_offline_sr" / "PongNoFrameskip-v4" / run_name
    ckpts = sorted(base.glob("*/*-seed-*/checkpoints/last.pt"), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        raise FileNotFoundError(f"No Simulus checkpoint found under {base}")
    return ckpts[-1]


def write(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"wrote {path}")


def dreamer_lines() -> list[str]:
    lines = [
        "# Dreamer zero-shot RL for TWISTER exp-repro replay offline SR WMs.",
        "# Auto-generated after dilated-k3 offline WM training.",
        "# Protocol: 20k AC updates, 64 envs, backup_every=15, horizon=512.",
        "",
    ]
    for cond, (slug, _) in WEIGHTS.items():
        ckpt_root = (
            HOME
            / "projects"
            / "dreamerv3-runs"
            / "twexp_offline_sr"
            / "logdir"
            / f"dreamer_twexp_mask1_spatial_{slug}_temporal_1"
            / "ckpt"
        )
        ckpt = latest_child(ckpt_root)
        run = f"twexp_replay_sr_dreamer_{cond}_ac20k_h512_rewq0p1"
        lines.append(
            "bash -lc 'cd \"$HOME/projects/dreamerv3-reborn\" && "
            "DREAMERV3_RUNS_ROOT=\"$HOME/projects/dreamerv3-runs\" "
            "DREAMERV3_AC_UPDATES=20000 DREAMERV3_SAVE_EVERY=10000 NUM_CKPT_TO_KEEP=2 "
            "DREAMERV3_LOG_EVERY=1000 DREAMERV3_EVAL_REAL_EVERY=2000 "
            "DREAMERV3_EVAL_REAL_VIDEO_EVERY=10000 DREAMERV3_VIDEO_EVERY=10000 "
            "DREAMERV3_PIXEL_RL_ENVS=64 DREAMERV3_PIXEL_RL_FRAMEWORK=jax "
            "DREAMERV3_PIXEL_RL_RESUME=False DREAMERV3_WM_HORIZON=512 "
            "DREAMERV3_WM_REWARD_QUANTIZE_THRESHOLD=0.1 "
            "DREAMERV3_WANDB_PROJECT=rl-in-pixel-env-dreamer "
            f"DREAMERV3_LOGGER_NAME={run} "
            f"bash scripts/experiments/pong_pixel_rl_in_env.sh wm \"{ckpt}\" "
            f"\"$HOME/projects/dreamerv3-runs/pong_pixel_rl_in_env/logdir/{run}\"'"
        )
    return lines


def diamond_lines() -> list[str]:
    lines = [
        "# DIAMOND zero-shot RL for TWISTER exp-repro replay offline SR WMs.",
        "# Auto-generated after dilated-k3 offline WM training.",
        "# Protocol: 20k AC updates, 64 envs, backup_every=15, horizon=512.",
        "",
    ]
    for cond in WEIGHTS:
        ckpt = f"outputs/twexp_offline_sr/diamond_twexp_{cond}/checkpoints/agent_versions/agent_epoch_01000.pt"
        run = f"twexp_replay_sr_diamond_{cond}_ac20k_h512"
        lines.append(
            "bash -lc 'cd \"$HOME/projects/diamond\" && "
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_PROJECT=rl-in-pixel-env-diamond "
            "WANDB_ENTITY=ssl-lab WANDB_MODE=online DIAMOND_PIXEL_RL_WANDB_ENABLED=1 "
            "DIAMOND_PIXEL_RL_AC_UPDATES=20000 DIAMOND_PIXEL_RL_ENVS=64 "
            "DIAMOND_PIXEL_RL_BACKUP_EVERY=15 DIAMOND_PIXEL_RL_WM_HORIZON=512 "
            "DIAMOND_PIXEL_RL_WM_INITIAL_SOURCE=real DIAMOND_PIXEL_RL_LOG_EVERY=1000 "
            "DIAMOND_PIXEL_RL_EVAL_REAL_EVERY=2000 DIAMOND_PIXEL_RL_EVAL_REAL_VIDEO_EVERY=10000 "
            "DIAMOND_PIXEL_RL_SAVE_EVERY=2000 DIAMOND_PIXEL_RL_CHECKPOINT_KEEP=10 "
            "DIAMOND_PIXEL_RL_RESUME=False conda run --no-capture-output -n diamond "
            f"bash scripts/experiments/pong_pixel_rl_in_diamond_env.sh wm {ckpt} {run} "
            "checkpoints/Pong.pt checkpoints/Pong.pt'"
        )
    return lines


def simulus_lines() -> list[str]:
    lines = [
        "# Simulus zero-shot RL for TWISTER exp-repro replay offline SR WMs.",
        "# Auto-generated after dilated-k3 offline WM training.",
        "# Protocol: 20k AC updates, 64 envs, backup_every=15, horizon=512.",
        "",
    ]
    for cond in WEIGHTS:
        ckpt = latest_simulus_ckpt(f"simulus_twexp_{cond}")
        run = f"twexp_replay_sr_simulus_{cond}_ac20k_h512_rewq0p1"
        lines.append(
            "bash -lc 'cd \"$HOME/projects/simulus\" && env "
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=\"$PWD/src\" "
            "WANDB_ENTITY=ssl-lab WANDB_MODE=online conda run --no-capture-output -n simulus "
            f"python -u -m src.pixel_rl.train --backend wm --wm-checkpoint \"{ckpt}\" "
            "--wm-initial-source real --wm-enable-curiosity False --wm-reward-quantize-threshold 0.1 "
            f"--run-name {run} --ac-updates 20000 --envs 64 --backup-every 15 --wm-horizon 512 "
            "--eval-real-every 2000 --eval-real-video-every 10000 --eval-real-eps 5 "
            "--save-every 5000 --checkpoint-keep 4 --resume False --wandb-enabled True "
            "--wandb-project rl-in-pixel-env-simulus --wandb-entity ssl-lab --wandb-mode online'"
        )
    return lines


def storm_lines() -> list[str]:
    lines = [
        "# STORM zero-shot RL for TWISTER exp-repro replay offline SR WMs.",
        "# Auto-generated after dilated-k3 offline WM training.",
        "# Protocol: 20k AC updates, 64 envs, backup_every=15, horizon=512.",
        "",
    ]
    for cond in WEIGHTS:
        run = f"twexp_replay_sr_storm_{cond}_ac20k_h512_rewq0p5"
        lines.append(
            "bash -lc 'cd \"$HOME/projects/oc-storm\" && "
            "CONDA_ENV_NAME=oc-storm WANDB_ENABLED=1 WANDB_PROJECT=rl-in-pixel-env-storm "
            "WANDB_ENTITY=ssl-lab WANDB_MODE=online STORM_SIZE_CONFIG=base "
            f"PIXEL_RL_CONFIG_PATH=runs/storm_twexp_{cond}/config.py "
            "PIXEL_RL_AC_UPDATES=20000 PIXEL_RL_ENVS=64 PIXEL_RL_BACKUP_EVERY=15 "
            "PIXEL_RL_LOG_EVERY=1000 PIXEL_RL_SAVE_EVERY=10000 PIXEL_RL_WM_VIDEO_EVERY=10000 "
            "PIXEL_RL_WM_HORIZON=512 PIXEL_RL_WM_DISABLE_KV_CACHE=True "
            "PIXEL_RL_WM_RESPECT_TERMINAL=True PIXEL_RL_WM_INITIAL_SOURCE=real "
            "PIXEL_RL_WM_REWARD_QUANTIZE_THRESHOLD=0.5 PIXEL_RL_EVAL_REAL_EVERY=2000 "
            "PIXEL_RL_EVAL_REAL_VIDEO_EVERY=10000 PIXEL_RL_EVAL_REAL_EPS=5 PIXEL_RL_RESUME=False "
            f"bash scripts/experiments/pong_pixel_rl_in_env.sh wm runs/storm_twexp_{cond}/ckpt/latest_agent.pth "
            f"pong_pixel_rl_in_env/logdir/{run}'"
        )
    return lines


def twister_lines() -> list[str]:
    lines = [
        "# TWISTER zero-shot RL for TWISTER exp-repro replay offline SR WMs.",
        "# Auto-generated after dilated-k3 offline WM training.",
        "# Protocol: 20k AC updates, 64 envs, backup_every=15, horizon=512.",
        "",
    ]
    for cond in WEIGHTS:
        run = f"twexp_replay_sr_twister_{cond}_ac20k_h512_rewq0p5"
        lines.append(
            "bash -lc 'cd \"$HOME/projects/twister\" && "
            "CONDA_ENV_NAME=twister WANDB_ENABLED=1 WANDB_PROJECT=rl-in-pixel-env-twister "
            "WANDB_ENTITY=ssl-lab WANDB_MODE=online PIXEL_RL_AC_UPDATES=20000 PIXEL_RL_ENVS=64 "
            "PIXEL_RL_BACKUP_EVERY=15 PIXEL_RL_LOG_EVERY=1000 PIXEL_RL_SAVE_EVERY=10000 "
            "PIXEL_RL_WM_ROLLOUT_VIDEO_EVERY=10000 PIXEL_RL_WM_HORIZON=512 "
            "PIXEL_RL_WM_RESPECT_TERMINAL=True PIXEL_RL_WM_INITIAL_SOURCE=real "
            "PIXEL_RL_WM_REWARD_QUANTIZE_THRESHOLD=0.5 PIXEL_RL_EVAL_REAL_EVERY=2000 "
            "PIXEL_RL_EVAL_REAL_VIDEO_EVERY=10000 PIXEL_RL_EVAL_REAL_EPS=5 PIXEL_RL_RESUME=False "
            f"bash scripts/experiments/pong_pixel_rl_in_env.sh wm runs/twexp_offline_sr/twister_twexp_{cond}/checkpoints/latest.ckpt "
            f"pong_pixel_rl_in_env/logdir/{run}'"
        )
    return lines


def main() -> None:
    files = {
        "dreamer_twexp_zero_shot_rl.commands.txt": dreamer_lines(),
        "diamond_twexp_zero_shot_rl.commands.txt": diamond_lines(),
        "simulus_twexp_zero_shot_rl.commands.txt": simulus_lines(),
        "storm_twexp_zero_shot_rl.commands.txt": storm_lines(),
        "twister_twexp_zero_shot_rl.commands.txt": twister_lines(),
    }
    all_lines: list[str] = []
    for filename, lines in files.items():
        write(COMMAND_DIR / filename, lines)
        all_lines.extend(line for line in lines if line and not line.startswith("#"))
    write(COMMAND_DIR / "all_twexp_zero_shot_rl.commands.txt", all_lines)


if __name__ == "__main__":
    main()
