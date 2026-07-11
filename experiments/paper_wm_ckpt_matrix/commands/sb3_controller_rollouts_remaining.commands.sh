#!/usr/bin/env bash
# Continue the fixed SB3 PPO Pong controller rollouts after the first batch.
# Protocol: 5 episodes, seeds 0..4, horizon 512, deterministic SB3 policy.

set -euo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
cd "$HOME/projects/CGSReg"

SB3_POLICY="$HOME/projects/CGSReg/artifacts/external_policies/sb3_pong/sb3_ppo_pong_policy_state.pt"
OUT_ROOT="artifacts/controller_policy_ablation/sb3_ppo_pong_h512_rollouts"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

conda activate simulus
CUDA_VISIBLE_DEVICES="$GPU" python scripts/eval/torch_play_rollout_videos.py \
  --wm simulus:simulus_repro="$HOME/projects/simulus/checkpoints/Pong.pt" \
  --policy sb3_ppo_pong="$SB3_POLICY" \
  --policy-format sb3_atari \
  --bootstrap-dataset "$HOME/projects/CGSReg/artifacts/bootstrap_datasets/diamond_pong_test_limit20/simulus/test" \
  --output-dir "$OUT_ROOT/simulus_repro" \
  --env-id PongNoFrameskip-v4 \
  --episodes 5 \
  --horizon 512 \
  --seed 0 \
  --device cuda \
  --fps 15 \
  --size 256 \
  --wm-initial-source dataset \
  --reward-threshold 0.1

CUDA_VISIBLE_DEVICES="$GPU" python scripts/eval/torch_play_rollout_videos.py \
  --wm simulus:offline_static_diamond_w0="$HOME/projects/simulus/external_wm_ckpts/offline_spatial/simulus_pong_offline_static_diamond_w0/checkpoints/last.pt" \
  --wm simulus:offline_static_diamond_w0p01="$HOME/projects/simulus/external_wm_ckpts/offline_spatial/simulus_pong_offline_static_diamond_spatial_0p01/checkpoints/last.pt" \
  --policy sb3_ppo_pong="$SB3_POLICY" \
  --policy-format sb3_atari \
  --bootstrap-dataset "$HOME/projects/CGSReg/artifacts/bootstrap_datasets/diamond_pong_test_limit20/simulus/test" \
  --output-dir "$OUT_ROOT/simulus_offline" \
  --env-id PongNoFrameskip-v4 \
  --episodes 5 \
  --horizon 512 \
  --seed 0 \
  --device cuda \
  --fps 15 \
  --size 256 \
  --wm-initial-source dataset \
  --extra wm_enable_curiosity=false \
  --reward-threshold 0.1

conda activate oc-storm
CUDA_VISIBLE_DEVICES="$GPU" python scripts/eval/torch_play_rollout_videos.py \
  --wm storm:storm_repro="$HOME/projects/oc-storm/runs/pong_atari100k_reproduction/logdir/Pong-STORM-base/ckpt/latest_agent.pth" \
  --policy sb3_ppo_pong="$SB3_POLICY" \
  --policy-format sb3_atari \
  --output-dir "$OUT_ROOT/storm_repro" \
  --env-id PongNoFrameskip-v4 \
  --episodes 5 \
  --horizon 512 \
  --seed 0 \
  --device cuda \
  --fps 15 \
  --size 256 \
  --wm-initial-source real \
  --extra config_path="$HOME/projects/oc-storm/runs/pong_atari100k_reproduction/logdir/Pong-STORM-base/config.py" \
  --extra wm_env_name=PongNoFrameskip-v4

CUDA_VISIBLE_DEVICES="$GPU" python scripts/eval/torch_play_rollout_videos.py \
  --wm storm:storm_w0="$HOME/projects/oc-storm/runs/pong_wm_reg_image_sweep/logdir/pong_wm_reg_base_mask1_spatial_0_temporal_1/ckpt/latest_agent.pth" \
  --policy sb3_ppo_pong="$SB3_POLICY" \
  --policy-format sb3_atari \
  --output-dir "$OUT_ROOT/storm_w0" \
  --env-id PongNoFrameskip-v4 \
  --episodes 5 \
  --horizon 512 \
  --seed 0 \
  --device cuda \
  --fps 15 \
  --size 256 \
  --wm-initial-source real \
  --extra config_path="$HOME/projects/oc-storm/runs/pong_wm_reg_image_sweep/logdir/pong_wm_reg_base_mask1_spatial_0_temporal_1/config.py" \
  --extra wm_env_name=PongNoFrameskip-v4

CUDA_VISIBLE_DEVICES="$GPU" python scripts/eval/torch_play_rollout_videos.py \
  --wm storm:storm_w0p01="$HOME/projects/oc-storm/runs/pong_wm_reg_image_sweep/logdir/pong_wm_reg_base_mask1_spatial_0p01_temporal_1/ckpt/latest_agent.pth" \
  --policy sb3_ppo_pong="$SB3_POLICY" \
  --policy-format sb3_atari \
  --output-dir "$OUT_ROOT/storm_w0p01" \
  --env-id PongNoFrameskip-v4 \
  --episodes 5 \
  --horizon 512 \
  --seed 0 \
  --device cuda \
  --fps 15 \
  --size 256 \
  --wm-initial-source real \
  --extra config_path="$HOME/projects/oc-storm/runs/pong_wm_reg_image_sweep/logdir/pong_wm_reg_base_mask1_spatial_0p01_temporal_1/config.py" \
  --extra wm_env_name=PongNoFrameskip-v4
