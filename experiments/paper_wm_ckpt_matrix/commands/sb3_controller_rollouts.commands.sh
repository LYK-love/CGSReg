#!/usr/bin/env bash
# Fixed SB3 PPO Pong controller rollouts for the paper target WM matrix.
# Protocol: 5 episodes, seeds 0..4, horizon 512, deterministic SB3 policy.

set -euo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
cd "$HOME/projects/CGSReg"

SB3_POLICY="$HOME/projects/CGSReg/artifacts/external_policies/sb3_pong/sb3_ppo_pong_policy_state.pt"
OUT_ROOT="artifacts/controller_policy_ablation/sb3_ppo_pong_h512_rollouts"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

conda activate dreamer
CUDA_VISIBLE_DEVICES="$GPU" python scripts/eval/pong_play_rollout_videos.py \
  --config "$HOME/projects/dreamerv3-runs/pong_wm_reg_sweep/logdir/pong_wm_reg_size200m_mask1_spatial_0_temporal_1/config.yaml" \
  --wm size200m_repro="$HOME/projects/dreamerv3-runs/pong_atari100k_reproduction/logdir/size200m/ckpt/latest" \
  --wm size200m_w0="$HOME/projects/dreamerv3-runs/pong_wm_reg_sweep/logdir/pong_wm_reg_size200m_mask1_spatial_0_temporal_1/ckpt/20260601T031422F628106" \
  --wm size200m_w001="$HOME/projects/dreamerv3-runs/pong_wm_reg_sweep/logdir/pong_wm_reg_size200m_mask1_spatial_0p01_temporal_1/ckpt/20260601T031013F757364" \
  --policy sb3_ppo_pong="$SB3_POLICY" \
  --policy-format sb3_atari \
  --bootstrap-dataset "$HOME/projects/dreamerv3-runs/datasets/pong_wm_reg_sweep/eval_replay" \
  --output-dir "$OUT_ROOT/dreamer" \
  --episodes 5 \
  --horizon 512 \
  --seed 0 \
  --task atari100k_pong \
  --jax-platform cuda \
  --fps 15 \
  --size 256

conda activate diamond
CUDA_VISIBLE_DEVICES="$GPU" python scripts/eval/torch_play_rollout_videos.py \
  --wm diamond:diamond_repro="$HOME/projects/diamond/checkpoints/reproduce_pong/agent_versions/agent_epoch_01000.pt" \
  --policy sb3_ppo_pong="$SB3_POLICY" \
  --policy-format sb3_atari \
  --bootstrap-dataset "$HOME/projects/diamond-assets/datasets/pong/test" \
  --output-dir "$OUT_ROOT/diamond_exp_repro" \
  --env-id PongNoFrameskip-v4 \
  --episodes 5 \
  --horizon 512 \
  --seed 0 \
  --device cuda \
  --fps 15 \
  --size 256 \
  --wm-initial-source dataset \
  --extra rew_end_model_ckpt="$HOME/projects/diamond/checkpoints/Pong.pt"

DIAMOND_W0="/data/luyukuan/projects/diamond/outputs/pong_wm_offline_base_denoiser_only/pong_wm_offline_size13m_b32_mask1_spatial_0p0_denoiser_only/checkpoints/agent_versions/agent_epoch_01000.pt"
if [[ -f "$DIAMOND_W0" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" python scripts/eval/torch_play_rollout_videos.py \
    --wm diamond:diamond_w0p0="$DIAMOND_W0" \
    --policy sb3_ppo_pong="$SB3_POLICY" \
    --policy-format sb3_atari \
    --bootstrap-dataset "$HOME/projects/diamond-assets/datasets/pong/test" \
    --output-dir "$OUT_ROOT/diamond" \
    --env-id PongNoFrameskip-v4 \
    --episodes 5 \
    --horizon 512 \
    --seed 0 \
    --device cuda \
    --fps 15 \
    --size 256 \
    --wm-initial-source dataset \
    --extra denoiser_ckpt="$DIAMOND_W0" \
    --extra rew_end_model_ckpt="$HOME/projects/diamond/checkpoints/Pong.pt"
else
  echo "SKIP missing DIAMOND offline lambda=0 checkpoint: $DIAMOND_W0" >&2
fi

DIAMOND_MASK13="/data/luyukuan/projects/diamond/outputs/pong_wm_offline_base_denoiser_only_fixed_sr_mask1_mask3/pong_wm_offline_size13m_b32_fixedsr_mask1_mask3_spatial_0p01_denoiser_only/checkpoints/agent_versions/agent_epoch_01000.pt"
CUDA_VISIBLE_DEVICES="$GPU" python scripts/eval/torch_play_rollout_videos.py \
  --wm diamond:diamond_mask13_w0p01="$DIAMOND_MASK13" \
  --policy sb3_ppo_pong="$SB3_POLICY" \
  --policy-format sb3_atari \
  --bootstrap-dataset "$HOME/projects/diamond-assets/datasets/pong/test" \
  --output-dir "$OUT_ROOT/diamond" \
  --env-id PongNoFrameskip-v4 \
  --episodes 5 \
  --horizon 512 \
  --seed 0 \
  --device cuda \
  --fps 15 \
  --size 256 \
  --wm-initial-source dataset \
  --extra denoiser_ckpt="$DIAMOND_MASK13" \
  --extra rew_end_model_ckpt="$HOME/projects/diamond/checkpoints/Pong.pt"

conda activate twister
CUDA_VISIBLE_DEVICES="$GPU" python scripts/eval/torch_play_rollout_videos.py \
  --wm twister:dyna_twister_repro="$HOME/projects/twister/callbacks/atari100k/atari100k-pong/checkpoints_epoch_50_step_100000.ckpt" \
  --wm twister:offline_diamond_w0="$HOME/projects/twister/runs/pong_offline_regu_no_ac_cpc_sweep/logdir/twister_pong_offline_no_ac_cpc_static_diamond_w0_model=base/checkpoints/latest.ckpt" \
  --wm twister:offline_diamond_w1="$HOME/projects/twister/runs/pong_offline_regu_no_ac_cpc_sweep/logdir/twister_pong_offline_no_ac_cpc_static_diamond_w1_model=base/checkpoints/latest.ckpt" \
  --policy sb3_ppo_pong="$SB3_POLICY" \
  --policy-format sb3_atari \
  --bootstrap-dataset "$HOME/projects/CGSReg/artifacts/bootstrap_datasets/diamond_pong_test_limit20/twister/test" \
  --output-dir "$OUT_ROOT/twister" \
  --env-id PongNoFrameskip-v4 \
  --episodes 5 \
  --horizon 512 \
  --seed 0 \
  --device cuda \
  --fps 15 \
  --size 256 \
  --wm-initial-source dataset

conda activate simulus
CUDA_VISIBLE_DEVICES="$GPU" python scripts/eval/torch_play_rollout_videos.py \
  --wm simulus:simulus_repro="/data/luyukuan/projects/Simulus/checkpoints/Pong.pt" \
  --wm simulus:offline_static_diamond_w0="$HOME/projects/simulus/external_wm_ckpts/offline_spatial/simulus_pong_offline_static_diamond_w0/checkpoints/last.pt" \
  --wm simulus:offline_static_diamond_w0p01="$HOME/projects/simulus/external_wm_ckpts/offline_spatial/simulus_pong_offline_static_diamond_spatial_0p01/checkpoints/last.pt" \
  --policy sb3_ppo_pong="$SB3_POLICY" \
  --policy-format sb3_atari \
  --bootstrap-dataset "$HOME/projects/CGSReg/artifacts/bootstrap_datasets/diamond_pong_test_limit20/simulus/test" \
  --output-dir "$OUT_ROOT/simulus" \
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
