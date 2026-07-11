#!/usr/bin/env bash
set -euo pipefail

COND="${1:?usage: run_simulus_chain.sh <cond> <lambda>}"
WEIGHT="${2:?usage: run_simulus_chain.sh <cond> <lambda>}"

SIM_ROOT="${SIM_ROOT:-$HOME/projects/simulus}"
WM_EVAL_ROOT="${WM_EVAL_ROOT:-$HOME/projects/CGSReg}"
DATASET="${OCSTORM_SIMULUS_DATASET:-$HOME/projects/shared_replay/oc_storm_pong_exp_repro_replay_sam2_dilated_k3_for_wm/simulus/train}"

WM_NAME="simulus_ocstorm_${COND}"
RL_RUN="ocstorm_replay_sr_simulus_${COND}_ac20k_h512_rewq0p1"
EVAL_OUT="$WM_EVAL_ROOT/artifacts/dataset_ablation_oc_storm_replay/fixed_real_eval/simulus/${COND}"

echo "[simulus-ocstorm-chain] host=$(hostname)"
echo "[simulus-ocstorm-chain] cuda=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[simulus-ocstorm-chain] cond=${COND} lambda=${WEIGHT}"
echo "[simulus-ocstorm-chain] dataset=${DATASET}"

test -d "$SIM_ROOT"
test -d "$WM_EVAL_ROOT"
test -d "$DATASET"

cd "$SIM_ROOT"

env \
  OMP_NUM_THREADS=2 \
  MKL_NUM_THREADS=2 \
  OPENBLAS_NUM_THREADS=2 \
  NUMEXPR_NUM_THREADS=2 \
  MALLOC_ARENA_MAX=2 \
  PYTHONPATH="$SIM_ROOT/src" \
  conda run --no-capture-output -n simulus python -u src/main.py \
    benchmark=atari \
    env.train.id=PongNoFrameskip-v4 \
    env.test.id=PongNoFrameskip-v4 \
    initialization.dataset.train_path="$DATASET" \
    initialization.dataset.test_path="$DATASET" \
    collection.train.stop_after_epochs=0 \
    evaluation.collect=false \
    common.epochs=600 \
    training.actor_critic.start_after_epochs=1000000 \
    +actor_critic.intrinsic_reward_weight=0.0 \
    training.world_model.replay_sampling_uniform_fraction=1.0 \
    world_model.enable_curiosity=false \
    training.world_model.spatial_regu.enabled=true \
    training.world_model.spatial_regu.weight="$WEIGHT" \
    training.world_model.spatial_regu.mask_weights.mask1=1.0 \
    training.world_model.spatial_regu.mask_weights.mask2=0.0 \
    training.world_model.spatial_regu.mask_weights.mask3=0.0 \
    wandb.group=ocstorm_replay_offline_sr \
    wandb.name="$WM_NAME" \
    outputs_dir_path=./outputs/ocstorm_replay_offline_sr \
    "hydra.run.dir=\${outputs_dir_path}/\${env.train.id}/\${wandb.name}/\${now:%Y-%m-%d}/\${now:%H-%M-%S}-seed-\${common.seed}"

WM_BASE="$SIM_ROOT/outputs/ocstorm_replay_offline_sr/PongNoFrameskip-v4/$WM_NAME"
WM_CKPT="$(find "$WM_BASE" -path '*/checkpoints/last.pt' -type f -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
test -f "$WM_CKPT"
echo "[simulus-ocstorm-chain] wm_ckpt=${WM_CKPT}"

env \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONPATH="$SIM_ROOT/src" \
  WANDB_ENTITY=ssl-lab \
  WANDB_MODE=online \
  conda run --no-capture-output -n simulus python -u -m src.pixel_rl.train \
    --backend wm \
    --wm-checkpoint "$WM_CKPT" \
    --wm-initial-source real \
    --wm-enable-curiosity False \
    --wm-reward-quantize-threshold 0.1 \
    --run-name "$RL_RUN" \
    --ac-updates 20000 \
    --envs 64 \
    --backup-every 15 \
    --wm-horizon 512 \
    --eval-real-every 2000 \
    --eval-real-video-every 10000 \
    --eval-real-eps 5 \
    --save-every 5000 \
    --checkpoint-keep 4 \
    --resume False \
    --wandb-enabled True \
    --wandb-project rl-in-pixel-env-simulus \
    --wandb-entity ssl-lab \
    --wandb-mode online

POLICY_CKPT="$SIM_ROOT/runs/$RL_RUN/pixel_rl_ckpt/latest.pt"
test -f "$POLICY_CKPT"
echo "[simulus-ocstorm-chain] policy_ckpt=${POLICY_CKPT}"

mkdir -p "$EVAL_OUT"
cd "$WM_EVAL_ROOT"
conda run --no-capture-output -n twister python scripts/eval/evaluate_torch_pong_real_policies.py \
  --episodes 20 \
  --reset-seeds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 \
  --deterministic-policy \
  --device cuda \
  --output-dir "$EVAL_OUT" \
  --policy "simulus_ocstorm_${COND}=$POLICY_CKPT"

echo "[simulus-ocstorm-chain] done ${COND}"

