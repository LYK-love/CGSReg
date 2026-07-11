#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <size_config> <lambda>" >&2
  echo "Example: $0 size200m 0.001" >&2
  exit 2
fi

SIZE="$1"
LAMBDA="$2"

weight_slug() {
  local value="$1"
  value="${value%.0}"
  value="${value//./p}"
  value="${value//-/m}"
  echo "$value"
}

SLUG="$(weight_slug "$LAMBDA")"
PROJECTS_ROOT="${PROJECTS_ROOT:-$HOME/projects}"
DREAMER_ROOT="${DREAMER_ROOT:-$PROJECTS_ROOT/dreamerv3-reborn}"
DREAMERV3_RUNS_ROOT="${DREAMERV3_RUNS_ROOT:-$PROJECTS_ROOT/dreamerv3-runs}"
WM_RUN="pong_wm_reg_${SIZE}_mask1_spatial_${SLUG}_temporal_1"
WM_LOGDIR="$DREAMERV3_RUNS_ROOT/pong_wm_reg_sweep/logdir/$WM_RUN"
WM_CKPT="$WM_LOGDIR/ckpt/latest"
RL_RUN="lambda_search_${SIZE}_w${SLUG}_ac20k_h512_rewq0p1"
RL_LOGDIR="$DREAMERV3_RUNS_ROOT/pong_pixel_rl_in_env/logdir/$RL_RUN"
EVAL_OUT="$PROJECTS_ROOT/CGSReg/artifacts/dreamer_lambda_search/zero_shot_20seed/${SIZE}_w${SLUG}"

echo "[lambda-search] host=$(hostname)"
echo "[lambda-search] cuda=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[lambda-search] size=$SIZE lambda=$LAMBDA slug=$SLUG"
echo "[lambda-search] wm_logdir=$WM_LOGDIR"
echo "[lambda-search] rl_logdir=$RL_LOGDIR"
echo "[lambda-search] eval_out=$EVAL_OUT"

cd "$DREAMER_ROOT"

resolve_dreamer_ckpt() {
  local ckpt="$1"
  if [ -d "$ckpt" ]; then
    printf '%s\n' "$ckpt"
    return 0
  fi
  if [ -f "$ckpt" ]; then
    local target
    target="$(cat "$ckpt")"
    if [ -n "$target" ] && [ -d "$(dirname "$ckpt")/$target" ]; then
      printf '%s\n' "$(dirname "$ckpt")/$target"
      return 0
    fi
  fi
  return 1
}

if [ -e "$WM_CKPT" ]; then
  echo "[lambda-search] Reusing existing WM checkpoint: $WM_CKPT"
else
  echo "[lambda-search] WM checkpoint missing; starting offline WM training."
  DREAMERV3_RUNS_ROOT="$DREAMERV3_RUNS_ROOT" \
  DREAMERV3_SIZE_CONFIG="$SIZE" \
  DREAMERV3_RUN_PREFIX="pong_wm_reg_${SIZE}" \
  DREAMERV3_SAVE_EVERY="${DREAMERV3_SAVE_EVERY:-3600}" \
  NUM_CKPT_TO_KEEP="${NUM_CKPT_TO_KEEP:-3}" \
  DREAMERV3_LOG_EVERY="${DREAMERV3_LOG_EVERY:-300}" \
  DREAMERV3_REPORT_EVERY="${DREAMERV3_REPORT_EVERY:-300}" \
  CONDA_ENV_NAME="${CONDA_ENV_NAME:-dreamer}" \
    bash scripts/experiments/pong_wm_reg_sweep.sh "$LAMBDA" 1 mask1
fi

if [ ! -e "$WM_CKPT" ]; then
  echo "[lambda-search] ERROR: WM checkpoint still missing after offline training: $WM_CKPT" >&2
  exit 1
fi

WM_CKPT_RESOLVED="$(resolve_dreamer_ckpt "$WM_CKPT")"
echo "[lambda-search] resolved_wm_ckpt=$WM_CKPT_RESOLVED"

echo "[lambda-search] Starting pixel-space zero-shot RL."
DREAMERV3_RUNS_ROOT="$DREAMERV3_RUNS_ROOT" \
DREAMERV3_AC_UPDATES=20000 \
DREAMERV3_SAVE_EVERY=10000 \
NUM_CKPT_TO_KEEP=2 \
DREAMERV3_LOG_EVERY=1000 \
DREAMERV3_EVAL_REAL_EVERY=2000 \
DREAMERV3_EVAL_REAL_VIDEO_EVERY=10000 \
DREAMERV3_VIDEO_EVERY=10000 \
DREAMERV3_PIXEL_RL_ENVS=64 \
DREAMERV3_PIXEL_RL_FRAMEWORK=jax \
DREAMERV3_PIXEL_RL_RESUME=False \
DREAMERV3_WM_HORIZON=512 \
DREAMERV3_WM_REWARD_QUANTIZE_THRESHOLD=0.1 \
DREAMERV3_WANDB_PROJECT=rl-in-pixel-env-dreamer \
DREAMERV3_LOGGER_NAME="$RL_RUN" \
CONDA_ENV_NAME="${CONDA_ENV_NAME:-dreamer}" \
  bash scripts/experiments/pong_pixel_rl_in_env.sh wm "$WM_CKPT_RESOLVED" "$RL_LOGDIR"

POLICY_CKPT="$RL_LOGDIR/pixel_rl_ckpt/latest.npz"
if [ ! -f "$POLICY_CKPT" ]; then
  echo "[lambda-search] ERROR: zero-shot RL policy missing: $POLICY_CKPT" >&2
  exit 1
fi

echo "[lambda-search] Starting 20-seed real-ALE policy eval."
cd "$PROJECTS_ROOT/CGSReg"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate dreamer
export PYTHONPATH="$DREAMER_ROOT:${PYTHONPATH:-}"
python scripts/eval/evaluate_pong_real_policies.py \
  --episodes 20 \
  --reset-seeds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 \
  --jax-platform cuda \
  --output-dir "$EVAL_OUT" \
  --policy "dreamer_${SIZE}_w${SLUG}=$POLICY_CKPT"

echo "[lambda-search] DONE size=$SIZE lambda=$LAMBDA"
