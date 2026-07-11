#!/usr/bin/env bash
set -euo pipefail

ROOT="$HOME/projects/CGSReg"
OFFLINE_SESSION="${OFFLINE_SESSION:-twexp_dilated_offline_all}"
ZSRL_SESSION="${ZSRL_SESSION:-twexp_dilated_zsrl_all}"
CUDA_DEVICES="${CUDA_DEVICES:-3}"

cd "$ROOT"

echo "[post] waiting for tmux session: ${OFFLINE_SESSION}"
while tmux has-session -t "$OFFLINE_SESSION" 2>/dev/null; do
  sleep 300
done

echo "[post] offline session finished; generating zero-shot RL commands"
python scripts/eval/prepare_twexp_dilated_zsrl_commands.py

echo "[post] launching zero-shot RL scheduler: ${ZSRL_SESSION}"
tmux new -d -s "$ZSRL_SESSION" \
  "cd \"$ROOT\" && tiny-exp-scheduler run experiments/dataset_ablation_twister_replay/commands/all_twexp_zero_shot_rl.commands.txt --cuda-devices ${CUDA_DEVICES} --cpu-threads 2 --logs-dir experiments/dataset_ablation_twister_replay/logs/zero_shot_rl_all --verbose --keep-job-tabs"

echo "[post] launched ${ZSRL_SESSION}"
