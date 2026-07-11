#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/projects/CGSReg"

while true; do
  gpu="$(
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
      awk -F, '
        {
          g=$1; m=$2; u=$3;
          g+=0; m+=0; u+=0;
          if (g >= 4 && g <= 7 && m < 64 && u == 0) {
            print g;
            exit;
          }
        }'
  )"
  if [[ -n "$gpu" ]]; then
    echo "[$(date -Is)] launching Simulus retry on cuda:${gpu}"
    exec tiny-exp-scheduler run \
      experiments/exp_repro_frozen_wm_rl_only_20seed/commands/pandaria_simulus_retry.commands.txt \
      --cuda-devices "$gpu" \
      --cpu-threads 2 \
      --logs-dir experiments/exp_repro_frozen_wm_rl_only_20seed/logs/pandaria_simulus_retry \
      --scheduler-name exp_repro_simulus_retry \
      --verbose --keep-job-tabs
  fi
  echo "[$(date -Is)] waiting for idle GPU 4-7"
  sleep 60
done
