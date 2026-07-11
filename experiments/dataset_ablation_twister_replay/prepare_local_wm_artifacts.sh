#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$HOME/projects/CGSReg/artifacts/dataset_ablation_twister_replay/wm_artifacts_20260621/unpacked}"

if [[ ! -d "$ROOT" ]]; then
  echo "Artifact root not found: $ROOT" >&2
  exit 1
fi

# Dreamer Elements checkpoints require a `done` marker next to agent.pkl.
# The compact Box archive stores agent.pkl and latest, so recreate the marker
# after extraction.
for ckpt_dir in "$ROOT"/dreamer/*/ckpt; do
  [[ -f "$ckpt_dir/agent.pkl" ]] || continue
  touch "$ckpt_dir/done"
done

echo "Prepared local TW exp-repro replay WM artifacts under: $ROOT"
