#!/usr/bin/env bash
set -euo pipefail

NAME="twexp_replay_sr_wm_artifacts_20260621"
PROJECTS_ROOT="${HOME}/projects"
EVAL_ROOT="${PROJECTS_ROOT}/CGSReg"
STAGE="${EVAL_ROOT}/artifacts/${NAME}"
ARCHIVES="${EVAL_ROOT}/artifacts/${NAME}_archives"
BOX_DEST="box:projects/wm-evaluation/dataset_ablation_twister_replay/wm_artifacts/${NAME}/"

mkdir -p "${STAGE}" "${ARCHIVES}"

copy_file() {
  local src="$1"
  local dst="$2"
  if [[ ! -f "${src}" ]]; then
    echo "[ERROR] missing file: ${src}" >&2
    exit 1
  fi
  mkdir -p "$(dirname "${dst}")"
  cp -a "${src}" "${dst}"
}

copy_dir_if_exists() {
  local src="$1"
  local dst="$2"
  if [[ -d "${src}" ]]; then
    mkdir -p "$(dirname "${dst}")"
    cp -a "${src}" "${dst}"
  fi
}

write_manifest() {
  local manifest="${STAGE}/MANIFEST.md"
  {
    echo "# TW exp-repro replay SR WM artifacts"
    echo
    echo "Created on pandaria on 2026-06-21."
    echo
    echo "Box destination: \`${BOX_DEST}\`"
    echo
    echo "Included projects: TWISTER, Dreamer, DIAMOND, Simulus, STORM."
    echo
    echo "Each run uses the TWISTER exp-repro replay dataset with SAM2 masks and mask1 spatial regularization weights w = 0, 0.01, 0.1, 1.0."
    echo
    echo "## Final WM Checkpoints"
    echo
    echo "| Project | w | Included checkpoint |"
    echo "| --- | ---: | --- |"
    echo "| TWISTER | 0 | twister/w0/checkpoints_100000.ckpt |"
    echo "| TWISTER | 0.01 | twister/w0p01/checkpoints_100000.ckpt |"
    echo "| TWISTER | 0.1 | twister/w0p1/checkpoints_100000.ckpt |"
    echo "| TWISTER | 1 | twister/w1/checkpoints_100000.ckpt |"
    echo "| Dreamer | 0 | dreamer/w0/ckpt/agent.pkl |"
    echo "| Dreamer | 0.01 | dreamer/w0p01/ckpt/agent.pkl |"
    echo "| Dreamer | 0.1 | dreamer/w0p1/ckpt/agent.pkl |"
    echo "| Dreamer | 1 | dreamer/w1/ckpt/agent.pkl |"
    echo "| DIAMOND | 0 | diamond/w0/checkpoints/agent_versions/agent_epoch_01000.pt and checkpoints/state.pt |"
    echo "| DIAMOND | 0.01 | diamond/w0p01/checkpoints/agent_versions/agent_epoch_01000.pt and checkpoints/state.pt |"
    echo "| DIAMOND | 0.1 | diamond/w0p1/checkpoints/agent_versions/agent_epoch_01000.pt and checkpoints/state.pt |"
    echo "| DIAMOND | 1 | diamond/w1/checkpoints/agent_versions/agent_epoch_01000.pt and checkpoints/state.pt |"
    echo "| Simulus | 0 | simulus/w0/checkpoints/last.pt |"
    echo "| Simulus | 0.01 | simulus/w0p01/checkpoints/last.pt |"
    echo "| Simulus | 0.1 | simulus/w0p1/checkpoints/last.pt |"
    echo "| Simulus | 1 | simulus/w1/checkpoints/last.pt |"
    echo "| STORM | 0 | storm/w0/ckpt/latest_agent.pth and agent_step_100000.pth |"
    echo "| STORM | 0.01 | storm/w0p01/ckpt/latest_agent.pth and agent_step_100000.pth |"
    echo "| STORM | 0.1 | storm/w0p1/ckpt/latest_agent.pth and agent_step_100000.pth |"
    echo "| STORM | 1 | storm/w1/ckpt/latest_agent.pth and agent_step_100000.pth |"
    echo
    echo "Zero-shot RL for TWISTER is already complete in W&B project \`ssl-lab/rl-in-pixel-env-twister\`."
  } > "${manifest}"
}

stage_twister() {
  local root="${PROJECTS_ROOT}/twister/runs/pong_offline_replay_ac_cpc_w_sweep/logdir"
  copy_file "${root}/offline-replay-ac-cpc-w0/checkpoints/checkpoints_100000.ckpt" "${STAGE}/twister/w0/checkpoints_100000.ckpt"
  copy_file "${root}/offline-replay-ac-cpc-w0p01/checkpoints/checkpoints_100000.ckpt" "${STAGE}/twister/w0p01/checkpoints_100000.ckpt"
  copy_file "${root}/offline-replay-ac-cpc-w0p1/checkpoints/checkpoints_100000.ckpt" "${STAGE}/twister/w0p1/checkpoints_100000.ckpt"
  copy_file "${root}/offline-replay-ac-cpc-w1/checkpoints/checkpoints_100000.ckpt" "${STAGE}/twister/w1/checkpoints_100000.ckpt"
  copy_dir_if_exists "${PROJECTS_ROOT}/twister/pong_pixel_rl_in_env/scheduler_logs/offline_replay_ac_cpc_w_sweep_zero_shot" "${STAGE}/twister/metadata/zero_shot_rl_logs/offline_replay_ac_cpc_w_sweep_zero_shot"
  copy_dir_if_exists "${PROJECTS_ROOT}/twister/pong_pixel_rl_in_env/scheduler_logs/offline_replay_ac_cpc_w0p1_zero_shot_retry1" "${STAGE}/twister/metadata/zero_shot_rl_logs/offline_replay_ac_cpc_w0p1_zero_shot_retry1"
}

stage_dreamer() {
  local root="${PROJECTS_ROOT}/dreamerv3-runs/twexp_offline_sr/logdir"
  local run
  for item in "w0:dreamer_twexp_mask1_spatial_0_temporal_1" "w0p01:dreamer_twexp_mask1_spatial_0p01_temporal_1" "w0p1:dreamer_twexp_mask1_spatial_0p1_temporal_1" "w1:dreamer_twexp_mask1_spatial_1_temporal_1"; do
    local w="${item%%:*}"
    run="${item#*:}"
    local ckpt_dir
    ckpt_dir="$(find "${root}/${run}/ckpt" -mindepth 1 -maxdepth 1 -type d | sort | tail -1)"
    copy_file "${ckpt_dir}/agent.pkl" "${STAGE}/dreamer/${w}/ckpt/agent.pkl"
    copy_file "${root}/${run}/ckpt/latest" "${STAGE}/dreamer/${w}/ckpt/latest"
    copy_file "${root}/${run}/config.yaml" "${STAGE}/dreamer/${w}/config.yaml"
    copy_file "${root}/${run}/metrics.jsonl" "${STAGE}/dreamer/${w}/metrics.jsonl"
  done
}

stage_diamond() {
  local root="${PROJECTS_ROOT}/diamond/outputs/twexp_offline_sr"
  for item in "w0:diamond_twexp_w0" "w0p01:diamond_twexp_w0p01" "w0p1:diamond_twexp_w0p1" "w1:diamond_twexp_w1"; do
    local w="${item%%:*}"
    local run="${item#*:}"
    copy_file "${root}/${run}/checkpoints/agent_versions/agent_epoch_01000.pt" "${STAGE}/diamond/${w}/checkpoints/agent_versions/agent_epoch_01000.pt"
    copy_file "${root}/${run}/checkpoints/state.pt" "${STAGE}/diamond/${w}/checkpoints/state.pt"
    copy_dir_if_exists "${root}/${run}/.hydra" "${STAGE}/diamond/${w}/.hydra"
  done
}

stage_simulus() {
  local root="${PROJECTS_ROOT}/simulus/outputs/twexp_offline_sr/PongNoFrameskip-v4"
  for item in "w0:simulus_twexp_w0/2026-06-18/15-37-58-seed-1" "w0p01:simulus_twexp_w0p01/2026-06-18/15-37-58-seed-1" "w0p1:simulus_twexp_w0p1/2026-06-19/05-55-28-seed-1" "w1:simulus_twexp_w1/2026-06-19/06-02-35-seed-1"; do
    local w="${item%%:*}"
    local run="${item#*:}"
    copy_file "${root}/${run}/checkpoints/last.pt" "${STAGE}/simulus/${w}/checkpoints/last.pt"
    copy_file "${root}/${run}/checkpoints/run_metadata.pt" "${STAGE}/simulus/${w}/checkpoints/run_metadata.pt"
    copy_dir_if_exists "${root}/${run}/.hydra" "${STAGE}/simulus/${w}/.hydra"
  done
}

stage_storm() {
  local root="${PROJECTS_ROOT}/oc-storm/runs"
  for item in "w0:storm_twexp_w0" "w0p01:storm_twexp_w0p01" "w0p1:storm_twexp_w0p1" "w1:storm_twexp_w1"; do
    local w="${item%%:*}"
    local run="${item#*:}"
    copy_file "${root}/${run}/ckpt/latest_agent.pth" "${STAGE}/storm/${w}/ckpt/latest_agent.pth"
    copy_file "${root}/${run}/ckpt/agent_step_100000.pth" "${STAGE}/storm/${w}/ckpt/agent_step_100000.pth"
    copy_file "${root}/${run}/config.py" "${STAGE}/storm/${w}/config.py"
  done
}

stage_metadata() {
  copy_file "${EVAL_ROOT}/experiments/dataset_ablation_twister_replay/README.md" "${STAGE}/metadata/README.md"
  copy_dir_if_exists "${EVAL_ROOT}/experiments/dataset_ablation_twister_replay/commands" "${STAGE}/metadata/commands"
}

make_archives() {
  rm -f "${ARCHIVES}/${NAME}"_*.tar.zst "${ARCHIVES}/SHA256SUMS"
  for part in twister dreamer diamond simulus storm metadata; do
    tar --zstd -cf "${ARCHIVES}/${NAME}_${part}.tar.zst" -C "${STAGE}" "${part}"
  done
  (cd "${ARCHIVES}" && sha256sum "${NAME}"_*.tar.zst > SHA256SUMS)
}

stage_twister
stage_dreamer
stage_diamond
stage_simulus
stage_storm
stage_metadata
write_manifest
make_archives

rclone copy "${ARCHIVES}/" "${BOX_DEST}" --progress
