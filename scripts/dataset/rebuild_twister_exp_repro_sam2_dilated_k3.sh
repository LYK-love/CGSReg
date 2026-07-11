#!/usr/bin/env bash
set -euo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate diamond

SRC_ROOT="${SRC_ROOT:-/data/luyukuan/projects/diamond-assets/datasets/twister_pong_exp_repro_replay_raw_nomask}"
OUT_ROOT="${OUT_ROOT:-/data/luyukuan/projects/diamond-assets/datasets/twister_pong_exp_repro_replay_sam2_dilated_k3_corrected_20260706}"
WM_OUTPUT_ROOT="${WM_OUTPUT_ROOT:-$HOME/projects/shared_replay/twister_pong_exp_repro_replay_sam2_dilated_k3_corrected_20260706_for_wm}"
SAM2_BACKEND_ENDPOINT="${SAM2_BACKEND_ENDPOINT:-http://209.137.198.192:7263}"
MAX_FRAMES_PER_CHUNK="${MAX_FRAMES_PER_CHUNK:-256}"
TARGET_STEPS="${TARGET_STEPS:-100000}"
FORCE="${FORCE:-0}"

DIAMOND_ROOT="${DIAMOND_ROOT:-$HOME/projects/diamond}"
WM_EVAL_ROOT="${WM_EVAL_ROOT:-$HOME/projects/CGSReg}"
LOG_DIR="$WM_EVAL_ROOT/experiments/dataset_ablation_twister_replay/logs/rebuild_twister_dataset_corrected_20260706"

SRC_TRAIN="$SRC_ROOT/train"
OUT_TRAIN="$OUT_ROOT/train"
export OUT_TRAIN TARGET_STEPS

mkdir -p "$LOG_DIR" "$(dirname "$OUT_ROOT")" "$(dirname "$WM_OUTPUT_ROOT")"

probe_endpoint() {
  local endpoint="$1"
  curl -m 10 -fsS \
    -X POST "$endpoint/graphql" \
    -H 'Content-Type: application/json' \
    --data '{"query":"query { __typename }"}' >/dev/null
}

if [[ ! -d "$SRC_TRAIN" ]]; then
  echo "[ERROR] Missing source train split: $SRC_TRAIN" >&2
  exit 1
fi

if ! probe_endpoint "$SAM2_BACKEND_ENDPOINT"; then
  echo "[ERROR] SAM2 endpoint is not reachable: $SAM2_BACKEND_ENDPOINT" >&2
  exit 1
fi

if [[ -e "$OUT_ROOT" || -e "$WM_OUTPUT_ROOT" ]]; then
  if [[ "$FORCE" != "1" ]]; then
    echo "[ERROR] Output already exists. Set FORCE=1 to overwrite." >&2
    echo "  OUT_ROOT=$OUT_ROOT" >&2
    echo "  WM_OUTPUT_ROOT=$WM_OUTPUT_ROOT" >&2
    exit 1
  fi
  rm -rf "$OUT_ROOT" "$WM_OUTPUT_ROOT"
fi

echo "[INFO] Source raw Diamond dataset: $SRC_ROOT"
echo "[INFO] Corrected masked Diamond dataset: $OUT_ROOT"
echo "[INFO] Corrected WM-format output root: $WM_OUTPUT_ROOT"
echo "[INFO] SAM2 backend: $SAM2_BACKEND_ENDPOINT"

echo "[INFO] Copying raw dataset to corrected output root."
mkdir -p "$OUT_ROOT"
cp -a "$SRC_TRAIN" "$OUT_TRAIN"
if [[ -f "$SRC_ROOT/manifest.json" ]]; then
  cp -a "$SRC_ROOT/manifest.json" "$OUT_ROOT/source_manifest.json"
fi

echo "[INFO] Trimming copied train split to TARGET_STEPS=$TARGET_STEPS and refreshing info.pt."
python - <<'PY'
from pathlib import Path
from collections import Counter
import os
import torch
import numpy as np

out_train = Path(os.environ["OUT_TRAIN"])
target_steps = int(os.environ["TARGET_STEPS"])

def episode_path(root: Path, episode_id: int) -> Path:
    parts = []
    for i in range(2, -1, -1):
        value = (episode_id % (10 ** (i + 1))) // (10 ** i) * (10 ** i)
        parts.append(f"{value:0{i + 1}d}")
    return root / "/".join(parts) / f"{episode_id}.pt"

paths = sorted(
    [p for p in out_train.rglob("*.pt") if p.name != "info.pt"],
    key=lambda p: int(p.stem),
)
if not paths:
    raise SystemExit(f"no episode files under {out_train}")

episodes = []
total = 0
for path in paths:
    ep = torch.load(path, map_location="cpu", weights_only=False)
    length = int(len(ep["rew"]))
    if total >= target_steps:
        path.unlink()
        continue
    keep = min(length, target_steps - total)
    if keep < length:
        for key, value in list(ep.items()):
            if torch.is_tensor(value) and len(value) == length:
                ep[key] = value[:keep].clone()
        torch.save(ep, path)
    episodes.append((path, ep, keep))
    total += keep

if total != target_steps:
    raise SystemExit(f"expected {target_steps} steps after trim, got {total}")

lengths = [int(length) for _, _, length in episodes]
starts = np.cumsum([0, *lengths[:-1]], dtype=np.int64)
counter_rew = Counter()
counter_end = Counter()
for _, ep, _ in episodes:
    counter_rew.update(ep["rew"].sign().to(torch.int64).tolist())
    counter_end.update(ep["end"].to(torch.int64).tolist())

info = {
    "is_static": False,
    "num_episodes": len(episodes),
    "num_steps": int(total),
    "start_idx": starts,
    "lengths": lengths,
    "counter_rew": counter_rew,
    "counter_end": counter_end,
    "game_name": "Pong",
}
torch.save(info, out_train / "info.pt")
print({"episodes": len(episodes), "steps": total, "last_length": lengths[-1], "terminal_count": int(counter_end.get(1, 0))})
PY

echo "[INFO] Adding SAM2 masks and applying dilation kernel size 3."
(
  cd "$DIAMOND_ROOT"
  python src/tools/segmentation/add_sam2_mask_to_dataset.py \
    --diamond-dataset-path "$OUT_TRAIN" \
    --guiding-data-dir "$DIAMOND_ROOT/third_party/Experimental-materials/aux" \
    --guiding-video-name pong_size64 \
    --selected-obj-ids 1,2,3 \
    --backend-endpoint "$SAM2_BACKEND_ENDPOINT" \
    --max-frames-per-chunk "$MAX_FRAMES_PER_CHUNK" \
    --dilation-kernel-size 3
)

cat > "$OUT_ROOT/manifest.json" <<EOF
{
  "name": "twister_pong_exp_repro_replay_sam2_dilated_k3_corrected_20260706",
  "source_dataset": "$SRC_ROOT",
  "source_split": "$SRC_TRAIN",
  "format": "diamond_dataset_with_sam2_masks",
  "game_name": "Pong",
  "split": "train",
  "target_steps": $TARGET_STEPS,
  "sam2_backend_endpoint": "$SAM2_BACKEND_ENDPOINT",
  "guiding_video_name": "pong_size64",
  "selected_obj_ids": [1, 2, 3],
  "mask_fields": {
    "mask1": "SAM2 object id 1",
    "mask2": "SAM2 object id 2",
    "mask3": "SAM2 object id 3"
  },
  "dilation_kernel_size": 3,
  "episode_boundaries": "copied from raw_nomask Diamond episodes; no reward-count re-splitting"
}
EOF

echo "[INFO] Inspecting corrected Diamond dataset."
(
  cd "$DIAMOND_ROOT"
  PYTHONPATH="$DIAMOND_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
    python src/tools/dataset/inspect_dataset.py \
      --path "$OUT_TRAIN" \
      --num-episodes 3 \
      --analyze-only
)

echo "[INFO] Converting corrected Diamond dataset to WM project formats."
bash "$WM_EVAL_ROOT/scripts/dataset/convert_twister_exp_repro_sam2_to_wm_datasets.sh" \
  --source-root "$OUT_ROOT" \
  --output-root "$WM_OUTPUT_ROOT" \
  --formats dreamer,simulus,twister,storm \
  --splits train \
  --image-size 64 \
  --force

echo "[INFO] Done."
echo "[INFO] Diamond dataset: $OUT_ROOT"
echo "[INFO] WM datasets: $WM_OUTPUT_ROOT"
