#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Convert the SAM2-masked TWISTER exp-repro replay, stored in DIAMOND dataset
format, into dataset formats for DreamerV3, Simulus, TWISTER, and OC-STORM.

Defaults assume the source dataset was produced by DIAMOND's SAM2 masking tool:

  $HOME/projects/diamond/datasets/twister_pong_exp_repro_replay_sam2_ball_dilated_k3/train

Usage:
  scripts/dataset/convert_twister_exp_repro_sam2_to_wm_datasets.sh [options]

Options:
  --source-root PATH     DIAMOND dataset root containing train/ [default:
                         $HOME/projects/diamond/datasets/twister_pong_exp_repro_replay_sam2_ball_dilated_k3]
  --output-root PATH     Output root [default:
                         $HOME/projects/shared_replay/twister_pong_exp_repro_replay_sam2_for_wm]
  --formats CSV         Any of dreamer,simulus,twister,storm [default: all four]
  --splits CSV          DIAMOND splits for converters that support split selection [default: train]
  --chunk-size N        Dreamer replay chunk size [default: 1024]
  --image-size N        Resize image/mask to NxN; 0 preserves source size [default: 64]
  --limit N             Optional max episodes for Simulus/TWISTER converter [default: 0]
  --include-ram         Preserve RAM when converter supports it
  --force               Overwrite existing output directories
  -h, --help            Show this help

Output layout:
  $OUTPUT_ROOT/dreamer/<split>/     DreamerV3 replay chunks
  $OUTPUT_ROOT/simulus/<split>/     Simulus native offline episodes
  $OUTPUT_ROOT/twister/<split>/     TWISTER offline WM episodes
  $OUTPUT_ROOT/storm/{train,eval}/  OC-STORM offline episodes
EOF
}

SOURCE_ROOT="$HOME/projects/diamond/datasets/twister_pong_exp_repro_replay_sam2_ball_dilated_k3"
OUTPUT_ROOT="$HOME/projects/shared_replay/twister_pong_exp_repro_replay_sam2_for_wm"
FORMATS="dreamer,simulus,twister,storm"
SPLITS="train"
CHUNK_SIZE="1024"
IMAGE_SIZE="64"
LIMIT="0"
INCLUDE_RAM="0"
FORCE="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-root)
      SOURCE_ROOT="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --formats)
      FORMATS="$2"
      shift 2
      ;;
    --splits)
      SPLITS="$2"
      shift 2
      ;;
    --chunk-size)
      CHUNK_SIZE="$2"
      shift 2
      ;;
    --image-size)
      IMAGE_SIZE="$2"
      shift 2
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --include-ram)
      INCLUDE_RAM="1"
      shift
      ;;
    --force)
      FORCE="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

SOURCE_ROOT="$(readlink -f "${SOURCE_ROOT/#\~/$HOME}")"
OUTPUT_ROOT="${OUTPUT_ROOT/#\~/$HOME}"
mkdir -p "$(dirname "$OUTPUT_ROOT")"
OUTPUT_ROOT="$(cd "$(dirname "$OUTPUT_ROOT")" && pwd)/$(basename "$OUTPUT_ROOT")"

DIAMOND_ROOT="${DIAMOND_ROOT:-$HOME/projects/diamond}"
DREAMERV3_ROOT="${DREAMERV3_ROOT:-$HOME/projects/dreamerv3-reborn}"
SIMULUS_ROOT="${SIMULUS_ROOT:-$HOME/projects/simulus}"
STORM_ROOT="${STORM_ROOT:-$HOME/projects/oc-storm}"

if [[ ! -d "$SIMULUS_ROOT" && -d "$HOME/projects/Simulus" ]]; then
  SIMULUS_ROOT="$HOME/projects/Simulus"
fi

require_dir() {
  local path="$1"
  local label="$2"
  if [[ ! -d "$path" ]]; then
    echo "Missing $label: $path" >&2
    exit 1
  fi
}

contains_format() {
  local needle="$1"
  case ",$FORMATS," in
    *",$needle,"*) return 0 ;;
    *) return 1 ;;
  esac
}

validate_formats() {
  local IFS=","
  read -ra requested <<< "$FORMATS"
  for fmt in "${requested[@]}"; do
    case "$fmt" in
      dreamer|simulus|twister|storm) ;;
      "")
        ;;
      *)
        echo "Unknown format in --formats: $fmt" >&2
        exit 2
        ;;
    esac
  done
}

csv_intersection() {
  local wanted="$1"
  local out=()
  local IFS=","
  read -ra requested <<< "$FORMATS"
  for fmt in "${requested[@]}"; do
    case ",$wanted," in
      *",$fmt,"*) out+=("$fmt") ;;
    esac
  done
  local joined=""
  for fmt in "${out[@]}"; do
    if [[ -z "$joined" ]]; then
      joined="$fmt"
    else
      joined="$joined,$fmt"
    fi
  done
  printf '%s' "$joined"
}

validate_formats
require_dir "$SOURCE_ROOT" "source DIAMOND dataset root"
require_dir "$SOURCE_ROOT/train" "source train split"

echo "Source DIAMOND dataset: $SOURCE_ROOT"
echo "Output root: $OUTPUT_ROOT"
echo "Formats: $FORMATS"
echo "Splits: $SPLITS"

COMMON_FORCE=()
if [[ "$FORCE" == "1" ]]; then
  COMMON_FORCE+=(--force)
fi

RAM_ARGS=()
if [[ "$INCLUDE_RAM" == "1" ]]; then
  RAM_ARGS+=(--include-ram)
fi

SIMULUS_TWISTER_FORMATS="$(csv_intersection "simulus,twister")"
if [[ -n "$SIMULUS_TWISTER_FORMATS" ]]; then
  require_dir "$SIMULUS_ROOT" "Simulus repo"
  LIMIT_ARGS=()
  if [[ "$LIMIT" != "0" ]]; then
    LIMIT_ARGS+=(--limit "$LIMIT")
  fi
  echo "[simulus/twister] $SOURCE_ROOT -> $OUTPUT_ROOT formats=$SIMULUS_TWISTER_FORMATS"
  (
    cd "$SIMULUS_ROOT"
    conda run --no-capture-output -n simulus \
      python -u scripts/dataset/convert_diamond_replay.py \
        --diamond-root "$SOURCE_ROOT" \
        --output-root "$OUTPUT_ROOT" \
        --formats "$SIMULUS_TWISTER_FORMATS" \
        --splits "$SPLITS" \
        --include-masks \
        --image-size "$IMAGE_SIZE" \
        "${LIMIT_ARGS[@]}" \
        "${RAM_ARGS[@]}" \
        "${COMMON_FORCE[@]}"
  )
fi

if contains_format dreamer; then
  require_dir "$DIAMOND_ROOT" "DIAMOND repo"
  require_dir "$DREAMERV3_ROOT" "DreamerV3 repo"
  IFS="," read -ra split_list <<< "$SPLITS"
  for split in "${split_list[@]}"; do
    [[ -z "$split" ]] && continue
    input_split="$SOURCE_ROOT/$split"
    if [[ ! -d "$input_split" ]]; then
      echo "[dreamer] skip missing optional split: $input_split"
      continue
    fi
    output_split="$OUTPUT_ROOT/dreamer/$split"
    echo "[dreamer] $input_split -> $output_split"
    (
      cd "$DIAMOND_ROOT"
      DREAMERV3_ROOT="$DREAMERV3_ROOT" \
      conda run --no-capture-output -n dreamer \
        python -u src/tools/dataset/convert_diamond_replay.py \
          --input-root "$input_split" \
          --output-root "$output_split" \
          --chunk-size "$CHUNK_SIZE" \
          --include-masks \
          "${RAM_ARGS[@]}" \
          "${COMMON_FORCE[@]}"
    )
  done
fi

if contains_format storm; then
  require_dir "$STORM_ROOT" "OC-STORM repo"
  echo "[storm] $SOURCE_ROOT -> $OUTPUT_ROOT/storm"
  (
    cd "$STORM_ROOT"
    conda run --no-capture-output -n oc-storm \
      python -u scripts/dataset/convert_diamond_replay.py \
        --diamond-root "$SOURCE_ROOT" \
        --output-root "$OUTPUT_ROOT/storm" \
        --include-masks \
        --image-size "$IMAGE_SIZE" \
        "${RAM_ARGS[@]}" \
        "${COMMON_FORCE[@]}"
  )
fi

cat > "$OUTPUT_ROOT/README.md" <<EOF
# TWISTER exp-repro replay converted datasets

Source DIAMOND dataset:

\`\`\`
$SOURCE_ROOT
\`\`\`

Generated by:

\`\`\`
$0 --source-root "$SOURCE_ROOT" --output-root "$OUTPUT_ROOT" --formats "$FORMATS" --splits "$SPLITS" --chunk-size "$CHUNK_SIZE" --image-size "$IMAGE_SIZE"$([[ "$INCLUDE_RAM" == "1" ]] && printf ' --include-ram')$([[ "$FORCE" == "1" ]] && printf ' --force')
\`\`\`

Output layout:

- \`dreamer/<split>/\`: DreamerV3 replay chunks.
- \`simulus/<split>/\`: Simulus native offline episodes.
- \`twister/<split>/\`: TWISTER offline WM episodes.
- \`storm/{train,eval}/\`: OC-STORM offline episodes. If the source has no \`test/\`, only \`storm/train/\` is written.

Mask fields \`mask1\`, \`mask2\`, and \`mask3\` are preserved by default.
EOF

echo "Done. Wrote outputs under: $OUTPUT_ROOT"
