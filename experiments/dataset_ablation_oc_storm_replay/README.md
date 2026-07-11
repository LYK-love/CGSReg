# OC-STORM Replay Dataset Ablation

This experiment follows the offline dataset ablation protocol for the validated
OC-STORM Pong exp-repro replay dataset with SAM2 dilated-k3 masks.

Dataset id:

```text
oc_storm_replay
```

Dataset status is recorded in `docs/dataset_registry.md`.

## Local Dataset Paths

Canonical local paths on pandaria:

```text
$HOME/projects/shared_replay/oc_storm_pong_exp_repro_replay_sam2_dilated_k3
$HOME/projects/shared_replay/oc_storm_pong_exp_repro_replay_sam2_dilated_k3_for_wm
$HOME/projects/diamond/datasets/oc_storm_pong_exp_repro_replay_sam2_dilated_k3
```

DIAMOND-format source:

```text
/data/luyukuan/projects/diamond-assets/datasets/oc_storm_pong_exp_repro_replay_sam2_dilated_k3
```

Registry source:

```text
docs/dataset_registry.md
box:zero-shot-rl/storm/oc_storm_pong_exp_repro_replay_sam2_dilated_k3_diamond_20260705/
```

Validation on pandaria:

```text
dataset_ok /home/lyk/projects/shared_replay/oc_storm_pong_exp_repro_replay_sam2_dilated_k3 episodes 38 first_T 1262
fields ['act', 'end', 'important_event_indicator', 'info', 'mask1', 'mask2', 'mask3', 'obs', 'rew', 'trunc']
```

Converted WM dataset outputs:

```text
DreamerV3: $HOME/projects/shared_replay/oc_storm_pong_exp_repro_replay_sam2_dilated_k3_for_wm/dreamer/train
TWISTER:   $HOME/projects/shared_replay/oc_storm_pong_exp_repro_replay_sam2_dilated_k3_for_wm/twister/train
DIAMOND:   $HOME/projects/diamond/datasets/oc_storm_pong_exp_repro_replay_sam2_dilated_k3
Simulus:   $HOME/projects/shared_replay/oc_storm_pong_exp_repro_replay_sam2_dilated_k3_for_wm/simulus/train
```

Dreamer conversion produced 98 replay chunks. TWISTER conversion produced 38
episodes and 100000 steps.

The Simulus-format split is generated with:

```bash
cd "$HOME/projects/CGSReg"
conda run --no-capture-output -n simulus python scripts/dataset/convert_diamond_dataset_to_simulus.py \
  --diamond-train-dir /data/luyukuan/projects/diamond-assets/datasets/oc_storm_pong_exp_repro_replay_sam2_dilated_k3/train \
  --output-train-dir "$HOME/projects/shared_replay/oc_storm_pong_exp_repro_replay_sam2_dilated_k3_for_wm/simulus/train"
```

## Stage 1 Offline WM Training

Project set for this completed launch:

```text
DreamerV3, DIAMOND, TWISTER
```

Grid:

```text
lambda_CGSReg in {0, 0.01, 0.1, 1.0}
mask preset = mask1
```

Command files:

```text
commands/dreamer_oc_storm_replay_offline_sr.commands.txt
commands/twister_oc_storm_replay_offline_sr.commands.txt
commands/diamond_oc_storm_replay_offline_sr.commands.txt
```

Initial scheduling plan on pandaria:

```text
CUDA 0,1,2,3: DreamerV3 offline WM jobs
CUDA 4,5,6,7: TWISTER offline WM jobs
DIAMOND: pending until one of the running schedulers frees GPUs
```

Launch commands:

```bash
cd "$HOME/projects/CGSReg"

tmux new -d -s ocstorm_dreamer_offline_sr '
  cd "$HOME/projects/CGSReg" &&
  tiny-exp-scheduler run \
    experiments/dataset_ablation_oc_storm_replay/commands/dreamer_oc_storm_replay_offline_sr.commands.txt \
    --cuda-devices 0,1,2,3 \
    --cpu-threads 2 \
    --logs-dir experiments/dataset_ablation_oc_storm_replay/logs/offline_sr_dreamer \
    --scheduler-name ocstorm-dreamer-offline \
    --verbose --keep-job-tabs
'

tmux new -d -s ocstorm_twister_offline_sr '
  cd "$HOME/projects/CGSReg" &&
  tiny-exp-scheduler run \
    experiments/dataset_ablation_oc_storm_replay/commands/twister_oc_storm_replay_offline_sr.commands.txt \
    --cuda-devices 4,5,6,7 \
    --cpu-threads 2 \
    --logs-dir experiments/dataset_ablation_oc_storm_replay/logs/offline_sr_twister \
    --scheduler-name ocstorm-twister-offline \
    --verbose --keep-job-tabs
'
```

Pending DIAMOND launch:

```bash
cd "$HOME/projects/CGSReg"
tmux new -d -s ocstorm_diamond_offline_sr '
  cd "$HOME/projects/CGSReg" &&
  tiny-exp-scheduler run \
    experiments/dataset_ablation_oc_storm_replay/commands/diamond_oc_storm_replay_offline_sr.commands.txt \
    --cuda-devices 0,1,2,3 \
    --cpu-threads 2 \
    --logs-dir experiments/dataset_ablation_oc_storm_replay/logs/offline_sr_diamond \
    --scheduler-name ocstorm-diamond-offline \
    --verbose --keep-job-tabs
'
```

## Simulus

The Simulus follow-up command runs one complete chain:

```text
offline WM training -> 20k-update zero-shot RL -> 20-seed real-ALE eval
```

Command file:

```text
experiments/dataset_ablation_oc_storm_replay/commands/simulus_ocstorm_chain.commands.txt
```

Launch on four local GPUs:

```bash
cd "$HOME/projects/CGSReg"
tiny-exp-scheduler run \
  experiments/dataset_ablation_oc_storm_replay/commands/simulus_ocstorm_chain.commands.txt \
  --cuda-devices 0,1,2,3 \
  --cpu-threads 2 \
  --logs-dir experiments/dataset_ablation_oc_storm_replay/logs/simulus_chain \
  --verbose \
  --keep-job-tabs
```

Fixed-real eval outputs:

```text
artifacts/dataset_ablation_oc_storm_replay/fixed_real_eval/simulus/<lambda_slug>/
```

## Completion Checklist

```text
[x] DreamerV3: 4 offline WM checkpoints
[x] TWISTER: 4 offline WM checkpoints
[x] DIAMOND: 4 offline WM checkpoints
[x] DreamerV3: 4 zero-shot RL policy checkpoints
[x] TWISTER: 4 zero-shot RL policy checkpoints
[x] DIAMOND: 4 zero-shot RL policy checkpoints
[x] DreamerV3: 4 fixed real-ALE 20-seed eval summaries
[x] TWISTER: 4 fixed real-ALE 20-seed eval summaries
[x] DIAMOND: 4 fixed real-ALE 20-seed eval summaries
[x] final aggregate summary CSV
```

## Runtime Notes

ZSRL/eval scheduling is handled by explicit command files plus tmux watchers.
Do not use `#include` pseudo-directives in command files; `tiny-exp-scheduler`
expects one runnable shell command per non-comment line.

DreamerV3 ZSRL must use the dated checkpoint directories under `ckpt/`, not the
`ckpt/latest` alias. The Dreamer loader checks for the concrete checkpoint path.

TWISTER `w0` retry2 stopped logging at update 9000/20000 while the process kept
running. It was stopped and restarted as:

```text
oc_storm_replay_twister_w0_retry3_ac20k_h512_rewq0p5
```

Stage 2/3 orchestration:

```text
ocstorm_twister_w0_zsrl_retry3
ocstorm_dreamer_zsrl_now
ocstorm_diamond_zsrl_now
ocstorm_eval_after_twister_dreamer_watch
ocstorm_diamond_eval_watch
```

## Final Results

All fixed real-ALE evaluations use 20 deterministic reset seeds, `0..19`.

Summary and per-rollout result files:

```text
oc_storm_replay_dataset_ablation_20seed_scores.md
oc_storm_replay_dataset_ablation_20seed_summary.csv
oc_storm_replay_dataset_ablation_20seed_scores.csv
```

Compact summary:

| Project | lambda_CGSReg | Mean return | Std return |
| --- | ---: | ---: | ---: |
| DreamerV3 | 0 | -18.55 | 2.4597 |
| DreamerV3 | 0.01 | -15.15 | 9.4271 |
| DreamerV3 | 0.1 | -20.20 | 0.6959 |
| DreamerV3 | 1.0 | -21.00 | 0.0000 |
| TWISTER | 0 | -21.00 | 0.0000 |
| TWISTER | 0.01 | -17.20 | 3.3966 |
| TWISTER | 0.1 | -20.80 | 0.4104 |
| TWISTER | 1.0 | -21.00 | 0.0000 |
| DIAMOND | 0 | -20.30 | 0.4702 |
| DIAMOND | 0.01 | -20.25 | 1.3328 |
| DIAMOND | 0.1 | -4.60 | 15.2605 |
| DIAMOND | 1.0 | -20.80 | 0.4104 |

## Checkpoint Archive

Final WM and zero-shot RL policy checkpoints are uploaded to UC Davis Box:

```text
box:zero-shot-rl/storm/oc_storm_replay_dataset_ablation_ckpts_20260707/
Box folder id: 397526027836
```

The checkpoint package contains final offline WM checkpoints and 20k zero-shot
RL policy checkpoints for DreamerV3, TWISTER, and DIAMOND. It also includes
`MANIFEST.md`, `SHA256SUMS.txt`, and copies of the final result CSV/MD metadata.
