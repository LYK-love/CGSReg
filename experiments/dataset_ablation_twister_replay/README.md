# TWISTER Exp-Repro Replay Dataset Ablation

This ablation trains offline Pong world models on the TWISTER exp-repro replay
dataset with SAM2 masks. The sweep uses the same small weight grid:

```text
w = [0.0, 0.01, 0.1, 1.0]
mask preset = mask1
```

## Paper-Use Status

The latest completed 20-seed fixed-real evaluations are tracked in:

```text
experiments/dataset_ablation_twister_replay/twister_dataset_ablation_20seed_eval_scores.md
```

Those numbers were synced into the paper-side CSVs:

```text
nips_paper/data/paper_results/appendix_twister_replay_dataset_ablation_20seed.csv
nips_paper/data/paper_results/appendix_offline_dataset_lambda_ablation_20seed.csv
```

Important status: all TWISTER-dataset ablation scores in this directory and in
the paper-side CSVs currently come from the old TWISTER conversion whose episode
boundaries were not repaired. `docs/dataset_registry.md` marks that conversion
invalid for new runs because some stored `.pt` episodes contain visual resets
inside a single episode. We nevertheless report these completed 20-seed RL
results for now as provisional appendix evidence. When the corrected TWISTER
replay conversion is ready, replace the whole TWISTER-dataset ablation block
with rerun offline-WM -> zero-shot-RL -> 20-seed fixed-real-eval results.

Historical converted dataset root used by the archived runs:

```bash
$HOME/projects/shared_replay/twister_pong_exp_repro_replay_sam2_for_wm
```

The DIAMOND project additionally used the original DIAMOND-format source:

```bash
$HOME/projects/diamond/datasets/twister_pong_exp_repro_replay_sam2
```

These old converted paths are not valid targets for new paper-facing runs. The
dataset has only a `train/` split. For projects that require a test path, the
archived commands either skip offline eval or point test/eval bookkeeping to
train.

## Data Sync

Large datasets and checkpoint bundles should be moved through Box/rclone, not
through direct `rsync` between servers. Restore the converted replay dataset and
WM artifacts from their Box archives, then keep the same logical local paths on
each machine.

For reproducing archived runs only, make DIAMOND's required `test/` split
resolve to train:

```bash
cd "$HOME/projects/diamond/datasets/twister_pong_exp_repro_replay_sam2"
if [ ! -e test ]; then ln -s train test; fi
```

## TWISTER Already Covered

Do not rerun TWISTER from this command set. W&B already has the TWISTER
zero-shot RL jobs for this dataset family in `ssl-lab/rl-in-pixel-env-twister`:

| w | W&B run | State | Final `eval_real/score_mean` |
| ---: | --- | --- | ---: |
| 0.0 | `jb2u331u` | finished | -10.8 |
| 0.01 | `oe7fu37x` | finished | -14.4 |
| 0.1 | retry `kmv64y6h` | finished | -10.6 |
| 1.0 | `sklo7o5n` | finished | -9.6 |

Those runs point to WM checkpoints under:

```text
runs/pong_offline_replay_ac_cpc_w_sweep/logdir/offline-replay-ac-cpc-w*/checkpoints/checkpoints_100000.ckpt
```

Local scheduler logs show all four TWISTER zero-shot RL jobs reached
`update=20000/20000`.

## Final WM Artifacts

All five WM projects have completed the TW exp-repro offline SR ablation on
pandaria:

| Project | Final WM runs |
| --- | --- |
| TWISTER | `offline-replay-ac-cpc-w0`, `w0p01`, `w0p1`, `w1` |
| Dreamer | `dreamer_twexp_mask1_spatial_{0,0p01,0p1,1}_temporal_1` |
| DIAMOND | `diamond_twexp_w0`, `w0p01`, `w0p1`, `w1` |
| Simulus | `simulus_twexp_w0`, `w0p01`, `w0p1`, `w1` |
| STORM | `storm_twexp_w0`, `w0p01`, `w0p1`, `w1` |

The WM-only artifact bundle is staged and uploaded by:

```bash
cd "$HOME/projects/CGSReg"
bash experiments/dataset_ablation_twister_replay/package_wm_artifacts.sh
```

Box destination:

```text
box:projects/wm-evaluation/dataset_ablation_twister_replay/wm_artifacts/twexp_replay_sr_wm_artifacts_20260621/
```

Archive names:

```text
twexp_replay_sr_wm_artifacts_20260621_twister.tar.zst
twexp_replay_sr_wm_artifacts_20260621_dreamer.tar.zst
twexp_replay_sr_wm_artifacts_20260621_diamond.tar.zst
twexp_replay_sr_wm_artifacts_20260621_simulus.tar.zst
twexp_replay_sr_wm_artifacts_20260621_storm.tar.zst
twexp_replay_sr_wm_artifacts_20260621_metadata.tar.zst
SHA256SUMS
```

DIAMOND archives include both `agent_epoch_01000.pt` and `state.pt`.

Local restore/check example:

```bash
cd "$HOME/projects/CGSReg"
mkdir -p artifacts/dataset_ablation_twister_replay/wm_artifacts_20260621
rclone copy \
  box:projects/wm-evaluation/dataset_ablation_twister_replay/wm_artifacts/twexp_replay_sr_wm_artifacts_20260621/ \
  artifacts/dataset_ablation_twister_replay/wm_artifacts_20260621/ \
  --progress
cd artifacts/dataset_ablation_twister_replay/wm_artifacts_20260621
sha256sum -c SHA256SUMS
mkdir -p unpacked
for archive in *.tar.zst; do tar --zstd -xf "$archive" -C unpacked; done
cd "$HOME/projects/CGSReg"
bash experiments/dataset_ablation_twister_replay/prepare_local_wm_artifacts.sh
```

## Zero-Shot RL Policy Training

TWISTER zero-shot RL is already complete, so only Dreamer, DIAMOND, Simulus,
and STORM are launched from this experiment directory. These jobs train the
pixel policy inside each frozen WM:

```text
ac_updates = 20000
envs = 64
backup_every = 15
wm_horizon = 512
eval_real_every = 2000
eval_real_eps = 5
```

The online `eval_real/*` metrics produced during these runs are useful
monitoring signals, but they are not the table source. For the paper, evaluate
the saved zero-shot policy checkpoint with the fixed real-ALE protocol below.

Command files:

```text
commands/dreamer_twexp_zero_shot_rl.commands.txt
commands/diamond_twexp_zero_shot_rl.commands.txt
commands/simulus_twexp_zero_shot_rl.commands.txt
commands/storm_twexp_zero_shot_rl.commands.txt
```

## Fixed Real-ALE Evaluation

Use the same local evaluation protocol as the main zero-shot MBRL table:

```text
episodes = 20
reset_seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
policy = deterministic / eval mode
```

Run the fixed-real-eval commands after the zero-shot policy checkpoints are
available. Skip a project if its
`artifacts/dataset_ablation_twister_replay/fixed_real_eval/<project>/pong_real_policy_eval_summary.csv`
already reports `episodes=20` for every policy row.

```bash
cd "$HOME/projects/CGSReg"
tiny-exp-scheduler run \
  experiments/dataset_ablation_twister_replay/commands/dreamer_twexp_fixed_real_eval.commands.txt \
  --cuda-devices auto \
  --cpu-threads 2 \
  --logs-dir experiments/dataset_ablation_twister_replay/logs/fixed_real_eval_dreamer \
  --verbose --keep-job-tabs

tiny-exp-scheduler run \
  experiments/dataset_ablation_twister_replay/commands/diamond_twexp_fixed_real_eval.commands.txt \
  --cuda-devices auto \
  --cpu-threads 2 \
  --logs-dir experiments/dataset_ablation_twister_replay/logs/fixed_real_eval_diamond \
  --verbose --keep-job-tabs

tiny-exp-scheduler run \
  experiments/dataset_ablation_twister_replay/commands/twister_twexp_fixed_real_eval.commands.txt \
  --cuda-devices auto \
  --cpu-threads 2 \
  --logs-dir experiments/dataset_ablation_twister_replay/logs/fixed_real_eval_twister \
  --verbose --keep-job-tabs
```

If evaluating from the archived policy bundle, extract it first:

```bash
cd "$HOME/projects/CGSReg/artifacts/twexp_replay_sr_zero_shot_rl_policy_ckpts_20260624"
tar --zstd -xf twexp_replay_sr_zero_shot_rl_policy_ckpts_20260624.tar.zst
```

Dreamer fixed eval requires the policy run `config.yaml` next to the policy
checkpoint. The original logdirs under
`$HOME/projects/dreamerv3-runs/pong_pixel_rl_in_env/logdir` contain these
configs. If using only the archive, make sure future policy packages include
the Dreamer policy-run `config.yaml` files.

### 2026-07-04 Dreamer/TWISTER Package

Pandaria completed and packaged the currently finished replay-ablation
offline-WM -> zero-shot-RL -> fixed-real-eval artifacts for DreamerV3 and
TWISTER:

```text
box:projects/wm-evaluation/dataset_ablation_twister_replay/completed_results/twexp_replay_ablation_completed_dreamer_twister_20260704/
```

Archive:

```text
twexp_replay_ablation_completed_dreamer_twister_20260704.tar.zst
twexp_replay_ablation_completed_dreamer_twister_20260704.tar.zst.sha256
```

Included contents:

- DreamerV3 dilated-k3 replay ablation: final offline WM checkpoints, 20k AC
  zero-shot policy checkpoints, scheduler logs, and 20-seed fixed-real eval.
- TWISTER replay ablation: final offline WM checkpoints, 20k AC zero-shot
  policy checkpoints, scheduler logs, and 20-seed fixed-real eval.

Dataset note: the DreamerV3 runs used the 2026-07-02 dilated-k3 Dreamer-format
dataset under
`$HOME/projects/shared_replay/twister_pong_exp_repro_replay_sam2_for_wm/dreamer`.
The TWISTER runs used the corresponding TWISTER-format dilated-k3 conversion.
Both converted TWISTER replay datasets are now marked invalid for new runs in
`docs/dataset_registry.md` because of episode-boundary issues. The paper
currently reports these completed results provisionally; replace them after the
corrected TWISTER replay dataset is regenerated and re-evaluated.

20-seed fixed-real scores:

| Project | w | score_mean |
| --- | ---: | ---: |
| DreamerV3 dilated-k3 | 0 | -20.80 |
| DreamerV3 dilated-k3 | 0.01 | -20.00 |
| DreamerV3 dilated-k3 | 0.1 | -19.85 |
| DreamerV3 dilated-k3 | 1.0 | -21.00 |
| TWISTER dilated-k3 | 0 | -17.20 |
| TWISTER dilated-k3 | 0.01 | -13.20 |
| TWISTER dilated-k3 | 0.1 | -9.20 |
| TWISTER dilated-k3 | 1.0 | 9.45 |

DIAMOND dilated-k3 replay ablation was started on pandaria separately in tmux
session `twexp_diamond_offline_sr_dilated_k3_20260704`. It was still in the
offline WM stage when this Dreamer/TWISTER package was created, so DIAMOND is
not included in this archive.

Summarize completed fixed evaluations with:

```bash
cd "$HOME/projects/CGSReg"
python scripts/eval/summarize_twexp_fixed_real_eval.py
```

This writes:

```text
results/dataset_ablation_twister_replay/twexp_replay_fixed_real_eval_20seed_scores.csv
```

Do not use `results/dataset_ablation_twister_replay/twexp_replay_zero_shot_wandb_scores.csv`
as the appendix table source; it records single-run W&B final `eval_real`
summaries, not the main-paper fixed-real protocol.

### Zero-Shot RL Policy Artifacts

The completed local zero-shot RL policy checkpoints and logs are staged in:

```text
$HOME/projects/CGSReg/artifacts/twexp_replay_sr_zero_shot_rl_policy_ckpts_20260624/
```

Box destination:

```text
box:projects/wm-evaluation/dataset_ablation_twister_replay/zero_shot_rl_policy_ckpts/twexp_replay_sr_zero_shot_rl_policy_ckpts_20260624/
```

Archive name:

```text
twexp_replay_sr_zero_shot_rl_policy_ckpts_20260624.tar.zst
```

Included local policy checkpoints:

| Family | Runs |
| --- | --- |
| TWISTER replay SR | w=0, 0.01, 0.1 retry, 1 |
| Dreamer replay SR | w=0, 0.01, 0.1, 1 |
| DIAMOND replay SR | w=0, 0.01, 0.1, 1 |
| DIAMOND mask1+mask3 | w=0.001, 0.01, 0.1, 1 |

Simulus and STORM replay SR zero-shot RL completed and their scheduler logs are
included in the archive, but their current local entrypoints did not leave
local policy checkpoint files. To include Simulus or STORM in the fixed-real
table, rerun or repackage their zero-shot jobs so that the final policy
checkpoint is saved locally, then evaluate it with the same fixed seeds above.

Recommended scheduler launch on pandaria:

```bash
tmux new -d -s twexp_zsrl_dreamer 'cd "$HOME/projects/dreamerv3-reborn" && tiny-exp-scheduler run "$HOME/projects/CGSReg/experiments/dataset_ablation_twister_replay/commands/dreamer_twexp_zero_shot_rl.commands.txt" --cuda-devices 0,1 --cpu-threads 2 --logs-dir "$HOME/projects/dreamerv3-runs/twexp_zero_shot_rl/scheduler_logs_retry1" --verbose --keep-job-tabs'
tmux new -d -s twexp_zsrl_diamond 'cd "$HOME/projects/diamond" && tiny-exp-scheduler run "$HOME/projects/CGSReg/experiments/dataset_ablation_twister_replay/commands/diamond_twexp_zero_shot_rl.commands.txt" --cuda-devices 2,3 --cpu-threads 2 --logs-dir outputs/twexp_zero_shot_rl/logs --verbose --keep-job-tabs'
tmux new -d -s twexp_zsrl_simulus 'cd "$HOME/projects/simulus" && tiny-exp-scheduler run "$HOME/projects/CGSReg/experiments/dataset_ablation_twister_replay/commands/simulus_twexp_zero_shot_rl.commands.txt" --cuda-devices 4,5 --cpu-threads 2 --logs-dir outputs/twexp_zero_shot_rl/logs --verbose --keep-job-tabs'
tmux new -d -s twexp_zsrl_storm 'cd "$HOME/projects/oc-storm" && tiny-exp-scheduler run "$HOME/projects/CGSReg/experiments/dataset_ablation_twister_replay/commands/storm_twexp_zero_shot_rl.commands.txt" --cuda-devices 6,7 --cpu-threads 2 --logs-dir runs/twexp_zero_shot_rl/scheduler_logs --verbose --keep-job-tabs'
```

## Scheduler Commands

Run each from the target project root on pandaria.

Dreamer:

```bash
cd "$HOME/projects/dreamerv3-reborn"
tiny-exp-scheduler run "$HOME/projects/CGSReg/experiments/dataset_ablation_twister_replay/commands/dreamer_twexp_offline_sr.commands.txt" \
  --cuda-devices auto \
  --cpu-threads 2 \
  --logs-dir "$HOME/projects/dreamerv3-runs/twexp_offline_sr/scheduler_logs" \
  --verbose --keep-job-tabs
```

DIAMOND:

```bash
cd "$HOME/projects/diamond"
tiny-exp-scheduler run "$HOME/projects/CGSReg/experiments/dataset_ablation_twister_replay/commands/diamond_twexp_offline_sr.commands.txt" \
  --cuda-devices auto \
  --cpu-threads 2 \
  --logs-dir outputs/twexp_offline_sr/logs \
  --verbose --keep-job-tabs
```

Simulus:

```bash
cd "$HOME/projects/simulus"
tiny-exp-scheduler run "$HOME/projects/CGSReg/experiments/dataset_ablation_twister_replay/commands/simulus_twexp_offline_sr.commands.txt" \
  --cuda-devices auto \
  --cpu-threads 2 \
  --logs-dir outputs/twexp_offline_sr/logs \
  --verbose --keep-job-tabs
```

STORM:

```bash
cd "$HOME/projects/oc-storm"
tiny-exp-scheduler run "$HOME/projects/CGSReg/experiments/dataset_ablation_twister_replay/commands/storm_twexp_offline_sr.commands.txt" \
  --cuda-devices auto \
  --cpu-threads 2 \
  --logs-dir runs/twexp_offline_sr/scheduler_logs \
  --verbose --keep-job-tabs
```

## Fixed-Policy Rollout Diagnostics

This is the appendix diagnostic that compares closed-loop generation after
changing the offline training replay dataset. It uses the same deterministic
real-env controller policy as the main rollout diagnostics.

Run the rollout jobs:

```bash
cd "$HOME/projects/CGSReg"
tiny-exp-scheduler run \
  experiments/dataset_ablation_twister_replay/commands/twexp_rollout_diagnostics.commands.txt \
  --cuda-devices auto \
  --cpu-threads 2 \
  --logs-dir experiments/dataset_ablation_twister_replay/logs/twexp_rollout_diagnostics \
  --verbose --keep-job-tabs
```

Build the compact video/metadata bundle:

```bash
python scripts/eval/build_rollout_eval_bundle.py \
  --manifest experiments/dataset_ablation_twister_replay/target_rollouts.json \
  --output-dir artifacts/twexp_replay_sr_rollout_eval_bundle
```

Run CUTIE segmentation on the bundle:

```bash
conda activate oc-storm
CUDA_VISIBLE_DEVICES=0 python scripts/eval/run_oc_storm_cutie_pong_tracks.py \
  --bundle-root artifacts/twexp_replay_sr_rollout_eval_bundle \
  --output-dir artifacts/twexp_replay_sr_rollout_eval_bundle/cutie_segmentations \
  --geometry square-to-atari \
  --fps 15
```

Aggregate the mask-based metrics:

```bash
python scripts/eval/make_cutie_rollout_quant.py \
  --bundle-root artifacts/twexp_replay_sr_rollout_eval_bundle \
  --output-dir artifacts/twexp_replay_sr_rollout_eval_bundle/results
```
