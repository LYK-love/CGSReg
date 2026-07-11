# Offline Dataset Ablation Protocol

This document defines the reusable protocol for dataset ablations in the Pong
world-model evaluation paper. It applies to any validated Pong offline dataset
stored in the DIAMOND dataset format.

For dataset download, archive names, checksums, and canonical local paths, see
the dataset-specific documentation, especially `docs/dataset_registry.md`.

## Scope

The ablation asks whether CGSReg remains useful when the offline world model is
trained from a different replay dataset.

For each target dataset, train offline world models for the selected project set
and evaluate the resulting zero-shot RL policies under the same fixed real-ALE
protocol used by the main paper.

Default project set:

```text
DreamerV3
DIAMOND
TWISTER
```

Optional project set, when needed:

```text
DreamerV3
DIAMOND
TWISTER
Simulus
STORM
```

Use this order whenever all five projects are listed:

```text
DreamerV3, DIAMOND, TWISTER, Simulus, STORM
```

## Required Dataset Format

The input dataset must follow `docs/diamond_dataset_standard.md`.

At minimum, the dataset must contain a `train/` split with episode `.pt` files
that include:

```text
obs, act, rew, end, trunc, mask1
```

For datasets used in current paper experiments, `mask2` and `mask3` should also
be present if available, but the default ablation only uses `mask1`.

For projects that require a `test/` split for bookkeeping, create a symlink to
`train/` unless the dataset-specific documentation provides a real test split:

```bash
cd "$DATASET_ROOT"
if [ ! -e test ]; then ln -s train test; fi
```

Before launching jobs, run a lightweight dataset sanity check:

```bash
python - <<'PY'
from pathlib import Path
import torch

root = Path("$DATASET_ROOT").expanduser()
files = sorted((root / "train").glob("*.pt"))
assert files, f"no .pt files under {root / 'train'}"
sample = torch.load(files[0], map_location="cpu")
required = ["obs", "act", "rew", "end", "trunc", "mask1"]
missing = [k for k in required if k not in sample]
assert not missing, f"missing fields in {files[0]}: {missing}"
T = len(sample["act"])
for k in required:
    assert len(sample[k]) == T, (k, len(sample[k]), T)
print("dataset_ok", root, "episodes", len(files), "first_T", T)
PY
```

Replace `$DATASET_ROOT` with the canonical local dataset path before running the
check.

## Experiment Matrix

Use this regularization grid:

```text
lambda_CGSReg in {0, 0.01, 0.1, 1.0}
mask preset = mask1
```

Do not include `lambda_CGSReg=0.001` in paper-facing dataset ablations.

DIAMOND defaults to `mask1` in this ablation protocol. Use a multi-mask DIAMOND
variant only when the experiment explicitly studies multi-mask supervision.

## Directory Layout

Create a dataset-specific experiment directory:

```text
experiments/dataset_ablation_<dataset_id>/
```

Use a short, stable `<dataset_id>`, for example:

```text
storm_replay
twister_replay
```

Recommended layout:

```text
experiments/dataset_ablation_<dataset_id>/
  README.md
  commands/
    dreamer_<dataset_id>_chain.commands.txt
    diamond_<dataset_id>_chain.commands.txt
    twister_<dataset_id>_chain.commands.txt
  logs/
```

For all-five-project ablations, add corresponding Simulus and STORM command
files.

Final normalized results should be stored under:

```text
artifacts/dataset_ablation_<dataset_id>/
  wm_artifacts/
  zero_shot_rl_policy_ckpts/
  fixed_real_eval/
    dreamer/
    diamond/
    twister/
  <dataset_id>_dataset_ablation_20seed_summary.csv
```

## Job Structure

The scheduling unit is one complete
`project x dataset_id x lambda_CGSReg` chain. Each line in a command file must
run all three stages for exactly one world-model checkpoint:

```text
offline world-model training
  -> pixel-space zero-shot RL inside the frozen WM
  -> fixed real-ALE 20-seed policy evaluation
```

Do not schedule the ablation as three global stage batches. In particular, do
not run all offline WM jobs first, then all zero-shot RL jobs, then all
fixed-real evaluations. That staging wastes GPUs when multiple project families
are queued: the first four jobs should each finish their own offline training,
zero-shot RL, and 20-seed eval before the scheduler starts later jobs.

For example, with four GPUs and eight `(project, lambda)` jobs, the scheduler
should run four complete chains first. Only after one full chain exits should a
new chain start on the freed GPU.

Each chain should:

```text
1. validate or locate the dataset
2. train or reuse the matching offline WM checkpoint
3. resolve the final WM checkpoint path
4. train the 20k-update zero-shot RL policy from that frozen WM
5. resolve the final policy checkpoint path
6. run deterministic real-ALE eval over reset seeds 0..19
7. write the per-job summary and append/update the aggregate summary CSV
```

Use separate chain command files per project, and one command line per
`lambda_CGSReg` value:

```text
commands/dreamer_<dataset_id>_chain.commands.txt
  line 1: DreamerV3 lambda=0 complete chain
  line 2: DreamerV3 lambda=0.01 complete chain
  line 3: DreamerV3 lambda=0.1 complete chain
  line 4: DreamerV3 lambda=1.0 complete chain
```

The sections below define the required stages inside each chain.

## Chain Stage 1: Offline World-Model Training

For every selected project and every value of `lambda_CGSReg`, train an offline
world model on the target dataset.

Use the same project-specific offline-training configuration as the main paper
unless the dataset-specific README states otherwise. The key controlled change is
the replay dataset.

Run names should make the dataset and weight unambiguous:

```text
paper-wm-<project>-<dataset_id>-offline-w0
paper-wm-<project>-<dataset_id>-offline-w0p01
paper-wm-<project>-<dataset_id>-offline-w0p1
paper-wm-<project>-<dataset_id>-offline-w1
```

Record each final WM checkpoint in the experiment README:

```text
project
dataset_id
lambda_CGSReg
mask preset
WM checkpoint path
W&B run id and run name
training log path
```

This stage must end by writing a machine-readable file, or printing an easily
grep-able line, that identifies the final WM checkpoint path used by the next
stage. Do not rely on manually editing the next command file.

## Chain Stage 2: Pixel-Space Zero-Shot RL

For every trained world model, train a pixel-space policy inside the frozen WM.
Use the fixed zero-shot RL protocol from the paper:

```text
ac_updates = 20000
envs = 64
backup_every = 15
wm_horizon = 512
eval_real_every = 2000
eval_real_eps = 5
```

The online W&B `eval_real/*` values during training are monitoring signals only.
They are not the paper-facing result.

Run names:

```text
paper-zsrl-<project>-<dataset_id>-offline-w0-h512
paper-zsrl-<project>-<dataset_id>-offline-w0p01-h512
paper-zsrl-<project>-<dataset_id>-offline-w0p1-h512
paper-zsrl-<project>-<dataset_id>-offline-w1-h512
```

Record each final policy checkpoint:

```text
project
dataset_id
lambda_CGSReg
WM checkpoint path
policy checkpoint path
W&B run id and run name
training log path
```

This stage must resolve the final policy checkpoint produced by the current
chain, not a hard-coded checkpoint path from an older run.

## Chain Stage 3: Fixed Real-ALE Evaluation

Evaluate every final policy checkpoint in the real Pong environment with the
same protocol used by the main zero-shot RL table:

```text
episodes = 20
reset seeds = 0..19
policy mode = deterministic / eval
environment = real ALE Pong
metric = total episode return
```

The final paper-facing numbers are:

```text
mean return over 20 seeds
population standard deviation over 20 seeds
per-seed returns
```

Each project should produce:

```text
artifacts/dataset_ablation_<dataset_id>/fixed_real_eval/<project>/pong_real_policy_eval_summary.csv
artifacts/dataset_ablation_<dataset_id>/fixed_real_eval/<project>/pong_real_policy_eval_scores.csv
```

The summary CSV must include enough provenance to trace every number:

```text
dataset_id
project
lambda_cgsreg
episodes
mean_return
std_return
wm_checkpoint
policy_checkpoint
wandb_run_id
wandb_run_name
scores_csv
```

## Scheduling

Use `tmux` and `tiny-exp-scheduler` for long jobs. Do not run large experiments
in a foreground shell. Schedule complete chains, not individual stages.

Example:

```bash
cd "$HOME/projects/CGSReg"
tmux new -d -s dataset_ablation_<dataset_id>_dreamer '
  cd "$HOME/projects/CGSReg" &&
  tiny-exp-scheduler run \
    experiments/dataset_ablation_<dataset_id>/commands/dreamer_<dataset_id>_chain.commands.txt \
    --cuda-devices auto \
    --cpu-threads 4 \
    --logs-dir experiments/dataset_ablation_<dataset_id>/logs/dreamer_chain \
    --verbose --keep-job-tabs
'
```

Each line in `dreamer_<dataset_id>_chain.commands.txt` should be equivalent to:

```bash
bash -lc '
  set -euo pipefail
  cd "$HOME/projects/CGSReg"
  bash experiments/dataset_ablation_<dataset_id>/scripts/run_dreamer_chain.sh \
    --dataset-root "$DATASET_ROOT" \
    --lambda-cgsreg 0.1 \
    --size size200m \
    --gpu "${CUDA_VISIBLE_DEVICES:-0}"
'
```

The exact helper script name is project-specific, but the behavior must match
the chain order above.

Before launching, check:

```bash
hostname
whoami
nvidia-smi
git -C "$HOME/projects/CGSReg" status --short
git -C "$HOME/projects/<project_repo>" status --short
```

Do not kill unrelated sessions or overwrite existing experiment directories.

## Completion Checklist

For each selected project, confirm:

```text
[ ] 4 offline WM checkpoints exist: lambda 0, 0.01, 0.1, 1.0
[ ] 4 zero-shot RL policy checkpoints exist
[ ] 4 fixed real-ALE 20-seed eval summaries exist
[ ] each eval row has episodes=20
[ ] each row has mean, std, and per-seed scores
[ ] all W&B run ids and names are recorded
[ ] command files and logs are preserved
[ ] final aggregate summary CSV exists
```

The final aggregate CSV should contain one row per
`project x lambda_CGSReg` pair:

```text
dataset_id,project,lambda_cgsreg,wm_checkpoint,policy_checkpoint,wandb_run_id,wandb_run_name,episodes,mean_return,std_return,scores_csv,summary_csv
```

## Reporting Back

When the run is launched or completed, report:

```text
dataset_id
canonical dataset path
dataset validation result
command file paths
tmux session names
W&B run ids and run names
completed/running/failed matrix
final summary CSV path
known issues or reruns needed
```
