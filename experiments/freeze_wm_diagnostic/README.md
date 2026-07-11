# Freeze-WM Diagnostic

This directory contains the Pong Dyna-style freeze-WM diagnostic for the five
world-model projects used in `CGSReg`.

The compact technical report for the five-project claim check is
[`dyna_freeze_wm_claim_report.md`](dyna_freeze_wm_claim_report.md).

The shared experiment contract is documented in
[`conventions.md`](conventions.md). Per-project adaptors live in
[`adaptors/`](adaptors/) and record each implementation's run names, local
output paths, freeze parameter mapping, W&B logging requirements, and runtime
notes.

## Directory

```text
experiments/freeze_wm_diagnostic/
  README.md
  commands/
    dreamer_pong_freeze_wm.commands.txt
    diamond_pong_freeze_wm.commands.txt
    simulus_pong_freeze_wm.commands.txt
    twister_pong_freeze_wm.commands.txt
    storm_pong_freeze_wm.commands.txt
  conventions.md
  adaptors/
    README.md
    dreamer.md
    diamond.md
    simulus.md
    twister.md
    storm.md
```

Each command file has four jobs:

```text
<project>_repro_nofreeze_1p5T
<project>_repro_f0p5_to1p5T
<project>_repro_f0p75_to1p5T
<project>_repro_f1p0_to1p5T
```

The freeze jobs stop only world-model optimizer updates. Policy/value training,
real-env collection, checkpointing, and real-env evaluation continue.

## Implementation Status

| Project | Freeze unit | Original T | Total length | Implementation |
| --- | ---: | ---: | ---: | --- |
| DreamerV3 | env step | 100000 | 150000 | existing `run.freeze_wm_after_step` |
| DIAMOND | collected train-env step | 100000 | 150000 collected steps + 75 final epochs | added `training.freeze_wm_after_collection_step` |
| Simulus | epoch | 600 | 900 | added `training.freeze_wm_after_epoch` / `training.freeze_wm_after_fraction` |
| TWISTER | model/env step | 100000 | 75 epochs / 150000 steps | existing `freeze_wm_after_step` |
| STORM | sample step | 100000 | 150020 | existing freeze step; added `--max-sample-steps` |

All projects log `dyna/freeze_progress`, normalized by the original training
length `T`. DIAMOND additionally logs exact `dyna/collection_step`; Simulus logs
`dyna/env_step_estimate` and `dyna/collection_step` because its native training
progress is epoch-based.

Real-env policy evaluation is standardized across projects as `eval_real/*`:
each eval runs 5 greedy ALE Pong episodes and logs mean/std/min/max score,
mean episode length, episode count, and collection step.

Older Simulus runs may only contain `test_dataset/return` rather than
`eval_real/*`. Those values are still valid real-env policy evaluations, but
they used the original Simulus evaluation policy (`should_sample=True`,
`temperature=0.5`) over 16 episodes. Treat them as auxiliary evidence, not as
the standardized five-project comparison metric. New Simulus diagnostic runs
should log `eval_real/*` and use 5 greedy episodes.

The command files use conservative runtime defaults for long multi-machine
runs: STORM does not export train/eval datasets by default and evaluates every
20000 sample steps, DIAMOND keeps the original real-eval cadence
(`evaluation.every=10`, `collection.test.num_episodes=5`) and checkpoints full
trainer state every 25 epochs, and Simulus uses the Atari default
`evaluation.every=20` with 5 greedy eval episodes. These settings reduce disk pressure and evaluation
overhead without changing the freeze-WM training intervention. TWISTER overrides
`eval_episodes=5` and uses `--eval_period_step 10000`; the repo default is 100
real-env eval episodes, which makes frequent step evals dominate runtime on Pong
once the policy becomes competent. TWISTER also uses an overall Dyna training
step for freeze progress and eval triggers; using the world-model optimizer step
would stall after WM freeze and would repeatedly trigger the same step-based
eval.

## Sync To Machines

The command files assume the projects live under `$HOME/projects`. Sync by
pulling each touched repository on the target machine:

```bash
for repo in CGSReg dreamerv3-reborn diamond simulus twister oc-storm; do
  git -C "$HOME/projects/$repo" pull
done
```

## Run On Pandaria

Run one project at a time unless you have enough free CPU cores. Each job caps
BLAS/OpenMP-style CPU pools at 4 threads:

```bash
cd "$HOME/projects/CGSReg"

tiny-exp-scheduler run experiments/freeze_wm_diagnostic/commands/dreamer_pong_freeze_wm.commands.txt \
  --cuda-devices 0,1,2,3 \
  --logs-dir experiments/freeze_wm_diagnostic/scheduler_logs/dreamer \
  --verbose --keep-job-tabs

tiny-exp-scheduler run experiments/freeze_wm_diagnostic/commands/diamond_pong_freeze_wm.commands.txt \
  --cuda-devices 0,1,2,3 \
  --logs-dir experiments/freeze_wm_diagnostic/scheduler_logs/diamond \
  --verbose --keep-job-tabs

tiny-exp-scheduler run experiments/freeze_wm_diagnostic/commands/simulus_pong_freeze_wm.commands.txt \
  --cuda-devices 0,1,2,3 \
  --logs-dir experiments/freeze_wm_diagnostic/scheduler_logs/simulus \
  --verbose --keep-job-tabs

tiny-exp-scheduler run experiments/freeze_wm_diagnostic/commands/twister_pong_freeze_wm.commands.txt \
  --cuda-devices 0,1,2,3 \
  --logs-dir experiments/freeze_wm_diagnostic/scheduler_logs/twister \
  --verbose --keep-job-tabs

tiny-exp-scheduler run experiments/freeze_wm_diagnostic/commands/storm_pong_freeze_wm.commands.txt \
  --cuda-devices 0,1,2,3 \
  --logs-dir experiments/freeze_wm_diagnostic/scheduler_logs/storm \
  --verbose --keep-job-tabs
```

Use fewer CUDA devices if the CPU is crowded. To make each job lighter, edit the
command files and change `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`,
`OPENBLAS_NUM_THREADS=4`, `NUMEXPR_NUM_THREADS=4`, and
`VECLIB_MAXIMUM_THREADS=4` to `2`.

DreamerV3 should be launched more conservatively than the PyTorch projects. JAX
can keep large asynchronous outputs alive around `report()` and W&B video
encoding, so prefer one Dreamer job per GPU and avoid running multiple Dreamer
jobs on the same GPU. The Dreamer commands disable the XLA GPU command buffer
with `XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="`; this avoids
the CUDA graph / stream-capture path that can fail near report/video sync while
keeping Dreamer's original `report()` enabled. If a Dreamer CUDA launch failure
happens, kill the failed processes and verify `jax.devices()` before restarting;
the CUDA context may be wedged until the GPU is reset or the machine is
rebooted.

## W&B Projects

```text
dreamer-dyna-freeze-wm
diamond-dyna-freeze-wm
simulus-dyna-freeze-wm
twister-dyna-freeze-wm
storm-dyna-freeze-wm
```

Compare:

```text
dyna/wm_frozen
dyna/freeze_progress
dyna/freeze_wm_after_step
dyna/freeze_wm_after_epoch
dyna/freeze_wm_after_collection_step
dyna/collection_step
dyna/env_step_estimate
eval_real/score_mean
eval/episode_return
score
```

The most important plot is real-env policy score against training progress,
with a vertical marker where `dyna/wm_frozen` flips from `0` to `1`.

## Plot Data Preparation

Use the W&B export helper to build the plotting table for the completed
diagnostics:

```bash
cd "$HOME/projects/CGSReg"
python experiments/freeze_wm_diagnostic/prepare_eval_curve_data.py
```

This writes local generated files under:

```text
experiments/freeze_wm_diagnostic/generated/
  eval_real_score_curves.csv
  eval_real_score_curves.json
  eval_real_score_curve_summary.csv
  freeze_wm_eval_real_score.svg
```

The canonical plotting table is `eval_real_score_curves.csv`. Its normalized
columns include:

```text
project, wandb_project, wandb_run_id, wandb_url, run_name, condition,
freeze_step, env_step, freeze_progress, wm_frozen, score_mean, source_metric,
source_quality
```

Project-specific x-axis handling is encoded in the script:

| Project | Score source | X-axis mapping |
| --- | --- | --- |
| DreamerV3 | `eval_real/score_mean` | W&B step divided by 4 to recover environment-equivalent step |
| Simulus | `test_dataset/return` | eval index mapped through `evaluation.every=20`, original `T=600` epochs |
| TWISTER | `eval_real/score_mean` | eval index mapped through `--eval_period_step 10000`; duplicate eval rows are averaged |
| STORM | `eval_real/score_mean` | `eval_real/collection_step` when available, otherwise eval index / final summary |

Current W&B API behavior makes STORM history unreliable: full history requests
can time out or return server errors, and sampled history often returns only
the final eval point. The script therefore marks STORM rows with
`source_quality=summary_only_wandb_history_failed` or
`summary_or_sparse_sampled_history` when it cannot recover the full curve.
Those rows are suitable for final-score comparison but not for a dense
post-freeze trajectory plot.
