# STORM Freeze-WM Adaptor

## Identity

| Field | Value |
| --- | --- |
| Repository | `$HOME/projects/oc-storm` |
| Conda env | `oc-storm` |
| Command file | `experiments/freeze_wm_diagnostic/commands/storm_pong_freeze_wm.commands.txt` |
| W&B project | `storm-dyna-freeze-wm` |
| W&B entity | `ssl-lab` |

## Run Names

```text
storm_repro_nofreeze_1p5T
storm_repro_f0p5_to1p5T
storm_repro_f0p75_to1p5T
storm_repro_f1p0_to1p5T
```

## Local Directories

STORM writes run outputs under:

```text
runs/storm_repro_nofreeze_1p5T
runs/storm_repro_f0p5_to1p5T
runs/storm_repro_f0p75_to1p5T
runs/storm_repro_f1p0_to1p5T
```

Scheduler logs should be written under:

```text
experiments/freeze_wm_diagnostic/scheduler_logs/storm/
```

## Freeze Mapping

STORM uses sample steps. The command uses `150020` rather than exactly `150000`
to preserve the repository's original `+20` offset.

| Contract | STORM argument |
| --- | --- |
| Total training `1.5T` | `--max-sample-steps 150020` |
| Original budget `T` | `--freeze-wm-original-step 100000` |
| No freeze | omit `--freeze-wm-after-step` |
| Freeze at `0.5T` | `--freeze-wm-after-step 50000` |
| Freeze at `0.75T` | `--freeze-wm-after-step 75000` |
| Freeze at `1.0T` | `--freeze-wm-after-step 100000` |

## Required W&B Logging

STORM should log:

```text
dyna/wm_frozen
dyna/collection_step
dyna/freeze_progress
dyna/freeze_wm_after_step
```

Real-environment evaluation is configured with:

```text
--eval_every_steps 20000
--eval_episodes 5
--eval_metrics_batches 1
WANDB_ENABLED=1
WANDB_PROJECT=storm-dyna-freeze-wm
```

## Runtime Notes

The command file disables dataset export to reduce disk pressure:

```text
--save_dataset False
--save_eval_dataset False
```

STORM uses `.cuda()` and `device="cuda"` internally. Under the scheduler this
resolves to the logical visible GPU and should respect `CUDA_VISIBLE_DEVICES`.
Use `train.py` for this diagnostic; `train_async.py` contains unrelated
hardcoded CUDA visibility behavior.
