# TWISTER Freeze-WM Adaptor

## Identity

| Field | Value |
| --- | --- |
| Repository | `$HOME/projects/twister` |
| Conda env | `twister` |
| Command file | `experiments/freeze_wm_diagnostic/commands/twister_pong_freeze_wm.commands.txt` |
| W&B project | `twister-dyna-freeze-wm` |
| W&B entity | `ssl-lab` |

## Run Names

```text
twister_repro_nofreeze_1p5T
twister_repro_f0p5_to1p5T
twister_repro_f0p75_to1p5T
twister_repro_f1p0_to1p5T
```

The command file sets both `run_name=<run>` and `WANDB_NAME=<run>`.

## Local Directories

TWISTER writes callbacks under its native callback root. For these commands the
run-local path is:

```text
callbacks/<run>/atari100k-pong
```

Local W&B files are written below the callback path by TWISTER's logger.

Scheduler logs should be written under:

```text
experiments/freeze_wm_diagnostic/scheduler_logs/twister/
```

## Freeze Mapping

TWISTER uses model/environment steps. The current reproduction config uses 75
epochs for 150000 steps.

| Contract | TWISTER override |
| --- | --- |
| Total training `1.5T` | `override_config={"epochs":75}` |
| Original budget `T` | `override_config={"freeze_wm_original_step":100000}` |
| No freeze | `override_config={"freeze_wm_after_step":-1}` |
| Freeze at `0.5T` | `override_config={"freeze_wm_after_step":50000}` |
| Freeze at `0.75T` | `override_config={"freeze_wm_after_step":75000}` |
| Freeze at `1.0T` | `override_config={"freeze_wm_after_step":100000}` |

## Required W&B Logging

TWISTER should log:

```text
dyna/wm_frozen
dyna/freeze_wm_after_step
dyna/freeze_progress
```

Real-environment evaluation is configured with:

```text
override_config={"eval_episodes":5}
--eval_period_step 10000
WANDB_PROJECT=twister-dyna-freeze-wm
WANDB_NAME=<run>
--wandb
```

TWISTER defaults to 100 real-environment eval episodes. The adaptor overrides
this to 5 to keep evaluation comparable and avoid eval runtime dominating Pong
once policies become competent.

## Runtime Notes

TWISTER uses the overall Dyna training step for freeze progress and step-based
eval triggers. Do not use the world-model optimizer step for those triggers,
because it stops advancing after WM freeze.

Before a clean rerun, remove stale callback directories for these run names so
checkpoints and W&B local files do not merge with previous attempts.
