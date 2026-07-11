# Simulus Freeze-WM Adaptor

## Identity

| Field | Value |
| --- | --- |
| Repository | `$HOME/projects/simulus` |
| Conda env | `simulus` |
| Command file | `experiments/freeze_wm_diagnostic/commands/simulus_pong_freeze_wm.commands.txt` |
| W&B project | `simulus-dyna-freeze-wm` |
| W&B entity | `ssl-lab` |

## Run Names

```text
simulus_repro_nofreeze_1p5T
simulus_repro_f0p5_to1p5T
simulus_repro_f0p75_to1p5T
simulus_repro_f1p0_to1p5T
```

## Local Directories

Hydra output directories:

```text
outputs/freeze_wm_diagnostic/PongNoFrameskip-v4/<run>/<date>/<time>-seed-<seed>
```

The command file also sets:

```text
outputs_dir_path=./outputs/freeze_wm_diagnostic
```

Scheduler logs should be written under:

```text
experiments/freeze_wm_diagnostic/scheduler_logs/simulus/
```

## Freeze Mapping

Simulus is epoch-based. The diagnostic maps `T = 100000` to the original 600
epoch training budget and extends to 900 epochs.

| Contract | Simulus override |
| --- | --- |
| Total training `1.5T` | `common.epochs=900` |
| Original budget `T` | `training.freeze_wm_original_epoch=600` |
| Collection extension | `collection.train.stop_after_epochs=750` |
| No freeze | omit `training.freeze_wm_after_epoch` |
| Freeze at `0.5T` | `training.freeze_wm_after_epoch=300` |
| Freeze at `0.75T` | `training.freeze_wm_after_epoch=450` |
| Freeze at `1.0T` | `training.freeze_wm_after_epoch=600` |

## Required W&B Logging

Simulus should log:

```text
dyna/wm_frozen
dyna/freeze_wm_after_epoch
dyna/env_step_estimate
dyna/collection_step
dyna/freeze_progress
```

Real-environment evaluation is configured with:

```text
evaluation.every=20
collection.test.config.num_episodes=5
collection.test.config.num_episodes_end=5
collection.test.config.should_sample=false
wandb.project=simulus-dyna-freeze-wm
wandb.name=<run>
```

Historical Simulus runs may only contain `test_dataset/return`. Treat those as
auxiliary unless the analysis explicitly maps them and notes that they were not
the standardized 5-episode greedy evaluation.

## Runtime Notes

The command file sets:

```text
PYTHONPATH="$HOME/projects/simulus/src"
```

Simulus uses `common.device=cuda:0` internally. Under the scheduler this is the
logical visible GPU and should respect `CUDA_VISIBLE_DEVICES`; do not add a
project-level override that rewrites `CUDA_VISIBLE_DEVICES`.
