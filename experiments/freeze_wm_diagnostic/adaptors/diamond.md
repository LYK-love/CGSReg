# DIAMOND Freeze-WM Adaptor

## Identity

| Field | Value |
| --- | --- |
| Repository | `$HOME/projects/diamond` |
| Conda env | `diamond` |
| Command file | `experiments/freeze_wm_diagnostic/commands/diamond_pong_freeze_wm.commands.txt` |
| W&B project | `diamond-dyna-freeze-wm` |
| W&B entity | `ssl-lab` |

## Run Names

```text
diamond_repro_nofreeze_1p5T
diamond_repro_f0p5_to1p5T
diamond_repro_f0p75_to1p5T
diamond_repro_f1p0_to1p5T
```

## Local Directories

Hydra output directories:

```text
outputs/freeze_wm_diagnostic/diamond_repro_nofreeze_1p5T
outputs/freeze_wm_diagnostic/diamond_repro_f0p5_to1p5T
outputs/freeze_wm_diagnostic/diamond_repro_f0p75_to1p5T
outputs/freeze_wm_diagnostic/diamond_repro_f1p0_to1p5T
```

Scheduler logs should be written under:

```text
experiments/freeze_wm_diagnostic/scheduler_logs/diamond/
```

## Freeze Mapping

DIAMOND uses collected train-environment steps as the freeze unit.

| Contract | DIAMOND override |
| --- | --- |
| Total collection `1.5T` | `collection.train.num_steps_total=150000` |
| Original budget `T` | `training.freeze_wm_original_collection_steps=100000` |
| No freeze | omit `training.freeze_wm_after_collection_step` |
| Freeze at `0.5T` | `training.freeze_wm_after_collection_step=50000` |
| Freeze at `0.75T` | `training.freeze_wm_after_collection_step=75000` |
| Freeze at `1.0T` | `training.freeze_wm_after_collection_step=100000` |
| Extra policy training | `training.num_final_epochs=75` |

## Fixed Reward/End Model

Unlike the other project adaptors, DIAMOND uses a fixed pretrained reward/end
model for this diagnostic. The reward/end model is not part of the Dyna-style
failure mechanism being tested, and skipping its training removes a substantial
runtime cost.

The command file loads only the reward/end component:

```text
initialization.path_to_ckpt="${DIAMOND_REW_END_CKPT:-$HOME/projects/diamond/checkpoints/Pong.pt}"
initialization.load_denoiser=False
initialization.load_rew_end_model=True
initialization.load_actor_critic=False
training.train_rew_end_model=False
```

Set `DIAMOND_REW_END_CKPT` on the target machine if the pretrained DIAMOND Pong
checkpoint lives somewhere other than `$HOME/projects/diamond/checkpoints/Pong.pt`.
The denoiser and actor-critic still train normally; the actor-critic still uses
the loaded reward/end model inside the world-model environment.

## Required W&B Logging

DIAMOND should log:

```text
dyna/wm_frozen
dyna/rew_end_model_training_enabled
dyna/freeze_progress
dyna/collection_step
dyna/freeze_wm_after_collection_step
```

Real-environment evaluation is configured with:

```text
evaluation.every=10
collection.test.num_episodes=5
wandb.project=diamond-dyna-freeze-wm
wandb.name=<run>
```

For standardized analysis, map DIAMOND real-eval metrics to `eval_real/*` when
they are not already logged in that namespace.

## Runtime Notes

The command file must keep:

```text
common.devices=all
```

`common.devices=all` preserves the scheduler-provided `CUDA_VISIBLE_DEVICES`.
Using `common.devices=0` overrides scheduler GPU assignment and can pin every
job to physical GPU 0.

The primary command file uses:

```text
training.compile_wm=True
```

This matches the original DIAMOND Pong reproduction speed path. Pandaria passed
the probe in `commands/diamond_compile_probe.commands.txt`: the first
actor-critic step paid the expected compile overhead and did not fail with
`-lcuda`. On machines where PyTorch Inductor cannot link CUDA, such as the
previous NV cloud failure mode, override this to `training.compile_wm=False`.
