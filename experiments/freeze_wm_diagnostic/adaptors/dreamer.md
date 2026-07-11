# DreamerV3 Freeze-WM Adaptor

## Identity

| Field | Value |
| --- | --- |
| Repository | `$HOME/projects/dreamerv3-reborn` |
| Conda env | `dreamer` |
| Command file | `experiments/freeze_wm_diagnostic/commands/dreamer_pong_freeze_wm.commands.txt` |
| W&B project | `dreamer-dyna-freeze-wm` |
| W&B entity | `ssl-lab` |

## Run Names

DreamerV3 includes the model size in the run name:

```text
dreamer_${DREAMER_SIZE_CONFIG:-size200m}_nofreeze_1p5T
dreamer_${DREAMER_SIZE_CONFIG:-size200m}_f0p5_to1p5T
dreamer_${DREAMER_SIZE_CONFIG:-size200m}_f0p75_to1p5T
dreamer_${DREAMER_SIZE_CONFIG:-size200m}_f1p0_to1p5T
```

This intentionally differs from the strict `<project>_repro_*` template so that
different DreamerV3 sizes do not share names.

## Local Directories

The command file writes checkpoints and logs under:

```text
${DREAMERV3_RUNS_ROOT:-$HOME/projects/dreamerv3-runs}/freeze_wm_diagnostic/<run>
```

Scheduler logs should be written under:

```text
experiments/freeze_wm_diagnostic/scheduler_logs/dreamer/
```

If a run directory already contains a checkpoint at `run.steps=150000`, DreamerV3
can load it and exit successfully without producing new training history. Use a
fresh run directory for a clean rerun.

## Freeze Mapping

| Contract | DreamerV3 argument |
| --- | --- |
| Total training `1.5T` | `--run.steps 150000` |
| Original budget `T` | `--run.freeze_wm_original_step 100000` |
| No freeze | `--run.freeze_wm_after_step -1` |
| Freeze at `0.5T` | `--run.freeze_wm_after_step 50000` |
| Freeze at `0.75T` | `--run.freeze_wm_after_step 75000` |
| Freeze at `1.0T` | `--run.freeze_wm_after_step 100000` |

The training loop passes `wm_frozen` into `agent.train()` and logs the Dyna state
from `embodied/run/train_eval.py`.

## Required W&B Logging

DreamerV3 should log:

```text
dyna/wm_frozen
dyna/freeze_wm_after_step
dyna/freeze_progress
```

Real-environment evaluation should be configured with:

```text
--run.eval_eps 5
--logger.project dreamer-dyna-freeze-wm
--logger.name <run>
```

Historical DreamerV3 runs may expose score summaries through native keys such as
`episode/score`. For cross-project plots, prefer normalized `eval_real/*` keys
when available, or document the native-key mapping used by the analysis script.

## Runtime Notes

The command file sets:

```text
XLA_PYTHON_CLIENT_PREALLOCATE=false
TF_GPU_ALLOCATOR=cuda_malloc_async
XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
DREAMERV3_SAVE_EVERY=5000
NUM_CKPT_TO_KEEP=3
```

Keep one DreamerV3 job per GPU. JAX report/video work can leave large
asynchronous outputs alive, so DreamerV3 should be scheduled more conservatively
than the PyTorch projects.
