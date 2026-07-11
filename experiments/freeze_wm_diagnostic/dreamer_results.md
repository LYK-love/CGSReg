# Dreamer Freeze-WM Diagnostic Results

Date summarized: 2026-06-16.

This summarizes the completed DreamerV3 Atari100K Pong Dyna-style freeze-WM
diagnostic on `lyk@pandaria`.

## Question

The intervention isolates continued world-model optimizer updates during the
extra Dyna-style training phase. Each job trains Dreamer to `1.5T`, where
`T = 100000` real environment steps. Freeze jobs stop only the world-model
optimizer after a chosen fraction of `T`; policy/value optimization, real-env
collection, checkpointing, and real-env evaluation continue.

If the Dyna-style failure were mainly caused by continuing to update the world
model after `T`, freezing the WM at or before `T` should improve or at least
stabilize real-env policy return. The observed result is the opposite.

## Runs

| Run | Freeze point | Final progress | Final WM frozen | Eval points | Last eval | Best eval | Last 10 eval mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dreamer_size200m_nofreeze_1p5T` | none | 1.49596 | 0 | 93 | 1 | 13 | 0.90 |
| `dreamer_size200m_f0p5_to1p5T` | 0.50T / 50k | 1.49310 | 1 | 87 | -19 | -11 | -17.50 |
| `dreamer_size200m_f0p75_to1p5T` | 0.75T / 75k | 1.48930 | 1 | 86 | -21 | -5 | -19.30 |
| `dreamer_size200m_f1p0_to1p5T` | 1.00T / 100k | 1.49446 | 1 | 100 | -21 | 16 | -21.00 |

Detailed machine-readable summary:
`experiments/freeze_wm_diagnostic/dreamer_results_summary.csv`.

## Main Finding

Freezing the WM does not rescue Dreamer's Dyna-style continuation on Pong.
The no-freeze `1.5T` baseline remains much stronger at the end than all freeze
variants: its final 10 real-env eval mean is `0.90`, while freeze-at-0.5T,
0.75T, and 1.0T end at `-17.50`, `-19.30`, and `-21.00`.

The `1.0T` freeze run is the sharpest evidence. It reaches a best real-env
score of `16`, then collapses to ten consecutive `-21` evals by the end despite
the WM being frozen only after the original `T`. That rules out a simple
explanation where post-`T` WM optimizer updates are the dominant failure cause.

The supported conclusion is narrower and stronger:

- Continued WM updates after `T` are not necessary for the failure.
- Frozen-WM policy/value continuation itself can degrade or fail to stabilize
  the policy.
- The likely failure surface is policy/value optimization against a stale or
  mismatched model, compounding model-policy distribution shift, or objective
  drift in the actor/value heads, not merely WM degradation from extra updates.

## Pre/Post Freeze Signal

| Run | Pre-freeze eval mean | Pre-freeze n | Post-freeze eval mean | Post-freeze n | Terminal pattern |
| --- | ---: | ---: | ---: | ---: | --- |
| `f0p5` | -18.93 | 28 | -17.53 | 58 | stays poor |
| `f0p75` | -15.34 | 44 | -19.32 | 41 | worsens after freeze |
| `f1p0` | -13.60 | 70 | -12.96 | 28 | post-freeze mean hides terminal collapse; last 10 are all -21 |

## Completion And Runtime Notes

Final local outputs are under:

```text
$HOME/projects/dreamerv3-runs/freeze_wm_diagnostic/
  dreamer_size200m_nofreeze_1p5T/
  dreamer_size200m_f0p5_to1p5T/
  dreamer_size200m_f0p75_to1p5T/
  dreamer_size200m_f1p0_to1p5T/
```

Each final run has `metrics.jsonl`, `scores.jsonl`, `config.yaml`, and a final
checkpoint. The final checkpoints are:

| Run | Final checkpoint |
| --- | --- |
| `nofreeze` | `20260614T235245F796007` |
| `f0p5` | `20260615T065501F038745` |
| `f0p75` | `20260614T231728F291338` |
| `f1p0` | `20260615T140021F919398` |

Scheduler logs are in:

```text
experiments/freeze_wm_diagnostic/scheduler_logs/dreamer/
experiments/freeze_wm_diagnostic/scheduler_logs/dreamer_remaining/
experiments/freeze_wm_diagnostic/scheduler_logs/dreamer_retry/
```

The initial Dreamer scheduler batch had JAX/CUDA launch or stream-capture
failures in several jobs. Follow-up `remaining` and `retry` jobs completed the
four target size200m runs. Local failed run directories and nonzero scheduler
log pairs were removed after the checkpoint archive was uploaded; the retained
scheduler exit files are successful runs only.

Uploaded Box archive:

```text
box:projects/wm-evaluation/freeze_wm_diagnostic/dreamer_size200m_20260616/
  dreamer_freeze_wm_size200m_ckpts_20260616.tar.zst
  dreamer_freeze_wm_size200m_ckpts_20260616.tar.zst.sha256
  manifest.txt
  dreamer_results.md
  dreamer_results_summary.csv
```

## W&B Runs

| Run | W&B |
| --- | --- |
| `nofreeze` | <https://wandb.ai/ssl-lab/dreamer-dyna-freeze-wm/runs/erz51wqh> |
| `f0p5` | <https://wandb.ai/ssl-lab/dreamer-dyna-freeze-wm/runs/r204k4y2> |
| `f0p75` | <https://wandb.ai/ssl-lab/dreamer-dyna-freeze-wm/runs/ygez8r7e> |
| `f1p0` | <https://wandb.ai/ssl-lab/dreamer-dyna-freeze-wm/runs/jg8ql9ou> |
