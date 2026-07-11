# External WM Generated-Video SAM2 Ball Tracking

This evaluates the same generated-video ball-tracking idea on four non-Dreamer Pong world-model reproduction checkpoints: DIAMOND, Simulus, TWISTER, and STORM.

## Protocol

- Rollout API: each project is called through its pixel WM play/RL API, i.e. `make_torch_wm_env(ctx).reset()` followed by `step(action)`.
- Action sequence: the fixed replay action sequence from `chunk_id=0`, `start=8` in `/data/luyukuan/projects/dreamerv3-runs/datasets/pong_wm_reg_sweep/eval_replay`.
- Horizons: `8, 16, 24, 32, 48`.
- SAM2 guiding source: the WM-generated video itself (`guiding_source=wm`).
- Prompt: a generated-frame heuristic ball point is searched in the WM video. If no generated ball prompt is found, the run is scored as zero coverage instead of falling back to a non-aligned real prompt.
- Alignment caveat: these external play APIs reset from each project’s own real/dataset initial source, not from the exact Dreamer replay hidden context. Thus the primary cross-project signal is SAM2 detectability/coverage. `ball_l2_rmse` is diagnostic only, not a strict GT-aligned trajectory error.

Artifacts:

- Rollouts, SAM2 masks, overlays, per-horizon tracks: `eval_outputs/pong_external_wm_sam2_ball_repro_h48_20260601`
- Aggregate CSV: `eval_outputs/pong_external_wm_sam2_ball_repro_h48_20260601/external_sam2_ball_summary.csv`

## Results

| Horizon | Model | SAM2 present | Missing rate ↓ | First present | Ball L2 RMSE ↓ |
| ---: | --- | ---: | ---: | ---: | ---: |
| 8 | DIAMOND repro | 0/8 | 1.000 | - | NaN |
| 16 | DIAMOND repro | 0/16 | 1.000 | - | NaN |
| 24 | DIAMOND repro | 0/24 | 1.000 | - | NaN |
| 32 | DIAMOND repro | 0/32 | 1.000 | - | NaN |
| 48 | DIAMOND repro | 0/48 | 1.000 | - | NaN |
| 8 | Simulus repro | 2/8 | 0.750 | 6 | 11.639 |
| 16 | Simulus repro | 10/16 | 0.375 | 6 | 11.370 |
| 24 | Simulus repro | 18/24 | 0.250 | 6 | 12.110 |
| 32 | Simulus repro | 26/32 | 0.188 | 6 | 13.782 |
| 48 | Simulus repro | 42/48 | 0.125 | 6 | 14.215 |
| 8 | STORM repro | 0/8 | 1.000 | - | NaN |
| 16 | STORM repro | 0/16 | 1.000 | - | NaN |
| 24 | STORM repro | 0/24 | 1.000 | - | NaN |
| 32 | STORM repro | 0/32 | 1.000 | - | NaN |
| 48 | STORM repro | 0/48 | 1.000 | - | NaN |
| 8 | TWISTER repro | 6/8 | 0.250 | 2 | 4.492 |
| 16 | TWISTER repro | 14/16 | 0.125 | 2 | 4.315 |
| 24 | TWISTER repro | 22/24 | 0.083 | 2 | 4.963 |
| 32 | TWISTER repro | 30/32 | 0.062 | 2 | 5.234 |
| 48 | TWISTER repro | 45/48 | 0.062 | 2 | 5.181 |

Interpretation:

- TWISTER is the strongest among these four reproduction checkpoints by generated-video ball detectability.
- Simulus also produces a trackable ball-like object, but it starts later and has lower coverage at short horizons.
- DIAMOND and STORM did not produce a generated-frame ball prompt under this strict protocol, so they are scored as zero coverage.
- Because starts are not exact replay-context aligned, the coverage and first-present fields are the meaningful comparison; `ball_l2_rmse` should not be used as a paper-facing GT trajectory metric here.
