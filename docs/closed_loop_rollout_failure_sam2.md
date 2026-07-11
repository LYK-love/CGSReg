# Closed-loop rollout failure analysis with SAM2

This note records the implemented SAM2-based analysis for the closed-loop Pong
rollout videos used in the paper notes. The inspected rollout artifacts are:

- `artifacts/dreamer_all5_real_policy_h512_rollouts`
- `artifacts/torch_selected_wms_real_policy_h512_rollouts`

The implementation is:

- `scripts/eval/pong_sam2_rollout_failure_events.py`

The completed run is:

- `eval_outputs/closed_loop_rollout_failure_sam2_hybrid`

## Protocol

Each rollout `.npz` contains one bootstrap frame followed by 512 generated
frames. The script drops the bootstrap frame and analyzes the 512 action-indexed
frames.

SAM2 is prompted from a generated-frame Pong-ball heuristic and run on the
generated rollout video. Event logic is evaluated in 64x64 Pong coordinates.
Because raw SAM2 tracking can briefly drop the ball on generated videos, the
default event track is `hybrid`: it uses SAM2 masks where available and fills
confirmed missing frames with a conservative Pong-ball image heuristic. The raw
SAM2 coverage and hybrid coverage are both recorded per rollout.

Detected event names:

- `ball_disappears`
- `missed_player_bounce`
- `spurious_x_bounce`

The script also records paddle-collision opportunity statistics for both
paddles:

- `expected_paddle_collisions`: total event-level opportunities where the ball
  enters either paddle collision zone while moving toward that paddle.
- `successful_paddle_collisions`: expected collisions where the ball x velocity
  reverses within the lookahead window.
- `failed_expected_paddle_collisions`: expected collisions where no x-velocity
  reversal is observed.
- `failed_expected_collision_rate`:
  `failed_expected_paddle_collisions / expected_paddle_collisions`.
- Left/right split columns are also written:
  `left_expected_paddle_collisions`,
  `left_failed_expected_paddle_collisions`,
  `left_failed_expected_collision_rate`,
  `right_expected_paddle_collisions`,
  `right_failed_expected_paddle_collisions`, and
  `right_failed_expected_collision_rate`.

The denominator is not frame-level. A single approach to a paddle is counted
once, then assigned to the success or failure bucket based on whether the x
velocity reverses shortly after contact.

Current thresholds are intentionally conservative to avoid counting SAM2
segmentation flicker as a world-model failure.

## Aggregate result

All event columns are mean counts over 5 episodes. The last column is the raw
sum over the three failure patterns across all 5 episodes.

| Source | Model | Ball disappears / ep | Missed player bounce / ep | Spurious x bounce / ep | Sum over 3 failure patterns |
| --- | --- | ---: | ---: | ---: | ---: |
| SAM2-hybrid | `size200m_repro` | 0.40 | 0.00 | 0.00 | 2 |
| SAM2-hybrid | `size200m_w0` | 0.80 | 0.00 | 0.20 | 5 |
| SAM2-hybrid | `size200m_w001` | 0.20 | 0.00 | 0.40 | 3 |
| SAM2-hybrid | `size400m_w0` | 0.80 | 0.00 | 0.00 | 4 |
| SAM2-hybrid | `size400m_w001` | 0.00 | 0.00 | 0.60 | 3 |
| SAM2-hybrid | `diamond_repro` | 0.00 | 0.00 | 0.00 | 0 |
| SAM2-hybrid | `simulus_repro` | 0.60 | 0.00 | 0.80 | 7 |
| SAM2-hybrid | `twister_repro` | 1.40 | 0.00 | 0.80 | 11 |

Mean hybrid present rates are available in the raw aggregate CSV.

Raw CSVs:

- `eval_outputs/closed_loop_rollout_failure_sam2_hybrid/aggregate_by_model.csv`
- `eval_outputs/closed_loop_rollout_failure_sam2_hybrid/rollout_summaries.csv`
- `eval_outputs/closed_loop_rollout_failure_sam2_hybrid/manual_dreamer_comparison.csv`

## Validation against manual Dreamer labels

The detector is useful as an automated, reproducible first pass, but it is not a
drop-in replacement for the manual labels yet.

Observed behavior:

- `ball_disappears` has a usable signal after suppressing short segmentation
  gaps and scoring-boundary events.
- `missed_player_bounce` is currently under-detected. The generated videos do
  not expose a reliable paddle mask with the simple deterministic detector, and
  the current implementation keeps this event conservative rather than counting
  many false contacts.
- `spurious_x_bounce` is also conservative after suppressing heuristic-only
  reacquisition jumps. It can miss visually obvious trajectory errors.

The Dreamer comparison CSV should be treated as the calibration report for
paper-facing use.

## Reproduction command

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n dreamer \
  python scripts/eval/pong_sam2_rollout_failure_events.py \
  --force-sam2 \
  --output-dir eval_outputs/closed_loop_rollout_failure_sam2_hybrid
```
