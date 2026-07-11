# Torch WM rollout failure analysis design

This note records the manual video inspection baseline and the planned SAM2-based detector for Pong rollout failure modes. The experiment is not run in this document.

## Manual baseline from visual inspection

User-provided manual inspection for the Dreamer rollout videos:

| wm ckpt | size200m repro | size200m w=0 | size200m w=0.01 | size400m w=0 | size400m w=0.01 |
| ------- | -------------- | ------------ | ---------------- | ------------ | ---------------- |
| ep0 | 球消失3次 | 球消失1次；未正确反弹1次 | 无 | 球轨迹错误（无故反弹）2次 | 无 |
| ep1 | 未正确反弹5次（轨迹固定） | 球未正确反弹3次 | 无 | 球未正确反弹2次 | 无 |
| ep2 | 球消失4次 | 球消失1次 | 未正确反弹1次 | 球未正确反弹1次 | 无 |
| ep3 | 未正确反弹5次（轨迹固定） | 球未正确反弹1次 | 无 | 球未正确反弹3次 | 无 |
| ep4 | 未正确反弹5次（轨迹固定） | 无 | 无 | 球轨迹错误（无故反弹）1次 | 无 |

Aggregated over the 5 inspected episodes:

| Source | Model | Ball disappears / ep | Missed player bounce / ep | Spurious x bounce / ep | Sum over 3 failure patterns |
| --- | --- | ---: | ---: | ---: | ---: |
| Human | `size200m_repro` | 1.40 | 3.00 | 0.00 | 22 |
| Human | `size200m_w0` | 0.40 | 1.00 | 0.00 | 7 |
| Human | `size200m_w001` | 0.00 | 0.20 | 0.00 | 1 |
| Human | `size400m_w0` | 0.00 | 1.20 | 0.60 | 9 |
| Human | `size400m_w001` | 0.00 | 0.00 | 0.00 | 0 |

SAM2/hybrid automated statistics from
`eval_outputs/closed_loop_rollout_failure_sam2_hybrid`:

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

Failure vocabulary:

- `ball_disappears`: the ball disappears during continuous play without leaving the playable field. In Pong, a legitimate disappearance/refresh should happen only through an out-of-bounds score event.
- `missed_player_bounce`: the ball reaches the player paddle collision zone but the x velocity does not reverse after contact.
- `spurious_x_bounce`: the ball x velocity reverses without any nearby paddle collision. This is the "球轨迹错误 / 无故反弹" mode.

## Target rollout inputs

For the Torch WM reproduction rollouts, the current completed rollout artifacts are:

- `artifacts/torch_selected_wms_real_policy_h512_rollouts/epXX_seedX/videos/{diamond_repro,twister_repro,simulus_repro}.mp4`
- `artifacts/torch_selected_wms_real_policy_h512_rollouts/epXX_seedX/rollouts/{diamond_repro,twister_repro,simulus_repro}.npz`

Each rollout contains `513` frames and `512` actions. The first frame is the post-reset/bootstrap observation; action-indexed transitions start at frame `1`.

## SAM2 tracking plan

Run SAM2 on each generated video and export at least these tracks:

- ball track: `ball_present`, `ball_x`, `ball_y`, `ball_area`
- player paddle track: `player_paddle_present`, bbox `[xmin, xmax, ymin, ymax]`
- optionally opponent paddle track, mainly to suppress false spurious-bounce detections

The existing generated-video SAM2 tooling already exports ball masks and centroid tracks. Paddle tracking should use the same SAM2 backend, but with a paddle prompt and bbox export. If paddle SAM2 is unreliable, fall back to deterministic image detection for paddles because Pong paddles are high-contrast vertical bars.

Use 64x64 game coordinates for event logic. SAM2 masks can be produced at 256x256, then downscaled by the same factor already used in `tracks_from_masks()`.

## Event detection design

### 1. Ball disappears

Detect gaps in `ball_present`:

1. Build contiguous visible segments from SAM2 ball track.
2. For every visible-to-missing-to-visible gap, classify it as a candidate disappearance.
3. Suppress legitimate out-of-bounds disappearances if the last visible ball center before the gap is already outside or very close to a horizontal scoring boundary.
4. Count the remaining gaps as `ball_disappears`.

Conservative rule:

- `gap_len >= 1`
- previous visible point and next visible point both exist
- `x_prev` and `x_next` are inside playable x range, not out-of-bounds
- no reward/score event is available in these WM rollouts, so boundary proximity is the main legitimacy filter

This maps to the existing `flicker_gaps` / `reappearance_teleport` logic in `scripts/eval/pong_ball_physical_consistency.py`, but the output should be renamed for Pong semantics.

### 2. Paddle collision opportunities and missed bounces

Use the RAM-event style from `scripts/eval/pong_wm_metrics.py::annotate_pong_events()` as the reference:

- A paddle collision is an x-velocity sign flip near a paddle y position.
- Here, because the generated WM video has no RAM, velocity comes from SAM2 ball centroids and paddle positions come from SAM2/deterministic paddle boxes.

Paddle-collision opportunity rule:

1. Estimate smoothed velocity from ball centers:
   - `vx[t] = x[t] - x[t-1]`
   - ignore steps with missing ball or speed below a small threshold
   - optionally use a 3-frame median/linear fit to reduce SAM2 centroid jitter
2. Identify contact candidates for both paddles:
   - for the left paddle, ball is moving left and enters the left paddle collision zone
   - for the right paddle, ball is moving right and enters the right paddle collision zone
   - vertical overlap with the corresponding paddle bbox after adding a small margin
3. Look ahead `1..3` frames after contact.
4. Expected behavior: `sign(vx)` reverses.
5. Count one event-level opportunity per approach:
   - if reversal occurs, count `successful_paddle_collisions`
   - if no reversal occurs and the ball remains visible, count `failed_expected_paddle_collisions`

The aggregate metric of interest is:

```text
failed_expected_collision_rate =
  failed_expected_paddle_collisions / expected_paddle_collisions
```

The implementation should also expose left/right splits:

- `left_expected_paddle_collisions`
- `left_failed_expected_paddle_collisions`
- `right_expected_paddle_collisions`
- `right_failed_expected_paddle_collisions`

The legacy `missed_player_bounce` event can remain for compatibility, but the
paper-facing collision-failure rate should use the two-sided opportunity
denominator above.

### 3. Spurious x bounce

This is an x-direction reversal without any valid paddle collision.

Rule:

1. Find `vx` sign flips where both previous and next speed magnitudes exceed the minimum speed threshold.
2. Suppress flips near:
   - player paddle bbox
   - opponent paddle bbox
   - score/out-of-bounds boundary
3. Count remaining x sign flips as `spurious_x_bounce`.

This is stricter and more Pong-specific than the existing `spontaneous_turn` metric in `scripts/eval/pong_ball_physical_consistency.py`, which uses general turn angle away from collision zones. For this task, prefer direct x-sign reversal because the manual label is specifically "无故反弹".

## Proposed outputs

Write one per-episode event CSV:

- `episode`
- `model`
- `event`
- `t`
- `x`
- `y`
- `gap_len`
- `vx_before`
- `vx_after`
- `player_side`
- `reason`

Write one aggregate CSV:

| model | episodes | ball_disappears | missed_player_bounce | spurious_x_bounce |
| --- | ---: | ---: | ---: | ---: |

Also save overlay videos with event markers:

- red vertical tick: `ball_disappears`
- yellow marker: `missed_player_bounce`
- purple marker: `spurious_x_bounce`

## Implementation notes

Add a new script rather than overloading the old Dreamer-aligned SAM2 metric:

- `scripts/eval/pong_sam2_rollout_failure_events.py`

Recommended structure:

1. Load rollout `.npz` frames.
2. Run or reuse SAM2 masks for ball and paddles.
3. Convert masks to centroid/bbox tracks.
4. Smooth/interpolate only short gaps for velocity estimation, but do not hide disappearance gaps from event counting.
5. Run the three event detectors.
6. Write event CSV, aggregate CSV, and overlays.

Use `scripts/eval/pong_ball_physical_consistency.py` for reusable primitives:

- `TrackPoint`
- `PaddleBox`
- `rows_to_tracks`
- `rows_to_paddle_boxes`
- collision-zone margin handling

Use `scripts/eval/pong_wm_metrics.py::annotate_pong_events()` as the semantic reference for paddle collision: x-velocity sign flip near paddle y.

## Validation against manual labels

Before using the detector as a metric, tune only on the five manually inspected Dreamer episodes above:

- Compare counts per episode and model against the manual table.
- Prefer conservative false-positive control. Missing one ambiguous event is better than counting SAM2 jitter as a failure.
- After thresholds are fixed, run unchanged on the Torch WM rollout set.
