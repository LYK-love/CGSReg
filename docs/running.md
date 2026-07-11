# Running Evaluations

## Protocol Boundaries

`CGSReg` owns metrics and reporting. It does not own world-model
training, policy training, or zero-shot RL policy evaluation.

The supported WM sources are:

- DreamerV3 checkpoints through the JAX Dreamer runtime.
- DIAMOND checkpoints through the DIAMOND Torch runtime.
- STORM/OC-STORM checkpoints through their Torch runtime.
- Twister checkpoints through the Twister Torch runtime.
- Simulus checkpoints through the Simulus Torch runtime.

## Replay-Aligned MSE

This protocol is meaningful only when the replay has object masks. The same
real replay context is used to initialize each model, and the same recorded
future action sequence is fed open-loop.

Important caveat: different architectures do not share a latent state. The
shared start means shared observed history and shared future actions, not
identical hidden representations.

For Dreamer checkpoints, run the formal visual metric entry point:

```bash
python scripts/eval/pong_wm_metrics.py visual-prediction \
  --wm baseline=/path/to/ckpt/latest \
  --wm w001=/path/to/ckpt/latest \
  --eval-replay-dir /path/to/dreamer_eval_replay_with_masks \
  --output-dir artifacts/short_horizon/visual_example \
  --horizons 12,32,48 \
  --context 5 \
  --samples 8 \
  --pixel-scale minus-one-one
```

It writes aggregate `visual_prediction_metrics.{json,csv}` plus
`visual_prediction_window_metrics.csv` with per-window global pixel MSE and
Ball/Left paddle/Right paddle masked MSE.

## Generated-Video Ball Tracking

This protocol evaluates whether generated videos contain a stable, trackable
ball-like object. SAM2 receives the generated video and a prompt derived from
the reference ball mask.

For external WMs, exact replay-context initialization may not be available.
When that is true, report detectability/coverage as the main metric and treat
center error as diagnostic.

## Generated-Video Physical Consistency

After ball tracking, run:

```bash
python scripts/eval/pong_ball_physical_consistency.py \
  --input-dir artifacts/sam2/run_name \
  --output-dir results/physical_consistency/run_name
```

The main paper-facing fields are:

- `detectability`
- `gap_frame_rate`
- `teleports_per_1k_eligible`
- `reappearance_teleport_rate`
- `spontaneous_turns_per_1k_eligible`
- `accel_p95_mean`

Collision exclusion:

- Preferred: provide generated-video left/right paddle SAM2 bbox tracks with
  `--paddle-track-csv model:left=...` and `--paddle-track-csv model:right=...`.
- Fallback: if no paddle tracks are provided, frames near the left/right edge
  are excluded using `--paddle-x-margin`.

Use real replay tracks only for threshold calibration, not as a trajectory
target. The default teleport and acceleration thresholds are derived from real
track speed/acceleration percentiles plus a margin.

## Zero-Shot RL

Zero-shot RL should be run in `rl-in-pixel-env`. This repo may store the final
tables under `results/zero_shot_rl/`, but the RL loop and policy evaluation
code should remain in that project or as a submodule reference.
