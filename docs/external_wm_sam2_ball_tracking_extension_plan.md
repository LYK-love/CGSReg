# External WM SAM2 Ball Tracking Extension Plan

本实验扩展 `generated_video_ball_tracking.md` 的 generated-video ball tracking 口径，用于衡量其他世界模型架构的 Pong base-size reproduction WM checkpoint。

当前待评估对象：

- DIAMOND reproduction base WM
- Simulus reproduction base WM
- TWISTER reproduction base WM
- STORM reproduction base WM

备注：原始实验需求里 `Twister` 出现两次；这里先按第四个模型为 `STORM` 设计。如果确实有两个不同 TWISTER base checkpoint，应在 `Model Registry` 中拆成 `twister_repro_a` 和 `twister_repro_b`。

## Goal

回答一个窄问题：在同一段 Pong action rollout 下，不同外部 WM 生成的视频中，球是否能被 SAM2 稳定跟踪。

主要指标是 ball detectability/coverage，而不是严格的真实轨迹误差。原因是外部 WM 的 play API 不能保证从 Dreamer replay 的完全相同 hidden context 初始化；因此 `ball_l2_rmse` 只能作为诊断指标，不作为跨架构主结论。

## Protocol

使用脚本：

```bash
python scripts/eval/pong_external_sam2_ball_experiment.py \
  --project diamond \
  --project simulus \
  --project twister \
  --project storm \
  --eval-replay-dir /data/luyukuan/projects/dreamerv3-runs/datasets/pong_wm_reg_sweep/eval_replay \
  --output-dir eval_outputs/pong_external_wm_sam2_ball_extension_base_YYYYMMDD \
  --chunk-id 0 \
  --start 8 \
  --context 5 \
  --horizons 8,16,24,32,48 \
  --prompt-source auto \
  --no-prompt-policy zero \
  --prompt-search-frames 12 \
  --guiding-source wm \
  --sam2-size 256 \
  --cuda-device 3
```

上面命令只是实验设计，不在本设计阶段运行。

### Fixed Inputs

- Replay/action source: `/data/luyukuan/projects/dreamerv3-runs/datasets/pong_wm_reg_sweep/eval_replay`
- Window: `chunk_id=0`, `start=8`
- Context: `5`
- Evaluation horizons: `8, 16, 24, 32, 48`
- Action sequence: replay window actions `action[start:start + max(horizons)]`
- SAM2 resolution: `256`
- SAM2 guiding source: `wm`
- Prompt source: `auto`
- No generated prompt policy: `zero`

### Model Registry

Default checkpoint registry comes from `scripts/eval/pong_external_sam2_ball_experiment.py`.

| Key | Name | Initial source | Default checkpoint |
| --- | --- | --- | --- |
| `diamond` | `diamond_repro` | `dataset` | `/data/luyukuan/projects/diamond-assets/checkpoints/Pong.pt` |
| `simulus` | `simulus_repro` | `real` | `/data/luyukuan/projects/Simulus/checkpoints/Pong.pt` |
| `twister` | `twister_repro` | `real` | `/data/luyukuan/projects/TWISTER/callbacks/atari100k/atari100k-pong/checkpoints_epoch_50_step_100000.ckpt` |
| `storm` | `storm_repro` | `real` | `/data/luyukuan/projects/oc-storm/runs/pong_atari100k_reproduction/logdir/Pong-STORM-base/ckpt/latest_agent.pth` |

If a checkpoint differs from the default registry, use `--override project:checkpoint=/path/to/ckpt` and record the exact override in the result document.

## Metrics

Primary:

- `sam2_present_frames`: number of horizon frames with a non-empty SAM2 ball mask.
- `present_rate`: `sam2_present_frames / horizon`.
- `missing_rate`: `1 - present_rate`.
- `first_present_t`: first generated frame index where SAM2 tracks the ball.
- `last_present_t`: last generated frame index where SAM2 tracks the ball.

Diagnostic:

- `ball_l2_rmse`
- `ball_l2_mean`
- `ball_vx_sign_accuracy`
- `ball_vy_sign_accuracy`

Interpretation rule: use coverage fields for cross-model ranking; use RMSE/sign metrics only after visually checking that model resets and trajectories are meaningfully comparable.

## Artifacts

Expected output layout:

- Rollouts: `eval_outputs/.../rollouts/*.npz`
- Raw generated videos: `eval_outputs/.../videos/*.mp4`
- Per-model frames: `eval_outputs/.../frames/<model>/`
- Per-horizon prompt files: `eval_outputs/.../h<horizon>/<model>/prompt.json`
- SAM2 masks: `eval_outputs/.../h<horizon>/<model>/sam2_ball.npz`
- Overlay videos: `eval_outputs/.../h<horizon>/<model>/sam2_ball_overlay.mp4`
- Per-horizon tracks: `eval_outputs/.../h<horizon>/<model>/sam2_tracks.csv`
- Aggregate summary: `eval_outputs/.../external_sam2_ball_summary.csv`
- Manifest: `eval_outputs/.../manifest.json`

## Result Table Template

Fill this table from `external_sam2_ball_summary.csv` after running the experiment.

| Horizon | Model | SAM2 present | Missing rate ↓ | First present | Ball L2 RMSE ↓ | Notes |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 8 | DIAMOND repro | TODO | TODO | TODO | TODO |  |
| 16 | DIAMOND repro | TODO | TODO | TODO | TODO |  |
| 24 | DIAMOND repro | TODO | TODO | TODO | TODO |  |
| 32 | DIAMOND repro | TODO | TODO | TODO | TODO |  |
| 48 | DIAMOND repro | TODO | TODO | TODO | TODO |  |
| 8 | Simulus repro | TODO | TODO | TODO | TODO |  |
| 16 | Simulus repro | TODO | TODO | TODO | TODO |  |
| 24 | Simulus repro | TODO | TODO | TODO | TODO |  |
| 32 | Simulus repro | TODO | TODO | TODO | TODO |  |
| 48 | Simulus repro | TODO | TODO | TODO | TODO |  |
| 8 | TWISTER repro | TODO | TODO | TODO | TODO |  |
| 16 | TWISTER repro | TODO | TODO | TODO | TODO |  |
| 24 | TWISTER repro | TODO | TODO | TODO | TODO |  |
| 32 | TWISTER repro | TODO | TODO | TODO | TODO |  |
| 48 | TWISTER repro | TODO | TODO | TODO | TODO |  |
| 8 | STORM repro | TODO | TODO | TODO | TODO |  |
| 16 | STORM repro | TODO | TODO | TODO | TODO |  |
| 24 | STORM repro | TODO | TODO | TODO | TODO |  |
| 32 | STORM repro | TODO | TODO | TODO | TODO |  |
| 48 | STORM repro | TODO | TODO | TODO | TODO |  |

## Acceptance Checks

Before interpreting results:

- Confirm `manifest.json` lists the intended checkpoint paths.
- Inspect at least one raw generated video per model.
- Inspect at least one SAM2 overlay per model at `H=48`.
- Confirm zero-coverage models are due to no generated-frame ball prompt or failed tracking, not a missing rollout.
- Keep `prompt_source` in the table notes if any model uses fallback behavior.

## Reporting Guidance

The final extension document should phrase conclusions around:

- which models produce a consistently trackable ball-like object,
- how early tracking begins,
- whether trackability survives to longer horizons,
- and whether visual overlays support the CSV coverage numbers.

Avoid paper-facing claims based only on RMSE unless an exact context-aligned initialization path is added for each external WM.
