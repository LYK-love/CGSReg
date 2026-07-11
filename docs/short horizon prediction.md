# Short-Horizon Pong WM Visual Evaluation Report
## Setup
- Eval replay: `runs/datasets/pong_wm_reg_sweep/eval_replay`
- Protocol: same replay context, same future action sequence, open-loop WM rollout.
- Context length: `5` frames.
- Prediction horizon: `12` frames.
- Number of sampled windows: `8`.
- Pixel scale for MSE: `[-1, 1]`.
- Object masks: dataset `mask1/mask2/mask3`, reported as Ball, Left paddle, Right paddle.
- Alignment note: the shared replay context is a conditioning prefix, not a
  shared latent state. Each WM consumes the same real replay history to build
  its own internal start state, then all WMs are rolled forward with the same
  future action sequence and compared against the same real future frames.

## Models
- `baseline`: `runs/pong_atari100k_reproduction/logdir/size200m/ckpt/latest`
- `w001`: `runs/pong_wm_reg_sweep/logdir/pong_wm_reg_size400m_mask1_spatial_0p01_temporal_1/ckpt/latest`
- `w0`: `runs/pong_wm_reg_sweep/logdir/pong_wm_reg_size400m_mask1_spatial_0_temporal_1/ckpt/latest`

## Main Results
| Model | Global pixel MSE ↓ | Ball masked MSE ↓ | Left paddle masked MSE ↓ | Right paddle masked MSE ↓ |
| --- | ---: | ---: | ---: | ---: |
| `baseline` | 0.002983 | 0.105385 | 0.052868 | 0.044612 |
| `w001` | 0.000911 | 0.065412 | 0.013577 | 0.015633 |
| `w0` | 0.000834 | 0.075488 | 0.014044 | 0.017237 |

## Cross-Scale Ball MSE Summary

This table records the H=12 Ball masked MSE values used for the
`repro`/`w=0`/`w=0.01` comparison across the 200m and 400m Dreamer WMs.
The `0.000885` value from the 200m run is the global pixel MSE for
`w=0.01`, not Ball masked MSE.

| Model | Size | Ball masked MSE ↓ |
| --- | ---: | ---: |
| `repro` | 200m | 0.105385 |
| `w=0` | 200m | 0.091289 |
| `w=0.01` | 200m | 0.086355 |
| `w=0` | 400m | 0.075488 |
| `w=0.01` | 400m | 0.065412 |

## Pairwise Prediction Difference
| Pair | Mean pairwise MSE |
| --- | ---: |
| `baseline_vs_w001` | 0.003101 |
| `baseline_vs_w0` | 0.003038 |
| `w001_vs_w0` | 0.000734 |

## Interpretation
- Best global pixel MSE: `w0`. In this short-horizon run, `w0` is 9.28% lower than `w001` on global MSE.
- Best `Ball` masked MSE: `w001`.
- Best `Left paddle` masked MSE: `w001`.
- Best `Right paddle` masked MSE: `w001`.
- Relative to `baseline`, both size400m WMs are substantially better on global and object-region MSE.
- `w001` improves over `w0` on the object masks, while `w0` is slightly better on full-frame MSE. This supports the interpretation that spatial regularization shifts error reduction toward the masked object regions rather than uniformly improving all pixels.

## Scope Note
- This short-horizon visual MSE protocol was useful for the `size400m` case
  summarized in this report, and its ranking was broadly consistent with our
  qualitative observations there.
- In the `size200m` offline `w=0.0` vs `w=0.01` checks, direct qualitative
  inspection in remote play showed that `w=0.01` produced clearly better ball
  dynamics and collisions, while the short-horizon MSE ranking did not
  consistently favor it.
- Because the same protocol does not transfer consistently across model scales,
  we do not plan to use it in the paper, even though it was informative for the
  `size400m` analysis.
- For `size200m`, the intended evidence is qualitative rollout comparison plus
  the SAM2-based object analysis on WM-generated videos.

核心结论：同一批 8 个 replay windows、context=5、短 horizon 12 下，两个 size400m WM 都明显优于 baseline。w0 的全局 pixel MSE 最低：

  | Model | Global MSE ↓ | Ball ↓ | Left paddle ↓ | Right paddle ↓ |
  |---|---:|---:|---:|---:|
  | baseline | 0.002983 | 0.105385 | 0.052868 | 0.044612 |
  | w=0.01 | 0.000911 | 0.065412 | 0.013577 | 0.015633 |
  | w=0 | 0.000834 | 0.075488 | 0.014044 | 0.017237 |

  但在对象区域 masked MSE 上，w=0.01 三个对象都优于 w=0，尤其是 Ball：0.065412 vs 0.075488。所以报告里的解释是：w=0.01 没有降低全局 MSE，但把误差改善
  更集中到了 mask 对象区域，这正好符合你关注视觉对象质量的论点。

## Generated-Video Object Metric With SAM2

上面的 MSE 指标是短序列 reconstruction：WM rollout 与对应 real-env ground-truth frames/masks 做逐像素对比。为了补充一个不直接做 pixel reconstruction 的指标，我另外对 **WM 生成视频本身** 做了 SAM2 球分割/跟踪：

- Rollout protocol: same replay context, same replay action sequence, open-loop WM rollout.
- Context length: `5` frames.
- Replay window: `chunk_id=0`, `start=8`, `prompt_t=0`.
- Horizons: `8, 16, 24, 32, 48`.
- SAM2 backend: `http://209.137.198.192:7263`.
- SAM2 prompt: a positive point from real `mask1` ball center at `prompt_t=0`.
- Important distinction: SAM2 only segments the WM-generated video. Real `mask1` is used to place the prompt and to compute trajectory error/missing denominator, but SAM2 does not see real future frames.

The real target video sanity check passed for every horizon: SAM2 tracked the real video ball for all frames (`8/8`, `16/16`, `24/24`, `32/32`, `48/48`). Thus the missing rates below are not caused by SAM2 failing on the corresponding real video.

### Metrics

- `SAM2 present`: number of WM-generated frames where SAM2 produced a non-empty ball mask.
- `missing rate`: fraction of real-ball-present frames where the WM video did not yield a SAM2 ball mask.
- `ball_center_mse`: squared center error between the SAM2 mask center in the WM video and the real `mask1` ball center, measured in 64x64 frame pixels and computed only on frames where SAM2 produced a ball mask. New runs also report `ball_l2_rmse` for the center-distance RMSE view.
- `first present`: first rollout timestep where SAM2 starts producing a ball mask. This is a diagnostic field, not the primary score.

### Horizon Sweep

| Horizon | Model | SAM2 present | Missing rate ↓ | First present | Archived ball L2 RMSE ↓ |
| ---: | --- | ---: | ---: | ---: | ---: |
| 8 | `baseline` | 0/8 | 1.000 | - | NaN |
| 8 | `w0` | 0/8 | 1.000 | - | NaN |
| 8 | `w001` | 2/8 | 0.750 | 6 | 12.693 |
| 16 | `baseline` | 0/16 | 1.000 | - | NaN |
| 16 | `w0` | 0/16 | 1.000 | - | NaN |
| 16 | `w001` | 10/16 | 0.375 | 6 | 12.752 |
| 24 | `baseline` | 0/24 | 1.000 | - | NaN |
| 24 | `w0` | 0/24 | 1.000 | - | NaN |
| 24 | `w001` | 18/24 | 0.250 | 6 | 13.128 |
| 32 | `baseline` | 0/32 | 1.000 | - | NaN |
| 32 | `w0` | 0/32 | 1.000 | - | NaN |
| 32 | `w001` | 26/32 | 0.188 | 6 | 15.169 |
| 48 | `baseline` | 0/48 | 1.000 | - | NaN |
| 48 | `w0` | 9/48 | 0.812 | 39 | 15.387 |
| 48 | `w001` | 42/48 | 0.125 | 6 | 15.664 |

Interpretation:

- `baseline` never yields a SAM2-trackable ball in this window.
- `w0` is also absent through horizon 32 and only becomes trackable late in horizon 48.
- `w001` produces a trackable ball-like object from timestep 6 onward, and the coverage improves with longer horizons because the same onset is included in a longer rollout.
- The strongest signal here is detectability/coverage, not the absolute center-distance value. The archived table uses the old L2 RMSE view; new runs should cite `ball_center_mse` for MSE and `ball_l2_rmse` for RMSE.
- Together with the masked-MSE result above, this supports the claim that `w001` learns a more object-like ball representation than `baseline` and `w0` in short open-loop WM rollouts.

## Artifacts
- Metrics JSON: `notebook_outputs/pong_wm_visual_three_way_short_horizon/three_way_short_horizon_metrics.json`
- Plots: `notebook_outputs/pong_wm_visual_three_way_short_horizon/metric_plots`
- Exported rollout frames: `notebook_outputs/pong_wm_visual_three_way_short_horizon/rollout_prediction/wm_frames`
- SAM2 sweep summary: `docs/dreamer_wm_experiment_results/sam2_ball_tracking_sweep_summary.csv`
- SAM2 real-target sanity: `docs/dreamer_wm_experiment_results/sam2_real_target_sanity.csv`
- Raw SAM2 rollout outputs: `eval_outputs/pong_sam2_ball_concept_sweep_h{8,16,24,32,48}_chunk0start8_sam2remote_20260530`
