# Generated-video ball tracking

本实验固定同一 **deterministic policy**，在不同 Dreamer WM ckpt 上各自 rollout；由于 policy 与初始热启动不同步，轨迹非同一 replay 序列，不共享同一段真实轨迹。  
`context=5` 仅用于 latent bootstrap（用前 5 帧条件化），从 `\hat{o}_6` 起为 open-loop 生成，不计入评估 horizon。

Recommended policy ckpt:

- `real_env_ac20k_backup15`
- Dreamer-compatible ckpt:
  `/data/luyukuan/projects/dreamerv3-runs/policies/rl_in_pixel_env/real_env_ac20k_backup15.dreamer_pixel_rl.npz`
- Converted JAX ckpt:
  `/data/luyukuan/projects/dreamerv3-runs/policies/rl_in_pixel_env/real_env_ac20k_backup15.jax.npz`
- Original rl-in-pixel-env Torch ckpt:
  `/scorpio/home/luyukuan/projects/rl-in-pixel-env/runs/backend=real_ac20k_envs64_backup15_cuda3/pixel_rl_ckpt/latest.pt`

This is the real-env trained Pong policy with the same Pixel-RL setup as the
WM-env policies except for the environment backend. For
`scripts/eval/pong_sam2_ball_experiment.py --policy`, use the
Dreamer-compatible `.dreamer_pixel_rl.npz` file above.

Archived policy notes:

- `.../pong_pixel_rl_in_env/logdir/runs/thr0p1_seed20260531/pixel_rl_ckpt/20260528T193744_update020000.npz`
  is a WM-backend policy from the size400m sweep alias.
- `/data/luyukuan/projects/dreamerv3-runs/pong_pixel_rl_in_env/logdir/wm_size200m_repro_envsteps20m_jax_horizon512_20260531_111215_evalfix_r1/pixel_rl_ckpt/20260531T152310_update020000.npz`
  is a WM-backend size200m policy and should not be described as real-env
  trained.

Metric note: current code reports `ball_center_mse` as true squared center
error, and also writes `ball_l2_mean` / `ball_l2_rmse`. The archived tables
below were produced before that rename/fix, so their center-error column should
be treated as historical L2-distance summaries unless rerun.

## size400m 结果（4 ckpt）

### H=8

| WM | k/H | Archived center L2 |
| --- | ---: | ---: |
| `size400m_w0` | 0/8 = 0.0000 | NaN |
| `size400m_w0.01` | 6/8 = 0.7500 | 4.7804624214 |
| `size400m_w0.1` | 3/8 = 0.3750 | 14.5904176650 |
| `size400m_w1` | 1/8 = 0.1250 | 13.7217875199 |

### H=16

| WM | k/H | Archived center L2 |
| --- | ---: | ---: |
| `size400m_w0` | 1/16 = 0.0625 | 23.7947120814 |
| `size400m_w0.01` | 14/16 = 0.8750 | 4.5959324752 |
| `size400m_w0.1` | 4/16 = 0.2500 | 17.5694018406 |
| `size400m_w1` | 9/16 = 0.5625 | 28.2490549070 |

### H=32

| WM | k/H | Archived center L2 |
| --- | ---: | ---: |
| `size400m_w0` | 1/32 = 0.03125 | 23.7947120814 |
| `size400m_w0.01` | 30/32 = 0.9375 | 5.6755720547 |
| `size400m_w0.1` | 4/32 = 0.1250 | 17.5694018406 |
| `size400m_w1` | 16/32 = 0.5000 | 33.5124026265 |

## size200m / size400m 汇总占位（按实验命名）

## size200m 结果（5 ckpt）

说明：`NaN` 表示该 horizon 下 SAM2 全程未检测到球。

| Model | H=8 detectability (k/H) | H=8 MSE | H=16 detectability (k/H) | H=16 MSE | H=32 detectability (k/H) | H=32 MSE |
|---|---:|---:|---:|---:|---:|---:|
| size200m_repro | 0/8 = 0.00000 | NaN | 0/16 = 0.00000 | NaN | 0/32 = 0.00000 | NaN |
| size200m_w0 | 0/8 = 0.00000 | NaN | 0/16 = 0.00000 | NaN | 0/32 = 0.00000 | NaN |
| size200m_w0.01 | 0/8 = 0.00000 | NaN | 0/16 = 0.00000 | NaN | 0/32 = 0.00000 | NaN |
| size200m_w0.1 | 0/8 = 0.00000 | NaN | 0/16 = 0.00000 | NaN | 0/32 = 0.00000 | NaN |
| size200m_w1 | 3/8 = 0.37500 | 10.18397 | 11/16 = 0.68750 | 9.90350 | 27/32 = 0.84375 | 11.61190 |

> 注：上表按固定同一 policy 与 `context=5` 进行 open-loop 生成后统计。
