# W&B Run Cleanup Candidates

Generated on 2026-07-04 from W&B entity `ssl-lab`. No runs have been deleted.
This file lists runs that look safe to review for deletion because they are not
part of the paper registry and are failed, crashed, killed, or very short setup
runs. Keep anything you still need for debugging.

Full inventory:
`artifacts/wandb_run_inventory/wm_wandb_runs_inventory.csv`.

## Delete Candidates

| Project | Run id | Name | Reason |
| --- | --- | --- | --- |
| `ssl-lab/dreamerv3-reborn` | `dstt3n6r` | `runs/pong_atari100k_reproduction/logdir/size800m` | Failed after 11s; no useful completed training. |
| `ssl-lab/dreamerv3-reborn` | `4izn0gnw` | `runs/pong_atari100k_reproduction/logdir/size400m` | Failed after 77s; no useful completed training. |
| `ssl-lab/dreamerv3-reborn` | `ttfmqmmg` | `pong_wm_reg_size400m_mask1_mask3_spatial_0p01_temporal_1` | Crashed duplicate/attempt; not in registry. |
| `ssl-lab/dreamerv3-reborn` | `i2b2w53a` | `pong_wm_reg_size400m_mask1_mask3_spatial_0p05_temporal_1` | Crashed duplicate/attempt; not in registry. |
| `ssl-lab/dreamerv3-reborn` | `e2jb5oic` | `pong_wm_reg_size400m_mask1_mask3_spatial_0p1_temporal_1` | Crashed duplicate/attempt; not in registry. |
| `ssl-lab/dreamerv3-reborn` | `pruyqma1` | `pong_wm_reg_size400m_mask1_mask3_spatial_1_temporal_1` | Crashed duplicate/attempt; not in registry. |
| `ssl-lab/dreamerv3-reborn` | `l9bft2fe` | `pong_wm_reg_size400m_mask1_mask3_spatial_1_temporal_1` | Crashed duplicate/attempt; not in registry. |
| `ssl-lab/dreamerv3-reborn` | `98in8mcc` | `pong_wm_reg_size200m_mask1_mask3_spatial_0_temporal_1` | Finished after only 18s; likely setup/probe run. |
| `ssl-lab/dreamerv3-reborn` | `b2zotrhl` | `pong_wm_reg_size200m_mask1_mask3_spatial_0p01_temporal_1` | Killed after 189s; no useful completed training. |
| `ssl-lab/DreamerV3 (New)` | `dc1vsvlx` | `atari_pong_original_guided_100` | Killed legacy guided run; not part of current WM/ZSRL registry. |
| `ssl-lab/DreamerV3 (New)` | `p4r5wdsj` | `atari_pong_original_guided_10000` | Killed legacy guided run; not part of current WM/ZSRL registry. |
| `ssl-lab/DreamerV3 (New)` | `4b6whckf` | `atari_pong_original_guided_10` | Killed legacy guided run; not part of current WM/ZSRL registry. |
| `ssl-lab/DreamerV3 (New)` | `41j3svfs` | `atari_pong_original_guided_1000` | Killed legacy guided run; not part of current WM/ZSRL registry. |
| `ssl-lab/DreamerV3 (New)` | `yoqdmn53` | `atari_pong_original` | Crashed legacy run; not part of current WM/ZSRL registry. |
| `ssl-lab/DreamerV3 (New)` | `swwlxvlm` | `atari_pong_original_guided_10_early_stop` | Killed legacy guided run; not part of current WM/ZSRL registry. |
| `ssl-lab/DreamerV3 (New)` | `yc0tjbqq` | `atari_pong_original_guided_100_early_stop` | Killed legacy guided run; not part of current WM/ZSRL registry. |
| `ssl-lab/DreamerV3 (New)` | `ztv80kn6` | `atari_pong_original_guided_1000_early_stop` | Killed legacy guided run; not part of current WM/ZSRL registry. |
| `ssl-lab/DreamerV3 (New)` | `zfrhso5k` | `atari_pong_original` | Killed legacy run; not part of current WM/ZSRL registry. |
| `ssl-lab/DreamerV3 (New)` | `i4kl1gss` | `atari100k_pong_original_larger_uniform_dino_recon_regu_ckpt` | Crashed legacy run; not part of current WM/ZSRL registry. |
| `ssl-lab/diamond` | `gjvvj9pp` | `diamond_twexp_w0p01` | Failed TWISTER-replay attempt superseded by dilated-k3 runs. |
| `ssl-lab/diamond` | `rtujcq2n` | `diamond_twexp_w0` | Failed TWISTER-replay attempt superseded by dilated-k3 runs. |
| `ssl-lab/diamond` | `yhkv8nka` | `diamond_twexp_w0p1` | Failed TWISTER-replay attempt superseded by dilated-k3 runs. |
| `ssl-lab/diamond` | `th0w5whu` | `diamond_twexp_w1` | Failed TWISTER-replay attempt superseded by dilated-k3 runs. |
| `ssl-lab/DIAMOND` | `gjvvj9pp` | `diamond_twexp_w0p01` | Same failed run as `ssl-lab/diamond/gjvvj9pp` exposed via project alias. |
| `ssl-lab/DIAMOND` | `rtujcq2n` | `diamond_twexp_w0` | Same failed run as `ssl-lab/diamond/rtujcq2n` exposed via project alias. |
| `ssl-lab/DIAMOND` | `yhkv8nka` | `diamond_twexp_w0p1` | Same failed run as `ssl-lab/diamond/yhkv8nka` exposed via project alias. |
| `ssl-lab/DIAMOND` | `th0w5whu` | `diamond_twexp_w1` | Same failed run as `ssl-lab/diamond/th0w5whu` exposed via project alias. |
| `ssl-lab/twister` | `8g4bkjo1` | `twister_pong_offline_regu_model=base_mask1_spatial_1` | Crashed old offline SR run; registry uses completed AC-CPC runs. |
| `ssl-lab/twister` | `n3c3zf5y` | `twister_pong_offline_regu_model=base_mask1_spatial_0p01` | Crashed old offline SR run; registry uses completed AC-CPC runs. |
| `ssl-lab/twister` | `rznsvbh4` | `twister_pong_offline_regu_model=base_mask1_spatial_10` | Crashed old offline SR run; registry uses completed AC-CPC runs. |
| `ssl-lab/twister` | `t8393u3c` | `twister_pong_offline_regu_model=base_mask1_spatial_0p1` | Crashed old offline SR run; registry uses completed AC-CPC runs. |
| `ssl-lab/rl-in-pixel-env-storm` | `2wg33nvw` | `pong_pixel_rl_in_env/logdir/wm_size200m_mask1_spatial_0p1_envsteps20m_torch_horizon27000` | Crashed old horizon-27000 ZSRL run; registry uses h128/h512 runs. |
| `ssl-lab/rl-in-pixel-env-storm` | `skpli6lt` | `pong_pixel_rl_in_env/logdir/wm_size400m_mask1_spatial_0_envsteps20m_torch_horizon27000` | Crashed old horizon-27000 ZSRL run; registry uses h128/h512 runs. |
| `ssl-lab/rl-in-pixel-env-storm` | `eerxuhri` | `pong_pixel_rl_in_env/logdir/wm_size400m_mask1_spatial_0p01_envsteps20m_torch_horizon27000` | Crashed old horizon-27000 ZSRL run; registry uses h128/h512 runs. |
| `ssl-lab/rl-in-pixel-env-storm` | `01fukmgz` | `pong_pixel_rl_in_env/logdir/wm_size400m_mask1_spatial_0p05_envsteps20m_torch_horizon27000` | Crashed old horizon-27000 ZSRL run; registry uses h128/h512 runs. |
| `ssl-lab/rl-in-pixel-env-storm` | `m4e0nngo` | `pong_pixel_rl_in_env/logdir/wm_size400m_mask1_spatial_0p1_envsteps20m_torch_horizon27000` | Crashed old horizon-27000 ZSRL run; registry uses h128/h512 runs. |
| `ssl-lab/rl-in-pixel-env-storm` | `9am62uk3` | `pong_pixel_rl_in_env/logdir/wm_size400m_mask1_spatial_1_envsteps20m_torch_horizon27000` | Crashed old horizon-27000 ZSRL run; registry uses h128/h512 runs. |
| `ssl-lab/rl-in-pixel-env-storm` | `omm5rczw` | `pong_pixel_rl_in_env/logdir/backend=wm_wm_family=storm_ckpt=pong_atari100k_repro_base_ac20k_envs64_backup15_horizon512_rewq0p5_reset=real_host=scorpio_gpu3_20260531` | Finished after 159s; likely incomplete setup attempt. |
| `ssl-lab/rl-in-pixel-env-storm` | `4xq6lgc5` | `pong_pixel_rl_in_env/logdir/backend=wm_wm_family=storm_ckpt=pong_atari100k_repro_base_ac20k_envs64_backup15_horizon128_rewq0p1_reset=real_host=pandaria` | Crashed; superseded by registry run `eoksijt6`. |
| `ssl-lab/rl-in-pixel-env-storm` | `pnuc2iqr` | `pong_pixel_rl_in_env/logdir/backend=wm_wm_family=storm_ckpt=pong_atari100k_repro_base_ac20k_envs64_backup15_horizon128_rewq0p001_reset=real_host=pandaria` | Crashed; superseded by registry run `9bx3w28b`. |

## Active Runs Renamed

These runs were active or pending at review time and were renamed to match the
shared convention. They are not deletion candidates.

| Project | Run id | Display name |
| --- | --- | --- |
| `ssl-lab/diamond` | `f01i94v4` | `paper-wm-diamond-twexp-dilated-k3-w0` |
| `ssl-lab/diamond` | `dyclkt69` | `paper-wm-diamond-twexp-dilated-k3-w0p01` |
| `ssl-lab/rl-in-pixel-env-twister` | `zwd9m1bt` | `paper-zsrl-twister-twexp-w0-h512-rewq0p5` |
| `ssl-lab/rl-in-pixel-env-twister` | `x0bkt2ot` | `paper-zsrl-twister-twexp-w0p01-h512-rewq0p5` |
| `ssl-lab/rl-in-pixel-env-twister` | `h4bqm2fm` | `paper-zsrl-twister-twexp-w0p1-h512-rewq0p5` |
| `ssl-lab/rl-in-pixel-env-twister` | `4qzb8na8` | `paper-zsrl-twister-twexp-w1-h512-rewq0p5` |
| `ssl-lab/rl-in-pixel-env-simulus` | `2cilzlwy` | `paper-zsrl-simulus-twexp-w0-h512` |
| `ssl-lab/rl-in-pixel-env-simulus` | `lc28ctea` | `paper-zsrl-simulus-twexp-w0p01-h512` |
| `ssl-lab/rl-in-pixel-env-simulus` | `s0odx1er` | `paper-zsrl-simulus-twexp-w0p1-h512` |

## Notes

- The `ssl-lab/diamond` and `ssl-lab/DIAMOND` entries can expose the same W&B
  run ids. Treat duplicate alias rows as one underlying run when deleting.
- One historical Simulus WM registry id, `v2h4k7yr`, was still unresolved under
  `ssl-lab/simulus` during this review.
