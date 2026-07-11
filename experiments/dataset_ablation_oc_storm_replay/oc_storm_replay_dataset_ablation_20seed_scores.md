# OC-STORM Replay Dataset Ablation 20-Seed Scores

Dataset: `oc_storm_replay`

Evaluation protocol: real ALE Pong, deterministic/eval policy mode, 20 reset seeds (`0..19`), total episode return.

## Summary

| Project | lambda_CGSReg | Mean return | Sample std | Min | Max | Episodes | Policy checkpoint |
|---|---:|---:|---:|---:|---:|---:|---|
| DreamerV3 | 0 | -18.55 | 2.46 | -21 | -10 | 20 | `/home/lyk/projects/dreamerv3-runs/pong_pixel_rl_in_env/logdir/oc_storm_replay_dreamer_w0_ac20k_h512_rewq0p1/pixel_rl_ckpt/20260706T070207_update020000.npz` |
| DreamerV3 | 0.01 | -15.15 | 9.43 | -21 | 11 | 20 | `/home/lyk/projects/dreamerv3-runs/pong_pixel_rl_in_env/logdir/oc_storm_replay_dreamer_w0p01_ac20k_h512_rewq0p1/pixel_rl_ckpt/20260706T071404_update020000.npz` |
| DreamerV3 | 0.1 | -20.20 | 0.70 | -21 | -19 | 20 | `/home/lyk/projects/dreamerv3-runs/pong_pixel_rl_in_env/logdir/oc_storm_replay_dreamer_w0p1_ac20k_h512_rewq0p1/pixel_rl_ckpt/20260706T065747_update020000.npz` |
| DreamerV3 | 1.0 | -21.00 | 0.00 | -21 | -21 | 20 | `/home/lyk/projects/dreamerv3-runs/pong_pixel_rl_in_env/logdir/oc_storm_replay_dreamer_w1_ac20k_h512_rewq0p1/pixel_rl_ckpt/20260706T070335_update020000.npz` |
| TWISTER | 0 | -21.00 | 0.00 | -21 | -21 | 20 | `/SSD_RAID0/lyk/twister/runs/pong_pixel_rl_in_env/logdir/oc_storm_replay_twister_w0_retry3_ac20k_h512_rewq0p5/pixel_rl_ckpt/latest.pt` |
| TWISTER | 0.01 | -17.20 | 3.40 | -21 | -14 | 20 | `/SSD_RAID0/lyk/twister/runs/pong_pixel_rl_in_env/logdir/oc_storm_replay_twister_w0p01_ac20k_h512_rewq0p5/pixel_rl_ckpt/latest.pt` |
| TWISTER | 0.1 | -20.80 | 0.41 | -21 | -20 | 20 | `/SSD_RAID0/lyk/twister/runs/pong_pixel_rl_in_env/logdir/oc_storm_replay_twister_w0p1_ac20k_h512_rewq0p5/pixel_rl_ckpt/latest.pt` |
| TWISTER | 1.0 | -21.00 | 0.00 | -21 | -21 | 20 | `/SSD_RAID0/lyk/twister/runs/pong_pixel_rl_in_env/logdir/oc_storm_replay_twister_w1_ac20k_h512_rewq0p5/pixel_rl_ckpt/latest.pt` |
| DIAMOND | 0 | -20.30 | 0.47 | -21 | -20 | 20 | `/home/lyk/projects/diamond/outputs/runs/oc_storm_replay_diamond_w0_ac20k_h512/pixel_rl_ckpt/latest.pt` |
| DIAMOND | 0.01 | -20.25 | 1.33 | -21 | -18 | 20 | `/home/lyk/projects/diamond/outputs/runs/oc_storm_replay_diamond_w0p01_ac20k_h512/pixel_rl_ckpt/latest.pt` |
| DIAMOND | 0.1 | -4.60 | 15.26 | -18 | 14 | 20 | `/home/lyk/projects/diamond/outputs/runs/oc_storm_replay_diamond_w0p1_ac20k_h512/pixel_rl_ckpt/latest.pt` |
| DIAMOND | 1.0 | -20.80 | 0.41 | -21 | -20 | 20 | `/home/lyk/projects/diamond/outputs/runs/oc_storm_replay_diamond_w1_ac20k_h512/pixel_rl_ckpt/latest.pt` |

## Per-Rollout Scores

### DreamerV3

| lambda_CGSReg | Policy | Seed | Score | Length |
|---:|---|---:|---:|---:|
| 0 | `dreamer_oc_storm_replay_w0` | 0 | -20 | 1334 |
| 0 | `dreamer_oc_storm_replay_w0` | 1 | -15 | 2237 |
| 0 | `dreamer_oc_storm_replay_w0` | 2 | -19 | 1675 |
| 0 | `dreamer_oc_storm_replay_w0` | 3 | -17 | 2519 |
| 0 | `dreamer_oc_storm_replay_w0` | 4 | -21 | 1842 |
| 0 | `dreamer_oc_storm_replay_w0` | 5 | -19 | 1799 |
| 0 | `dreamer_oc_storm_replay_w0` | 6 | -20 | 1823 |
| 0 | `dreamer_oc_storm_replay_w0` | 7 | -19 | 1913 |
| 0 | `dreamer_oc_storm_replay_w0` | 8 | -21 | 2120 |
| 0 | `dreamer_oc_storm_replay_w0` | 9 | -19 | 2640 |
| 0 | `dreamer_oc_storm_replay_w0` | 10 | -19 | 1733 |
| 0 | `dreamer_oc_storm_replay_w0` | 11 | -10 | 1997 |
| 0 | `dreamer_oc_storm_replay_w0` | 12 | -18 | 2014 |
| 0 | `dreamer_oc_storm_replay_w0` | 13 | -20 | 2242 |
| 0 | `dreamer_oc_storm_replay_w0` | 14 | -19 | 1499 |
| 0 | `dreamer_oc_storm_replay_w0` | 15 | -18 | 1650 |
| 0 | `dreamer_oc_storm_replay_w0` | 16 | -18 | 2447 |
| 0 | `dreamer_oc_storm_replay_w0` | 17 | -19 | 1615 |
| 0 | `dreamer_oc_storm_replay_w0` | 18 | -19 | 2582 |
| 0 | `dreamer_oc_storm_replay_w0` | 19 | -21 | 1907 |
| 0.01 | `dreamer_oc_storm_replay_w0p01` | 0 | -21 | 814 |
| 0.01 | `dreamer_oc_storm_replay_w0p01` | 1 | -21 | 789 |
| 0.01 | `dreamer_oc_storm_replay_w0p01` | 2 | -18 | 1343 |
| 0.01 | `dreamer_oc_storm_replay_w0p01` | 3 | -17 | 1528 |
| 0.01 | `dreamer_oc_storm_replay_w0p01` | 4 | -21 | 818 |
| 0.01 | `dreamer_oc_storm_replay_w0p01` | 5 | -20 | 896 |
| 0.01 | `dreamer_oc_storm_replay_w0p01` | 6 | -14 | 1714 |
| 0.01 | `dreamer_oc_storm_replay_w0p01` | 7 | -16 | 1585 |
| 0.01 | `dreamer_oc_storm_replay_w0p01` | 8 | -21 | 787 |
| 0.01 | `dreamer_oc_storm_replay_w0p01` | 9 | -13 | 1908 |
| 0.01 | `dreamer_oc_storm_replay_w0p01` | 10 | -6 | 2821 |
| 0.01 | `dreamer_oc_storm_replay_w0p01` | 11 | 9 | 2763 |
| 0.01 | `dreamer_oc_storm_replay_w0p01` | 12 | -21 | 788 |
| 0.01 | `dreamer_oc_storm_replay_w0p01` | 13 | -18 | 1284 |
| 0.01 | `dreamer_oc_storm_replay_w0p01` | 14 | 11 | 2559 |
| 0.01 | `dreamer_oc_storm_replay_w0p01` | 15 | -20 | 954 |
| 0.01 | `dreamer_oc_storm_replay_w0p01` | 16 | -20 | 925 |
| 0.01 | `dreamer_oc_storm_replay_w0p01` | 17 | -21 | 787 |
| 0.01 | `dreamer_oc_storm_replay_w0p01` | 18 | -14 | 1761 |
| 0.01 | `dreamer_oc_storm_replay_w0p01` | 19 | -21 | 788 |
| 0.1 | `dreamer_oc_storm_replay_w0p1` | 0 | -21 | 818 |
| 0.1 | `dreamer_oc_storm_replay_w0p1` | 1 | -20 | 895 |
| 0.1 | `dreamer_oc_storm_replay_w0p1` | 2 | -20 | 860 |
| 0.1 | `dreamer_oc_storm_replay_w0p1` | 3 | -20 | 832 |
| 0.1 | `dreamer_oc_storm_replay_w0p1` | 4 | -20 | 837 |
| 0.1 | `dreamer_oc_storm_replay_w0p1` | 5 | -20 | 838 |
| 0.1 | `dreamer_oc_storm_replay_w0p1` | 6 | -20 | 840 |
| 0.1 | `dreamer_oc_storm_replay_w0p1` | 7 | -19 | 914 |
| 0.1 | `dreamer_oc_storm_replay_w0p1` | 8 | -21 | 847 |
| 0.1 | `dreamer_oc_storm_replay_w0p1` | 9 | -19 | 974 |
| 0.1 | `dreamer_oc_storm_replay_w0p1` | 10 | -20 | 837 |
| 0.1 | `dreamer_oc_storm_replay_w0p1` | 11 | -21 | 782 |
| 0.1 | `dreamer_oc_storm_replay_w0p1` | 12 | -21 | 788 |
| 0.1 | `dreamer_oc_storm_replay_w0p1` | 13 | -21 | 818 |
| 0.1 | `dreamer_oc_storm_replay_w0p1` | 14 | -20 | 842 |
| 0.1 | `dreamer_oc_storm_replay_w0p1` | 15 | -20 | 836 |
| 0.1 | `dreamer_oc_storm_replay_w0p1` | 16 | -20 | 839 |
| 0.1 | `dreamer_oc_storm_replay_w0p1` | 17 | -21 | 819 |
| 0.1 | `dreamer_oc_storm_replay_w0p1` | 18 | -19 | 974 |
| 0.1 | `dreamer_oc_storm_replay_w0p1` | 19 | -21 | 904 |
| 1.0 | `dreamer_oc_storm_replay_w1` | 0 | -21 | 758 |
| 1.0 | `dreamer_oc_storm_replay_w1` | 1 | -21 | 761 |
| 1.0 | `dreamer_oc_storm_replay_w1` | 2 | -21 | 759 |
| 1.0 | `dreamer_oc_storm_replay_w1` | 3 | -21 | 759 |
| 1.0 | `dreamer_oc_storm_replay_w1` | 4 | -21 | 759 |
| 1.0 | `dreamer_oc_storm_replay_w1` | 5 | -21 | 760 |
| 1.0 | `dreamer_oc_storm_replay_w1` | 6 | -21 | 762 |
| 1.0 | `dreamer_oc_storm_replay_w1` | 7 | -21 | 758 |
| 1.0 | `dreamer_oc_storm_replay_w1` | 8 | -21 | 759 |
| 1.0 | `dreamer_oc_storm_replay_w1` | 9 | -21 | 762 |
| 1.0 | `dreamer_oc_storm_replay_w1` | 10 | -21 | 759 |
| 1.0 | `dreamer_oc_storm_replay_w1` | 11 | -21 | 764 |
| 1.0 | `dreamer_oc_storm_replay_w1` | 12 | -21 | 760 |
| 1.0 | `dreamer_oc_storm_replay_w1` | 13 | -21 | 758 |
| 1.0 | `dreamer_oc_storm_replay_w1` | 14 | -21 | 764 |
| 1.0 | `dreamer_oc_storm_replay_w1` | 15 | -21 | 758 |
| 1.0 | `dreamer_oc_storm_replay_w1` | 16 | -21 | 761 |
| 1.0 | `dreamer_oc_storm_replay_w1` | 17 | -21 | 759 |
| 1.0 | `dreamer_oc_storm_replay_w1` | 18 | -21 | 758 |
| 1.0 | `dreamer_oc_storm_replay_w1` | 19 | -21 | 760 |

### TWISTER

| lambda_CGSReg | Policy | Seed | Score | Length |
|---:|---|---:|---:|---:|
| 0 | `twister_oc_storm_replay_w0` | 0 | -21 | 1481 |
| 0 | `twister_oc_storm_replay_w0` | 1 | -21 | 997 |
| 0 | `twister_oc_storm_replay_w0` | 2 | -21 | 1243 |
| 0 | `twister_oc_storm_replay_w0` | 3 | -21 | 1301 |
| 0 | `twister_oc_storm_replay_w0` | 4 | -21 | 997 |
| 0 | `twister_oc_storm_replay_w0` | 5 | -21 | 1478 |
| 0 | `twister_oc_storm_replay_w0` | 6 | -21 | 1004 |
| 0 | `twister_oc_storm_replay_w0` | 7 | -21 | 1240 |
| 0 | `twister_oc_storm_replay_w0` | 8 | -21 | 998 |
| 0 | `twister_oc_storm_replay_w0` | 9 | -21 | 999 |
| 0 | `twister_oc_storm_replay_w0` | 10 | -21 | 997 |
| 0 | `twister_oc_storm_replay_w0` | 11 | -21 | 1480 |
| 0 | `twister_oc_storm_replay_w0` | 12 | -21 | 1298 |
| 0 | `twister_oc_storm_replay_w0` | 13 | -21 | 1239 |
| 0 | `twister_oc_storm_replay_w0` | 14 | -21 | 1478 |
| 0 | `twister_oc_storm_replay_w0` | 15 | -21 | 1240 |
| 0 | `twister_oc_storm_replay_w0` | 16 | -21 | 998 |
| 0 | `twister_oc_storm_replay_w0` | 17 | -21 | 1480 |
| 0 | `twister_oc_storm_replay_w0` | 18 | -21 | 1002 |
| 0 | `twister_oc_storm_replay_w0` | 19 | -21 | 1000 |
| 0.01 | `twister_oc_storm_replay_w0p01` | 0 | -14 | 2834 |
| 0.01 | `twister_oc_storm_replay_w0p01` | 1 | -14 | 2833 |
| 0.01 | `twister_oc_storm_replay_w0p01` | 2 | -21 | 2552 |
| 0.01 | `twister_oc_storm_replay_w0p01` | 3 | -18 | 2837 |
| 0.01 | `twister_oc_storm_replay_w0p01` | 4 | -14 | 2833 |
| 0.01 | `twister_oc_storm_replay_w0p01` | 5 | -21 | 2546 |
| 0.01 | `twister_oc_storm_replay_w0p01` | 6 | -14 | 2840 |
| 0.01 | `twister_oc_storm_replay_w0p01` | 7 | -21 | 2549 |
| 0.01 | `twister_oc_storm_replay_w0p01` | 8 | -14 | 2834 |
| 0.01 | `twister_oc_storm_replay_w0p01` | 9 | -14 | 2835 |
| 0.01 | `twister_oc_storm_replay_w0p01` | 10 | -14 | 2833 |
| 0.01 | `twister_oc_storm_replay_w0p01` | 11 | -21 | 2548 |
| 0.01 | `twister_oc_storm_replay_w0p01` | 12 | -18 | 2834 |
| 0.01 | `twister_oc_storm_replay_w0p01` | 13 | -21 | 2548 |
| 0.01 | `twister_oc_storm_replay_w0p01` | 14 | -21 | 2546 |
| 0.01 | `twister_oc_storm_replay_w0p01` | 15 | -21 | 2549 |
| 0.01 | `twister_oc_storm_replay_w0p01` | 16 | -14 | 2834 |
| 0.01 | `twister_oc_storm_replay_w0p01` | 17 | -21 | 2548 |
| 0.01 | `twister_oc_storm_replay_w0p01` | 18 | -14 | 2838 |
| 0.01 | `twister_oc_storm_replay_w0p01` | 19 | -14 | 2836 |
| 0.1 | `twister_oc_storm_replay_w0p1` | 0 | -21 | 763 |
| 0.1 | `twister_oc_storm_replay_w0p1` | 1 | -21 | 817 |
| 0.1 | `twister_oc_storm_replay_w0p1` | 2 | -21 | 791 |
| 0.1 | `twister_oc_storm_replay_w0p1` | 3 | -21 | 821 |
| 0.1 | `twister_oc_storm_replay_w0p1` | 4 | -21 | 817 |
| 0.1 | `twister_oc_storm_replay_w0p1` | 5 | -20 | 956 |
| 0.1 | `twister_oc_storm_replay_w0p1` | 6 | -21 | 852 |
| 0.1 | `twister_oc_storm_replay_w0p1` | 7 | -21 | 788 |
| 0.1 | `twister_oc_storm_replay_w0p1` | 8 | -21 | 846 |
| 0.1 | `twister_oc_storm_replay_w0p1` | 9 | -21 | 819 |
| 0.1 | `twister_oc_storm_replay_w0p1` | 10 | -21 | 817 |
| 0.1 | `twister_oc_storm_replay_w0p1` | 11 | -20 | 958 |
| 0.1 | `twister_oc_storm_replay_w0p1` | 12 | -21 | 758 |
| 0.1 | `twister_oc_storm_replay_w0p1` | 13 | -21 | 787 |
| 0.1 | `twister_oc_storm_replay_w0p1` | 14 | -20 | 956 |
| 0.1 | `twister_oc_storm_replay_w0p1` | 15 | -21 | 788 |
| 0.1 | `twister_oc_storm_replay_w0p1` | 16 | -21 | 846 |
| 0.1 | `twister_oc_storm_replay_w0p1` | 17 | -20 | 958 |
| 0.1 | `twister_oc_storm_replay_w0p1` | 18 | -21 | 850 |
| 0.1 | `twister_oc_storm_replay_w0p1` | 19 | -21 | 848 |
| 1.0 | `twister_oc_storm_replay_w1` | 0 | -21 | 758 |
| 1.0 | `twister_oc_storm_replay_w1` | 1 | -21 | 757 |
| 1.0 | `twister_oc_storm_replay_w1` | 2 | -21 | 763 |
| 1.0 | `twister_oc_storm_replay_w1` | 3 | -21 | 761 |
| 1.0 | `twister_oc_storm_replay_w1` | 4 | -21 | 757 |
| 1.0 | `twister_oc_storm_replay_w1` | 5 | -21 | 758 |
| 1.0 | `twister_oc_storm_replay_w1` | 6 | -21 | 764 |
| 1.0 | `twister_oc_storm_replay_w1` | 7 | -21 | 760 |
| 1.0 | `twister_oc_storm_replay_w1` | 8 | -21 | 758 |
| 1.0 | `twister_oc_storm_replay_w1` | 9 | -21 | 759 |
| 1.0 | `twister_oc_storm_replay_w1` | 10 | -21 | 757 |
| 1.0 | `twister_oc_storm_replay_w1` | 11 | -21 | 760 |
| 1.0 | `twister_oc_storm_replay_w1` | 12 | -21 | 758 |
| 1.0 | `twister_oc_storm_replay_w1` | 13 | -21 | 759 |
| 1.0 | `twister_oc_storm_replay_w1` | 14 | -21 | 758 |
| 1.0 | `twister_oc_storm_replay_w1` | 15 | -21 | 760 |
| 1.0 | `twister_oc_storm_replay_w1` | 16 | -21 | 758 |
| 1.0 | `twister_oc_storm_replay_w1` | 17 | -21 | 760 |
| 1.0 | `twister_oc_storm_replay_w1` | 18 | -21 | 762 |
| 1.0 | `twister_oc_storm_replay_w1` | 19 | -21 | 760 |

### DIAMOND

| lambda_CGSReg | Policy | Seed | Score | Length |
|---:|---|---:|---:|---:|
| 0 | `diamond_oc_storm_replay_w0` | 0 | -20 | 884 |
| 0 | `diamond_oc_storm_replay_w0` | 1 | -20 | 882 |
| 0 | `diamond_oc_storm_replay_w0` | 2 | -21 | 763 |
| 0 | `diamond_oc_storm_replay_w0` | 3 | -21 | 820 |
| 0 | `diamond_oc_storm_replay_w0` | 4 | -20 | 882 |
| 0 | `diamond_oc_storm_replay_w0` | 5 | -20 | 915 |
| 0 | `diamond_oc_storm_replay_w0` | 6 | -20 | 889 |
| 0 | `diamond_oc_storm_replay_w0` | 7 | -21 | 760 |
| 0 | `diamond_oc_storm_replay_w0` | 8 | -20 | 883 |
| 0 | `diamond_oc_storm_replay_w0` | 9 | -20 | 884 |
| 0 | `diamond_oc_storm_replay_w0` | 10 | -20 | 882 |
| 0 | `diamond_oc_storm_replay_w0` | 11 | -20 | 917 |
| 0 | `diamond_oc_storm_replay_w0` | 12 | -21 | 817 |
| 0 | `diamond_oc_storm_replay_w0` | 13 | -21 | 759 |
| 0 | `diamond_oc_storm_replay_w0` | 14 | -20 | 915 |
| 0 | `diamond_oc_storm_replay_w0` | 15 | -21 | 760 |
| 0 | `diamond_oc_storm_replay_w0` | 16 | -20 | 883 |
| 0 | `diamond_oc_storm_replay_w0` | 17 | -20 | 917 |
| 0 | `diamond_oc_storm_replay_w0` | 18 | -20 | 887 |
| 0 | `diamond_oc_storm_replay_w0` | 19 | -20 | 885 |
| 0.01 | `diamond_oc_storm_replay_w0p01` | 0 | -21 | 758 |
| 0.01 | `diamond_oc_storm_replay_w0p01` | 1 | -21 | 786 |
| 0.01 | `diamond_oc_storm_replay_w0p01` | 2 | -21 | 791 |
| 0.01 | `diamond_oc_storm_replay_w0p01` | 3 | -21 | 789 |
| 0.01 | `diamond_oc_storm_replay_w0p01` | 4 | -21 | 786 |
| 0.01 | `diamond_oc_storm_replay_w0p01` | 5 | -18 | 1444 |
| 0.01 | `diamond_oc_storm_replay_w0p01` | 6 | -18 | 1727 |
| 0.01 | `diamond_oc_storm_replay_w0p01` | 7 | -21 | 788 |
| 0.01 | `diamond_oc_storm_replay_w0p01` | 8 | -21 | 787 |
| 0.01 | `diamond_oc_storm_replay_w0p01` | 9 | -21 | 788 |
| 0.01 | `diamond_oc_storm_replay_w0p01` | 10 | -21 | 786 |
| 0.01 | `diamond_oc_storm_replay_w0p01` | 11 | -18 | 1446 |
| 0.01 | `diamond_oc_storm_replay_w0p01` | 12 | -21 | 758 |
| 0.01 | `diamond_oc_storm_replay_w0p01` | 13 | -21 | 787 |
| 0.01 | `diamond_oc_storm_replay_w0p01` | 14 | -18 | 1444 |
| 0.01 | `diamond_oc_storm_replay_w0p01` | 15 | -21 | 788 |
| 0.01 | `diamond_oc_storm_replay_w0p01` | 16 | -21 | 787 |
| 0.01 | `diamond_oc_storm_replay_w0p01` | 17 | -18 | 1446 |
| 0.01 | `diamond_oc_storm_replay_w0p01` | 18 | -21 | 791 |
| 0.01 | `diamond_oc_storm_replay_w0p01` | 19 | -21 | 789 |
| 0.1 | `diamond_oc_storm_replay_w0p1` | 0 | 10 | 2525 |
| 0.1 | `diamond_oc_storm_replay_w0p1` | 1 | -18 | 2717 |
| 0.1 | `diamond_oc_storm_replay_w0p1` | 2 | 14 | 2120 |
| 0.1 | `diamond_oc_storm_replay_w0p1` | 3 | -18 | 2690 |
| 0.1 | `diamond_oc_storm_replay_w0p1` | 4 | -18 | 2717 |
| 0.1 | `diamond_oc_storm_replay_w0p1` | 5 | 10 | 2525 |
| 0.1 | `diamond_oc_storm_replay_w0p1` | 6 | -18 | 2724 |
| 0.1 | `diamond_oc_storm_replay_w0p1` | 7 | 14 | 2117 |
| 0.1 | `diamond_oc_storm_replay_w0p1` | 8 | -18 | 2718 |
| 0.1 | `diamond_oc_storm_replay_w0p1` | 9 | -18 | 2719 |
| 0.1 | `diamond_oc_storm_replay_w0p1` | 10 | -18 | 2717 |
| 0.1 | `diamond_oc_storm_replay_w0p1` | 11 | 10 | 2527 |
| 0.1 | `diamond_oc_storm_replay_w0p1` | 12 | -18 | 2687 |
| 0.1 | `diamond_oc_storm_replay_w0p1` | 13 | 14 | 2116 |
| 0.1 | `diamond_oc_storm_replay_w0p1` | 14 | 10 | 2525 |
| 0.1 | `diamond_oc_storm_replay_w0p1` | 15 | 14 | 2117 |
| 0.1 | `diamond_oc_storm_replay_w0p1` | 16 | -18 | 2718 |
| 0.1 | `diamond_oc_storm_replay_w0p1` | 17 | 10 | 2527 |
| 0.1 | `diamond_oc_storm_replay_w0p1` | 18 | -18 | 2722 |
| 0.1 | `diamond_oc_storm_replay_w0p1` | 19 | -18 | 2720 |
| 1.0 | `diamond_oc_storm_replay_w1` | 0 | -20 | 839 |
| 1.0 | `diamond_oc_storm_replay_w1` | 1 | -21 | 785 |
| 1.0 | `diamond_oc_storm_replay_w1` | 2 | -20 | 841 |
| 1.0 | `diamond_oc_storm_replay_w1` | 3 | -21 | 789 |
| 1.0 | `diamond_oc_storm_replay_w1` | 4 | -21 | 785 |
| 1.0 | `diamond_oc_storm_replay_w1` | 5 | -21 | 786 |
| 1.0 | `diamond_oc_storm_replay_w1` | 6 | -21 | 792 |
| 1.0 | `diamond_oc_storm_replay_w1` | 7 | -21 | 760 |
| 1.0 | `diamond_oc_storm_replay_w1` | 8 | -21 | 786 |
| 1.0 | `diamond_oc_storm_replay_w1` | 9 | -21 | 787 |
| 1.0 | `diamond_oc_storm_replay_w1` | 10 | -21 | 785 |
| 1.0 | `diamond_oc_storm_replay_w1` | 11 | -20 | 837 |
| 1.0 | `diamond_oc_storm_replay_w1` | 12 | -21 | 786 |
| 1.0 | `diamond_oc_storm_replay_w1` | 13 | -21 | 759 |
| 1.0 | `diamond_oc_storm_replay_w1` | 14 | -21 | 786 |
| 1.0 | `diamond_oc_storm_replay_w1` | 15 | -21 | 760 |
| 1.0 | `diamond_oc_storm_replay_w1` | 16 | -21 | 786 |
| 1.0 | `diamond_oc_storm_replay_w1` | 17 | -20 | 837 |
| 1.0 | `diamond_oc_storm_replay_w1` | 18 | -21 | 790 |
| 1.0 | `diamond_oc_storm_replay_w1` | 19 | -21 | 788 |

## Files

- Aggregate summary CSV: `experiments/dataset_ablation_oc_storm_replay/oc_storm_replay_dataset_ablation_20seed_summary.csv`
- Per-rollout score CSV: `experiments/dataset_ablation_oc_storm_replay/oc_storm_replay_dataset_ablation_20seed_scores.csv`
- DreamerV3 source summary: `artifacts/dataset_ablation_oc_storm_replay/fixed_real_eval/dreamer/pong_real_policy_eval_summary.csv`
- DreamerV3 source episodes: `artifacts/dataset_ablation_oc_storm_replay/fixed_real_eval/dreamer/pong_real_policy_eval_episodes.csv`
- TWISTER source summary: `artifacts/dataset_ablation_oc_storm_replay/fixed_real_eval/twister/pong_real_policy_eval_summary.csv`
- TWISTER source episodes: `artifacts/dataset_ablation_oc_storm_replay/fixed_real_eval/twister/pong_real_policy_eval_episodes.csv`
- DIAMOND source summary: `artifacts/dataset_ablation_oc_storm_replay/fixed_real_eval/diamond/pong_real_policy_eval_summary.csv`
- DIAMOND source episodes: `artifacts/dataset_ablation_oc_storm_replay/fixed_real_eval/diamond/pong_real_policy_eval_episodes.csv`
