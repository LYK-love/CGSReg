# DreamerV3 Lambda Search

Purpose: fill the missing DreamerV3 CGSReg hyperparameter points on the main
DIAMOND replay dataset for both `size200m` and `size400m`.

Grid:

- sizes: `size200m`, `size400m`
- `lambda_CGSReg`: `0.001`, `0.1`, `1.0`

Each command runs the full chain:

1. offline WM training if `ckpt/latest` is missing,
2. pixel-space zero-shot RL with 20k actor-critic updates, horizon 512, reward
   quantization threshold 0.1,
3. deterministic real-ALE 20-seed evaluation with reset seeds `0..19`.

Run on longjing-2:

```bash
cd "$HOME/projects/CGSReg"
tmux new -s dreamer_lambda_search
tiny-exp-scheduler run experiments/dreamer_lambda_search/commands.txt \
  --cuda-devices 0,1,2 \
  --cpu-threads 4 \
  --logs-dir experiments/dreamer_lambda_search/scheduler_logs \
  --verbose \
  --keep-job-tabs
```

Outputs:

- offline WMs:
  `$HOME/projects/dreamerv3-runs/pong_wm_reg_sweep/logdir/pong_wm_reg_<size>_mask1_spatial_<slug>_temporal_1`
- zero-shot RL policies:
  `$HOME/projects/dreamerv3-runs/pong_pixel_rl_in_env/logdir/lambda_search_<size>_w<slug>_ac20k_h512_rewq0p1`
- 20-seed eval:
  `artifacts/dreamer_lambda_search/zero_shot_20seed/<size>_w<slug>`
