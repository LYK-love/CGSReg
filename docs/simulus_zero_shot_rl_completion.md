# Simulus Zero-Shot RL Restore And Eval Notes

This note records how the Simulus rows in the three-way zero-shot RL comparison
were restored or can be reproduced:

- offline `w=0`
- offline recommended CGSReg weight, currently `w=0.01`

The exp-repro Simulus baseline and the offline `w=0`/`w=0.01` rows are recorded
in `docs/zero_shot_mbrl.md`.

## Target Rows

| Row | WM | Policy |
| --- | --- | --- |
| `w=0` | `offline_static_diamond_w0` | `runs/backend=wm_wm_family=simulus_ckpt=offline_static_diamond_w0_ac20k_envs64_backup15_horizon512_rewq0p1_reset=real/pixel_rl_ckpt/latest.pt` |
| `w=0.01` | `offline_static_diamond_w0p01` | `runs/backend=wm_wm_family=simulus_ckpt=offline_static_diamond_w0p01_ac20k_envs64_backup15_horizon512_rewq0p1_reset=real/pixel_rl_ckpt/latest.pt` |

Both policy checkpoints were trained with `20k` AC updates, `64` envs,
backup length `15`, horizon `512`, reset source `real`, and reward
quantization threshold `0.1`.

## Locate Repositories

Use lowercase repo names on pandaria if that is how the machine is set up.
The commands below auto-detect `simulus` vs `Simulus`.

```bash
set -euo pipefail

WM_EVAL_ROOT="$HOME/projects/CGSReg"
if [ -d "$HOME/projects/simulus" ]; then
  SIMULUS_ROOT="$HOME/projects/simulus"
else
  SIMULUS_ROOT="$HOME/projects/Simulus"
fi
TWISTER_ROOT="${TWISTER_ROOT:-$HOME/projects/twister}"

test -d "$WM_EVAL_ROOT"
test -d "$SIMULUS_ROOT"
test -d "$TWISTER_ROOT"
```

## Restore Existing Policy Checkpoints

This is the preferred path. It avoids retraining the two policies and only
runs the standardized real-env evaluation.

```bash
set -euo pipefail

WM_EVAL_ROOT="$HOME/projects/CGSReg"
if [ -d "$HOME/projects/simulus" ]; then
  SIMULUS_ROOT="$HOME/projects/simulus"
else
  SIMULUS_ROOT="$HOME/projects/Simulus"
fi

cd "$SIMULUS_ROOT"
mkdir -p archives

rclone copy \
  box:zero-shot-rl/simulus/simulus_offline_wm_policy_ckpts_latest.tar.gz \
  archives/ \
  --progress

sha1sum archives/simulus_offline_wm_policy_ckpts_latest.tar.gz
tar -tzf archives/simulus_offline_wm_policy_ckpts_latest.tar.gz | sed -n '1,40p'
tar -xzf archives/simulus_offline_wm_policy_ckpts_latest.tar.gz -C .
```

Expected SHA1 prefix:

```text
d56a389836328b69b455c370840b3ad905432e9c
```

Verify the two policy checkpoints:

```bash
set -euo pipefail

if [ -d "$HOME/projects/simulus" ]; then
  SIMULUS_ROOT="$HOME/projects/simulus"
else
  SIMULUS_ROOT="$HOME/projects/Simulus"
fi

test -f "$SIMULUS_ROOT/runs/backend=wm_wm_family=simulus_ckpt=offline_static_diamond_w0_ac20k_envs64_backup15_horizon512_rewq0p1_reset=real/pixel_rl_ckpt/latest.pt"
test -f "$SIMULUS_ROOT/runs/backend=wm_wm_family=simulus_ckpt=offline_static_diamond_w0p01_ac20k_envs64_backup15_horizon512_rewq0p1_reset=real/pixel_rl_ckpt/latest.pt"
```

## Real-Env Policy Evaluation

Run the shared 5-episode deterministic ALE Pong evaluation. The evaluator uses
the TWISTER real-env adapter only as the real Atari Pong environment; the
policy checkpoints remain Simulus zero-shot RL policies in the shared
`rl-in-pixel-env` Torch actor-critic format.

```bash
set -euo pipefail

WM_EVAL_ROOT="$HOME/projects/CGSReg"
if [ -d "$HOME/projects/simulus" ]; then
  SIMULUS_ROOT="$HOME/projects/simulus"
else
  SIMULUS_ROOT="$HOME/projects/Simulus"
fi
TWISTER_ROOT="${TWISTER_ROOT:-$HOME/projects/twister}"

cd "$WM_EVAL_ROOT"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
TWISTER_ROOT="$TWISTER_ROOT" \
conda run --no-capture-output -n twister python -u \
  scripts/eval/evaluate_torch_pong_real_policies.py \
  --policy "simulus_offline_w0=$SIMULUS_ROOT/runs/backend=wm_wm_family=simulus_ckpt=offline_static_diamond_w0_ac20k_envs64_backup15_horizon512_rewq0p1_reset=real/pixel_rl_ckpt/latest.pt" \
  --policy "simulus_offline_w0p01=$SIMULUS_ROOT/runs/backend=wm_wm_family=simulus_ckpt=offline_static_diamond_w0p01_ac20k_envs64_backup15_horizon512_rewq0p1_reset=real/pixel_rl_ckpt/latest.pt" \
  --output-dir eval_outputs/simulus_offline_zsrl_policy_real_env_eval \
  --episodes 5 \
  --reset-seeds 0,1,2,3,4 \
  --device cuda \
  --deterministic-policy \
  --max-steps-per-episode 30000
```

The main files to copy into `docs/zero_shot_mbrl.md` are:

```text
eval_outputs/simulus_offline_zsrl_policy_real_env_eval/pong_real_policy_eval_summary.csv
eval_outputs/simulus_offline_zsrl_policy_real_env_eval/pong_real_policy_eval_episodes.csv
eval_outputs/simulus_offline_zsrl_policy_real_env_eval/pong_real_policy_eval.json
```

## If The Policy Archive Is Missing

Only use this fallback if the Box archive cannot be restored. It retrains the
offline zero-shot RL policies. The existing command file includes `w=0`,
`w=0.01`, and `w=0.1`; for the paper table, keep `w=0` and `w=0.01`.

```bash
set -euo pipefail

if [ -d "$HOME/projects/simulus" ]; then
  SIMULUS_ROOT="$HOME/projects/simulus"
else
  SIMULUS_ROOT="$HOME/projects/Simulus"
fi

cd "$SIMULUS_ROOT"

tiny-exp-scheduler run scripts/experiments/pong_pixel_rl_offline_wm_sweep.commands.txt \
  --cuda-devices 0 \
  --cpu-threads 2 \
  --logs-dir runs/pong_pixel_rl_offline_wm_sweep/scheduler_logs \
  --verbose \
  --keep-job-tabs
```

After the scheduler finishes, run the real-env policy evaluation section above.

## W&B Runs From The Existing Archive

These runs correspond to the restored policy checkpoints:

| Weight | W&B run |
| --- | --- |
| `w=0` | [`paper-zsrl-simulus-offline-w0-h512-rewq0p1`](https://wandb.ai/ssl-lab/rl-in-pixel-env-simulus/runs/7i0qq7ho) |
| `w=0.01` | [`paper-zsrl-simulus-offline-w0p01-h512-rewq0p1`](https://wandb.ai/ssl-lab/rl-in-pixel-env-simulus/runs/y5n1ev0v) |
| `w=0.1` | [`paper-zsrl-simulus-offline-w0p1-h512-rewq0p1`](https://wandb.ai/ssl-lab/rl-in-pixel-env-simulus/runs/wg1jc761) |

## Summary Table

The Simulus section of `docs/zero_shot_mbrl.md` now has the same three-way
comparison shape as Dreamer and TWISTER:

```text
exp-repro baseline / offline w=0 / offline recommended w=0.01
```
