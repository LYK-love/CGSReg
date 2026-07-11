# Exp-Repro Frozen-WM RL-Only Diagnostic

This runbook is for pandaria. It launches the diagnostic experiment used by the
main-paper frozen-WM gap figure:

1. take the world-model checkpoint produced by each reproduced Dyna-style MBRL
   run;
2. freeze that world model;
3. train only a new RL policy/value function inside the frozen model, using the
   corresponding project's native pixel-space MBRL/RL protocol;
4. evaluate the resulting policy in real ALE Pong with the common deterministic
   20-seed protocol.

This is not the unified zero-shot RL protocol used in the main CGSReg result
figure. It is the project-native RL-only continuation/retraining diagnostic.

## Definition of "Frozen WM"

For this experiment, "frozen WM" means the whole learned simulator used by the
project's pixel-space WM environment is fixed. No world-model optimizer runs.
Only the newly initialized RL policy/value parameters are updated.

Project-specific boundary:

| Project | Frozen components |
| --- | --- |
| DreamerV3 | RSSM/dynamics, decoder, reward/continuation heads from the exp-repro checkpoint |
| DIAMOND | full exp-repro agent checkpoint loaded through `--wm-checkpoint`, including denoiser and `rew_end_model`; do not mix in HF `checkpoints/Pong.pt` reward/end weights |
| TWISTER | full Transformer WM checkpoint used by its `pixel_rl` WM env |
| Simulus | full Simulus checkpoint used by its `src.pixel_rl` WM env |
| STORM | full STORM checkpoint used by its `pixel_rl` WM env |

The DIAMOND command file intentionally sets:

```bash
DIAMOND_PIXEL_RL_WM_CHECKPOINT=checkpoints/reproduce_pong/agent_versions/agent_epoch_01000.pt
```

This forces the adapter to call `agent.load(path)` and load the full checkpoint.
The positional component arguments at the end of the DIAMOND command are then
ignored by the adapter.

## Target Matrix

| Project | Exp-repro WM checkpoint | RL-only protocol |
| --- | --- | --- |
| DreamerV3 | `dreamerv3-reborn/runs/pong_atari100k_reproduction/logdir/size200m/ckpt/latest` | Dreamer pixel RL in WM, 20k AC updates, envs 64, horizon 512 |
| DIAMOND | `diamond/checkpoints/reproduce_pong/agent_versions/agent_epoch_01000.pt` | DIAMOND pixel RL in full exp-repro WM, 20k AC updates, envs 64, horizon 512 |
| TWISTER | `twister/callbacks/atari100k/atari100k-pong/checkpoints_epoch_50_step_100000.ckpt` | TWISTER pixel RL in WM, 20k AC updates, envs 64, horizon 512 |
| Simulus | `simulus/checkpoints/Pong.pt` | Simulus pixel RL in WM, 20k AC updates, envs 64, horizon 512 |
| STORM | `oc-storm/runs/pong_atari100k_reproduction/logdir/Pong-STORM-base/ckpt/latest_agent.pth` | STORM pixel RL in WM, 20k AC updates, envs 64, horizon 512 |

Each run uses the project's existing reward quantization and WM reset/context
settings from the previous horizon-512 pixel-RL-in-WM experiments.

## Expected Runtime

These are estimates from earlier 20k-update pixel-RL-in-WM runs on pandaria or
nearby servers. Start DIAMOND first because it is the slowest.

| Project | Reference speed | Estimated runtime |
| --- | ---: | ---: |
| DreamerV3 | W&B `qa32kfzt`, `_runtime ~= 15124s`, `progress/fps ~= 1286` | 4.0-4.5 h |
| DIAMOND | W&B `v4y77brt` / `uwqo1qel`, `_runtime ~= 39704-42232s`, `progress/fps ~= 458-491` | 11-12 h |
| TWISTER | W&B `q0uu7f2c` / local `q8c8bege`, `_runtime ~= 6225-10729s` | 1.8-3.0 h |
| Simulus | W&B summary runtime is unreliable, but `progress/fps ~= 1480-1500` | 3.5-4.0 h |
| STORM | W&B `phzb9xat` / `imj7qsz4`, `_runtime ~= 7906-13356s` | 2.5-4.0 h |

If all five run concurrently on separate GPUs, the wall time should be dominated
by DIAMOND, roughly half a day. Sequential runtime is roughly 23-28 hours.

## Preflight on Pandaria

Run this first:

```bash
hostname
whoami
nvidia-smi

cd "$HOME/projects/CGSReg"
git status --short
test -f experiments/exp_repro_frozen_wm_rl_only_20seed/commands/diamond_exp_repro_rl_only.commands.txt
rg -n "DIAMOND_PIXEL_RL_WM_CHECKPOINT" \
  experiments/exp_repro_frozen_wm_rl_only_20seed/commands/diamond_exp_repro_rl_only.commands.txt
```

Check that the five WM checkpoints exist:

```bash
test -e "$HOME/projects/dreamerv3-reborn/runs/pong_atari100k_reproduction/logdir/size200m/ckpt/latest"
test -f "$HOME/projects/diamond/checkpoints/reproduce_pong/agent_versions/agent_epoch_01000.pt"
test -f "$HOME/projects/twister/callbacks/atari100k/atari100k-pong/checkpoints_epoch_50_step_100000.ckpt"
test -f "$HOME/projects/simulus/checkpoints/Pong.pt"
test -e "$HOME/projects/oc-storm/runs/pong_atari100k_reproduction/logdir/Pong-STORM-base/ckpt/latest_agent.pth"
```

Check environments:

```bash
cd "$HOME/projects/dreamerv3-reborn" && conda run -n dreamer python - <<'PY'
print("dreamer env ok")
PY
cd "$HOME/projects/diamond" && conda run -n diamond python - <<'PY'
print("diamond env ok")
PY
cd "$HOME/projects/twister" && conda run -n twister python - <<'PY'
print("twister env ok")
PY
cd "$HOME/projects/simulus" && conda run -n simulus python - <<'PY'
print("simulus env ok")
PY
cd "$HOME/projects/oc-storm" && conda run -n oc-storm python - <<'PY'
print("oc-storm env ok")
PY
```

## Smoke Test

Before the 20k-update run, create temporary 200-update smoke command files.
This checks checkpoint loading, WM reset/context collection, W&B setup, real-env
eval, and policy checkpoint writing.

```bash
cd "$HOME/projects/CGSReg"
mkdir -p experiments/exp_repro_frozen_wm_rl_only_20seed/commands_smoke

for f in experiments/exp_repro_frozen_wm_rl_only_20seed/commands/*_exp_repro_rl_only.commands.txt; do
  base="$(basename "$f")"
  perl -pe '
    s/AC_UPDATES=20000/AC_UPDATES=200/g;
    s/EVAL_REAL_EVERY=2000/EVAL_REAL_EVERY=100/g;
    s/EVAL_REAL_VIDEO_EVERY=10000/EVAL_REAL_VIDEO_EVERY=0/g;
    s/VIDEO_EVERY=10000/VIDEO_EVERY=0/g;
    s/WM_VIDEO_EVERY=10000/WM_VIDEO_EVERY=0/g;
    s/ROLLOUT_VIDEO_EVERY=10000/ROLLOUT_VIDEO_EVERY=0/g;
    s/SAVE_EVERY=10000/SAVE_EVERY=200/g;
    s/SAVE_EVERY=2000/SAVE_EVERY=200/g;
    s/CHECKPOINT_KEEP=10/CHECKPOINT_KEEP=2/g;
    s/h512_ac20k/h512_smoke200/g;
    s/ac20k/ac200/g;
  ' "$f" > "experiments/exp_repro_frozen_wm_rl_only_20seed/commands_smoke/$base"
done
```

Launch the smoke tests exactly like the full runs, but point each scheduler to
`commands_smoke`. Use a separate logs directory with `_smoke`.

The smoke tests should reach `update=200/200`, write a `pixel_rl_ckpt`, and exit
with code 0. If a smoke test fails, fix that project before launching the 20k
run.

## Full 20k Runs

Run from pandaria after preflight and smoke tests pass. Each command file
assumes the working directory shown in its header.

Start DIAMOND first because it is the slowest.

### DreamerV3

```bash
cd "$HOME/projects/dreamerv3-reborn"
tiny-exp-scheduler run \
  "$HOME/projects/CGSReg/experiments/exp_repro_frozen_wm_rl_only_20seed/commands/dreamer_exp_repro_rl_only.commands.txt" \
  --cuda-devices auto \
  --cpu-threads 2 \
  --logs-dir "$HOME/projects/dreamerv3-runs/exp_repro_rl_only_20seed/scheduler_logs" \
  --verbose --keep-job-tabs
```

### DIAMOND

```bash
cd "$HOME/projects/diamond"
tiny-exp-scheduler run \
  "$HOME/projects/CGSReg/experiments/exp_repro_frozen_wm_rl_only_20seed/commands/diamond_exp_repro_rl_only.commands.txt" \
  --cuda-devices auto \
  --cpu-threads 2 \
  --logs-dir outputs/exp_repro_rl_only_20seed/logs \
  --verbose --keep-job-tabs
```

### TWISTER

```bash
cd "$HOME/projects/twister"
tiny-exp-scheduler run \
  "$HOME/projects/CGSReg/experiments/exp_repro_frozen_wm_rl_only_20seed/commands/twister_exp_repro_rl_only.commands.txt" \
  --cuda-devices auto \
  --cpu-threads 2 \
  --logs-dir pong_pixel_rl_in_env/scheduler_logs/exp_repro_rl_only_20seed \
  --verbose --keep-job-tabs
```

### Simulus

```bash
cd "$HOME/projects/simulus"
tiny-exp-scheduler run \
  "$HOME/projects/CGSReg/experiments/exp_repro_frozen_wm_rl_only_20seed/commands/simulus_exp_repro_rl_only.commands.txt" \
  --cuda-devices auto \
  --cpu-threads 2 \
  --logs-dir outputs/exp_repro_rl_only_20seed/logs \
  --verbose --keep-job-tabs
```

### STORM

```bash
cd "$HOME/projects/oc-storm"
tiny-exp-scheduler run \
  "$HOME/projects/CGSReg/experiments/exp_repro_frozen_wm_rl_only_20seed/commands/storm_exp_repro_rl_only.commands.txt" \
  --cuda-devices auto \
  --cpu-threads 2 \
  --logs-dir runs/pong_pixel_rl_in_env/scheduler_logs/exp_repro_rl_only_20seed \
  --verbose --keep-job-tabs
```

## Monitoring

Healthy training logs should look like this:

```text
Pixel RL framework=torch backend=wm device=cuda envs=64 ... updates=20000 ...
[torch:wm] update=1000/20000 env_steps=960000/19200000 ...
[torch:wm] update=2000/20000 ... eval_real/score_mean=...
```

For DreamerV3 JAX, the prefix may be different, but it should still report AC
updates, env steps, fps, and `eval_real/*` metrics.

Monitor scheduler exits:

```bash
find "$HOME/projects" -path '*exp_repro_rl_only_20seed*' -name '*.exit' -print -exec cat {} \;
```

Monitor latest logs:

```bash
tail -f "$HOME/projects/dreamerv3-runs/exp_repro_rl_only_20seed/scheduler_logs"/*.log
tail -f "$HOME/projects/diamond/outputs/exp_repro_rl_only_20seed/logs"/*.log
tail -f "$HOME/projects/twister/pong_pixel_rl_in_env/scheduler_logs/exp_repro_rl_only_20seed"/*.log
tail -f "$HOME/projects/simulus/outputs/exp_repro_rl_only_20seed/logs"/*.log
tail -f "$HOME/projects/oc-storm/runs/pong_pixel_rl_in_env/scheduler_logs/exp_repro_rl_only_20seed"/*.log
```

## Known Risks and Fixes

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| DIAMOND logs mention only `--denoiser-ckpt` component loading | command file is stale | sync this repo or ensure `DIAMOND_PIXEL_RL_WM_CHECKPOINT=checkpoints/reproduce_pong/agent_versions/agent_epoch_01000.pt` is present |
| checkpoint path not found | repo/checkpoints not synced on pandaria | restore/symlink checkpoint, then rerun smoke |
| run resumes from an old policy checkpoint | old run dir with `latest.pt` exists and resume was not disabled | current commands set resume false; if still ambiguous, change run suffix before rerun |
| W&B upload hangs after training | video or network upload delay | local checkpoint is usually already written; wait or rerun with `WANDB_MODE=offline` |
| W&B warning about video fps | W&B behavior for uploaded video files | harmless |
| training-time `eval_real` uses 5 episodes | by design for monitoring | final paper values come from the separate 20-seed eval below |
| no `eval_real` during 200-update smoke | eval cadence was not reduced | ensure smoke command replaced `EVAL_REAL_EVERY=2000` with `100` |

The main logging risk is not `wm_updates=0`, because these commands do not run
the original Dyna-style WM training loop. They run frozen-WM pixel-RL runners
whose counters are `ac_updates` and `env_steps`.

## Fixed 20-Seed Real-Env Eval

After the five RL-only policy checkpoints exist, run:

```bash
cd "$HOME/projects/CGSReg"
tiny-exp-scheduler run \
  experiments/exp_repro_frozen_wm_rl_only_20seed/commands/eval_exp_repro_rl_only_20seed.commands.txt \
  --cuda-devices auto \
  --cpu-threads 2 \
  --logs-dir experiments/exp_repro_frozen_wm_rl_only_20seed/logs/eval_20seed \
  --verbose --keep-job-tabs
```

The eval protocol is:

```text
episodes = 20
reset_seeds = 0..19
policy = deterministic / eval mode
metric = real ALE Pong return
```

Write final paper values from:

```bash
$HOME/projects/CGSReg/artifacts/exp_repro_frozen_wm_rl_only_20seed/eval/*/pong_real_policy_eval_summary.csv
```

The eval command file uses default checkpoint paths matching the run names
above. If any project writes a checkpoint under a different restored path on
pandaria, override that path with:

```bash
export DREAMER_EXP_REPRO_RL_ONLY_POLICY=/path/to/dreamer/pixel_rl_ckpt/latest.npz
export DIAMOND_EXP_REPRO_RL_ONLY_POLICY=/path/to/diamond/pixel_rl_ckpt/latest.pt
export TWISTER_EXP_REPRO_RL_ONLY_POLICY=/path/to/twister/pixel_rl_ckpt/latest.pt
export SIMULUS_EXP_REPRO_RL_ONLY_POLICY=/path/to/simulus/pixel_rl_ckpt/latest.pt
export STORM_EXP_REPRO_RL_ONLY_POLICY=/path/to/storm/pixel_rl_ckpt/latest.pt
```

Then rerun the eval scheduler command.

## Archived Policy Checkpoints

The five pandaria RL-only jobs completed on 2026-06-30. Their policy
checkpoints, command files, scheduler logs, monitoring configs, and short videos
are archived in Box:

```text
box:projects/wm-evaluation/exp_repro_frozen_wm_rl_only_20seed/policy_ckpts/exp_repro_frozen_wm_rl_only_policy_ckpts_20260630/
```

Archive files:

```text
exp_repro_frozen_wm_rl_only_policy_ckpts_20260630.tar.zst
exp_repro_frozen_wm_rl_only_policy_ckpts_20260630.tar.zst.sha256
```

Training-time 5-episode real-env eval summaries:

| Project | W&B run | Final `eval_real/score_mean` | Archived final policy |
| --- | --- | ---: | --- |
| DreamerV3 | `8tcwls9u` | -21.0 | `dreamer/pixel_rl_ckpt/20260630T122117_update020000.npz` |
| DIAMOND | `yrshnfym` | -10.8 | `diamond/pixel_rl_ckpt/20260630T190647_update020000.pt` |
| TWISTER | `85tyxmx7` | -14.2 | `twister/pixel_rl_ckpt/20260630T091105_update020000.pt` |
| Simulus | `j2346z74` | -3.0 | `simulus/pixel_rl_ckpt/20260630T163821_update020000.pt` |
| STORM | `64k1abna` | -21.0 | `storm/pixel_rl_ckpt/20260630T101119_update020000.pt` |

Restore with:

```bash
cd "$HOME/projects/CGSReg"
mkdir -p artifacts/exp_repro_frozen_wm_rl_only_20seed
rclone copy \
  box:projects/wm-evaluation/exp_repro_frozen_wm_rl_only_20seed/policy_ckpts/exp_repro_frozen_wm_rl_only_policy_ckpts_20260630/ \
  artifacts/exp_repro_frozen_wm_rl_only_20seed/ \
  --progress

cd artifacts/exp_repro_frozen_wm_rl_only_20seed
sha256sum -c exp_repro_frozen_wm_rl_only_policy_ckpts_20260630.tar.zst.sha256
tar --zstd -xf exp_repro_frozen_wm_rl_only_policy_ckpts_20260630.tar.zst
cd exp_repro_frozen_wm_rl_only_policy_ckpts_20260630
sha256sum -c SHA256SUMS
```

## Completion Checklist

- [x] Preflight checkpoint paths pass.
- [x] DIAMOND command includes `DIAMOND_PIXEL_RL_WM_CHECKPOINT`.
- [x] 200-update smoke test passes for DreamerV3, DIAMOND, TWISTER, Simulus,
  and STORM.
- [x] Full 20k RL-only run exits with code 0 for all five projects.
- [x] Each run has a final policy checkpoint under `pixel_rl_ckpt/latest.*`.
- [x] Policy checkpoint archive uploaded to Box.
- [ ] Fixed 20-seed real-env eval exits with code 0.
- [ ] Summary CSVs are copied or recorded for the paper.
- [ ] `nips_paper/scripts/make_frozen_wm_gap_figure.py` is updated with the
  final 20-seed values.
