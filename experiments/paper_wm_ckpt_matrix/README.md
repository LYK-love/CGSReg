# Paper WM Checkpoint Matrix

Target comparison for the paper experiment:

- Per WM project: `w=0`, recommended spatial weight, and exp-repro reference.
- Protocol: 5 closed-loop Pong rollouts, horizon 512, seeds 0..4.
- Policy: matched real-env pixel RL policy where available.
- Zero-shot RL policy evaluation is a separate real-ALE protocol: 20 full
  deterministic episodes with reset seeds 0..19. Do not mix these numbers with
  the 5 closed-loop rollout diagnostics above.

Canonical files:

- `target_rollouts.json`: machine-readable target matrix.
- `results/paper_wm_ckpt_matrix/closed_loop_failure_quant.csv`: compact quantitative table for the one non-RL experiment.
- `results/paper_wm_ckpt_matrix/closed_loop_failure_quant.md`: markdown version of the same table.
- `results/paper_wm_ckpt_matrix/rollout_frame_panel.png`: qualitative rollout panel for the main improving models.
- `results/paper_wm_ckpt_matrix/target_rollout_summary.csv`: WM-imagined return sanity summary. Do not use as a main result.
- `results/paper_wm_ckpt_matrix/target_rollout_failure_summary.csv`: full target summary joined with current heuristic failure metrics.
- `eval_outputs/paper_wm_ckpt_matrix_failure_heuristic/`: raw per-rollout event outputs.
- `artifacts/zero_shot_mbrl_eval_20seed/`: refreshed 20-seed real-ALE
  zero-shot RL outputs for policy checkpoints that were not already evaluated
  under this protocol.

Current status:

- Available: Dreamer, DIAMOND, Simulus, Twister, STORM.
- Main qualitative panel: Dreamer, DIAMOND, Twister.

To reproduce or refill STORM rollout artifacts:

```bash
bash experiments/paper_wm_ckpt_matrix/commands/storm_target_rollouts.commands.txt
```

To run the 20-seed real-ALE zero-shot RL evals for main-paper policy
checkpoints that still need the updated protocol:

```bash
cd "$HOME/projects/CGSReg"
tiny-exp-scheduler run \
  experiments/paper_wm_ckpt_matrix/commands/main_dreamer_zero_shot_20seed_eval.commands.txt \
  --cuda-devices auto \
  --cpu-threads 2 \
  --logs-dir experiments/paper_wm_ckpt_matrix/logs/main_dreamer_zero_shot_20seed_eval \
  --verbose --keep-job-tabs

tiny-exp-scheduler run \
  experiments/paper_wm_ckpt_matrix/commands/main_torch_zero_shot_20seed_eval.commands.txt \
  --cuda-devices auto \
  --cpu-threads 2 \
  --logs-dir experiments/paper_wm_ckpt_matrix/logs/main_torch_zero_shot_20seed_eval \
  --verbose --keep-job-tabs
```

Skip a project if its target summary already reports `episodes=20` for every
policy row. DIAMOND already has 20-seed local summaries under
`$HOME/projects/diamond/debug_outputs/pong_pixel_rl_policy_real_env_eval/`.

To refresh the compact paper artifacts after rerunning failure detection:

```bash
python scripts/eval/summarize_target_rollouts.py \
  --failure-aggregate eval_outputs/paper_wm_ckpt_matrix_failure_heuristic/aggregate_by_model.csv

python scripts/eval/make_paper_rollout_artifacts.py
```

The current failure table uses heuristic-only ball tracking. It is useful for
the paper's compact quantitative result if described as a conservative
closed-loop tracking failure metric. Use SAM2/hybrid tracking for stricter
collision-failure claims.
