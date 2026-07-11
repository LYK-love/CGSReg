# Zero-Shot RL Table Completion Status

This note tracks the five-project three-way zero-shot RL table:

```text
exp-repro baseline / offline w=0 / offline recommended CGSReg weight
```

The canonical evaluation standard is defined in
`docs/zero_shot_mbrl.md`: 5 ALE Pong episodes, reset seeds
`0,1,2,3,4`, and deterministic policy actions.

## Status

| Project | Status | Evidence |
| --- | --- | --- |
| Dreamer | complete | `docs/zero_shot_mbrl.md` records exp-repro, `w=0`, and recommended `w=0.01` rows with local fixed-seed eval. |
| DIAMOND | complete for table, but exp-repro policy archive is not restored locally | HF official and offline fixed-SR rows have local fixed-seed eval. Exp-repro zero-shot RL is recorded from W&B run `paper-zsrl-diamond-exp-repro-h512` (`v4y77brt`) with `eval_real/episodes=5`. |
| Simulus | complete for table | Exp-repro rows and offline `w=0`/`w=0.01` rows are recorded. Offline policy archive: `box:zero-shot-rl/simulus/simulus_offline_wm_policy_ckpts_latest.tar.gz`. |
| TWISTER | complete for table | Exp-repro, DIAMOND-static offline sweep, and TWISTER-replay offline evidence are recorded. |
| STORM | complete with W&B evidence | `paper-zsrl-storm-offline-w0-h512-rewq0p1` and `paper-zsrl-storm-offline-w0p01-h512-rewq0p1` finished with W&B `eval_real/episodes=5`; policy checkpoints were archived locally under `oc-storm/archives/zero_shot_rl/`. |

## Remaining Caveats

- DIAMOND exp-repro zero-shot RL is in W&B, but the final zero-shot policy
  checkpoint is not currently restored locally in `CGSReg`.
- STORM table rows use W&B final `eval_real/score_mean`; a separate local
  fixed-seed evaluator output is not yet present in this repo.
- Larger dataset-ablation sweeps are tracked separately and are not required to
  complete the core three-way table.

## Reference Commands

The old Simulus/STORM command notes are kept in:

```text
docs/simulus_zero_shot_rl_completion.md
```

Use those only if a policy archive needs to be restored or a row needs to be
reproduced from scratch.
