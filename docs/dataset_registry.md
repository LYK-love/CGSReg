# Dataset Registry

This document records offline datasets that are safe to use for paper-facing
experiments and datasets that have been invalidated.

## Model Artifacts And Checkpoints

### DreamerV3 Pong Atari100K Size400M Repro, Cloud Single A6000, 2026-07-08

- Host:
  `lyk@0136-ict-prxmx50038` (`cloud`)
- Remote run root:
  `~/projects/dreamerv3-runs/pong_atari100k_reproduction/`
- Box path:
  `box:projects/wm-evaluation/dreamerv3_size400m_pong_repro_cloud_20260708/`
- Box folder id:
  `398211423680`
- Archive:
  `dreamerv3_size400m_pong_repro_cloud_20260708.tar.zst`
- Archive size:
  `5367698902` bytes
- SHA256:
  `388cbf97b0f0e9a7fc38a846724a72a98ac767f61b18be5cfcff5a5f3b965867`
- SHA256 sidecar:
  `dreamerv3_size400m_pong_repro_cloud_20260708.tar.zst.sha256`
- Lightweight result summary:
  `results/dreamerv3_size400m_pong_repro_cloud_20260708.json`

Archive contents:

- `checkpoint/20260707T233830F168241/`
  - `agent.pkl`
  - `step.pkl`
  - `replay_train.pkl`
  - `replay_eval.pkl`
  - `done`
- `logs/config.yaml`
- `logs/metrics.jsonl`
- `logs/scores.jsonl`
- `logs/size400m_bs4_nofull_resume_20260708_072638.log`
- `MANIFEST.md`
- `file_sizes.txt`

Run summary:

- Task: DreamerV3 `size400m` Atari100K Pong experiment reproduction.
- Hardware: single NVIDIA RTX A6000 on `cloud`.
- Final successful command used `batch_size=4`, JSONL logging only,
  `--agent.report False`, `--run.report_every 999999`,
  `--run.eval_envs 1`, `--run.save_full_checkpoint False`,
  `DREAMERV3_SAVE_EVERY=999999`, and
  `XLA_PYTHON_CLIENT_MEM_FRACTION=0.82`.
- Final log:
  `~/projects/dreamerv3-runs/pong_atari100k_reproduction/scheduler_logs/size400m_bs4_nofull_resume_20260708_072638.log`
- Final status:
  `END status=0` at `2026-07-08T12:03:15+00:00`.
- Final metric step:
  `399800`.
- Final reported training episode score:
  `-12.0` at step `398260`.
- Best observed episode score in the successful run:
  `-6.0` at step `347224`.
- Final `train/dyna/freeze_progress`:
  `0.998155`.
- Final `replay/replay_ratio`:
  `260`.

Checkpoint caveat:

- The final no-full-checkpoint resume was required because full checkpoint
  writes were killed by host RAM pressure on the single-A6000 cloud machine.
- Therefore the archived checkpoint is the last complete checkpoint used to
  resume the final successful run:
  `ckpt/20260707T233830F168241/`.
- No new full checkpoint was written at the final `399800` metric step.
- This artifact records the completed repro run and preserves the latest
  complete restorable checkpoint available from that run lineage.
- No downstream zero-shot RL or 20-seed policy evaluation has been run from
  this artifact yet.

## Validated Datasets

### OC-STORM Pong Exp-Repro Replay, SAM2, Dilated k3

- Local path:
  `/data/luyukuan/projects/diamond-assets/datasets/oc_storm_pong_exp_repro_replay_sam2_dilated_k3`
- Box path:
  `box:zero-shot-rl/storm/oc_storm_pong_exp_repro_replay_sam2_dilated_k3_diamond_20260705/`
- Box folder:
  `zero-shot-rl/storm/oc_storm_pong_exp_repro_replay_sam2_dilated_k3_diamond_20260705`
- Archive:
  `oc_storm_pong_exp_repro_replay_sam2_dilated_k3_diamond_20260705.tar.zst`
- SHA256:
  `782b03d2abfef1f4ceb4efd49ca78b7e630d5c5ee58339fb233f9e9e55a151e5`
- Manifest:
  `oc_storm_pong_exp_repro_replay_sam2_dilated_k3_diamond_20260705.manifest.json`

Validation summary:

- Format: Diamond-style episode `.pt` files.
- Required fields present: `obs`, `act`, `rew`, `end`, `trunc`, `mask1`, `mask2`, `mask3`.
- Episodes: 38.
- Steps: 100000.
- Complete terminal episodes: 37.
- Final incomplete tail: 1 budget-truncated segment with no terminal flag.
- Return range: `[-21, 18]`.
- Terminal checks: 0 bad records.
- Mask checks: binary masks, consistent first dimension across all episode fields.
- Spot visual checks: lowest-return, highest-return, and final tail episodes were inspected
  with contact sheets; score display and terminal/reward alignment looked consistent.

Generated local summaries:

- `/scorpio/home/luyukuan/projects/CGSReg/artifacts/debug/dataset_summary/oc_storm_exp_repro_dilated_k3_diamond/summary.json`
- `/scorpio/home/luyukuan/projects/CGSReg/artifacts/debug/dataset_episode_inspector/oc_storm_boundary_probe/`

Associated paper experiment artifacts:

- Experiment:
  `experiments/dataset_ablation_oc_storm_replay/`
- Result document:
  `experiments/dataset_ablation_oc_storm_replay/oc_storm_replay_dataset_ablation_20seed_scores.md`
- Aggregate summary CSV:
  `experiments/dataset_ablation_oc_storm_replay/oc_storm_replay_dataset_ablation_20seed_summary.csv`
- Per-rollout score CSV:
  `experiments/dataset_ablation_oc_storm_replay/oc_storm_replay_dataset_ablation_20seed_scores.csv`
- Checkpoint Box path:
  `box:zero-shot-rl/storm/oc_storm_replay_dataset_ablation_ckpts_20260707/`
- Checkpoint Box folder id:
  `397526027836`
- Checkpoint package contents:
  final offline WM checkpoints and 20k zero-shot RL policy checkpoints for
  DreamerV3, TWISTER, and DIAMOND, plus `MANIFEST.md`, `SHA256SUMS.txt`, and
  copied result metadata.

## Invalidated Datasets

### TWISTER Exp-Repro Replay, Old SAM2/Dilated Conversions

Do not use the old TWISTER exp-repro converted datasets for new paper
experiments.
The conversion incorrectly used TWISTER replay `model_step` values as globally
unique step identifiers and a later repair pass re-split episodes by Pong reward
count. This produced files where the video contains a visual environment reset
inside a single stored `.pt` episode.

Deleted local invalid paths:

- `/data/luyukuan/projects/diamond-assets/datasets/twister_pong_exp_repro_replay_sam2_ball_dilated_k3_diamond_standard`
- `/data/luyukuan/projects/shared_replay/twister_pong_exp_repro_replay_sam2_diamond_standard_for_wm`
- `/data/luyukuan/projects/diamond-assets/datasets/twister_pong_exp_repro_replay_sam2_dilated_k3_original_boundaries`

Box invalidation:

- Folder renamed to:
  `INVALID_DO_NOT_USE_twister_pong_exp_repro_replay_sam2_for_wm_dreamer_dilated_k3_20260702`
- Box folder URL:
  `https://ucdavis.app.box.com/folder/396115475753`

The TWISTER dataset should be regenerated from the original TWISTER replay with
a corrected conversion path before any new dataset-ablation experiments use it.

Paper reporting status:

- The existing TWISTER-dataset ablation numbers are explicitly marked as
  coming from these old, episode-boundary-unfixed conversions.
- We currently report those completed 20-seed RL results as provisional
  appendix evidence, because they are the latest completed runs.
- When a corrected TWISTER replay conversion is available, replace the whole
  TWISTER-dataset ablation block rather than mixing old and corrected rows.
