# Paper Evidence Registry

This registry maps the paper's adopted numeric tables to raw local artifacts.
The paper-side compact tables live in:

`~/projects/nips_paper/data/paper_results/`

Large raw outputs remain in this repository under `artifacts/` and
`experiments/`.

## Main-Paper Figures And Tables

### Figure 2 / Frozen-WM Diagnostic

Paper table:

`nips_paper/data/paper_results/fig2_frozen_wm_gap_20episode.csv`

Raw sources:

- Dyna-style policy eval:
  `artifacts/dyna_style_policy_20seed_eval/{dreamer,diamond,twister,simulus,storm}/pong_real_policy_eval_summary.csv`
- Frozen reproduced-WM RL-only eval:
  `artifacts/exp_repro_frozen_wm_rl_only_20seed/eval/{dreamer,diamond,twister,simulus,storm}/pong_real_policy_eval_summary.csv`

The Dyna-style evals are 20 real-ALE evaluation episodes. The frozen-WM
RL-only evals use the same real-ALE return metric over 20 deterministic
evaluation episodes/seeds.

### Figure 4 / Unified Pixel-Space Zero-Shot RL

Paper table:

`nips_paper/data/paper_results/fig4_zero_shot_main_20seed.csv`

Adopted Table 6 values:

| World model | Exp-repro WM | Offline lambda=0 | Offline CGSReg | Adopted CGSReg setting | Source |
| --- | ---: | ---: | ---: | --- | --- |
| DreamerV3 | -20.95 +/- 0.22 | -21.00 +/- 0.00 | -11.90 +/- 5.66 | `size200m`, `lambda=0.1` | `artifacts/dreamer_lambda_search/zero_shot_20seed/size200m_w0p1/pong_real_policy_eval_summary.csv` |
| DIAMOND | -19.65 +/- 1.53 | -13.90 +/- 4.51 | -5.80 +/- 6.63 | `mask1`, `lambda=0.01` | `artifacts/zero_shot_mbrl_eval_20seed/diamond_mask1_ablation/pong_real_policy_eval_summary.csv` |
| TWISTER | -8.40 +/- 6.41 | -21.00 +/- 0.00 | -1.90 +/- 21.26 | `lambda=1.0` | `artifacts/zero_shot_mbrl_eval_20seed/twister/pong_real_policy_eval_summary.csv` |
| Simulus | -9.30 +/- 9.57 | -15.80 +/- 6.57 | -20.75 +/- 0.44 | DIAMOND replay `lambda=1.0` completed result | `artifacts/zero_shot_mbrl_eval_20seed/simulus_diamondreplay_w1_20260703_134938/pong_real_policy_eval_summary.csv` |
| STORM | -21.00 +/- 0.00 | -21.00 +/- 0.00 | -21.00 +/- 0.00 | `lambda=0.01` | `artifacts/zero_shot_mbrl_eval_20seed/storm/pong_real_policy_eval_summary.csv` |

DreamerV3 details for Table 6:

| Setting | W&B run id | W&B display name | 20-seed score | Artifact |
| --- | --- | --- | ---: | --- |
| Exp-repro WM | `qa32kfzt` | `paper-zsrl-dreamer-repro-size200m-h512` | -20.95 +/- 0.22 | `artifacts/zero_shot_mbrl_eval_20seed/dreamer/pong_real_policy_eval_summary.csv` |
| Offline `lambda=0` | `2xw3qk3y` | `paper-zsrl-dreamer-size200m-w0-h512` | -21.00 +/- 0.00 | `artifacts/zero_shot_mbrl_eval_20seed/dreamer_size200m/pong_real_policy_eval_summary.csv` |
| Offline `lambda=0.1` | `y00l3khw` | `paper-zsrl-dreamer-size200m-w0p1-h512` | -11.90 +/- 5.66 | `artifacts/dreamer_lambda_search/zero_shot_20seed/size200m_w0p1/pong_real_policy_eval_summary.csv` |

Raw sources:

- DreamerV3 exp-repro and offline `lambda=0`:
  `artifacts/zero_shot_mbrl_eval_20seed/dreamer/pong_real_policy_eval_summary.csv`
- DreamerV3 adopted CGSReg row:
  `artifacts/dreamer_lambda_search/zero_shot_20seed/size200m_w0p1/pong_real_policy_eval_summary.csv`
- DIAMOND exp-repro and offline `lambda=0`:
  `artifacts/zero_shot_mbrl_eval_20seed/diamond_main/pong_real_policy_eval_summary.csv`
- DIAMOND adopted CGSReg row:
  `artifacts/zero_shot_mbrl_eval_20seed/diamond_mask1_ablation/pong_real_policy_eval_summary.csv`
- TWISTER:
  `artifacts/zero_shot_mbrl_eval_20seed/twister/pong_real_policy_eval_summary.csv`
- Simulus:
  `artifacts/zero_shot_mbrl_eval_20seed/simulus/pong_real_policy_eval_summary.csv`
- Simulus DIAMOND-replay `lambda=0.01` and `lambda=0.1`:
  `artifacts/zero_shot_mbrl_eval_20seed/simulus_diamondreplay_w0p01_20260704_220023/pong_real_policy_eval_summary.csv`
  and
  `artifacts/zero_shot_mbrl_eval_20seed/simulus_diamondreplay_w0p1_20260704_220023/pong_real_policy_eval_summary.csv`.
  The synced 20-seed scores are `-5.10 +/- 5.68` and
  `-4.10 +/- 15.78`, respectively.
- STORM:
  `artifacts/zero_shot_mbrl_eval_20seed/storm/pong_real_policy_eval_summary.csv`

Important: the main paper uses DreamerV3 `size200m` so that the offline
CGSReg row matches the reproduced DreamerV3 reference model size. The stronger
`size400m` result remains in the regularization-strength appendix table but is
not the main-paper DreamerV3 row.

Important: the main paper uses DIAMOND `mask1` for CGSReg. The `mask1+mask3`
results are local artifacts but are not reported in the current paper.

## Appendix Ablations

### Offline Dataset and Regularization Strength

Paper table:

`nips_paper/data/paper_results/appendix_offline_dataset_lambda_ablation_20seed.csv`

Raw sources:

- DreamerV3 reported lambda search:
  `artifacts/dreamer_lambda_search/zero_shot_20seed/{size200m_w0p1,size200m_w1,size400m_w0p1,size400m_w1}/pong_real_policy_eval_summary.csv`
- DreamerV3 `size200m lambda=0.01` 20-seed fill-in:
  `artifacts/dreamer_lambda_search/zero_shot_20seed/size200m_w0p01/pong_real_policy_eval_summary.csv`.
  The synced 20-seed score is `-21.00 +/- 0.00`; the evaluated policy is
  `dreamerv3-runs/pong_pixel_rl_in_env/logdir/lambda_search_size200m_w0p01_ac20k_h512_rewq0p1/pixel_rl_ckpt/20260705T223838_update020000.npz`.
- DreamerV3 main `lambda=0`, main selected `size200m lambda=0.1`, and extra
  `size400m lambda=0.01`:
  `artifacts/zero_shot_mbrl_eval_20seed/dreamer/pong_real_policy_eval_summary.csv`
  `artifacts/dreamer_lambda_search/zero_shot_20seed/size200m_w0p1/pong_real_policy_eval_summary.csv`,
  and `artifacts/zero_shot_mbrl_eval_20seed/dreamer_size400m_ac20k_cgsreg_20260531/pong_real_policy_eval_summary.csv`
- DIAMOND:
  `artifacts/zero_shot_mbrl_eval_20seed/diamond_mask1_ablation/pong_real_policy_eval_summary.csv`
- TWISTER:
  `artifacts/zero_shot_mbrl_eval_20seed/twister/pong_real_policy_eval_summary.csv`
- Simulus:
  `artifacts/zero_shot_mbrl_eval_20seed/simulus/pong_real_policy_eval_summary.csv`
  `artifacts/zero_shot_mbrl_eval_20seed/simulus_diamondreplay_w0p01_20260704_220023/pong_real_policy_eval_summary.csv`
  `artifacts/zero_shot_mbrl_eval_20seed/simulus_diamondreplay_w0p1_20260704_220023/pong_real_policy_eval_summary.csv`
  `artifacts/zero_shot_mbrl_eval_20seed/simulus_diamondreplay_w1_20260703_134938/pong_real_policy_eval_summary.csv`
- STORM:
  `artifacts/zero_shot_mbrl_eval_20seed/storm/pong_real_policy_eval_summary.csv`
- STORM with OC-STORM exp-repro replay:
  `artifacts/dataset_ablation_ocstorm_dataset/fixed_real_eval/storm/storm_ocstorm_dataset_20seed_summary.csv`.
  This fixed real-ALE 20-seed evaluation covers
  `lambda in {0, 0.01, 0.1, 1.0}` and all four settings score
  `-21.00 +/- 0.00`. Per-setting episode CSVs are under
  `artifacts/dataset_ablation_ocstorm_dataset/fixed_real_eval/storm/{w0,w0p01,w0p1,w1}/`.
  The corresponding pixel-RL checkpoints are in
  `~/projects/oc-storm/runs/pong_pixel_rl_in_env/logdir/ocstorm_dataset_storm_w{0,0p01,0p1,1}_ac20k_h512_rewq0p5/pixel_rl_ckpt/latest.pt`.
  Local W&B summaries/logs confirm final online 5-episode evals of `-21`
  for `w0`, `w0p01`, and `w0p1`, and `-20.6` for `w1`; the adopted
  paper value is the formal fixed real-ALE 20-seed result.
- `experiments/dataset_ablation_twister_replay/README.md`

The appendix table merges the previous regularization-strength and
TWISTER-replay dataset ablation tables. Running entries correspond to launched
runs whose final synced 20-seed evaluations have not landed yet; remaining
pending entries have not been launched or integrated under the same protocol.

Latest TWISTER-replay 20-seed score extract:

`experiments/dataset_ablation_twister_replay/twister_dataset_ablation_20seed_eval_scores.md`

Paper-side compact extract:

`nips_paper/data/paper_results/appendix_twister_replay_dataset_ablation_20seed.csv`

2026-07-05 paper sync:

| Project | lambda=0 | lambda=0.01 | lambda=0.1 | lambda=1.0 | Paper status |
| --- | ---: | ---: | ---: | ---: | --- |
| DreamerV3 TWISTER replay | -20.80 +/- 0.41 | -20.00 +/- 1.56 | -19.85 +/- 1.04 | -21.00 +/- 0.00 | latest completed |
| DIAMOND TWISTER replay | -20.70 +/- 0.92 | -20.65 +/- 0.49 | -21.00 +/- 0.00 | -20.55 +/- 0.51 | latest completed |
| TWISTER TWISTER replay | -17.20 +/- 4.79 | -13.20 +/- 9.81 | -9.20 +/- 11.62 | 9.45 +/- 15.82 | latest completed |
| Simulus TWISTER replay | -17.60 +/- 1.47 | -20.00 +/- 1.30 | -18.90 +/- 1.02 | -15.70 +/- 3.26 | latest completed |
| STORM TWISTER replay | pending | pending | pending | pending | pending |

Important caveat: the TWISTER-replay rows above come from the old converted
TWISTER exp-repro replay datasets whose episode boundaries were not repaired.
`docs/dataset_registry.md` invalidates those datasets for new experiments
because some stored `.pt` episodes contain a visual environment reset inside the
same episode. The paper currently reports these completed 20-seed RL results
provisionally. When the corrected TWISTER replay conversion is ready, replace
the whole TWISTER-replay block with rerun results.
