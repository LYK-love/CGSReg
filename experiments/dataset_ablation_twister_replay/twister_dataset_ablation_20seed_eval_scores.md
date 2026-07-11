# TWISTER Dataset Ablation 20-Seed RL Scores

This document records the fixed-real 20-seed Pong evaluation scores for the
TWISTER-dataset ablation runs. The source CSV/JSON artifacts are local
generated outputs under `artifacts/dataset_ablation_twister_replay/fixed_real_eval/`
and `artifacts/zero_shot_mbrl_eval_20seed/`; this Markdown file is tracked so
the summary scores are available from GitHub without uploading large artifacts.

Paper sync status: these summary values were copied into
`nips_paper/data/paper_results/appendix_twister_replay_dataset_ablation_20seed.csv`
and `nips_paper/data/paper_results/appendix_offline_dataset_lambda_ablation_20seed.csv`
on 2026-07-05 and updated with the Simulus `lambda=1.0` row on 2026-07-06.

Dataset status: all scores in this file come from the old TWISTER replay
conversion whose episode boundaries were not repaired. `docs/dataset_registry.md`
marks that conversion invalid for new experiments because some stored `.pt`
episodes contain visual resets inside a single episode. We currently report
these completed 20-seed RL results provisionally in the paper appendix. When the
corrected TWISTER replay conversion is available, replace this complete result
block instead of mixing old and corrected rows.

## Summary

| family | policy | episodes | score_mean | score_std_sample | score_min | score_max | length_mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w0` | 20 | -20.8 | 0.410391 | -21 | -20 | 931.7 |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w0p01` | 20 | -20 | 1.55597 | -21 | -16 | 992.7 |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w0p1` | 20 | -19.85 | 1.03999 | -21 | -18 | 1542.3 |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w1` | 20 | -21 | 0 | -21 | -21 | 759.9 |
| Twister dilated k3 | `twister_twexp_dilated_k3_w0` | 20 | -17.2 | 4.78594 | -21 | -11 | 1566.15 |
| Twister dilated k3 | `twister_twexp_dilated_k3_w0p01` | 20 | -13.2 | 9.80655 | -21 | -1 | 2408.35 |
| Twister dilated k3 | `twister_twexp_dilated_k3_w0p1` | 20 | -9.2 | 11.6239 | -21 | 5 | 2293.95 |
| Twister dilated k3 | `twister_twexp_dilated_k3_w1` | 20 | 9.45 | 15.8163 | -21 | 20 | 2130.5 |
| Simulus dilated k3 | `simulus_twisterdataset_w0` | 20 | -17.6 | 1.46539 | -20 | -16 | 1668.1 |
| Simulus dilated k3 | `simulus_twisterdataset_w0p01` | 20 | -20 | 1.29777 | -21 | -18 | 1041.8 |
| Simulus dilated k3 | `simulus_twisterdataset_w0p1` | 20 | -18.9 | 1.02084 | -21 | -17 | 1222.5 |
| Simulus dilated k3 | `simulus_twisterdataset_w1` | 20 | -15.7 | 3.26222 | -20 | -12 | 1960.6 |

Simulus source summaries:

- `artifacts/zero_shot_mbrl_eval_20seed/simulus_twisterdataset_w0_20260703_234144/pong_real_policy_eval_summary.csv`
- `artifacts/zero_shot_mbrl_eval_20seed/simulus_twisterdataset_w0p01_20260703_234156/pong_real_policy_eval_summary.csv`
- `artifacts/zero_shot_mbrl_eval_20seed/simulus_twisterdataset_w0p1_20260703_234157/pong_real_policy_eval_summary.csv`
- `artifacts/zero_shot_mbrl_eval_20seed/simulus_twisterdataset_w1_20260704_164631/pong_real_policy_eval_summary.csv`

## Per-Rollout Scores

| family | policy | rollout | seed | score | agent_score | opponent_score | length | terminal |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w0` | 1 | 1 | -21 | 0 | 21 | 758 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w0` | 2 | 2 | -21 | 0 | 21 | 761 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w0` | 3 | 3 | -21 | 0 | 21 | 759 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w0` | 4 | 4 | -21 | 0 | 21 | 759 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w0` | 5 | 5 | -21 | 0 | 21 | 759 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w0` | 6 | 6 | -21 | 0 | 21 | 1225 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w0` | 7 | 7 | -21 | 0 | 21 | 762 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w0` | 8 | 8 | -21 | 0 | 21 | 758 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w0` | 9 | 9 | -21 | 0 | 21 | 787 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w0` | 10 | 10 | -21 | 0 | 21 | 886 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w0` | 11 | 11 | -21 | 0 | 21 | 902 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w0` | 12 | 12 | -20 | 1 | 21 | 1714 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w0` | 13 | 13 | -21 | 0 | 21 | 760 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w0` | 14 | 14 | -20 | 1 | 21 | 1034 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w0` | 15 | 15 | -21 | 0 | 21 | 903 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w0` | 16 | 16 | -20 | 1 | 21 | 1171 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w0` | 17 | 17 | -21 | 0 | 21 | 1401 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w0` | 18 | 18 | -21 | 0 | 21 | 759 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w0` | 19 | 19 | -20 | 1 | 21 | 1016 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w0` | 20 | 20 | -21 | 0 | 21 | 760 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w0p01` | 1 | 1 | -18 | 3 | 21 | 1303 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w0p01` | 2 | 2 | -21 | 0 | 21 | 761 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w0p01` | 3 | 3 | -18 | 3 | 21 | 1279 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w0p01` | 4 | 4 | -21 | 0 | 21 | 837 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w0p01` | 5 | 5 | -20 | 1 | 21 | 993 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w0p01` | 6 | 6 | -21 | 0 | 21 | 760 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w0p01` | 7 | 7 | -21 | 0 | 21 | 855 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w0p01` | 8 | 8 | -18 | 3 | 21 | 1132 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w0p01` | 9 | 9 | -19 | 2 | 21 | 1512 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w0p01` | 10 | 10 | -21 | 0 | 21 | 762 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w0p01` | 11 | 11 | -21 | 0 | 21 | 759 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w0p01` | 12 | 12 | -21 | 0 | 21 | 764 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w0p01` | 13 | 13 | -18 | 3 | 21 | 1676 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w0p01` | 14 | 14 | -21 | 0 | 21 | 758 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w0p01` | 15 | 15 | -21 | 0 | 21 | 764 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w0p01` | 16 | 16 | -21 | 0 | 21 | 758 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w0p01` | 17 | 17 | -21 | 0 | 21 | 761 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w0p01` | 18 | 18 | -21 | 0 | 21 | 759 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w0p01` | 19 | 19 | -21 | 0 | 21 | 758 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w0p01` | 20 | 20 | -16 | 5 | 21 | 1903 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w0p1` | 1 | 1 | -18 | 3 | 21 | 2365 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w0p1` | 2 | 2 | -18 | 3 | 21 | 2217 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w0p1` | 3 | 3 | -21 | 0 | 21 | 759 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w0p1` | 4 | 4 | -21 | 0 | 21 | 759 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w0p1` | 5 | 5 | -18 | 3 | 21 | 2168 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w0p1` | 6 | 6 | -20 | 1 | 21 | 2252 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w0p1` | 7 | 7 | -21 | 0 | 21 | 762 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w0p1` | 8 | 8 | -21 | 0 | 21 | 758 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w0p1` | 9 | 9 | -20 | 1 | 21 | 2137 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w0p1` | 10 | 10 | -21 | 0 | 21 | 762 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w0p1` | 11 | 11 | -20 | 1 | 21 | 1970 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w0p1` | 12 | 12 | -20 | 1 | 21 | 2010 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w0p1` | 13 | 13 | -20 | 1 | 21 | 1847 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w0p1` | 14 | 14 | -20 | 1 | 21 | 1037 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w0p1` | 15 | 15 | -20 | 1 | 21 | 1167 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w0p1` | 16 | 16 | -20 | 1 | 21 | 1039 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w0p1` | 17 | 17 | -20 | 1 | 21 | 1647 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w0p1` | 18 | 18 | -20 | 1 | 21 | 2243 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w0p1` | 19 | 19 | -20 | 1 | 21 | 1157 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w0p1` | 20 | 20 | -18 | 3 | 21 | 1790 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w1` | 1 | 1 | -21 | 0 | 21 | 758 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w1` | 2 | 2 | -21 | 0 | 21 | 761 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w1` | 3 | 3 | -21 | 0 | 21 | 759 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w1` | 4 | 4 | -21 | 0 | 21 | 759 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w1` | 5 | 5 | -21 | 0 | 21 | 759 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w1` | 6 | 6 | -21 | 0 | 21 | 760 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w1` | 7 | 7 | -21 | 0 | 21 | 762 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w1` | 8 | 8 | -21 | 0 | 21 | 758 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w1` | 9 | 9 | -21 | 0 | 21 | 759 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w1` | 10 | 10 | -21 | 0 | 21 | 762 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w1` | 11 | 11 | -21 | 0 | 21 | 759 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w1` | 12 | 12 | -21 | 0 | 21 | 764 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w1` | 13 | 13 | -21 | 0 | 21 | 760 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w1` | 14 | 14 | -21 | 0 | 21 | 758 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w1` | 15 | 15 | -21 | 0 | 21 | 764 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w1` | 16 | 16 | -21 | 0 | 21 | 758 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w1` | 17 | 17 | -21 | 0 | 21 | 761 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w1` | 18 | 18 | -21 | 0 | 21 | 759 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w1` | 19 | 19 | -21 | 0 | 21 | 758 | True |
| Dreamer dilated k3 | `dreamer_twexp_dilated_k3_w1` | 20 | 20 | -21 | 0 | 21 | 760 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w0` | 0 | 0 | -21 | 0 | 21 | 1348 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w0` | 1 | 1 | -21 | 0 | 21 | 1335 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w0` | 2 | 2 | -11 | 10 | 21 | 2020 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w0` | 3 | 3 | -21 | 0 | 21 | 1349 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w0` | 4 | 4 | -21 | 0 | 21 | 1335 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w0` | 5 | 5 | -12 | 9 | 21 | 1793 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w0` | 6 | 6 | -21 | 0 | 21 | 1342 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w0` | 7 | 7 | -11 | 10 | 21 | 2017 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w0` | 8 | 8 | -21 | 0 | 21 | 1336 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w0` | 9 | 9 | -21 | 0 | 21 | 1337 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w0` | 10 | 10 | -21 | 0 | 21 | 1335 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w0` | 11 | 11 | -12 | 9 | 21 | 1795 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w0` | 12 | 12 | -21 | 0 | 21 | 1346 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w0` | 13 | 13 | -11 | 10 | 21 | 2016 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w0` | 14 | 14 | -12 | 9 | 21 | 1793 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w0` | 15 | 15 | -11 | 10 | 21 | 2017 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w0` | 16 | 16 | -21 | 0 | 21 | 1336 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w0` | 17 | 17 | -12 | 9 | 21 | 1795 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w0` | 18 | 18 | -21 | 0 | 21 | 1340 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w0` | 19 | 19 | -21 | 0 | 21 | 1338 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w0p01` | 0 | 0 | -21 | 0 | 21 | 1583 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w0p01` | 1 | 1 | -21 | 0 | 21 | 1576 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w0p01` | 2 | 2 | -2 | 19 | 21 | 3589 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w0p01` | 3 | 3 | -21 | 0 | 21 | 1547 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w0p01` | 4 | 4 | -21 | 0 | 21 | 1576 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w0p01` | 5 | 5 | -1 | 20 | 21 | 3735 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w0p01` | 6 | 6 | -21 | 0 | 21 | 1583 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w0p01` | 7 | 7 | -2 | 19 | 21 | 3586 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w0p01` | 8 | 8 | -21 | 0 | 21 | 1577 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w0p01` | 9 | 9 | -21 | 0 | 21 | 1578 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w0p01` | 10 | 10 | -21 | 0 | 21 | 1576 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w0p01` | 11 | 11 | -1 | 20 | 21 | 3737 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w0p01` | 12 | 12 | -21 | 0 | 21 | 1544 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w0p01` | 13 | 13 | -2 | 19 | 21 | 3585 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w0p01` | 14 | 14 | -1 | 20 | 21 | 3735 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w0p01` | 15 | 15 | -2 | 19 | 21 | 3586 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w0p01` | 16 | 16 | -21 | 0 | 21 | 1577 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w0p01` | 17 | 17 | -1 | 20 | 21 | 3737 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w0p01` | 18 | 18 | -21 | 0 | 21 | 1581 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w0p01` | 19 | 19 | -21 | 0 | 21 | 1579 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w0p1` | 0 | 0 | 5 | 21 | 16 | 3154 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w0p1` | 1 | 1 | -19 | 2 | 21 | 1581 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w0p1` | 2 | 2 | 5 | 21 | 16 | 3160 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w0p1` | 3 | 3 | -21 | 0 | 21 | 1531 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w0p1` | 4 | 4 | -19 | 2 | 21 | 1581 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w0p1` | 5 | 5 | 1 | 21 | 20 | 3196 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w0p1` | 6 | 6 | -19 | 2 | 21 | 1588 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w0p1` | 7 | 7 | 5 | 21 | 16 | 3157 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w0p1` | 8 | 8 | -19 | 2 | 21 | 1582 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w0p1` | 9 | 9 | -19 | 2 | 21 | 1583 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w0p1` | 10 | 10 | -19 | 2 | 21 | 1581 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w0p1` | 11 | 11 | 1 | 21 | 20 | 3198 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w0p1` | 12 | 12 | -21 | 0 | 21 | 1528 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w0p1` | 13 | 13 | 5 | 21 | 16 | 3156 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w0p1` | 14 | 14 | 1 | 21 | 20 | 3196 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w0p1` | 15 | 15 | 5 | 21 | 16 | 3157 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w0p1` | 16 | 16 | -19 | 2 | 21 | 1582 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w0p1` | 17 | 17 | 1 | 21 | 20 | 3198 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w0p1` | 18 | 18 | -19 | 2 | 21 | 1586 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w0p1` | 19 | 19 | -19 | 2 | 21 | 1584 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w1` | 0 | 0 | -21 | 0 | 21 | 1515 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w1` | 1 | 1 | 20 | 21 | 1 | 2005 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w1` | 2 | 2 | -2 | 19 | 21 | 3441 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w1` | 3 | 3 | -21 | 0 | 21 | 1516 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w1` | 4 | 4 | 20 | 21 | 1 | 2005 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w1` | 5 | 5 | 20 | 21 | 1 | 1561 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w1` | 6 | 6 | 20 | 21 | 1 | 2012 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w1` | 7 | 7 | -2 | 19 | 21 | 3438 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w1` | 8 | 8 | 20 | 21 | 1 | 2006 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w1` | 9 | 9 | 20 | 21 | 1 | 2007 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w1` | 10 | 10 | 20 | 21 | 1 | 2005 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w1` | 11 | 11 | 20 | 21 | 1 | 1563 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w1` | 12 | 12 | -21 | 0 | 21 | 1513 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w1` | 13 | 13 | -2 | 19 | 21 | 3437 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w1` | 14 | 14 | 20 | 21 | 1 | 1561 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w1` | 15 | 15 | -2 | 19 | 21 | 3438 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w1` | 16 | 16 | 20 | 21 | 1 | 2006 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w1` | 17 | 17 | 20 | 21 | 1 | 1563 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w1` | 18 | 18 | 20 | 21 | 1 | 2010 | True |
| Twister dilated k3 | `twister_twexp_dilated_k3_w1` | 19 | 19 | 20 | 21 | 1 | 2008 | True |
