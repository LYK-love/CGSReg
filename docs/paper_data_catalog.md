# Paper Data Catalog

This catalog gives the canonical locations for paper-facing numeric results.
It separates adopted paper tables, raw evaluation artifacts, and experiment
family notes so future updates do not require searching through unrelated logs.

## Main-Paper RL Results

| Result family | Adopted compact table | Raw source registry |
| --- | --- | --- |
| Frozen-WM diagnostic / Figure 2 | `~/projects/nips_paper/data/paper_results/fig2_frozen_wm_gap_20episode.csv` | `docs/paper_evidence_registry.md` |
| Pixel-space zero-shot RL / Figure 4 and Table 6 | `~/projects/nips_paper/data/paper_results/fig4_zero_shot_main_20seed.csv` | `docs/paper_evidence_registry.md` |

All main-paper RL scores are reported as real-environment Atari Pong returns
over 20 deterministic evaluation seeds unless a row is explicitly marked
pending or provisional.

## Appendix Dataset And Lambda Ablations

| Result family | Adopted compact table | Detailed local record |
| --- | --- | --- |
| Combined offline dataset and `lambda_CGSReg` ablation | `~/projects/nips_paper/data/paper_results/appendix_offline_dataset_lambda_ablation_20seed.csv` | `docs/paper_evidence_registry.md` |
| TWISTER replay dataset ablation extract | `~/projects/nips_paper/data/paper_results/appendix_twister_replay_dataset_ablation_20seed.csv` | `experiments/dataset_ablation_twister_replay/twister_dataset_ablation_20seed_eval_scores.md` |

TWISTER replay status: these rows currently come from the old TWISTER replay
conversion whose episode boundaries were not repaired. The paper reports the
completed 20-seed RL results provisionally. The dataset is invalid for new runs;
see `docs/dataset_registry.md`. When the corrected conversion is available,
replace the entire TWISTER-replay ablation block.

## Dataset Records

| Dataset family | Status document | Notes |
| --- | --- | --- |
| Validated DIAMOND-format datasets | `docs/dataset_registry.md` | Canonical local paths, Box paths, checksums, and validation summaries. |
| DreamerV3 size400m Pong repro checkpoint artifacts | `docs/dataset_registry.md` and `results/dreamerv3_size400m_pong_repro_cloud_20260708.json` | Cloud single-A6000 repro log, Box archive pointer, checksum, and checkpoint caveat. No downstream zero-shot RL or 20-seed policy evaluation has been run from this artifact yet. |
| Reusable dataset-ablation protocol | `docs/offline_dataset_ablation_protocol.md` | Defines the chain schedule: offline WM -> zero-shot RL -> 20-seed eval per `(project, dataset, lambda)`. |
| TWISTER replay ablation archive | `experiments/dataset_ablation_twister_replay/README.md` | Historical commands, package locations, and provisional result status. |

## Raw Artifact Conventions

Use these roots for raw outputs rather than adding large files to Git:

```text
~/projects/CGSReg/artifacts/
~/projects/CGSReg/results/
~/projects/<project>/runs/
~/projects/<project>/outputs/
```

Small paper-facing CSVs live in:

```text
~/projects/nips_paper/data/paper_results/
```

Before changing a paper number, update the compact CSV and the corresponding
source note in `docs/paper_evidence_registry.md`.
