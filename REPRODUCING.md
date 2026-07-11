# Reproducing CGSReg Experiments

This file gives the top-level reproduction map. Detailed commands are delegated
to `experiments/` and `scripts/`.

## 1. Bootstrap Repositories

```bash
bash scripts/bootstrap_repos.sh
bash scripts/status_repos.sh
```

## 2. Restore Large Artifacts

Large files are not checked into git. Restore datasets, checkpoints, and policy
checkpoints using the manifests in:

```text
docs/dataset_registry.md
docs/paper_data_catalog.md
docs/paper_evidence_registry.md
experiments/*/README.md
```

## 3. Fixed Real-ALE Policy Evaluation

Paper-facing zero-shot RL scores use:

```text
episodes = 20
reset seeds = 0..19
policy mode = deterministic / eval
environment = real ALE Pong
metric = total episode return
```

Dreamer/JAX policies:

```bash
python scripts/eval/evaluate_pong_real_policies.py --help
```

Torch policies:

```bash
python scripts/eval/evaluate_torch_pong_real_policies.py --help
```

## 4. Offline Dataset Ablations

The canonical protocol is:

```text
docs/offline_dataset_ablation_protocol.md
```

Experiment-specific command files are under:

```text
experiments/dataset_ablation_oc_storm_replay/commands/
experiments/dataset_ablation_twister_replay/commands/
```

## 5. SAM2 Rollout Analysis

Cross-model generated-video and physical-consistency analysis lives in:

```text
scripts/eval/pong_sam2_ball_experiment.py
scripts/eval/pong_external_sam2_ball_experiment.py
scripts/eval/pong_ball_physical_consistency.py
scripts/eval/masks_to_tracks.py
```

## 6. Backend-Specific Training

Backend training code remains in its pinned fork:

```text
third_party/dreamerv3-reborn
third_party/diamond
third_party/twister
third_party/simulus
third_party/oc-storm
```

Use the command files in `experiments/` as the paper-facing launch source.
