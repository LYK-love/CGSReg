# CGSReg

This is the paper codebase for CGSReg experiments.

The repository is intentionally organized as a single public entry point. It
does not ask readers to navigate multiple unrelated links. Evaluation,
experiments, dataset conversion, SAM2 analysis, and paper registries live in
this main repository. The five world-model backends are published as clean
public forks and pinned as git submodules under `third_party/`.

## What This Repo Contains

```text
CGSReg/
  README.md
  REPRODUCING.md
  docs/
    codebase_layout.md
    release_checklist.md
  scripts/
    bootstrap_repos.sh
    status_repos.sh
    dataset/
    eval/
    figures/
  third_party/
    backend_repos.lock
  artifacts/
    README.md
  experiments/
    ...
```

After bootstrapping, `third_party/` contains:

```text
third_party/dreamerv3-reborn   # DreamerV3 backend
third_party/diamond            # DIAMOND backend
third_party/twister            # TWISTER backend
third_party/simulus            # Simulus backend
third_party/oc-storm           # STORM/OC-STORM backend
```

## Quick Start

Clone this repository and fetch the pinned backend repos:

```bash
git clone https://github.com/LYK-love/CGSReg.git
cd CGSReg
bash scripts/bootstrap_repos.sh
bash scripts/status_repos.sh
```

The bootstrap script checks out the exact commits listed in
`third_party/backend_repos.lock`.

## Main Workflows

The codebase supports three levels of reproduction.

1. Inspect paper numbers:

```text
docs/paper_data_catalog.md
docs/paper_evidence_registry.md
experiments/
```

2. Re-run fixed real-ALE 20-seed evaluation:

```text
scripts/eval/evaluate_pong_real_policies.py
scripts/eval/evaluate_torch_pong_real_policies.py
```

3. Re-run full chains:

```text
dataset conversion
-> offline world-model training
-> zero-shot RL inside frozen WM
-> fixed real-ALE 20-seed eval
-> summary CSV generation
```

Command files live under:

```text
experiments/*/commands/
```

## Backend Ownership

| Component | Location after bootstrap |
|---|---|
| Evaluation, protocols, result registries, SAM2 analysis | `scripts/`, `experiments/`, `docs/` |
| DreamerV3 | `third_party/dreamerv3-reborn` |
| DIAMOND | `third_party/diamond` |
| TWISTER | `third_party/twister` |
| Simulus | `third_party/simulus` |
| STORM/OC-STORM | `third_party/oc-storm` |

## Large Artifacts

Datasets, checkpoints, W&B logs, generated videos, and SAM2 masks are not stored
in git. They are tracked by manifests and checksums in `docs/` and
`experiments/`, and should be restored from the archive locations described
there.

## Paper Link

Use this repository as the single code URL in the paper.
