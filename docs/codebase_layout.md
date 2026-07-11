# Codebase Layout

CGSReg is an umbrella repository. The top-level project is the only URL that
needs to appear in the paper. Paper-facing orchestration lives directly in this
repo, while exact backend world-model code is pinned as submodules under
`third_party/`.

## Layers

### Layer 1: Paper Entry Point

```text
CGSReg/
  README.md
  REPRODUCING.md
  docs/
  scripts/
  experiments/
  third_party/backend_repos.lock
```

This layer explains the project and provides stable bootstrap commands. It also
owns evaluation, dataset conversion, SAM2 analysis, experiment command files,
paper evidence registries, and artifact manifests.

### Layer 2: Evaluation And Experiment Orchestration

```text
scripts/
experiments/
docs/
```

These directories own:

- paper protocols
- experiment command files
- fixed real-ALE 20-seed evaluation
- SAM2 rollout analysis
- dataset registries
- paper evidence registries
- large artifact manifests

### Layer 3: Backend World Models

```text
third_party/dreamerv3-reborn
third_party/diamond
third_party/twister
third_party/simulus
third_party/oc-storm
```

These repos own backend-specific model code, offline training entry points, and
policy-in-WM training implementations.

## Why Not One Giant Source Tree?

The five WM projects use different dependency stacks, training scripts, config
systems, and checkpoint formats. Keeping them as pinned backend repositories
preserves their working layouts while giving readers one paper codebase.

## Public Backend Submodules

The private development repos are not published directly. Each backend is first
copied into a minimal clean public repository, then pinned here as a submodule.
The mapping from private source commit to public release commit is tracked in
`third_party/backend_repos.lock`.
