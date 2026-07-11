# Dyna Freeze-WM Diagnostic Conventions

This document is the cross-project contract for the Dyna-style freeze world
model diagnostic. Project-specific differences belong in `adaptors/*.md`.

## Scope

The diagnostic asks whether a policy can keep improving from model-generated
training once world-model updates stop. Each run keeps real-environment
collection, policy/value updates, checkpointing, and real-environment
evaluation active. Only the world-model optimizer updates are disabled after the
freeze point.

Project adaptors may define extra speed-oriented constraints when they do not
change the policy-training question being tested. DIAMOND, for example, uses a
fixed pretrained reward/end model for this diagnostic and does not train its
reward/end component.

The baseline training budget is `T = 100000` environment-equivalent steps. The
diagnostic extends training to `1.5T = 150000` environment-equivalent steps.

## W&B Projects

Each world-model implementation owns one W&B project:

| Project | W&B project |
| --- | --- |
| DreamerV3 | `dreamer-dyna-freeze-wm` |
| DIAMOND | `diamond-dyna-freeze-wm` |
| Simulus | `simulus-dyna-freeze-wm` |
| TWISTER | `twister-dyna-freeze-wm` |
| STORM | `storm-dyna-freeze-wm` |

All runs use entity `ssl-lab` and online logging unless intentionally debugging
offline.

## Run Names

The canonical run-name template is:

```text
<project>_repro_<condition>
```

where `<condition>` is one of:

| Condition | Meaning | Freeze point | Total training |
| --- | --- | ---: | ---: |
| `nofreeze_1p5T` | No world-model freeze. | none | `1.5T` |
| `f0p5_to1p5T` | Freeze WM after `0.5T`; continue policy to `1.5T`. | `50000` | `150000` |
| `f0p75_to1p5T` | Freeze WM after `0.75T`; continue policy to `1.5T`. | `75000` | `150000` |
| `f1p0_to1p5T` | Freeze WM after `1.0T`; continue policy for another `0.5T`. | `100000` | `150000` |

Current project prefixes:

```text
diamond_repro_...
simulus_repro_...
twister_repro_...
storm_repro_...
```

DreamerV3 includes the size config in the prefix to avoid mixing model sizes:

```text
dreamer_<size>_...
```

For the current size200m diagnostic this becomes:

```text
dreamer_size200m_nofreeze_1p5T
dreamer_size200m_f0p5_to1p5T
dreamer_size200m_f0p75_to1p5T
dreamer_size200m_f1p0_to1p5T
```

## Local Output Directories

Local output directories must contain the W&B run name so that checkpoints,
Hydra outputs, local W&B files, and scheduler logs can be associated without
consulting W&B.

Canonical scheduler logs live under:

```text
experiments/freeze_wm_diagnostic/scheduler_logs/<project>/
```

Project-local output roots are documented in each adaptor:

```text
adaptors/dreamer.md
adaptors/diamond.md
adaptors/simulus.md
adaptors/twister.md
adaptors/storm.md
```

## Required W&B Logging

Every adaptor should log or provide a deterministic mapping to the following
canonical fields.

### Dyna State

| Canonical key | Meaning |
| --- | --- |
| `dyna/wm_frozen` | `0` before the freeze point, `1` after WM updates are disabled. |
| `dyna/freeze_progress` | Current training progress divided by the original `T`. |
| `dyna/collection_step` | Environment-equivalent progress used for comparisons. |
| `dyna/freeze_wm_after_step` | Step freeze threshold when the project is step-based. |
| `dyna/freeze_wm_after_epoch` | Epoch freeze threshold when the project is epoch-based. |
| `dyna/freeze_wm_after_collection_step` | Collection-step freeze threshold when collection is the native unit. |

For no-freeze runs, the freeze threshold should be `-1`, omitted, or documented
as not applicable by the adaptor. `dyna/wm_frozen` must remain `0`.

### Real-Environment Evaluation

The comparison metric is greedy ALE Pong real-environment policy evaluation.
Each standardized evaluation should use 5 episodes.

Preferred canonical keys:

| Canonical key | Meaning |
| --- | --- |
| `eval_real/score_mean` | Mean Pong return over eval episodes. |
| `eval_real/score_std` | Return standard deviation. |
| `eval_real/score_min` | Minimum return. |
| `eval_real/score_max` | Maximum return. |
| `eval_real/episode_length_mean` | Mean episode length. |
| `eval_real/num_episodes` | Number of eval episodes, normally `5`. |
| `eval_real/collection_step` | Environment-equivalent step at evaluation time. |
| `eval_real/policy` | `greedy`. |
| `eval_real/env` | ALE Pong real environment identifier. |

If an implementation still logs native keys, the adaptor must document the
mapping. Native keys are acceptable for historical analysis only when the
mapping is explicit.

## Command Files

Canonical command files live in:

```text
experiments/freeze_wm_diagnostic/commands/
```

Each primary command file must contain exactly the four run conditions above.
The command files may include project-specific runtime controls, such as CPU
thread caps, checkpoint retention, or compile toggles, but those controls should
be documented in the matching adaptor.

## Comparison Rules

1. Compare real-environment policy score against `dyna/freeze_progress`.
2. Mark the freeze point where `dyna/wm_frozen` first changes from `0` to `1`.
3. Treat all project-native progress units as adapters to the same original
   budget `T = 100000`.
4. Do not compare sampled-policy evaluation to greedy-policy evaluation unless
   the run is explicitly marked historical or auxiliary.
5. Do not merge runs across different model sizes into the same DreamerV3 run
   name.
