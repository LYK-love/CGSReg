# CGSReg Code Release Organization

Use `CGSReg` as the only code URL in the paper.

## Repository Roles

| Repository | Visibility | Role |
|---|---|---|
| `CGSReg` | Public | Paper entry point, experiments, evaluation, SAM2 analysis, dataset conversion, result registries |
| `CGSReg-dreamerv3-reborn` | Public | Clean DreamerV3 backend submodule |
| `CGSReg-diamond` | Public | Clean DIAMOND backend submodule |
| `CGSReg-twister` | Public | Clean TWISTER backend submodule |
| `CGSReg-simulus` | Public | Clean Simulus backend submodule |
| `CGSReg-oc-storm` | Public | Clean STORM/OC-STORM backend submodule |

The private development repositories are source inputs only. They are not
linked from the paper and should not be mirrored directly.

## Main Repository Layout

```text
CGSReg/
  docs/          # protocols, registries, release notes
  experiments/   # paper command files and result summaries
  scripts/       # evaluation, SAM2, dataset conversion, release tooling
  third_party/   # pinned backend submodules
```

## Clean Backend Release Flow

1. Record the private source commits in `third_party/backend_repos.lock`.
2. Generate filtered local snapshots:

```bash
bash scripts/build_all_clean_backends.sh --init-git
```

3. Review `build/clean_backends/*` manually.
4. Push each clean snapshot to its public `CGSReg-*` repository.
5. Replace `TBD` public commits in `third_party/backend_repos.lock`.
6. Add and pin submodules:

```bash
bash scripts/add_public_submodules.sh
```

The clean snapshot filters live in `release/clean_manifests/`.
