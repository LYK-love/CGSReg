# Release Checklist

- [ ] Create the public `CGSReg` GitHub repository.
- [ ] Create the five public clean backend repositories.
- [ ] Build clean backend snapshots with `scripts/build_all_clean_backends.sh --init-git`.
- [ ] Review each snapshot for local-only files, private notes, checkpoints,
      archives, notebooks, W&B logs, and generated media.
- [ ] Push the clean backend snapshots to their public repositories.
- [ ] Update `third_party/backend_repos.lock` with the public backend commits.
- [ ] Add backend repositories as git submodules with
      `scripts/add_public_submodules.sh`.
- [ ] Replace SSH URLs with HTTPS URLs in public-facing files if needed.
- [ ] Ensure large artifacts are represented by manifests and checksums, not
      committed files.
- [ ] Verify `scripts/bootstrap_repos.sh` on a fresh machine.
- [ ] Verify fixed real-ALE eval scripts run from a clean checkout.
- [ ] Verify paper tables point to raw artifacts in
      `docs/paper_evidence_registry.md`.
- [ ] Tag the release, for example `paper-v1.0`.
