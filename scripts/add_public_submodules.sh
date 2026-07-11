#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK="$ROOT/third_party/backend_repos.lock"

while IFS='|' read -r name public_url branch public_commit submodule_path source_repo source_commit role; do
  [[ -z "${name:-}" ]] && continue
  [[ "$name" =~ ^# ]] && continue

  if [[ "$public_commit" == "TBD" ]]; then
    echo "[ERROR] $name has no public commit in $LOCK yet." >&2
    echo "        Publish the clean backend repo first, then update backend_repos.lock." >&2
    exit 1
  fi

  if [[ -e "$ROOT/$submodule_path" ]]; then
    echo "==> $name already exists at $submodule_path"
  else
    git -C "$ROOT" submodule add -b "$branch" "$public_url" "$submodule_path"
  fi

  git -C "$ROOT/$submodule_path" fetch --all --tags
  git -C "$ROOT/$submodule_path" checkout "$public_commit"
  echo
done < "$LOCK"
