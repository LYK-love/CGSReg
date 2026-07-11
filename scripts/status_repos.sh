#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK="$ROOT/third_party/backend_repos.lock"

while IFS='|' read -r name url branch commit path source_repo source_commit role; do
  [[ -z "${name:-}" ]] && continue
  [[ "$name" =~ ^# ]] && continue

  dest="$ROOT/$path"
  echo "==> $name"
  echo "    role:          $role"
  echo "    public url:    $url"
  echo "    source repo:   $source_repo"
  echo "    source commit: $source_commit"
  if [[ "$commit" == "TBD" ]]; then
    echo "    public commit: TBD"
    echo "    status:        pending public clean repo publication"
    echo
    continue
  fi
  if ! git -C "$dest" rev-parse --git-dir >/dev/null 2>&1; then
    echo "    missing: $dest"
    echo
    continue
  fi
  actual="$(git -C "$dest" rev-parse HEAD)"
  current_branch="$(git -C "$dest" branch --show-current || true)"
  echo "    expected: $commit"
  echo "    actual:   $actual"
  echo "    branch:   ${current_branch:-detached}"
  if [[ "$actual" != "$commit" ]]; then
    echo "    status:   MISMATCH"
  else
    echo "    status:   OK"
  fi
  dirty="$(git -C "$dest" status --short)"
  if [[ -n "$dirty" ]]; then
    echo "    dirty:"
    echo "$dirty" | sed 's/^/      /'
  fi
  echo
done < "$LOCK"
