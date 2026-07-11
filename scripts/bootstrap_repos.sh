#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK="$ROOT/third_party/backend_repos.lock"

if [[ ! -f "$LOCK" ]]; then
  echo "[ERROR] Missing lock file: $LOCK" >&2
  exit 1
fi

while IFS='|' read -r name url branch commit path source_repo source_commit role; do
  [[ -z "${name:-}" ]] && continue
  [[ "$name" =~ ^# ]] && continue

  dest="$ROOT/$path"
  echo "==> $name"
  echo "    role:          $role"
  echo "    public url:    $url"
  echo "    branch:        $branch"
  echo "    public commit: $commit"
  echo "    path:          $dest"

  if [[ "$commit" == "TBD" ]]; then
    echo "    status:        pending public clean repo publication"
    echo
    continue
  fi

  if git -C "$dest" rev-parse --git-dir >/dev/null 2>&1; then
    git -C "$dest" fetch --all --tags
  else
    mkdir -p "$(dirname "$dest")"
    git clone "$url" "$dest"
  fi

  git -C "$dest" checkout "$commit"
  echo
done < "$LOCK"
