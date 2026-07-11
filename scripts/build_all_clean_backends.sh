#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK="$ROOT/third_party/backend_repos.lock"

init_git=0
if [[ "${1:-}" == "--init-git" ]]; then
  init_git=1
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--init-git]" >&2
  exit 1
fi

while IFS='|' read -r name public_url branch public_commit submodule_path source_repo source_commit role; do
  [[ -z "${name:-}" ]] && continue
  [[ "$name" =~ ^# ]] && continue
  if [[ "$init_git" == "1" ]]; then
    bash "$ROOT/scripts/build_clean_backend.sh" "$name" --init-git
  else
    bash "$ROOT/scripts/build_clean_backend.sh" "$name"
  fi
  echo
done < "$LOCK"
