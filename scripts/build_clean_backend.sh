#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK="$ROOT/third_party/backend_repos.lock"
MANIFEST_DIR="$ROOT/release/clean_manifests"
BUILD_ROOT="$ROOT/build/clean_backends"

usage() {
  echo "Usage: $0 BACKEND_NAME [--init-git]" >&2
  echo "Example: $0 dreamerv3-reborn --init-git" >&2
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

name_arg="$1"
init_git=0
if [[ "${2:-}" == "--init-git" ]]; then
  init_git=1
elif [[ $# -gt 1 ]]; then
  usage
  exit 1
fi

manifest="$MANIFEST_DIR/$name_arg.rsync"
if [[ ! -f "$manifest" ]]; then
  echo "[ERROR] Missing clean manifest: $manifest" >&2
  exit 1
fi

line="$(awk -F'|' -v name="$name_arg" '($1 == name) {print; found=1} END {if (!found) exit 1}' "$LOCK")" || {
  echo "[ERROR] Backend not found in $LOCK: $name_arg" >&2
  exit 1
}

IFS='|' read -r name public_url branch public_commit submodule_path source_repo source_commit role <<< "$line"
source_label="$source_repo"
if [[ "$source_repo" == private:* ]]; then
  source_repo="${CGSREG_PRIVATE_REPOS_ROOT:-$HOME/projects}/${source_repo#private:}"
fi

if [[ ! -d "$source_repo/.git" ]]; then
  echo "[ERROR] Source repo is missing or not a git repo: $source_repo" >&2
  exit 1
fi

if ! git -C "$source_repo" cat-file -e "$source_commit^{commit}" 2>/dev/null; then
  echo "[ERROR] Source commit is not available locally: $source_repo $source_commit" >&2
  exit 1
fi

mkdir -p "$ROOT/build"
tmp_root="$(mktemp -d "$ROOT/build/${name}.archive.XXXXXX")"
archive_dir="$tmp_root/source"
dest="$BUILD_ROOT/$name"
mkdir -p "$archive_dir" "$dest"

cleanup() {
  rm -rf "$tmp_root"
}
trap cleanup EXIT

git -C "$source_repo" archive --format=tar "$source_commit" | tar -x -C "$archive_dir"

rm -rf "$dest"
mkdir -p "$dest"
rsync -a --delete --prune-empty-dirs --filter="merge $manifest" --exclude='*' "$archive_dir/" "$dest/"

cat > "$dest/CGSREG_SOURCE.md" <<EOF
# CGSReg Backend Source

Backend: $name
Role: $role

Public repository: $public_url
Planned branch: $branch
Planned submodule path: $submodule_path

This public snapshot was generated from the private development repository:

- source repo: $source_label
- source commit: $source_commit

The snapshot is filtered by:

\`\`\`text
release/clean_manifests/$name.rsync
\`\`\`
EOF

if [[ "$init_git" == "1" ]]; then
  git -C "$dest" init -q
  git -C "$dest" branch -M "$branch"
  git -C "$dest" add .
  if ! git -C "$dest" diff --cached --quiet; then
    git -C "$dest" commit -m "Initial CGSReg public backend release"
  fi
fi

echo "Built clean backend snapshot:"
echo "  name:        $name"
echo "  source:      $source_repo@$source_commit"
echo "  destination: $dest"
if [[ "$init_git" == "1" ]]; then
  echo "  git commit:  $(git -C "$dest" rev-parse HEAD)"
fi
