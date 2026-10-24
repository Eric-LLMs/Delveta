#!/usr/bin/env bash
# Move a Docker-managed model volume between machines without re-downloading.
#
# A named volume lives inside Docker's own storage, so it cannot be copied with
# a file manager. This script wraps it in a single tar.gz using a throwaway
# container:
#
#   export  <volume> <archive.tar.gz>
#   import  <volume> <archive.tar.gz>
#
# Export keeps the layout the bootstrap expects (versions/<revision>,
# the `current` symlink, and .delveta-model.json), so an imported volume is
# reused as-is and the next start performs no network request. Partial downloads
# and the lock file are excluded: they are not part of a published snapshot.
#
# The target volume must already exist. Compose creates it; a manual
# `docker volume create` would make the volume invisible to `down -v`.
set -euo pipefail

# Pinned by digest; alpine is only used as a tar runner.
IMAGE="alpine@sha256:28bd5fe8b56d1bd048e5babf5b10710ebe0bae67db86916198a6eec434943f8b"

usage() {
  cat >&2 <<'EOF'
usage: model_volume_bundle.sh export <volume> <archive.tar.gz>
       model_volume_bundle.sh import <volume> <archive.tar.gz>
EOF
  exit 64
}

[ "$#" -eq 3 ] || usage
action="$1"
volume="$2"
archive="$3"
case "$action" in
  export|import) ;;
  *) usage ;;
esac

if ! docker volume inspect "$volume" >/dev/null 2>&1; then
  echo "volume '$volume' does not exist." >&2
  if [ "$action" = "import" ]; then
    echo "create it with compose first, e.g.: docker compose create <service>" >&2
  fi
  exit 1
fi

# Docker Desktop wants a Windows-style path; Git Bash's pwd prints /e/... which
# it does not understand.
host_path() {
  if command -v cygpath >/dev/null 2>&1; then
    cygpath -m "$1"
  else
    printf '%s' "$1"
  fi
}

archive_dir="$(host_path "$(cd "$(dirname "$archive")" && pwd)")"
archive_name="$(basename "$archive")"
archive_path="$archive_dir/$archive_name"

# Keep the in-container paths (/data, /backup) out of MSYS's path rewriting.
export MSYS_NO_PATHCONV=1

case "$action" in
  export)
    if [ -e "$archive_path" ]; then
      echo "refusing to overwrite existing $archive_path" >&2
      exit 1
    fi
    docker run --rm \
      -v "$volume:/data:ro" \
      -v "$archive_dir:/backup" \
      "$IMAGE" \
      tar czf "/backup/$archive_name" \
        --exclude='./.fetch.lock' \
        --exclude='./.incoming-*' \
        --exclude='./.current.tmp' \
        -C /data .
    echo "exported $volume -> $archive_path"
    ;;
  import)
    if [ ! -f "$archive_path" ]; then
      echo "archive not found: $archive_path" >&2
      exit 1
    fi
    docker run --rm \
      -v "$volume:/data" \
      -v "$archive_dir:/backup:ro" \
      "$IMAGE" \
      tar xzf "/backup/$archive_name" -C /data
    echo "imported $archive_path -> $volume"
    ;;
esac
