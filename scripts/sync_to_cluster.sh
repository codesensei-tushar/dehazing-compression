#!/bin/bash
# Sync the project to the GPU cluster via rsync over sshpass.
# Source of truth is local; cluster-side edits may be overwritten.
# Usage:
#   ./scripts/sync_to_cluster.sh              # safe sync (no delete)
#   ./scripts/sync_to_cluster.sh --delete     # mirror (wipes stale remote files)

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$PROJECT_DIR/.env"

if [ ! -f "$ENV_FILE" ]; then
    echo "Missing $ENV_FILE" >&2
    exit 1
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

: "${CLUSTER_USER:?}"
: "${CLUSTER_HOST:?}"
: "${CLUSTER_PASSWORD:?}"
REMOTE_DIR="${CLUSTER_REMOTE_DIR:-dehazing-compression}"

command -v sshpass >/dev/null || { echo "sshpass not installed" >&2; exit 1; }
command -v rsync   >/dev/null || { echo "rsync not installed"   >&2; exit 1; }

EXTRA_FLAGS=()
if [ "${1:-}" = "--delete" ]; then
    EXTRA_FLAGS+=(--delete)
    echo "WARNING: --delete set; remote files missing locally will be removed."
fi

echo "Syncing $PROJECT_DIR/ -> $CLUSTER_USER@$CLUSTER_HOST:$REMOTE_DIR/"

sshpass -p "$CLUSTER_PASSWORD" rsync -avz "${EXTRA_FLAGS[@]}" \
  --exclude=".env" \
  --exclude=".git/" \
  --exclude="data/" \
  --exclude="experiments/" \
  --exclude="wandb/" \
  --exclude="__pycache__/" \
  --exclude="*.pth" --exclude="*.pt" --exclude="*.pkl" \
  --exclude=".mypy_cache/" --exclude=".pytest_cache/" --exclude=".ruff_cache/" \
  -e "ssh -o StrictHostKeyChecking=accept-new" \
  "$PROJECT_DIR/" "$CLUSTER_USER@$CLUSTER_HOST:$REMOTE_DIR/"

echo "Sync complete."
