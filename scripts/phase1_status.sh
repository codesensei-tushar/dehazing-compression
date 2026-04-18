#!/bin/bash
# Quick status checker for the Phase-1 tmux run on the cluster.
# Run LOCALLY (uses ./gpu to ssh in).

set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU="$PROJECT_DIR/gpu"

$GPU "cd dehazing-compression && \
    echo '--- uptime ---'; uptime; \
    echo '--- tmux ---'; tmux ls 2>/dev/null || echo 'no tmux sessions'; \
    echo '--- phase1_status.txt ---'; tail -20 results/phase1_status.txt 2>/dev/null || echo 'no status file'; \
    echo '--- last runall lines ---'; tail -3 results/phase1_runall.log 2>/dev/null | tr '\\r' '\\n' | tail -3 || true; \
    echo '--- last block_static lines ---'; tail -3 results/phase1_block_static.log 2>/dev/null | tr '\\r' '\\n' | tail -3 || true; \
    echo '--- result JSONs ---'; ls -lh results/*.json 2>/dev/null"
