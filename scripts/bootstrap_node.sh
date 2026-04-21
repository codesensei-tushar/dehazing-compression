#!/bin/bash
# Bootstrap a fresh teaching@172.18.40.* node by scp'ing data + code from
# the SOURCE node (where this script runs) to the TARGET node, then
# installing Python deps.
#
# Run ON the source node (172.18.40.119) after sshing in with ./gpu.
#
# Usage:
#   bash scripts/bootstrap_node.sh <target_ip>
#   bash scripts/bootstrap_node.sh 172.18.40.131
#
# Requires:
#   - sshpass on source node
#   - Same `teaching` user + password `ds123` on target (teaching-node convention)

set -euo pipefail

TARGET="${1:?target IP required}"
USER="${TARGET_USER:-teaching}"
PASS="${TARGET_PASSWORD:-ds123}"
REMOTE_DIR="${REMOTE_DIR:-dehazing-compression}"

SSH_OPTS="-o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
SRC="/home/teaching/${REMOTE_DIR}"

if [ ! -d "$SRC" ]; then
    echo "ERROR: source dir $SRC not found. Are we on the source node?" >&2
    exit 1
fi

echo "=== [1/6] sanity-check target ==="
sshpass -p "$PASS" ssh $SSH_OPTS "$USER@$TARGET" "hostname && nvidia-smi -L | head -2"

echo "=== [2/6] create remote dir layout ==="
sshpass -p "$PASS" ssh $SSH_OPTS "$USER@$TARGET" "mkdir -p ~/${REMOTE_DIR}/{data/RESIDE,experiments/teachers/dehamer/ckpts/indoor,experiments/soft_labels,results,experiments/students,experiments/ptq}"

echo "=== [3/6] scp code (rsync over ssh w/ excludes) ==="
sshpass -p "$PASS" rsync -az \
    --exclude=".env" --exclude=".env.*" --exclude="servers.csv" \
    --exclude=".git/" --exclude="__pycache__/" \
    --exclude="data/RESIDE/*" --exclude="data/dummy/*" \
    --exclude="experiments/*" \
    --exclude="results/*" \
    --exclude="*.pt" --exclude="*.pth" --exclude="*.pkl" --exclude="*.zip" \
    -e "ssh $SSH_OPTS" \
    "$SRC/" "$USER@$TARGET:${REMOTE_DIR}/"

echo "=== [4/6] scp datasets (zips are smaller than unpacked) ==="
for z in data/RESIDE/ITS.zip data/RESIDE/SOTS.zip; do
    if [ -f "$SRC/$z" ]; then
        echo "  -> $z"
        sshpass -p "$PASS" scp $SSH_OPTS "$SRC/$z" "$USER@$TARGET:${REMOTE_DIR}/$z"
    fi
done

echo "=== [5/6] scp teacher indoor ckpt + soft labels (tar'd) ==="
CKPT="$SRC/experiments/teachers/dehamer/ckpts/indoor/PSNR3663_ssim09881.pt"
if [ -f "$CKPT" ]; then
    sshpass -p "$PASS" scp $SSH_OPTS "$CKPT" \
        "$USER@$TARGET:${REMOTE_DIR}/experiments/teachers/dehamer/ckpts/indoor/PSNR3663_ssim09881.pt"
fi

SOFT="$SRC/experiments/soft_labels/dehamer_indoor"
if [ -d "$SOFT" ]; then
    echo "  tar'ing + streaming soft labels (~2-3 GB)..."
    tar -C "$SRC/experiments/soft_labels" -cf - dehamer_indoor \
        | sshpass -p "$PASS" ssh $SSH_OPTS "$USER@$TARGET" \
            "cd ${REMOTE_DIR}/experiments/soft_labels && tar -xf -"
fi

echo "=== [6/6] remote unzip + pip install ==="
sshpass -p "$PASS" ssh $SSH_OPTS "$USER@$TARGET" bash -s <<'REMOTE'
set -e
cd ~/dehazing-compression
# Unzip ITS + SOTS if not already.
if [ ! -d data/RESIDE/ITS-Train ] && [ -f data/RESIDE/ITS.zip ]; then
    cd data/RESIDE && unzip -q -n ITS.zip && cd ../..
fi
if [ ! -d data/RESIDE/SOTS-Test ] && [ -f data/RESIDE/SOTS.zip ]; then
    cd data/RESIDE && unzip -q -n SOTS.zip && cd ../..
fi

# Locate the adu env python (same path on all teaching nodes).
PY="/home/teaching/miniconda3/envs/adu/bin/python"
PIP="/home/teaching/miniconda3/envs/adu/bin/pip"
if [ ! -x "$PY" ]; then
    echo "ERROR: adu env python not found at $PY" >&2
    exit 2
fi

# Install needed pip deps (idempotent).
"$PIP" install --quiet lmdb scipy scikit-image einops gdown opencv-python-headless 2>&1 | tail -3

# Quick smoke: can we import the student?
"$PY" -c "
import sys; sys.path.insert(0, '.')
from models.students.nafnet_student import build_student, count_params
import torch
for w in (16, 32):
    m = build_student(width=w)
    n, mM = count_params(m)
    print(f'student width={w}: {n:,} ({mM:.2f}M) OK')
"
echo BOOTSTRAP_DONE
REMOTE

echo ""
echo "=== DONE: target $TARGET is provisioned ==="
