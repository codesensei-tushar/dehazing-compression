#!/usr/bin/env bash
# Train + eval AOD-Net and FFA-Net on RESIDE ITS, then evaluate on
# SOTS-indoor + SOTS-outdoor. Apples-to-apples baseline rows for the
# main comparison table.
#
# Runtime budget on RTX A5000:
#   AOD-Net  : ~30 min train + 1 min eval × 2 splits
#   FFA-Net  : ~3-5 h train + 5 min eval × 2 splits
#
# Run on 172.18.40.103 (has ITS + SOTS + adu env + indoor teacher already):
#   ./scripts/sync_to_cluster.sh    # if local code changed
#   ./gpu "cd dehazing-compression && bash scripts/run_baselines.sh 2>&1 | tee results/baselines.log"

set -euo pipefail
PY=/home/teaching/miniconda3/envs/adu/bin/python
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ITS_HAZY="$ROOT/data/RESIDE/ITS-Train/train_indoor/haze"
ITS_GT="$ROOT/data/RESIDE/ITS-Train/train_indoor/clear_images"

mkdir -p experiments/baselines results

# ── Pre-flight ──────────────────────────────────────────────────────────────
[ -d "$ITS_HAZY" ] || { echo "FAIL: $ITS_HAZY missing"; exit 1; }
[ -d "$ITS_GT" ]   || { echo "FAIL: $ITS_GT missing";   exit 1; }
$PY -c "import torch; assert torch.cuda.is_available(), 'no CUDA'"

AOD_CKPT="$ROOT/experiments/baselines/aodnet_indoor.pth"
FFA_CKPT="$ROOT/experiments/baselines/ffanet_indoor.pth"

# ── AOD-Net ─────────────────────────────────────────────────────────────────
if [ ! -f "$AOD_CKPT" ]; then
    echo "=== $(date) train AOD-Net (200 ep) ==="
    $PY evaluate/train_aodnet.py \
        --its-hazy "$ITS_HAZY" --its-gt "$ITS_GT" \
        --out "$AOD_CKPT" --epochs 200 --batch 8 --patch 240 --lr 1e-3
else
    echo "=== AOD-Net ckpt already present: $AOD_CKPT ==="
fi

echo "=== $(date) eval AOD-Net (SOTS-indoor) ==="
$PY evaluate/eval_baseline.py --model aodnet --ckpt "$AOD_CKPT" \
    --tag aodnet --split indoor

echo "=== $(date) eval AOD-Net (SOTS-outdoor) ==="
$PY evaluate/eval_baseline.py --model aodnet --ckpt "$AOD_CKPT" \
    --tag aodnet --split outdoor

# ── FFA-Net ─────────────────────────────────────────────────────────────────
if [ ! -f "$FFA_CKPT" ]; then
    echo "=== $(date) train FFA-Net (100 ep) ==="
    $PY evaluate/train_ffanet.py \
        --its-hazy "$ITS_HAZY" --its-gt "$ITS_GT" \
        --out "$FFA_CKPT" --epochs 100 --batch 1 --patch 240 --lr 1e-4
else
    echo "=== FFA-Net ckpt already present: $FFA_CKPT ==="
fi

echo "=== $(date) eval FFA-Net (SOTS-indoor) ==="
$PY evaluate/eval_baseline.py --model ffanet --ckpt "$FFA_CKPT" \
    --tag ffanet --split indoor

echo "=== $(date) eval FFA-Net (SOTS-outdoor) ==="
$PY evaluate/eval_baseline.py --model ffanet --ckpt "$FFA_CKPT" \
    --tag ffanet --split outdoor

echo "=== $(date) BASELINES DONE ==="
echo "results:"
ls -la results/eval_baseline_*.json | tail -10
