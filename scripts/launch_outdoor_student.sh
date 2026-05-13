#!/usr/bin/env bash
# Train a dedicated outdoor haze student (mirrors Node B: w32, GT target).
#
# Pipeline:
#   Step 1 — Generate outdoor DeHamer soft labels for 50K OTS subset (~2-3h)
#   Step 2 — Train outdoor NAFNet-32 student on 50K OTS subset (~8-10h)
#   Step 3 — Evaluate best.pt on SOTS-outdoor
#
# OTS paths (313K pairs). Adjust if your cluster layout differs:
#   hazy:  data/RESIDE/OTS/haze/
#   clean: data/RESIDE/OTS/clear/
# SOTS-outdoor eval paths:
#   hazy:  data/RESIDE/SOTS-Test/valid_outdoor/input/
#   gt:    data/RESIDE/SOTS-Test/valid_outdoor/gt/
#
# Outdoor DeHamer checkpoint:
#   experiments/teachers/dehamer/ckpts/outdoor/PSNR3518_SSIM09860.pt
#
# Run on the cluster (inside tmux or nohup):
#   ./scripts/sync_to_cluster.sh
#   ./gpu "cd dehazing-compression && bash scripts/launch_outdoor_student.sh 2>&1 | tee results/phase2_haze_outdoor_b.log"

set -euo pipefail
PY=/home/teaching/miniconda3/envs/adu/bin/python
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TAG="haze_outdoor_b"

OTS_HAZY="${ROOT}/data/RESIDE/OTS/haze"
OTS_CLEAN="${ROOT}/data/RESIDE/OTS/clear"
OUTDOOR_CKPT="${ROOT}/experiments/teachers/dehamer/ckpts/outdoor/PSNR3518_SSIM09860.pt"
PSEUDO_DIR="${ROOT}/experiments/soft_labels/dehamer_outdoor"
SOTS_OUT_HAZY="${ROOT}/data/RESIDE/SOTS-Test/valid_outdoor/input"
SOTS_OUT_GT="${ROOT}/data/RESIDE/SOTS-Test/valid_outdoor/gt"

# ── Pre-flight checks ─────────────────────────────────────────────────────────
if [ ! -d "$OTS_HAZY" ]; then
    echo "ERROR: OTS hazy dir not found: $OTS_HAZY"
    echo "Download OTS: https://bit.ly/3k8a0Gf and extract under data/RESIDE/OTS/"
    exit 1
fi
if [ ! -f "$OUTDOOR_CKPT" ]; then
    echo "ERROR: outdoor DeHamer checkpoint not found: $OUTDOOR_CKPT"
    echo "Run scripts/download_dehamer_ckpts.sh or place the checkpoint manually."
    exit 1
fi

echo "=== Step 1: Generate outdoor soft labels (50K OTS subset) ==="
echo "    ckpt : $OUTDOOR_CKPT"
echo "    src  : $OTS_HAZY"
echo "    out  : $PSEUDO_DIR"
echo "    (skip-existing=true; safe to rerun if interrupted)"
$PY "${ROOT}/scripts/gen_soft_labels.py" \
    --ckpt "$OUTDOOR_CKPT" \
    --hazy-dir "$OTS_HAZY" \
    --out-dir "$PSEUDO_DIR" \
    --max 50000

echo ""
echo "=== Step 2: Train outdoor student (haze_outdoor_b) ==="
echo "    width=32, GT target, lambda_feat=0.05, lambda_perc=0.05, 200 epochs"
echo "    training data: 50K OTS pairs"
echo "    val set: SOTS-outdoor"
$PY "${ROOT}/phase2_distill/train.py" \
    --tag "$TAG" \
    --hazy-dir "$OTS_HAZY" \
    --clean-dir "$OTS_CLEAN" \
    --pseudo-dir "$PSEUDO_DIR" \
    --sots-hazy "$SOTS_OUT_HAZY" \
    --sots-gt "$SOTS_OUT_GT" \
    --width 32 \
    --epochs 200 --batch 8 --patch 128 --workers 4 \
    --lr-hi 1e-3 --lr-lo 1e-6 \
    --lambda-feat 0.05 --lambda-perc 0.05 \
    --max-samples 50000 \
    --val-interval 5 --ckpt-interval 10 \
    --wandb

echo ""
echo "=== Step 3: Evaluate best.pt on SOTS-outdoor ==="
BEST_CKPT="${ROOT}/experiments/students/${TAG}/best.pt"
$PY "${ROOT}/phase2_distill/eval_student.py" \
    --ckpt "$BEST_CKPT" \
    --tag "$TAG" \
    --width 32 \
    --split outdoor

echo ""
echo "Done. Results at results/eval_student_${TAG}_outdoor.json"
echo "Status log at results/phase2_${TAG}_status.txt"
