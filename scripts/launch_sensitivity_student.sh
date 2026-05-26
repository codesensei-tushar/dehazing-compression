#!/usr/bin/env bash
# Train the sensitivity-driven distillation student.
#
# Pipeline (assumes indoor soft labels already exist or are not needed —
# we use GT target, not pseudo):
#   1. Verify ITS data, teacher ckpt, sensitivity JSON present
#   2. Train haze_b_sens (w32, GT, 200 ep) with multi-tap loss weighted by
#      Phase-1 sensitivity ranking
#   3. Eval best.pt on SOTS-indoor + SOTS-outdoor
#
# Compare against haze_b_large_tight (same student, single-tap pseudo-target loss).
#
# Run on 172.18.40.113 (or any free A5000):
#   ./scripts/sync_to_cluster.sh
#   ./gpu "cd dehazing-compression && bash scripts/launch_sensitivity_student.sh 2>&1 | tee results/phase2_haze_b_sens.log"

set -euo pipefail
PY=/home/teaching/miniconda3/envs/adu/bin/python
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

TAG="haze_b_sens"
ITS_HAZY="$ROOT/data/RESIDE/ITS-Train/train_indoor/haze"
ITS_GT="$ROOT/data/RESIDE/ITS-Train/train_indoor/clear_images"
TEACHER="$ROOT/experiments/teachers/dehamer/ckpts/indoor/PSNR3663_ssim09881.pt"
SENS_JSON="$ROOT/results/dehamer_sensitivity_indoor.json"

[ -d "$ITS_HAZY" ]  || { echo "FAIL: ITS hazy dir missing: $ITS_HAZY"; exit 1; }
[ -d "$ITS_GT" ]    || { echo "FAIL: ITS gt dir missing:   $ITS_GT";  exit 1; }
[ -f "$TEACHER" ]   || { echo "FAIL: teacher ckpt missing: $TEACHER"; exit 1; }
[ -f "$SENS_JSON" ] || { echo "FAIL: sensitivity JSON missing: $SENS_JSON"; exit 1; }

echo "=== $(date) Step 1: Train haze_b_sens (sensitivity-driven multi-tap) ==="
$PY phase2_distill/train_sensitivity_taps.py \
    --tag "$TAG" \
    --hazy-dir "$ITS_HAZY" --clean-dir "$ITS_GT" \
    --teacher-ckpt "$TEACHER" --sensitivity-json "$SENS_JSON" \
    --width 32 \
    --epochs 200 --batch 8 --patch 128 --workers 4 \
    --lr-hi 1e-3 --lr-lo 1e-6 \
    --lambda-feat 0.05 --lambda-perc 0.05 \
    --val-interval 5 --ckpt-interval 10 \
    --wandb

echo "=== $(date) Step 2: eval on SOTS-indoor ==="
BEST="$ROOT/experiments/students/$TAG/best.pt"
[ -f "$BEST" ] || { echo "FAIL: $BEST not produced"; exit 1; }
$PY phase2_distill/eval_student.py --ckpt "$BEST" --tag "$TAG" \
    --width 32 --split indoor

echo "=== $(date) Step 3: eval on SOTS-outdoor ==="
$PY phase2_distill/eval_student.py --ckpt "$BEST" --tag "$TAG" \
    --width 32 --split outdoor || echo "outdoor eval failed (non-fatal — outdoor split may be missing)"

echo "=== $(date) DONE ==="
ls -la results/eval_student_${TAG}*.json
