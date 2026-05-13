#!/usr/bin/env bash
# Evaluate all three indoor students (A, B, C) on every remaining split,
# and run the DeHamer teacher with its per-split specialized checkpoint.
#
# Run on the cluster after syncing:
#   ./scripts/sync_to_cluster.sh && ./gpu "cd dehazing-compression && bash scripts/eval_all_splits.sh"
#
# Results land in results/  as JSON files:
#   eval_student_<tag>_<split>.json   — student PSNR/SSIM/latency per split
#   dehamer_fp32_<split>.json         — teacher reference per split
#
# Dense-Haze and NH-HAZE: pass --hazy-dir / --gt-dir if your dataset layout
# differs from the defaults below. Edit the DATA_DENSE / DATA_NH vars.

set -euo pipefail
PY=/home/teaching/miniconda3/envs/adu/bin/python
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Adjust these if Dense-Haze / NH-HAZE live at different paths.
DATA_DENSE_HAZY="${ROOT}/data/Dense-Haze/hazy"
DATA_DENSE_GT="${ROOT}/data/Dense-Haze/GT"
DATA_NH_HAZY="${ROOT}/data/NH-HAZE/hazy"
DATA_NH_GT="${ROOT}/data/NH-HAZE/GT"

STUDENTS=(
    "haze_a_small_tight 16"
    "haze_b_large_tight 32"
    "haze_c_large_pseudo 32"
)

# ── 1. Teacher baselines on all splits ──────────────────────────────────────
echo "=== DeHamer teacher: outdoor ==="
$PY "${ROOT}/evaluate/benchmark_dehamer.py" --split outdoor

if [ -d "$DATA_DENSE_HAZY" ]; then
    echo "=== DeHamer teacher: dense ==="
    $PY "${ROOT}/evaluate/benchmark_dehamer.py" --split dense \
        --hazy-dir "$DATA_DENSE_HAZY" --gt-dir "$DATA_DENSE_GT"
else
    echo "[SKIP] Dense-Haze not found at $DATA_DENSE_HAZY"
    echo "       Download and set DATA_DENSE_HAZY / DATA_DENSE_GT in this script."
fi

if [ -d "$DATA_NH_HAZY" ]; then
    echo "=== DeHamer teacher: nh ==="
    $PY "${ROOT}/evaluate/benchmark_dehamer.py" --split nh \
        --hazy-dir "$DATA_NH_HAZY" --gt-dir "$DATA_NH_GT"
else
    echo "[SKIP] NH-HAZE not found at $DATA_NH_HAZY"
    echo "       Download and set DATA_NH_HAZY / DATA_NH_GT in this script."
fi

# ── 2. Student evals on outdoor ──────────────────────────────────────────────
echo ""
echo "=== Students: outdoor (cross-domain) ==="
for entry in "${STUDENTS[@]}"; do
    TAG=$(echo "$entry" | cut -d' ' -f1)
    W=$(echo "$entry" | cut -d' ' -f2)
    CKPT="${ROOT}/experiments/students/${TAG}/best.pt"
    if [ ! -f "$CKPT" ]; then
        echo "[SKIP] $CKPT not found"
        continue
    fi
    echo "--- $TAG (w=$W) → outdoor ---"
    $PY "${ROOT}/phase2_distill/eval_student.py" \
        --ckpt "$CKPT" --tag "$TAG" --width "$W" --split outdoor
done

# ── 3. Student evals on dense ────────────────────────────────────────────────
if [ -d "$DATA_DENSE_HAZY" ]; then
    echo ""
    echo "=== Students: dense (cross-domain) ==="
    for entry in "${STUDENTS[@]}"; do
        TAG=$(echo "$entry" | cut -d' ' -f1)
        W=$(echo "$entry" | cut -d' ' -f2)
        CKPT="${ROOT}/experiments/students/${TAG}/best.pt"
        if [ ! -f "$CKPT" ]; then
            echo "[SKIP] $CKPT not found"
            continue
        fi
        echo "--- $TAG (w=$W) → dense ---"
        $PY "${ROOT}/phase2_distill/eval_student.py" \
            --ckpt "$CKPT" --tag "$TAG" --width "$W" --split dense \
            --hazy-dir "$DATA_DENSE_HAZY" --gt-dir "$DATA_DENSE_GT"
    done
fi

# ── 4. Student evals on nh ───────────────────────────────────────────────────
if [ -d "$DATA_NH_HAZY" ]; then
    echo ""
    echo "=== Students: nh (cross-domain) ==="
    for entry in "${STUDENTS[@]}"; do
        TAG=$(echo "$entry" | cut -d' ' -f1)
        W=$(echo "$entry" | cut -d' ' -f2)
        CKPT="${ROOT}/experiments/students/${TAG}/best.pt"
        if [ ! -f "$CKPT" ]; then
            echo "[SKIP] $CKPT not found"
            continue
        fi
        echo "--- $TAG (w=$W) → nh ---"
        $PY "${ROOT}/phase2_distill/eval_student.py" \
            --ckpt "$CKPT" --tag "$TAG" --width "$W" --split nh \
            --hazy-dir "$DATA_NH_HAZY" --gt-dir "$DATA_NH_GT"
    done
fi

echo ""
echo "All done. JSON results in ${ROOT}/results/"
