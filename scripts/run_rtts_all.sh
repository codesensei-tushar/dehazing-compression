#!/usr/bin/env bash
# Run the real-world (RTTS) qualitative + NIQE/BRISQUE pipeline on
# teacher + all 3 indoor students (+ baselines if their ckpts are present).
#
# Assumes:
#   - data/RTTS/ populated (run scripts/download_rtts.sh first if not)
#   - pyiqa installed in adu env (auto-installs below)
#   - DeHamer indoor teacher + student ckpts present
#
# Output:
#   results/rtts_<tag>/<image>.png    dehazed PNGs (one folder per model)
#   results/rtts_<tag>.json           mean NIQE + BRISQUE
#
# Run on 172.18.40.139 (or any clean A5000 node):
#   ./gpu "cd dehazing-compression && bash scripts/run_rtts_all.sh 2>&1 | tee results/rtts_all.log"
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY=/home/teaching/miniconda3/envs/adu/bin/python
PIP=/home/teaching/miniconda3/envs/adu/bin/pip
cd "$ROOT"

# pyiqa is the only extra dep beyond the base adu env
$PY -c "import pyiqa" 2>/dev/null || {
    echo "=== installing pyiqa ==="
    $PIP install --quiet pyiqa
}

# Optional baselines (skip silently if ckpt absent)
AOD_CKPT="$ROOT/experiments/baselines/aodnet_indoor.pth"
FFA_CKPT="$ROOT/experiments/baselines/ffanet_indoor.pth"

DEHAMER_INDOOR="$ROOT/experiments/teachers/dehamer/ckpts/indoor/PSNR3663_ssim09881.pt"
STUDENT_A="$ROOT/experiments/students/haze_a_small_tight/best.pt"
STUDENT_B="$ROOT/experiments/students/haze_b_large_tight/best.pt"
STUDENT_C="$ROOT/experiments/students/haze_c_large_pseudo/best.pt"

# Sanity check
[ -f "$DEHAMER_INDOOR" ] || { echo "FAIL: teacher ckpt missing: $DEHAMER_INDOOR"; exit 1; }
[ -d "$ROOT/data/RTTS" ] || { echo "FAIL: data/RTTS missing — run scripts/download_rtts.sh"; exit 1; }

# Hazy passthrough — gives a NIQE/BRISQUE baseline on the raw RTTS images
echo "=== $(date) RTTS: hazy passthrough ==="
$PY evaluate/eval_rtts.py --model hazy_passthrough --tag rtts_hazy

# Teacher
echo "=== $(date) RTTS: DeHamer teacher (indoor) ==="
$PY evaluate/eval_rtts.py --model teacher --tag dehamer_indoor --ckpt "$DEHAMER_INDOOR"

# Students
for entry in "haze_a_small_tight:16:$STUDENT_A" \
             "haze_b_large_tight:32:$STUDENT_B" \
             "haze_c_large_pseudo:32:$STUDENT_C"; do
    IFS=":" read -r tag w ckpt <<< "$entry"
    if [ -f "$ckpt" ]; then
        echo "=== $(date) RTTS: student $tag (w=$w) ==="
        $PY evaluate/eval_rtts.py --model student --tag "$tag" --width "$w" --ckpt "$ckpt"
    else
        echo "skip $tag (no ckpt at $ckpt)"
    fi
done

# Baselines (optional)
if [ -f "$AOD_CKPT" ]; then
    echo "=== $(date) RTTS: AOD-Net ==="
    $PY evaluate/eval_rtts.py --model aodnet --tag aodnet --ckpt "$AOD_CKPT"
fi
if [ -f "$FFA_CKPT" ]; then
    echo "=== $(date) RTTS: FFA-Net ==="
    $PY evaluate/eval_rtts.py --model ffanet --tag ffanet --ckpt "$FFA_CKPT"
fi

echo "=== $(date) RTTS ALL DONE ==="
echo "results:"
ls -la results/rtts_*.json
