#!/bin/bash
# Launch Phase-2 pipeline (soft labels -> student training) in a detached tmux
# session. Runs sequentially, writes logs to results/, survives SSH drops.
#
# Requires: ITS already unzipped at data/RESIDE/ITS-Train/train_indoor/{haze,clear_images}.
# Uses CLUSTER_PYTHON env var, else /home/teaching/miniconda3/envs/adu/bin/python.
#
# Usage (on the cluster):
#   bash scripts/launch_phase2_tmux.sh                       # default: full pipeline
#   bash scripts/launch_phase2_tmux.sh 10 haze_test           # epochs=10, tag=haze_test
#
# Check:   tmux attach -t phase2          (detach with Ctrl-b d)
# Status:  cat results/phase2_status.txt
# Kill:    tmux kill-session -t phase2

set -euo pipefail

EPOCHS="${1:-200}"
TAG="${2:-haze_s1}"
SESSION="phase2"
PY="${CLUSTER_PYTHON:-/home/teaching/miniconda3/envs/adu/bin/python}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

mkdir -p results experiments/soft_labels experiments/students

# Sanity-check ITS is present.
if [ ! -d "data/RESIDE/ITS-Train/train_indoor/haze" ]; then
    echo "ERROR: ITS hazy dir missing at data/RESIDE/ITS-Train/train_indoor/haze" >&2
    echo "Download + unzip ITS first (gdown id 1lE6FyHS-1MHoV6iM_s7phgf3Z3XJeC9E)." >&2
    exit 1
fi

if ! command -v tmux >/dev/null; then
    echo "ERROR: tmux not installed on this node." >&2
    exit 1
fi

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -c "$PROJECT_DIR"

tmux send-keys -t "$SESSION" "cd ${PROJECT_DIR}" C-m
tmux send-keys -t "$SESSION" "echo STARTED \$(date -Iseconds) > results/phase2_status.txt" C-m

# Stage 1: soft labels (DeHamer on all 13,990 ITS hazy images, GPU).
tmux send-keys -t "$SESSION" "echo === STAGE 1: soft labels === | tee -a results/phase2_status.txt" C-m
tmux send-keys -t "$SESSION" "${PY} scripts/gen_soft_labels.py \\
    --ckpt experiments/teachers/dehamer/ckpts/indoor/PSNR3663_ssim09881.pt \\
    --hazy-dir data/RESIDE/ITS-Train/train_indoor/haze \\
    --out-dir experiments/soft_labels/dehamer_indoor \\
    2>&1 | tee results/phase2_soft_labels.log" C-m
tmux send-keys -t "$SESSION" "echo STAGE1_DONE \$(date -Iseconds) | tee -a results/phase2_status.txt" C-m

# Stage 2: student training.
tmux send-keys -t "$SESSION" "echo === STAGE 2: student training === | tee -a results/phase2_status.txt" C-m
tmux send-keys -t "$SESSION" "${PY} phase2_distill/train.py \\
    --tag ${TAG} \\
    --epochs ${EPOCHS} \\
    --batch 8 --patch 128 --workers 4 \\
    --lr-hi 1e-3 --lr-lo 1e-6 \\
    --lambda-feat 0.01 \\
    --pseudo-dir experiments/soft_labels/dehamer_indoor \\
    --val-interval 5 --ckpt-interval 10 \\
    2>&1 | tee results/phase2_train.log" C-m
tmux send-keys -t "$SESSION" "echo STAGE2_DONE \$(date -Iseconds) | tee -a results/phase2_status.txt" C-m

# Finalize.
tmux send-keys -t "$SESSION" "echo PHASE2_DONE \$(date -Iseconds) | tee -a results/phase2_status.txt" C-m

echo "Launched tmux session '$SESSION' on $(hostname)."
echo "Logs    : results/phase2_{status,soft_labels,train}.{txt,log}"
echo "Attach  : tmux attach -t $SESSION     (detach with Ctrl-b d)"
echo "Kill    : tmux kill-session -t $SESSION"
