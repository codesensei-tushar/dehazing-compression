#!/bin/bash
# Launch the full Phase-1 PTQ suite inside a detached tmux session on the cluster.
# Runs sequentially, writes logs to results/, persists across SSH disconnects.
#
# Usage (on the cluster, inside the project dir):
#   bash scripts/launch_phase1_tmux.sh            # default: 500 eval, 100 calib
#   bash scripts/launch_phase1_tmux.sh 100 50     # custom: --n-eval 100, --n-calib 50
#
# Check status:  tmux attach -t phase1     (detach with Ctrl-b d)
# Kill:          tmux kill-session -t phase1

set -euo pipefail

N_EVAL="${1:-0}"        # 0 == full 500
N_CALIB="${2:-100}"
SESSION="phase1"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

if ! command -v tmux >/dev/null; then
    echo "tmux not installed; falling back to nohup." >&2
    nohup bash -c "
        set -e
        source ~/miniconda3/etc/profile.d/conda.sh
        conda activate dehaze
        python phase1_quantize/run_all_ptq.py --variants fp32 dyn_all dyn_mixed --top-k 5 --n-eval ${N_EVAL} \\
            > results/phase1_runall.log 2>&1
        python phase1_quantize/block_static_ptq.py --n-calib ${N_CALIB} --n-eval ${N_EVAL} --top-k 5 \\
            > results/phase1_block_static.log 2>&1
        echo PHASE1_DONE > results/phase1_status.txt
    " > results/phase1_driver.log 2>&1 &
    echo "nohup driver pid=$!"
    echo "logs: results/phase1_{runall,block_static,driver}.log"
    exit 0
fi

# Kill any stale session with the same name.
tmux kill-session -t "$SESSION" 2>/dev/null || true

mkdir -p results

tmux new-session -d -s "$SESSION" -c "$PROJECT_DIR"

# Limit Torch thread pool to something modest to coexist with other cluster users.
# Default is #physical cores; on a heavily shared node we get better throughput
# with 8 threads than fighting for 40.
tmux send-keys -t "$SESSION" "export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8" C-m
tmux send-keys -t "$SESSION" "source ~/miniconda3/etc/profile.d/conda.sh" C-m
tmux send-keys -t "$SESSION" "conda activate dehaze" C-m
tmux send-keys -t "$SESSION" "cd ${PROJECT_DIR}" C-m
tmux send-keys -t "$SESSION" "echo STARTED \$(date -Iseconds) > results/phase1_status.txt" C-m
tmux send-keys -t "$SESSION" "uptime | tee -a results/phase1_status.txt" C-m

# Stage 1: fp32 + dyn_all + dyn_mixed, full SOTS-indoor.
tmux send-keys -t "$SESSION" "echo === STAGE 1: run_all_ptq === | tee -a results/phase1_status.txt" C-m
tmux send-keys -t "$SESSION" "python phase1_quantize/run_all_ptq.py --variants fp32 dyn_all dyn_mixed --top-k 5 --n-eval ${N_EVAL} 2>&1 | tee results/phase1_runall.log" C-m
tmux send-keys -t "$SESSION" "echo STAGE1_DONE \$(date -Iseconds) | tee -a results/phase1_status.txt" C-m

# Stage 2: block-static + mixed-final compositions.
tmux send-keys -t "$SESSION" "echo === STAGE 2: block_static_ptq === | tee -a results/phase1_status.txt" C-m
tmux send-keys -t "$SESSION" "python phase1_quantize/block_static_ptq.py --n-calib ${N_CALIB} --n-eval ${N_EVAL} --top-k 5 2>&1 | tee results/phase1_block_static.log" C-m
tmux send-keys -t "$SESSION" "echo STAGE2_DONE \$(date -Iseconds) | tee -a results/phase1_status.txt" C-m

# Finalize.
tmux send-keys -t "$SESSION" "echo PHASE1_DONE \$(date -Iseconds) | tee -a results/phase1_status.txt" C-m
tmux send-keys -t "$SESSION" "uptime | tee -a results/phase1_status.txt" C-m

echo "Launched tmux session '$SESSION'."
echo "Status file : results/phase1_status.txt"
echo "Stage logs  : results/phase1_runall.log, results/phase1_block_static.log"
echo "Attach      : tmux attach -t $SESSION     (detach with Ctrl-b d)"
echo "Kill        : tmux kill-session -t $SESSION"
