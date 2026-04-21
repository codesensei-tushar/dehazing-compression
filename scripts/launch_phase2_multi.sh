#!/bin/bash
# Launch three Phase-2 distillation variants across three teaching nodes.
#
# Run ON the SOURCE node (172.18.40.119) after `./gpu` from local.
#
# Variants (tag -> node):
#   haze_a_small_tight  -> 172.18.40.119  (source; already bootstrapped)
#   haze_b_large_tight  -> 172.18.40.131
#   haze_c_large_pseudo -> 172.18.40.137
#
# Each run creates a tmux session `phase2_<tag>` on its node.

set -euo pipefail

NODE_A="172.18.40.119"
NODE_B="172.18.40.131"
NODE_C="172.18.40.137"
USER="teaching"
PASS="ds123"

SSH_OPTS="-o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
PY="/home/teaching/miniconda3/envs/adu/bin/python"
REMOTE_DIR="dehazing-compression"

EPOCHS="${EPOCHS:-200}"

# ---- Node B & C bootstrap (idempotent) --------------------------------------
for T in "$NODE_B" "$NODE_C"; do
    echo "=== bootstrap $T ==="
    bash "$(dirname "$0")/bootstrap_node.sh" "$T"
done

# ---- Helper: launch one training job in a remote tmux session ---------------
launch_remote() {
    local IP="$1" TAG="$2" ARGS="$3"
    echo "=== launching $TAG on $IP ==="
    # Build the training command.
    local CMD="cd ~/${REMOTE_DIR} && \
       echo STARTED \$(date -Iseconds) > results/phase2_${TAG}_status.txt && \
       ${PY} phase2_distill/train.py \
         --tag ${TAG} \
         --epochs ${EPOCHS} \
         --batch 8 --patch 128 --workers 4 \
         --lr-hi 1e-3 --lr-lo 1e-6 \
         --pseudo-dir experiments/soft_labels/dehamer_indoor \
         --val-interval 5 --ckpt-interval 10 \
         ${ARGS} \
         2>&1 | tee results/phase2_${TAG}.log && \
       echo DONE \$(date -Iseconds) >> results/phase2_${TAG}_status.txt"

    # Send to a new tmux session on the target.
    sshpass -p "$PASS" ssh $SSH_OPTS "$USER@$IP" \
        "tmux kill-session -t phase2_${TAG} 2>/dev/null; \
         tmux new-session -d -s phase2_${TAG} && \
         tmux send-keys -t phase2_${TAG} \"${CMD//\"/\\\"}\" C-m && \
         echo launched phase2_${TAG} on \$(hostname)"
}

# ---- The three configurations ------------------------------------------------
#  A: small student (width=16), stronger losses, GT target
launch_remote "$NODE_A" "haze_a_small_tight" \
  "--width 16 --lambda-feat 0.05 --lambda-perc 0.05"

#  B: large student (width=32), stronger losses, GT target
launch_remote "$NODE_B" "haze_b_large_tight" \
  "--width 32 --lambda-feat 0.05 --lambda-perc 0.05"

#  C: large student (width=32), pseudo-target distillation, perceptual only
launch_remote "$NODE_C" "haze_c_large_pseudo" \
  "--width 32 --lambda-feat 0.00 --lambda-perc 0.05 --use-pseudo-as-target"

echo ""
echo "=== ALL 3 LAUNCHED ==="
echo "A: haze_a_small_tight  on $NODE_A  (width=16, GT, L1+0.05 L_feat+0.05 L_perc)"
echo "B: haze_b_large_tight  on $NODE_B  (width=32, GT, L1+0.05 L_feat+0.05 L_perc)"
echo "C: haze_c_large_pseudo on $NODE_C  (width=32, pseudo, L1+0.05 L_perc)"
echo ""
echo "Attach to any:  ssh teaching@<IP> ; tmux attach -t phase2_<tag>"
