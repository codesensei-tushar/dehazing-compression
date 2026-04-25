# Active Training Runs

Live register of Phase-2 distillation runs across cluster nodes. Updated as we go.

## Teaching-node password
All teaching accounts share `ds123`. To SSH directly:

```
sshpass -p ds123 ssh -o StrictHostKeyChecking=accept-new teaching@<IP>
# or (local project convenience, uses .env credentials):
./gpu                         # currently points at 172.18.40.119
./gpu "tmux ls"               # one-shot remote command
```

Switch `./gpu` to another node: edit `.env` (`CLUSTER_HOST`) and re-source.

## Current Phase-2 parallel runs (started 2026-04-20)

| Tag | Node / IP | Student | Target | λ_feat | λ_perc | Session | Best ckpt (remote) |
|-----|-----------|---------|--------|-------:|-------:|---------|--------------------|
| `haze_s1` *(completed)* | `teaching@172.18.40.119` | 4.35M (w16) | GT | 0.01 | 0.00 | `phase2` | 29.78 dB / 0.9675 @ ep194 |
| `haze_a_small_tight` *(resumed)* | `teaching@172.18.40.103` | 4.35M (w16) | GT | 0.05 | 0.05 | nohup (no tmux) | resumed from ep104 best.pt |
| `haze_b_large_tight` *(completed)* | `teaching@172.18.40.103` | 17.11M (w32) | GT | 0.05 | 0.05 | nohup (no tmux) | 34.40 dB / 0.9865 @ best |
| `haze_c_large_pseudo` *(completed)* | `teaching@172.18.40.103` | 17.11M (w32) | Pseudo (teacher output) | 0.00 | 0.05 | `phase2_haze_c_large_pseudo` | 33.87 dB / 0.9834 @ best |

Each training run persists:
- `results/phase2_<tag>.log` — full stdout tee
- `results/phase2_<tag>_status.txt` — STARTED / DONE markers
- `experiments/students/<tag>/best.pt`, `epoch_{010,...}.pt`, `training_summary.json`

## SSH quick-access snippets

```bash
# Attach to the running tmux on any node
sshpass -p ds123 ssh -o StrictHostKeyChecking=accept-new teaching@172.18.40.119 -t "tmux attach -t phase2_haze_a_small_tight"
sshpass -p ds123 ssh -o StrictHostKeyChecking=accept-new teaching@172.18.40.115 -t "tmux attach -t phase2_haze_b_large_tight"
sshpass -p ds123 ssh -o StrictHostKeyChecking=accept-new teaching@172.18.40.103 -t "tmux attach -t phase2_haze_c_large_pseudo"

# Concurrent multi-node status poll (runs LOCALLY)
python scripts/phase2_multi_status.py --tail 5

# Tail one node's log
sshpass -p ds123 ssh -o StrictHostKeyChecking=accept-new teaching@172.18.40.115 \
    "tail -20 dehazing-compression/results/phase2_haze_b_large_tight.log | tr '\\r' '\\n' | tail -10"

# Pull best.pt + summary from a node when it finishes
sshpass -p ds123 rsync -az -e "ssh -o StrictHostKeyChecking=accept-new" \
    teaching@172.18.40.115:dehazing-compression/experiments/students/haze_b_large_tight/ \
    ./experiments/students/haze_b_large_tight/
```

## Kickoff procedure (for re-runs)

1. Sync latest code from local to node A: `./scripts/sync_to_cluster.sh`
2. SSH into node A: `./gpu`
3. `cd dehazing-compression && bash scripts/launch_phase2_multi.sh`

The multi-launcher will (a) bootstrap nodes B & C via scp from node A (idempotent), then (b) start tmux sessions on all three.

## History

- **2026-04-19 22:57 → 2026-04-20 07:27** — `haze_s1` full 200-epoch run on node A (172.18.40.119). Best 29.78 dB. Loss weighting + adapter were the bottleneck; this motivated the three-way split above.
- **2026-04-21 10:48 → 22:xx** — haze_a_small_tight on 119, haze_b_large_tight on 115, haze_c_large_pseudo on 103.
- **2026-04-22** — `haze_c_large_pseudo` finished @ **33.87 / 0.9833** (paper winner so far). `haze_b_large_tight` on 115 crashed during checkpoint save — disk full (1.1 TB used of 1.2 TB, other users). Relaunched B on 103 (137 GB free). A is still at epoch 106/200 with best 31.48 dB so far.
- **2026-04-26** — Node A host `172.18.40.119` became unusable (`/home` full, repo path missing). A resumed on `172.18.40.103` from `haze_a_small_tight/best.pt` (epoch 104, best 31.48 dB). Resume bug fixed for torch >=2.6 (`torch.load(..., weights_only=False)`).
