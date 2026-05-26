# Parallel BMVC Strengthening Runs

Per-node assignment + live status for the four parallel BMVC-strengthening
tracks. Updated by hand as each job's state changes.

Snapshot: **2026-05-26 12:48 IST**.

## Topology

| Tier | Node | Role |
|---|---|---|
| Big-file hub | `tushar@10.8.48.242` (rudra) | All teacher + student .pt files at `/data1/tushar/bmvc/experiments/`. Cross-node ckpt rsync source. |
| Compute (active) | `teaching@172.18.40.119` | Outdoor student training (`haze_outdoor_b`) |
| Compute (active) | `teaching@172.18.40.133` | Lightweight CNN baselines (AOD-Net + FFA-Net) |
| Compute (active) | `teaching@172.18.40.113` | Sensitivity-driven distillation (`haze_b_sens`) |
| Compute (planned) | `teaching@172.18.40.139` | RTTS qualitative + NIQE/BRISQUE — **blocked on RTTS data** |

All four teaching nodes are RTX A5000 / 24 GB / `adu` conda env at
`/home/teaching/miniconda3/envs/adu`. rudra is RTX 5000 Ada / 32 GB —
currently used only as hub, not as a worker.

## Per-job registry

### Job 0 — Outdoor student (running)
- **Node:** `teaching@172.18.40.119`
- **Tag:** `haze_outdoor_b`
- **Code:** `scripts/launch_outdoor_student.sh` → `phase2_distill/train.py`
- **Data:** `/DATA/datasets/dehazing/RESIDE/OTS/` (313,947 hazy + 8,970 clean) + indoor soft labels reused, outdoor DeHamer teacher
- **Launched:** 2026-05-24 16:26 IST (tmux session `outdoor`)
- **Output:** `results/phase2_haze_outdoor_b.log`, `experiments/students/haze_outdoor_b/best.pt`
- **Realistic ETA:** ~36–48 h from launch (50K OTS pairs × 200 ep, ~3.5 it/s)
- **Status:** training in progress

### Job A — Lightweight baselines (running)
- **Node:** `teaching@172.18.40.133`
- **Tags:** `aodnet` + `ffanet` (eval JSONs)
- **Code:** `scripts/run_baselines.sh` → `evaluate/train_aodnet.py` (200 ep) → `evaluate/train_ffanet.py` (100 ep) → `evaluate/eval_baseline.py`
- **Data on node:** ITS-Train (4.3 GB) + SOTS-Test (419 MB) under `/DATA/datasets/dehazing/RESIDE/` (gdown'd this session)
- **Ckpts on node:** DeHamer indoor teacher (537 MB, pushed from local relay 2026-05-25 before the no-local rule landed)
- **Launched:** 2026-05-26 12:32 IST (nohup, no tmux on 133)
- **Output:**
  - `results/baselines.log`
  - `experiments/baselines/aodnet_indoor.pth`, `experiments/baselines/ffanet_indoor.pth`
  - `results/eval_baseline_{aodnet,ffanet}_{indoor,outdoor}.json` (×4)
- **Realistic ETA:** AOD-Net ~30 min train + 2 min eval × 2 splits, then FFA-Net ~3–5 h train + 5 min × 2 evals. Total ~4–6 h.
- **Status:** AOD-Net training, epoch 1/200 at 41% (5 it/s)

### Job B — Pareto + sensitivity heatmap + cross-domain figures (complete)
- **Node:** LOCAL (no GPU needed)
- **Code:** `scripts/gen_bmvc_figures.py`
- **Outputs:** `results/figures/{pareto_with_baselines,sensitivity_heatmap,cross_domain_bars}.png`
- **Status:** done 2026-05-25; committed in `9c31f66`.

### Job C — RTTS + no-reference (BLOCKED)
- **Node:** `teaching@172.18.40.139`
- **Tags:** `rtts_hazy`, `dehamer_indoor`, `haze_a_small_tight`, `haze_b_large_tight`, `haze_c_large_pseudo`
- **Code:** `scripts/run_rtts_all.sh` → `evaluate/eval_rtts.py` → `evaluate/fade.py`
- **Data on node:** DeHamer indoor teacher ✓, 3 student best.pt's ✓ (pushed rudra→139 2026-05-26 12:24)
- **Blocker:** RTTS dataset (4,322 real hazy images, ~280 MB) — placeholder GDrive ID in `scripts/download_rtts.sh` is unverified and may fail. Need either:
  - A verified RTTS GDrive ID, or
  - A direct download URL, or
  - Kaggle credentials + dataset slug
- **Realistic ETA once unblocked:** ~30 min (pyiqa install + ~5 models × ~1 min/run).
- **Status:** all dependencies in place, awaiting data source.

### Job D — Sensitivity-driven distillation (running)
- **Node:** `teaching@172.18.40.113`
- **Tag:** `haze_b_sens`
- **Code:** `scripts/launch_sensitivity_student.sh` → `phase2_distill/train_sensitivity_taps.py`
- **Data on node:** ITS-Train (4.3 GB) + SOTS-Test (419 MB) — gdown'd this session
- **Ckpts on node:** DeHamer indoor teacher (pushed 2026-05-25)
- **Launched:** 2026-05-26 12:43 IST (nohup, no tmux on 113)
- **Output:**
  - `results/phase2_haze_b_sens.log`, `results/phase2_haze_b_sens_status.txt`
  - `experiments/students/haze_b_sens/best.pt`, `epoch_*.pt`, `training_summary.json`
  - Then `results/eval_student_haze_b_sens_{indoor,outdoor}.json`
- **Realistic ETA:** ~16 h training (1748 steps/ep × 200 ep at ~6 it/s) + eval
- **Status:** ep 0 at 26% (6 it/s, loss 0.15–0.22)

## SSH quick-attach

```
# 119  (tmux 'outdoor')
sshpass -p ds123 ssh -t teaching@172.18.40.119 "tmux attach -t outdoor"

# 133  (no tmux — tail log)
sshpass -p ds123 ssh teaching@172.18.40.133 "tail -f dehazing-compression/results/baselines.log | tr '\\r' '\\n'"

# 113  (no tmux — tail log)
sshpass -p ds123 ssh teaching@172.18.40.113 "tail -f dehazing-compression/results/phase2_haze_b_sens.log | tr '\\r' '\\n'"

# rudra (hub — no jobs, file inspection only)
ssh tushar@10.8.48.242 "ls -lah /data1/tushar/bmvc/experiments/teachers/dehamer/ckpts/ /data1/tushar/bmvc/experiments/students/"
```

## After-completion handoff

When a job ends, pull only the small artefacts back to local (JSONs + small
PNGs). Big files stay on rudra:

```
# 133 baselines
sshpass -p ds123 rsync -avz "teaching@172.18.40.133:dehazing-compression/results/eval_baseline_*.json" results/
ssh tushar@10.8.48.242 'rsync -avz -e "ssh -i ~/.bmvc_to_teaching" \
    teaching@172.18.40.133:dehazing-compression/experiments/baselines/ \
    /data1/tushar/bmvc/experiments/baselines/'

# 113 sensitivity-distill
sshpass -p ds123 rsync -avz \
    "teaching@172.18.40.113:dehazing-compression/results/eval_student_haze_b_sens*.json" \
    "teaching@172.18.40.113:dehazing-compression/results/phase2_haze_b_sens.log" \
    results/
ssh tushar@10.8.48.242 'rsync -avz -e "ssh -i ~/.bmvc_to_teaching" \
    teaching@172.18.40.113:dehazing-compression/experiments/students/haze_b_sens/ \
    /data1/tushar/bmvc/experiments/students/haze_b_sens/'

# 119 outdoor student (when training done)
sshpass -p ds123 rsync -avz \
    "teaching@172.18.40.119:dehazing-compression/results/eval_student_haze_outdoor_b*.json" \
    results/
ssh tushar@10.8.48.242 'rsync -avz -e "ssh -i ~/.bmvc_to_teaching" \
    teaching@172.18.40.119:dehazing-compression/experiments/students/haze_outdoor_b/ \
    /data1/tushar/bmvc/experiments/students/haze_outdoor_b/'
```
