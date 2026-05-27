# Parallel BMVC Strengthening Runs

Per-node assignment + live status for the four parallel BMVC-strengthening
tracks. Updated by hand as each job's state changes.

Snapshot: **2026-05-27 15:30 IST**.

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

### Job 0 — Outdoor student (running — relaunched 2026-05-27)
- **Node:** `teaching@172.18.40.119`
- **Tag:** `haze_outdoor_b`
- **Code:** `scripts/launch_outdoor_student.sh` → `phase2_distill/train.py`
- **Data:** `/DATA/datasets/dehazing/RESIDE/OTS/` (313,947 hazy + 8,970 clean) + outdoor DeHamer soft labels (50K, pregenerated)
- **Launched:** 2026-05-27 15:27 IST (nohup; first launch on 2026-05-24 crashed at validation — `SOTSEvalDataset` only globbed `*.png`, SOTS-outdoor hazy files are `.jpg` → 0 val pairs → fixed in `data/reside.py`)
- **Output:** `results/phase2_haze_outdoor_b.log`, `experiments/students/haze_outdoor_b/best.pt`
- **Realistic ETA:** ~36–48 h from relaunch
- **Status:** training in progress (ep 000, ~3-4 it/s)

### Job A — Lightweight baselines (running — relaunched 2026-05-27)
- **Node:** `teaching@172.18.40.133`
- **Tags:** `aodnet` + `ffanet` (eval JSONs)
- **Code:** `scripts/run_baselines.sh` → `evaluate/train_aodnet.py` (200 ep) → `evaluate/train_ffanet.py` (100 ep) → `evaluate/eval_baseline.py`
- **Data on node:** ITS-Train (4.3 GB) + SOTS-Test (419 MB) under `/DATA/datasets/dehazing/RESIDE/`
- **Ckpts on node:** DeHamer indoor teacher (pushed rudra→133 via LAN)
- **Launched:** relaunched 2026-05-27 15:29 IST after `scikit-image` install; original run on 2026-05-26 12:32 completed AOD-Net training (200 ep) then failed eval: `ModuleNotFoundError: No module named 'skimage'`
- **Output:**
  - `results/baselines.log`
  - `experiments/baselines/aodnet_indoor.pth` (already done), `experiments/baselines/ffanet_indoor.pth` (pending)
  - `results/eval_baseline_{aodnet,ffanet}_{indoor,outdoor}.json` (×4, pending)
- **Realistic ETA:** AOD-Net eval ~4 min, then FFA-Net ~3–5 h train + eval
- **Status:** AOD-Net eval running; FFA-Net train next

### Job B — Pareto + sensitivity heatmap + cross-domain figures (complete)
- **Node:** LOCAL (no GPU needed)
- **Code:** `scripts/gen_bmvc_figures.py`
- **Outputs:** `results/figures/{pareto_with_baselines,sensitivity_heatmap,cross_domain_bars}.png`
- **Status:** done 2026-05-25; committed in `9c31f66`.

### Job C — RTTS + no-reference (COMPLETE)
- **Node:** `teaching@172.18.40.139`
- **Tags:** `rtts_hazy`, `dehamer_indoor`, `haze_a_small_tight`, `haze_b_large_tight`, `haze_c_large_pseudo`
- **Code:** `scripts/run_rtts_all.sh` → `evaluate/eval_rtts.py` → `evaluate/fade.py`
- **Completed:** 2026-05-26 ~13:30 IST
- **Results (NIQE ↓ / BRISQUE ↓, 4322 real hazy images):**
  | Model | NIQE | BRISQUE |
  |---|---:|---:|
  | Hazy passthrough | 4.94 | 30.78 |
  | DeHamer teacher | **4.80** | **29.10** |
  | haze_a_small_tight (w16) | 12.87 | 101.74 |
  | haze_b_large_tight (w32) | 31.74 | 139.59 |
  | haze_c_large_pseudo (w32 pseudo) | 61.98 | 153.82 |
- **Finding:** students overtrained on synthetic ITS degrade severely on real haze (domain gap). Teacher barely improves over hazy. Framed as limitation in paper.
- **Eval JSONs:** `results/rtts_*.json` (5 files) committed locally.
- **Status:** DONE

### Job D — Sensitivity-driven distillation (COMPLETE)
- **Node:** `teaching@172.18.40.113`
- **Tag:** `haze_b_sens`
- **Code:** `scripts/launch_sensitivity_student.sh` → `phase2_distill/train_sensitivity_taps.py`
- **Completed:** 2026-05-27 03:08 IST (launched 2026-05-26 12:43 IST, ~14.4 h)
- **Results:**
  - Indoor SOTS: **34.555 dB / 0.9875 SSIM** (ep 199 best)
  - Cross-domain outdoor SOTS: **20.29 dB / 0.854 SSIM**
  - Latency: **23 ms / 43 FPS @ 256²**, 33 ms / 30 FPS @ 512²
  - Params: 17.11M
- **Interpretation:** +0.15 dB over `haze_b_large_tight` (34.40) — sensitivity-weighted taps marginally outperform uniform taps. Ties Phase 1 sensitivity analysis into Phase 2 method.
- **Artifacts:** eval JSONs in `results/`; `best.pt` pushed to `rudra:/data1/tushar/bmvc/experiments/students/haze_b_sens/`
- **Status:** DONE

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
