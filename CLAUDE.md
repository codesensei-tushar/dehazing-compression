# Dehazing Model Compression — Project Context

Quantization + Condition-Specific Distillation for Real-Time Inference.

## Target venues

Three tracks; pick by appetite and timeline. Detailed tier table + per-venue
fit assessment lives in `Checklist.md` §3.

### Journal track (original target list)
- Signal Processing: Image Communication (IEEEtran, 8 pages, double-column)
- The Visual Computer (Springer journal style, 6–8 pages)
- Optik

### Conference track
- **Tier-2 (sweet spot for current shape):** WACV / BMVC / ACCV — values practical ML + systems insight; matches the engineering contribution.
- **Tier-1 (reachable with extensions):** CVPR / ICCV / ECCV — needs Restormer-teacher track, SOTS-outdoor split, deployment study, sharper novelty framing.
- **Workshops (fast-feedback fallback):** CVPR-W / ECCV-W — strong fit for the PTQ + distillation engineering story without the deeper SOTA-comparison expansion.

### Preprint
- **arXiv** — recommended regardless of which of the above is chosen; do *not* arXiv until the same-GPU teacher latency is in place (v1 is permanent record).

## Contribution claim
Systematic compression of large dehazing transformers (DeHamer, Restormer) into condition-specific lightweight students (haze student, rain student), achieving near-teacher PSNR/SSIM at 5–10× fewer parameters and demonstrable real-time FPS on standard GPU hardware. Two sequential strategies:
1. Post-training quantization (PTQ) — fast baseline.
2. Condition-specific knowledge distillation — deployable, minimal-tradeoff final result.

Framing for reviewers: condition-specific angle + autonomous driving / license plate recognition / surveillance motivation. Target venues respond to application framing.

## Live project docs
- `README.md` — paper-facing narrative, methods, and result tables.
- `Update.md` — chronological engineering changelog.
- `RUNS.md` — active/completed run registry across nodes.
- `Checklist.md` — submission-readiness checklist (venue-wise status + blockers).

## Teacher models

### DeHamer (primary teacher for haze)
- Paper: CVPR 2022 (Guo et al., 2022), pages 5812–5820
- Repo: github.com/Li-Chongyi/Dehamer
- Params: 132.5M
- Arch: CNN encoder + ViT with 3D position embedding (transmission-aware) + feature modulation + CNN decoder
- Trained on dehazing only. 4 pretrained checkpoints ready to use (no fine-tune needed):
  - `ckpts/dense/PSNR1662_SSIM05602.pt` — dense haze
  - `ckpts/NH/PSNR2066_SSIM06844.pt` — non-homogeneous haze
  - `ckpts/indoor/PSNR3663_ssim09881.pt` — SOTS indoor (use this for haze student)
  - `ckpts/outdoor/PSNR3518_SSIM09860.pt` — SOTS outdoor (use this for haze student)

### Restormer (secondary teacher; rain teacher after fine-tune)
- Paper: arxiv.org/abs/2111.09881 (CVPR 2022, Oral)
- Repo: github.com/swz30/Restormer
- Params: 26.1M, MACs 564G (512×512)
- Arch: 4-level encoder-decoder, channel self-attention (MDTA), GDFN feed-forward
- Per-task checkpoints (deraining, deblurring, defocus deblurring, denoising). NO dehazing checkpoint — fine-tune the deraining checkpoint on RESIDE ITS (50K iters, patch 128, lr=2e-4, AdamW) → `restormer_dehaze.pth`.

Teacher selection decision: DeHamer as primary for haze (dehazing-specific, physics-based position embedding, ready checkpoints). Restormer as rain teacher and secondary comparison for haze.

## Datasets

### RESIDE (primary dehazing)
- Official: sites.google.com/view/reside-dehaze-datasets
- Links repo: github.com/Boyiliee/RESIDE-dataset-link
- ITS (indoor, 13,990 pairs, synthetic) — training haze student indoor
- OTS (outdoor, 313,950 pairs, synthetic) — final training runs to match published baselines
- SOTS indoor/outdoor (500 each, synthetic) — evaluation
- RTTS (4,332, real, no GT) — qualitative only
- Dense-Haze (55 pairs, real) — optional heavy-haze eval
- NH-HAZE (55 pairs, real) — optional non-homogeneous eval

Downloads: ITS https://bit.ly/3iwHmh0 · OTS https://bit.ly/3k8a0Gf · SOTS https://bit.ly/2XZH498 · SOTS Kaggle mirror https://www.kaggle.com/datasets/balraj98/synthetic-objective-testing-set-sots-reside

Start with ITS only (13,990) for initial experiments; OTS only for final training.

### Rain13K (rain student)
- Source: github.com/swz30/Restormer (Deraining section)
- 13,711 training pairs combined from Rain100H, Rain100L, Rain1200, Rain1400, Rain2800
- Test: Rain100H (100), Rain100L (100), Test100, Test1200, Test2800
- Patch-cropped to 128×128 for training

### Snow100K (optional — only if adding 3rd student)
snow100k.github.io

### GoPro (optional — PTQ calibration if pursuing joint-task BHNet angle)
seungjunnah.github.io/Datasets/gopro.html

## Phase 1 — PTQ

Goal: Quantize pretrained DeHamer and Restormer to INT8, measure PSNR/SSIM drop vs speedup. Standalone publishable (sensitivity study on dehazing transformers) + performance floor for Phase 2.

Workflow:
1. Prepare teachers (DeHamer pretrained; Restormer fine-tuned on ITS → `restormer_dehaze.pth`).
2. Calibration set: 500 RESIDE ITS images (not test set) to collect activation statistics for static PTQ.
3. Apply PTQ:
   - Dynamic: `torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)`
   - Static (`fbgemm`): `prepare` → run calibration forward passes → `convert`
   - For deployment-quality results, use NVIDIA `pytorch-quantization` (TensorRT PTQ toolkit).
4. Sensitivity analysis: per-layer ablation, revert one layer at a time to FP32, measure PSNR recovery. Attention projection + normalization layers typically most sensitive in transformers.
5. Mixed-precision: keep top 10–15% most sensitive layers in FP32, quantize rest to INT8.
6. Latency: warm up GPU, time 100 forward passes on single 256×256 image. Report GPU + batch size.

Expected: INT8 static → 1.5–2.5× latency speedup, 0.3–0.8 dB PSNR drop on transformers; mixed-precision cuts drop to <0.2 dB.

Results table rows: DeHamer FP32 / INT8 / INT8 Mixed; Restormer FP32 / INT8 / INT8 Mixed. Columns: Precision, PSNR Indoor, SSIM Indoor, Latency (ms/img), Params.

## Phase 2 — Condition-specific distillation

Do NOT design a new architecture. Use NAFNet (github.com/megvii-research/NAFNet) as student backbone. Inside BasicSR.

Student config: **NAFNet-32** (width=32, ~2.6M params, target >60 FPS @ 256×256). `enc_blks=[1,1,1,28]`, `middle_blk_num=1`, `dec_blks=[1,1,1,1]`. Random init; teacher guides. Avoid NAFNet-64 (~17M, too close to teacher, not useful); NAFNet-16 (~0.5M, >100 FPS) only for extreme-lightweight demo.

Distillation loss:
```
L_total = L_pixel (L1 to GT) + λ_feat * L_feat (L2, matched decoder features) + λ_perc * L_perceptual (VGG)
# Start: λ_feat=0.01, λ_perc=0 (add 0.05 only if output is blurry)
```
Feature matching: forward hooks on teacher & student decoder at matched resolution; 1×1 conv adapter on student side if channel dims differ. Teacher frozen (`eval()`, `requires_grad=False`).

Training protocol (haze student):
- Dataset: RESIDE ITS (13,990)
- Patch 128×128, batch 8, AdamW betas=(0.9, 0.9), lr 1e-3 cosine → 1e-6
- 200 epochs (or 400K iterations)
- Augmentation: random H/V flip, random 90° rotation
- Single GPU, RTX 3060+ (batch 8 fits 10GB)

Soft-label trick: run DeHamer offline on all ITS images, save dehazed outputs as PNGs → pseudo-clean targets. Avoids DeHamer in training loop (it's slow).

Rain student: identical procedure, teacher = `restormer_dehaze.pth`, data = Rain13K, eval on Rain100H/L.

Target: haze student PSNR within 1.5–2 dB of teacher. If gap larger → increase λ_feat to 0.05, add perceptual loss, train longer.

## Evaluation

Metrics:
- PSNR / SSIM via `skimage.metrics` — primary quality (synthetic: SOTS indoor/outdoor)
- Latency via `torch.cuda.Event` — mean of 100 passes
- FPS = 1000/latency_ms
- Params = `sum(p.numel() for p in model.parameters())/1e6`
- MACs via `fvcore` or `ptflops` @ 256×256
- FADE (no-reference) via github.com/thaolmk54/haze-evaluation — real-world qualitative on RTTS

Report latency + FPS at BOTH 256×256 AND 512×512 to show scaling.

Baselines in comparison table:
- AOD-Net (lightweight speed baseline)
- FFA-Net (CNN dehazing-specific)
- DeHamer FP32 (ceiling)
- Restormer fine-tuned FP32 (ceiling)
- DeHamer INT8 Mixed (Phase 1 result — distillation must beat this)
- One prior distillation-on-dehazing paper (e.g., KDDN if available)

Ablations (minimum two):
1. Feature loss contribution: L_pixel only vs L_pixel + L_feat.
2. Teacher choice: DeHamer teacher vs Restormer teacher for same student.

## Repo layout (planned, per README)
```
configs/        ptq_{dehamer,restormer}.yaml, distill_{haze,rain}.yaml
data/           reside.py, rain13k.py
models/
  teachers/     dehamer.py, restormer.py (thin wrappers + hooks)
  students/     nafnet_student.py (NAFNet-32 + adapter)
  quantized/    quant_utils.py (PTQ helpers, sensitivity scan, mixed-precision config)
phase1_quantize/ run_ptq.py, sensitivity.py
phase2_distill/  train.py, losses.py (L_pixel, L_feat, L_perceptual)
evaluate/        metrics.py (PSNR, SSIM, FADE, FPS), benchmark.py
scripts/         download_reside.sh, download_rain13k.sh, run_all_baselines.sh
experiments/     gitignored: checkpoints, logs
results/         committed: tables, figures
```

## Timeline (8 weeks)
1. Download RESIDE ITS/SOTS. Clone DeHamer, Restormer. Verify pretrained inference. Measure FP32 baselines.
2. Dynamic PTQ on both teachers. Measure PSNR + latency. Identify sensitive layers.
3. Static mixed-precision PTQ. Finish Phase 1 table. Start writing §3–4.
4. Fine-tune Restormer on ITS (50K iters, 12–18h on single GPU). Set up NAFNet-32. Begin haze distillation.
5–6. Finish haze student training. Evaluate SOTS. Download Rain13K. Start rain distillation.
7. Finish rain student. Run ablations. Full results table.
8. Finish remaining sections. Qualitative figures. Submit.

## Third-party dependencies (git submodules)

No pip packages for DeHamer / Restormer / NAFNet — pin via submodules so versions freeze at the commit used for the paper (reproducibility).

```
git submodule add https://github.com/Li-Chongyi/Dehamer    third_party/dehamer
git submodule add https://github.com/swz30/Restormer       third_party/restormer
git submodule add https://github.com/megvii-research/NAFNet third_party/nafnet
```

In `models/teachers/dehamer.py`, `models/teachers/restormer.py`, `models/students/nafnet_student.py`: `sys.path.insert(0, 'third_party/<name>')` then import the model class. Only the model definitions + checkpoints are consumed; training code never touches the third-party repos.

## Compute workflow — local dev + remote cluster

Two machines, strict separation:

**Local (`/home/tushar/dehazing-compression`)**
- Code editing, Claude Code sessions, unit-level smoke tests.
- Submodules cloned locally (code only, small — DeHamer 9.3M, Restormer 4.7M, NAFNet 11M).
- A tiny dummy dataset under `data/dummy/` (a handful of 256×256 hazy/clean pairs) for fast syntax/shape checks. Never the full RESIDE.
- No checkpoints, no full datasets, no training runs.

**Why keep a local base (hybrid rationale)**
- Claude can only read/edit files locally. Without a local copy, every iteration becomes a sync + cluster-roundtrip.
- Submodules are code-only (tens of MB total) — negligible disk cost for a large productivity win.
- Local dummy-data smoke tests catch shape/import bugs before wasting cluster GPU time.
- Cluster stays purely for the expensive work: full datasets, training, PTQ eval, long baselines.

**Remote GPU cluster (connected via project-local `./gpu` script)**
- Full datasets (RESIDE ITS/OTS/SOTS, Rain13K, GoPro) live ONLY here.
- Teacher checkpoints, all training runs, PTQ eval.
- Claude does NOT run on the cluster.
- Transport is rsync, not GitHub — local is source of truth, pushed directly to cluster folder `dehazing-compression/`.

**Credentials**
- Stored in project-local `.env` (gitignored). `.env.example` at project root shows required keys: `CLUSTER_USER`, `CLUSTER_HOST`, `CLUSTER_PASSWORD`, `CLUSTER_REMOTE_DIR`.
- `./gpu` and `./scripts/sync_to_cluster.sh` both source `.env` — do not hardcode credentials in any script.

**Sync + run loop**
```
local:  git add … && git commit -m "…"
        ./scripts/sync_to_cluster.sh                 # rsync project → cluster
local:  ./gpu "cd dehazing-compression && python phase1_quantize/run_ptq.py"
   or:  ./gpu                                        # interactive shell on cluster
```

Rsync excludes `.env`, `.git/`, `data/`, `experiments/`, `wandb/`, checkpoints. Pass `--delete` only when you want to mirror (wipes stale remote files). Data and checkpoints are never sent over rsync — they're downloaded directly on the cluster.

**Pulling results back**
```
sshpass -p "$CLUSTER_PASSWORD" rsync -avz \
  "$CLUSTER_USER@$CLUSTER_HOST:dehazing-compression/results/" ./results/
```

## Cluster Python environment

### Active compute target: `teaching@172.18.40.119` (dslab)
- **Reason:** `10.8.1.106` (cs671 host) is saturated by other users (load avg 180+). Teaching nodes at `172.18.40.*` are separate physical hosts with idle CPU + GPU.
- Env: pre-existing **`adu`** conda env at `/home/teaching/miniconda3/envs/adu/` (Python 3.11, torch 2.7.1 + cu118).
- GPU: 1× NVIDIA RTX A5000. CPU: 32 cores. RAM: 125 GB.
- No `conda` binary on PATH — invoke the env's Python directly: `/home/teaching/miniconda3/envs/adu/bin/python`.
- Extras installed: `scikit-image`, `einops`, `gdown` (pre-existing: `torch timm PIL tqdm yaml`).

### Credentials layout
- `.env` holds the **active** cluster's creds (gitignored).
- `.env.10.8.1.106` is a parked backup for the saturated primary cluster.
- Full cluster roster (30+ `teaching@172.18.40.*` nodes + cs671 users on 10.8.1.106) lives in `servers.csv` (gitignored).
- `scripts/monitor_nodes.py` probes the full roster and reports CPU / RAM / GPU load; use this to re-select when the active node gets loaded.

### Switching nodes
```bash
# Probe everything
python scripts/monitor_nodes.py --json results/cluster_status.json
# Pick best row, edit .env to point at its user/host/password
./gpu "echo connected as \$(whoami) on \$(hostname)"     # verify
./scripts/sync_to_cluster.sh
# Re-install extras if the new node's env lacks them.
```

### Legacy (10.8.1.106)
- Conda env: `dehaze` (cloned from `myenv`, py 3.10, torch 2.11 + cu130).
- Still populated with the full teacher-checkpoint set + ITS.zip + unzipped SOTS under `/usershome/cs671_user16/dehazing-compression/`.
- Switch back by restoring `.env.10.8.1.106` → `.env`.

## GPU cluster allocation

Free GPUs: 1 and 4 (~11GB used of 48GB). GPUs 0, 2, 3, 5, 6, 7 are occupied.

| Job | GPU(s) | Command |
|---|---|---|
| PTQ eval (no training) | 1 | `CUDA_VISIBLE_DEVICES=1 python phase1_quantize/run_ptq.py` |
| Student distillation | 4 | `CUDA_VISIBLE_DEVICES=4 python phase2_distill/train.py` |
| Multi-GPU distillation (only if needed) | 1+4 | `CUDA_VISIBLE_DEVICES=1,4 torchrun --nproc_per_node=2 train.py` |

NAFNet-32, batch 8, patch 128×128 fits comfortably in 11GB — student does NOT need multi-GPU. Reserve multi-GPU only if teacher is in the training loop.

Preferred pattern: pre-generate teacher soft labels offline on GPU 1 (DeHamer forward on all ITS images → save PNGs), then train student on GPU 4 reading from disk. Decouples teacher inference from student training.

## Experiment tracking — Weights & Biases

Use W&B free academic tier. In `phase2_distill/train.py`:
```python
import wandb
wandb.init(project="dehazing-compression", config=cfg)
wandb.log({"psnr": val_psnr, "loss": loss})
```
Run comparisons + shareable links > manual log files when juggling experiments across the cluster.

## .gitignore essentials
```
experiments/
data/
*.pth
*.pt
*.pkl
__pycache__/
wandb/
```
Commit: configs, code, `results/` tables. Never commit checkpoints or datasets.

## Risks & mitigations
- Restormer fine-tune too slow → ITS only, 50K iters.
- Student PSNR gap >2 dB → λ_feat 0.05, add perceptual loss, +50–100 epochs.
- OTS too large → train on ITS only (most published works do).
- DeHamer inference too slow → generate soft labels offline, train against saved PNGs.
- PTQ PSNR drop >1 dB → mixed precision; frame sensitivity map as contribution.
- Novelty concerns → lean on condition-specific angle + autonomous driving framing.
