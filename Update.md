# Update Log

One-liners; most recent at top. Times approximate (local).

## 2026-04-18

- **15:35** · Pulled `results/dehamer_fp32_indoor.json` back locally via rsync. ITS.zip (4.3G) finished downloading in background on cluster.
- **15:30** · Ran Phase-1 FP32 baseline on cluster GPU 1 (A6000): **PSNR 36.58 / SSIM 0.9862** over 500 SOTS-indoor images in 94.2s. Latency 25.9 ms @256² (38.6 FPS), 86.4 ms @512² (11.6 FPS). Matches published numbers within 0.06 dB. → task #7 ✓
- **15:25** · Added `evaluate/benchmark_dehamer.py` (full SOTS-indoor FP32 eval + latency) and `evaluate/metrics.py` (psnr/ssim/latency_ms via CUDA events).
- **15:20** · Ran `scripts/verify_dehamer_sots.py` on cluster with real indoor ckpt: 35.13 dB / 0.9877 on `1400_1.png`. Confirms wrapper + checkpoint work end-to-end. → task #6 ✓
- **15:15** · Started ITS download in background from DeHamer GDrive mirror (id `1lE6FyHS…`, 4.5G). SOTS (`1IyZPih5…`, 435M) downloaded and unzipped: 500 indoor + 500 outdoor hazy/gt. → task #3 ✓
- **15:12** · RESIDE `bit.ly` links from spec are rotted (redirect to dead UTexas Box 404). Scraped Google Sites page, then found DeHamer README's own GDrive dataset mirror table — used that instead.
- **15:10** · Downloaded DeHamer pretrained checkpoints via `gdown`: indoor / outdoor / NH / dense, 4× 537MB, under `experiments/teachers/dehamer/ckpts/`. → task #4 ✓
- **15:05** · Ran `scripts/smoke_dehamer_local.py` on cluster — SMOKE OK, 132.45M params, CUDA forward works.
- **15:00** · Created cluster conda env **`dehaze`** by cloning `myenv` (fresh create failed: SSL cert error on `repo.anaconda.com`). Installed extras from pypi: `timm scikit-image einops gdown ptflops` (needed `--trusted-host pypi.org --trusted-host files.pythonhosted.org`).
- **14:58** · `./scripts/sync_to_cluster.sh` sent project to `cs671_user16@10.8.1.106:dehazing-compression/` (9MB, mostly submodule code). Cluster: 8× A6000.
- **14:55** · Connected to cluster via `./gpu` (sshpass + ssh). `/home/tushar/gpu` untouched per user; project-local `./gpu` sources `.env`.
- **14:50** · Local smoke test passes on GPU: `UNet_emb` loads (132.45M params), forward on 256² dummy image works, output shape matches input. `timm` installed locally.
- **14:47** · Wrote `models/teachers/dehamer.py` — sys.path hack for `third_party/dehamer/src`, loads UNet_emb, strips `module.` state_dict prefix, exposes `load_dehamer/preprocess/dehaze/count_params`. Normalization from DeHamer's `val_data.py`: mean (0.64, 0.60, 0.58), std (0.14, 0.15, 0.152). → task #5 ✓
- **14:45** · Created `scripts/make_dummy_data.py` → generated 8 synthetic hazy/clean pairs at 256² under `data/dummy/` via atmospheric scattering `I = J·t + A(1-t)`. → task #8 ✓
- **14:42** · Updated CLAUDE.md with hybrid rationale (local base for editing/smoke tests, cluster for compute).
- **14:40** · Added 3 submodules under `third_party/`: Dehamer (9.3M), Restormer (4.7M), NAFNet (11M). Staged `.gitmodules` + 3 pointers; not committed. → task #2 ✓
- **14:35** · Updated CLAUDE.md: replaced git-based sync workflow with rsync-over-sshpass. Added env names, credentials-in-.env, and sync+run loop.
- **14:33** · Created `./gpu` (project-local, sources `.env`) + `scripts/sync_to_cluster.sh` (rsync with safe excludes: `.env .git/ data/ experiments/ wandb/ __pycache__/ *.pt *.pth *.pkl`).
- **14:32** · Created `.env` (cluster creds: user, host, password, remote_dir) + `.env.example` template. `.env` already in `.gitignore`.
- **14:25** · Created 7 Week-1 tasks via TaskCreate; later added 2 more (#8 dummy data, #9 local smoke).
- **14:22** · Added compute-workflow section to CLAUDE.md (local vs cluster split).
- **14:20** · Scaffolded repo skeleton: `configs/ data/ models/{teachers,students,quantized} phase1_quantize/ phase2_distill/ evaluate/ scripts/ experiments/ results/ third_party/` + `__init__.py` stubs + `requirements.txt`. → task #1 ✓
- **14:15** · Added third-party submodule plan, GPU allocation table, W&B snippet, and `.gitignore` essentials to CLAUDE.md.
- **14:14** · Deleted all memory files per user; kept everything in project-local CLAUDE.md.
- **14:10** · Created initial memory files under `~/.claude-atlas/.../memory/` (user_researcher, project_paper_framing, project_two_phase) — reverted at user's request.
- **14:05** · Created `CLAUDE.md` with full project spec: teachers (DeHamer/Restormer), datasets (RESIDE/Rain13K/Snow100K/GoPro), Phase-1 PTQ workflow, Phase-2 distillation workflow with NAFNet-32 student, evaluation protocol, baselines, ablations, timeline, risks.
