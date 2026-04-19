# Update Log

One-liners; most recent at top. Times approximate (local).

## 2026-04-18 — Phase-1 LOCKED

- **18:30** · **Phase-1 numbers locked.** README §5 rewritten with the full 6-row table on SOTS-indoor 500 pairs: FP32 36.576 dB, INT8-dyn-all 36.470 (−0.105), INT8-dyn-mixed-top5 **36.551 (−0.025, 1.27× CPU speedup — paper headline)**, INT8-block-static 34.545 (−2.031), block+dyn_all 34.487 (−2.089), block+dyn_mixed 34.524 (−2.052). §6 expanded with recommendation + documented negative result on block-wise static PTQ of CNN blocks. → tasks #14, #15 ✓
- **18:15** · `PHASE1_DONE` on teaching@172.18.40.119 at 18:09:41 IST. Stage 1 took 22 min (17:14→17:37), Stage 2 took 32 min (17:37→18:09). Pulled 11 artefacts to local `results/`.

## 2026-04-18 — Week 2 (Phase-1 PTQ, in-session)

- **18:20** · Launched Phase-1 full suite in tmux `phase1` on teaching@172.18.40.119. Stage 1 (`run_all_ptq.py fp32 dyn_all dyn_mixed --top-k 5 --n-eval 0`) started. Stage 2 (`block_static_ptq.py --n-calib 100 --n-eval 0 --top-k 5`) will run sequentially. Status file `results/phase1_status.txt`; logs `phase1_runall.log`, `phase1_block_static.log`. Attach: `./gpu "tmux attach -t phase1"`.
- **18:15** · Bootstrapped new node: synced project (~9 MB), downloaded DeHamer indoor ckpt (512 MB) and SOTS (435 MB, unzipped → 500 indoor pairs) via gdown. Smoke test passed: 132.45M params, CUDA forward OK on A5000.
- **18:05** · Installed missing pip extras in `adu`: `scikit-image`, `einops`, `gdown` (torch/timm/PIL/yaml/tqdm already present).
- **17:55** · Switched active compute target to `teaching@172.18.40.119` (hostname `dslab`). 32 cores, 125 GB RAM, 1× RTX A5000, load 0.13, pre-existing `adu` conda env with torch 2.7.1+cu118. Parked 10.8.1.106 creds as `.env.10.8.1.106`.
- **17:50** · Wrote `scripts/monitor_nodes.py` (paramiko concurrent probe of all 49 credentials in `servers.csv`). Report: 10.8.1.106 at 100% CPU & load 183 on 64 cores (other users); ~30 `teaching@172.18.40.*` nodes at load < 1, each 32 cores + 1 GPU. Picker uses load/core + CPU + RAM + GPU% fitness score.
- **17:45** · Saved full cluster roster (49 rows) as `servers.csv` (gitignored). Added `scripts/launch_phase1_tmux.sh` (OMP_NUM_THREADS=16, tee'd logs, status file) and `scripts/phase1_status.sh` (remote status probe).

- **17:45** · Paused Phase-1 runs — cluster load avg 216 (another `cs671_*` user running multi-worker PyTorch job, 1300% CPU per worker). Our run slowed from 1 s/img to 35 s/img at image 70/500. Killed background run; will resume when cluster cools.
- **17:40** · Drafted paper-style README.md with full Phase-1 explanation (motivation, background on DeHamer + PTQ modes, setup, method §4.1–4.5, results table with placeholders, findings/limitations, repo layout). Sensitivity table already filled; quantization rows to fill after 500-image re-runs. → task #16 partial
- **17:30** · Wrote `phase1_quantize/block_static_ptq.py` for eager-mode static PTQ on 9 CNN Sequential blocks (`E_block1..4`, `_block{1,3,4,5,7}`). Captures per-block calibration inputs via forward hooks on FP32 model, wraps each block in `QuantStub`/`DeQuantStub`, fuses Conv+ReLU pairs, prepare→calibrate→convert→splice. Saves converted state to `experiments/ptq/dehamer_block_static_indoor.pt`. → task #15 coded, not yet executed
- **17:20** · Wrote `phase1_quantize/run_all_ptq.py` unifying FP32/dyn_all/dyn_mixed evaluation with JSON+CSV output. Mixed-precision variant reads top-K from `results/dehamer_sensitivity_indoor.json`. → task #14 coded; run paused


- **17:15** · Pulled sensitivity JSON back; saved to `results/dehamer_sensitivity_indoor.json`.
- **17:10** · Ran full sensitivity scan (`phase1_quantize/sensitivity.py`, 30 SOTS-indoor pairs, 26 Linear modules, ~9 min on cluster CPU). FP32 34.099 dB → all-INT8 34.050 dB (**Δ = -0.049 dB**). Top-5 most sensitive: `swin_1.layers.0.blocks.0.mlp.fc1` (+0.021), `.blocks.1.mlp.fc1` (+0.012), `.blocks.1.mlp.fc2` (+0.011), `layers.1.blocks.1.attn.proj` (+0.008), `layers.2.blocks.0.mlp.fc1` (+0.006). Findings: earliest stage most sensitive; MLP fc1 (4× expansion) > fc2 (projection). → task #13 ✓
- **16:45** · Rewrote sensitivity scan to use quantize-all-then-swap-one-FP32 (class-swap trick didn't work — `quantize_dynamic` matches subclasses).
- **16:35** · **FX static PTQ is blocked on DeHamer.** `prepare_fx` hits data-dependent control flow in `third_party/dehamer/src/swin.py` (runtime padding checks at L245/284/432/434). Patched DarkChannel via `models/teachers/dehamer_fx_patch.py`, but subsequent tracing errors mean patching the rest would be forking the model. Deferred. → task #12 paused
- **16:25** · Wrote `phase1_quantize/static_ptq.py` (FX graph mode static PTQ scaffold with fbgemm, ITS calibration). Kept for later (eager-mode or torchao pt2e path).



- **16:05** · Pulled `results/dehamer_int8_dynamic_indoor.json` back locally.
- **16:00** · Ran full `phase1_quantize/run_ptq.py --mode dynamic --max-pairs 100` on cluster CPU: FP32 PSNR 35.046 / INT8 PSNR 34.979 (**ΔPSNR -0.067 dB**); CPU synth-latency @256² 211→196 ms (**1.08× speedup**). Only 26/354 layers quantized (all Swin Linear; 328 Conv2d stay FP32). Result persisted to `results/dehamer_int8_dynamic_indoor.json`. → tasks #10, #11 ✓
- **15:55** · Wrote `phase1_quantize/run_ptq.py` supporting `--mode dynamic` (Linear→qint8); static mode stubbed for next step. Apples-to-apples CPU FP32 vs CPU INT8 (PyTorch dynamic PTQ is CPU-only). → task #10
- **15:50** · Phase-1 finding staged: on a conv-dominated transformer, dynamic PTQ barely moves latency. Motivates static PTQ that can also quantize Conv2d.

## 2026-04-18 — Week 1

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
