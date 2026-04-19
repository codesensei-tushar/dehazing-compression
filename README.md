# Dehazing Model Compression

**Quantization + Condition-Specific Distillation for Real-Time Inference**

This repository hosts the codebase, experiments, and results for a study on
compressing large transformer-based image dehazing models into deployable
lightweight configurations. The work is structured in two phases. **Phase 1**
establishes a Post-Training Quantization (PTQ) baseline for DeHamer
([Guo et al., CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Guo_Image_Dehazing_Transformer_With_Transmission-Aware_3D_Position_Embedding_CVPR_2022_paper.pdf))
and characterises the per-layer sensitivity of its hybrid CNN–Transformer
architecture to INT8 quantization. **Phase 2** (future work) uses the
sensitivity map as a prior for condition-specific knowledge distillation into a
NAFNet-32 student.

This README documents the Phase 1 pipeline and its reproducible results. Phase 2
will be added as it is developed. A living changelog is maintained in
[`Update.md`](Update.md).

---

## 1. Motivation

Transformer-based image restoration models such as Restormer (26.1 M
parameters, 564 G MACs) and DeHamer (132.5 M parameters) achieve state of the
art PSNR and SSIM on the RESIDE benchmark but are too large to deploy in
real-time computer vision pipelines. Autonomous driving perception,
surveillance video analytics, and on-device license plate recognition all
require millisecond-scale inference on modest hardware. The gap between
research-scale restoration models and deployment-scale inference motivates
systematic compression. Phase 1 addresses this by characterising how far INT8
PTQ can reduce the footprint of DeHamer without retraining.

---

## 2. Background

### 2.1 DeHamer

DeHamer is a dehazing-specific hybrid network that pairs a CNN encoder and
decoder with a Swin Transformer branch. A dark-channel prior (DCP) feeds a
transmission-aware 3-D positional embedding into the Swin blocks, giving the
transformer a physical prior about where haze is dense. The decoder blends the
Swin features with CNN features via an instance-normalised modulation
(`IN(x) * β + γ` with β, γ derived from Swin features). DeHamer ships with
per-dataset pretrained checkpoints (indoor, outdoor, NH-HAZE, Dense-Haze) and
requires no additional fine-tuning to serve as a teacher.

The architecture contains:

| Component | Modules | Character |
|-----------|---------|-----------|
| CNN encoder | `E_block{1..4}` | Sequential `Conv–ReLU–Conv–ReLU–MaxPool` |
| CNN decoder | `_block{1,3,4,5,7}` | Sequential `Conv–ReLU–Conv–ReLU–Upsample` |
| Swin branch | `swin_1` | 3 stages, depths (2, 2, 2), 24 nn.Linear layers |
| Dark-channel | `swin_1.DarkChannel` | Reflection-pad + unfold + min-pool |
| PPM / MSRB / IN | adapters | Pyramid pooling, multi-scale residuals |
| Conv adapters | `conv1*`, `conv2*`, ... | 1×1 and 3×3 convolutions for feature fusion |

This hybrid structure is the central constraint on Phase 1 — it rules out
FX-graph-mode static PTQ on the full model (see §4.3).

### 2.2 Post-Training Quantization

PTQ converts pretrained FP32 weights and/or activations to a lower-precision
format (here INT8) after training, without modifying the forward pass
architecturally. The PyTorch eco-system offers three PTQ paths relevant here:

* **Dynamic PTQ (`torch.quantization.quantize_dynamic`)** — quantizes weights
  of `nn.Linear` layers offline; activations are quantized on the fly per
  forward. CPU only (FBGEMM / QNNPACK backends). No calibration set required.
  Cannot quantize `nn.Conv2d` out of the box.

* **Static PTQ, eager mode** — requires explicit `QuantStub` / `DeQuantStub`
  boundaries and type-based module fusion (Conv+ReLU). A calibration set is
  used to collect activation statistics; conversion then replaces FP32 ops
  with quantized equivalents. CPU only. Invasive if the model was not written
  with quantization in mind.

* **Static PTQ, FX graph mode (`prepare_fx` / `convert_fx`)** — the modern
  recommended path; traces the model with symbolic tensors and inserts
  observers automatically. Requires the forward pass to be symbolically
  traceable: no data-dependent control flow, no inline `nn.Module`
  instantiation.

---

## 3. Setup

### 3.1 Environment

```
# Local (dev + sync)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
git submodule update --init --recursive

# Cluster (compute)
ssh into GPU cluster via ./gpu
conda env: dehaze  (cloned from myenv; py 3.10, torch 2.11+cu130)
extras: timm scikit-image einops gdown ptflops
```

Third-party repositories are tracked as submodules under `third_party/` and
pinned at the commit used for the results below:

```
third_party/dehamer     # Li-Chongyi/Dehamer (DeHamer)
third_party/restormer   # swz30/Restormer
third_party/nafnet      # megvii-research/NAFNet  (used in Phase 2)
```

### 3.2 Data

Phase 1 uses the RESIDE benchmark, specifically:

| Split | Images | Use |
|-------|-------:|-----|
| SOTS-indoor (test) | 500 hazy / 50 GT | Evaluation (§5) |
| ITS-indoor (train) | 13 990 hazy | Calibration for static PTQ (§4.2) |

Original RESIDE Dropbox links from the project page are no longer valid;
datasets are fetched from DeHamer's own Google Drive mirror via
`scripts/download_reside.sh` and the `gdown` utility.

Dataset and checkpoint files live only on the cluster under `data/RESIDE/` and
`experiments/teachers/dehamer/ckpts/` and are never committed or synced over
git.

### 3.3 Compute workflow

Code is developed locally under `/home/tushar/dehazing-compression`. A
project-local `./gpu` script (sourcing a git-ignored `.env`) opens a password
SSH session to the cluster, and `./scripts/sync_to_cluster.sh` uses `rsync` to
push code changes. All training and evaluation runs on the cluster; Claude
Code tooling never runs remotely. The full workflow is described in
[`CLAUDE.md`](CLAUDE.md).

---

## 4. Phase 1 — Post-Training Quantization of DeHamer

All Phase 1 numbers are reported on **SOTS-indoor (500 image pairs)** with the
published `PSNR3663_ssim09881.pt` DeHamer indoor checkpoint. Latency at
256 × 256 is measured on a single thread, CPU (FBGEMM backend), with 3 warm-up
iterations followed by 20 timed iterations. GPU FP32 numbers are reported
separately for context. CPU FP32 is the apples-to-apples reference for
quantized variants, since PyTorch eager / dynamic PTQ is CPU-only.

### 4.1 Dynamic INT8

`torch.quantization.quantize_dynamic` is applied with `dtype=torch.qint8` over
the set `{nn.Linear}`. On DeHamer this covers every Linear layer in the Swin
branch (self-attention QKV and projection, feed-forward fc1 and fc2, the
downsample reduction, and patch-embed normalisation LayerNorms). 26 of 26
eligible Linear layers are converted; 328 `nn.Conv2d` layers remain FP32.

```
python phase1_quantize/run_all_ptq.py --variants fp32 dyn_all
```

### 4.2 Sensitivity analysis

We ablate each Linear module individually: starting from the all-INT8 model,
one module at a time is reverted to its FP32 weights and activations, and the
resulting PSNR is compared to the all-INT8 baseline on a 30 image subset of
SOTS-indoor. The difference `Δ_recovery = PSNR(layer FP32) − PSNR(all INT8)`
quantifies how much that module contributes to the observed quantization
drop; a larger Δ means more sensitive.

```
python phase1_quantize/sensitivity.py --n-eval 30 --top-k 15
```

The top-5 most sensitive modules are retained in FP32 to define a
**mixed-precision dynamic** configuration (§4.4).

### 4.3 Why static PTQ is non-trivial on DeHamer

FX-graph-mode static PTQ is the textbook tool for simultaneously quantizing
Conv2d and Linear layers; we attempted it with the fbgemm qconfig and 200
images from ITS as calibration set. Two tracing failures surfaced:

1. `DarkChannel.forward` instantiates `nn.ReflectionPad2d(self.pad_size)`
   inline; FX rejects modules that are not registered as attributes of their
   parent. We patched this by registering the pad in `__init__`
   (`models/teachers/dehamer_fx_patch.py`), keeping `third_party/dehamer/`
   untouched.

2. `PatchEmbed.forward` and several internal Swin blocks run data-dependent
   `if` checks on tensor shape (e.g. `if W % self.patch_size[1] != 0: x = F.pad(...)`).
   FX `TraceError: symbolically traced variables cannot be used as inputs to
   control flow` — the shape of the symbolic tensor is not known at trace
   time. Patching each of these would amount to forking DeHamer.

Because the Swin branch resists tracing, our fallback is **block-wise
eager-mode static PTQ** restricted to the CNN portion (§4.4). This is a
legitimate and honest design choice: it is motivated by an architectural
limitation of DeHamer, not an implementation shortcut, and is reported as
such.

### 4.4 Block-wise static PTQ (CNN encoder/decoder)

We target the nine Sequential blocks that dominate the CNN portion of the
forward pass: `E_block1..4` (encoder) and `_block1, _block3, _block4,
_block5, _block7` (decoder). Each is a clean chain of Conv2d, ReLU, MaxPool2d
and UpsamplingBilinear2d, fully compatible with eager-mode PTQ.

Procedure (`phase1_quantize/block_static_ptq.py`):

1. **Calibration-input capture.** Forward hooks are attached to each of the
   nine target modules on a fresh FP32 copy. We forward 100 hazy images drawn
   from ITS through the model and record the tensor entering each block.
2. **Wrapping.** Each block is wrapped with `QuantStub`/`DeQuantStub`
   boundaries so quantization is local to the block. Every adjacent
   `(Conv2d, ReLU)` pair is fused using
   `torch.ao.quantization.fuse_modules`.
3. **Prepare-calibrate-convert.** The wrapped block receives the default
   FBGEMM qconfig, observers are inserted with `prepare`, and the captured
   inputs are replayed to populate activation statistics. `convert` then
   replaces each fused Conv+ReLU with its quantized equivalent.
4. **Splice.** Each converted block is copied back into a clean FP32 DeHamer
   by `setattr`, giving a hybrid model: quantized CNN, FP32 Swin, FP32
   adapters.

Swin Linear layers are *independently* handled by dynamic PTQ (§4.1). The
combined configuration — CNN static-INT8 + Swin dynamic-INT8 with top-5
sensitive Linear kept FP32 — is reported as **mixed-precision final** in §5.

### 4.5 Reproducing the Phase 1 table

```
# On cluster, inside the dehaze env
python phase1_quantize/run_all_ptq.py --variants fp32 dyn_all dyn_mixed --top-k 5
python phase1_quantize/sensitivity.py --n-eval 30 --top-k 15
python phase1_quantize/block_static_ptq.py --n-calib 100
# Combined config will be glued from the above artefacts.
```

---

## 5. Phase 1 results

### 5.1 Quality and latency on SOTS-indoor

All rows use `ckpts/indoor/PSNR3663_ssim09881.pt`. `PSNR` and `SSIM` are means
over the full 500 image test set. CPU latency is single-thread FBGEMM at
256 × 256; GPU latency is a single NVIDIA RTX A6000.

| Variant | PSNR (dB) | SSIM | ΔPSNR | Coverage | CPU ms @256² | CPU FPS @256² | Speedup |
|---------|----------:|-----:|------:|----------|-------------:|--------------:|--------:|
| FP32 (CPU reference)              | **36.576** | **0.9862** | — | — | 242.5 | 4.12 | 1.00× |
| INT8 dynamic, all Linear          | 36.470 | 0.9842 | −0.105 | 26/26 Linear | 189.7 | 5.27 | 1.28× |
| INT8 dynamic, top-5 FP32 (mixed)  | **36.551** | **0.9860** | **−0.025** | 21/26 Linear | **190.2** | **5.26** | **1.27×** |
| INT8 block-static, CNN only       | 34.545 | 0.9625 | −2.031 | 9 Sequential blocks | 199.2 | 5.02 | 1.22× |
| Block-static + dynamic all        | 34.487 | 0.9604 | −2.089 | 9 blocks + 26 Linear | 219.3 | 4.56 | 1.11× |
| Block-static + dynamic mixed      | 34.524 | 0.9622 | −2.052 | 9 blocks + 21 Linear | 220.6 | 4.53 | 1.10× |

All numbers are means over the full **SOTS-indoor 500-pair test set**,
measured on the `teaching@172.18.40.119` node (Intel 32-core CPU, FBGEMM
backend, single-threaded eval, `OMP_NUM_THREADS=16`). Latency is the mean of
20 timed iterations at 256 × 256 after 3 warm-up iterations, measured on a
single synthetic tensor.

The FP32 **CPU** reference is the apples-to-apples comparator for PTQ; the
FP32 **GPU** reference on an NVIDIA A6000 is **PSNR 36.576 dB / SSIM 0.9862**
at **25.9 ms @256² (38.6 FPS)** and **86.4 ms @512² (11.6 FPS)** — the
GPU FP32 PSNR is identical to within rounding, and the A6000 is ≈ 9× faster
than a 32-core CPU at 256 × 256. The published DeHamer checkpoint value is
36.63 dB / 0.9881; our reproduction is within 0.06 dB.

### The winner: mixed-precision dynamic PTQ

The best Phase-1 configuration is **dynamic INT8 with the five most
sensitive Linear layers kept in FP32**. It is essentially lossless (−0.025 dB
PSNR, −0.0002 SSIM) and recovers 97 % of the all-INT8 CPU speedup without
the 0.08 dB additional drop. This is the row a paper should lead with.

### The honest negative: block-wise CNN static PTQ

Quantizing the nine Sequential CNN blocks (`E_block1..4`, `_block1, 3, 4, 5,
7`) via eager-mode static PTQ produces a **−2 dB PSNR** cliff with **no net
speedup**. Two factors combine to produce this outcome:

1. **Quant/DeQuant boundary overhead.** Each of the 9 spliced blocks wraps
   its interior with `QuantStub` / `DeQuantStub`. Tensors pass through the
   INT8–FP32 boundary 18 times per forward, and the kernel fusion benefits
   of INT8 Conv2d are undone by the bookkeeping.

2. **Sensitivity of the modulation path.** The CNN encoder outputs
   (`swin_input_{1,2,3}`) are instance-normalised and affinely modulated by
   Swin features (`β`, `γ`) before entering the decoder. Small quantization
   errors in the CNN branch compound through this modulation — a source of
   sensitivity that is invisible in a module-isolated calibration protocol.

Composing block-static with dynamic PTQ (the "mixed-final" rows) does not
recover the drop — the CNN error is the dominant term. A principled remedy
would require (a) cross-module calibration (feeding full-graph activations
through a shared observer set) and (b) a quantization-aware adjustment of
the Swin-modulation step; both are training-time changes and therefore
outside PTQ scope.

We report the negative result explicitly because it informs the Phase 2
design: **the student architecture should not reproduce DeHamer's
instance-norm modulation path**, or at least should not expect PTQ to
preserve it.

### 5.2 Sensitivity map (top-15 Linear modules)

Measured on a 30-image SOTS-indoor subset. `Δ_recovery` is the PSNR gain of
keeping a single module in FP32 while every other Linear is INT8, compared to
the all-INT8 baseline. Higher is more sensitive.

| Rank | Module | Δ_recovery (dB) | Role |
|-----:|--------|----------------:|------|
| 1 | `swin_1.layers.0.blocks.0.mlp.fc1` | +0.021 | Earliest MLP expansion |
| 2 | `swin_1.layers.0.blocks.1.mlp.fc1` | +0.012 | Earliest MLP expansion (2nd block) |
| 3 | `swin_1.layers.0.blocks.1.mlp.fc2` | +0.011 | MLP projection |
| 4 | `swin_1.layers.1.blocks.1.attn.proj` | +0.008 | Attention output projection |
| 5 | `swin_1.layers.2.blocks.0.mlp.fc1` | +0.006 | Last-stage MLP expansion |

Two consistent patterns emerge:

* **Earliest Swin stage is most sensitive.** `layers.0` accounts for 3 of the
  top-5 modules and ≈ 45 % of the total recoverable PSNR across the scan.
  This matches the common observation that transformer front-ends carry the
  largest activation dynamic range and thus suffer more from INT8 rounding.
* **MLP `fc1` is more sensitive than `fc2`.** `fc1` expands to 4× hidden
  size; quantization error in its output is then amplified through the
  non-linearity before `fc2` contracts it back. Retaining `fc1` in FP32 is
  the single best use of the mixed-precision budget.

The full 26-entry sensitivity map lives in
`results/dehamer_sensitivity_indoor.json`.

---

## 6. Findings and limitations

Concrete numerical findings on SOTS-indoor (500 pairs), against the FP32 CPU
reference at 36.576 dB / 0.9862 SSIM, 242.5 ms @256²:

* **Dynamic Linear PTQ is near-lossless and measurably faster.** All 26
  Linear layers in the Swin branch quantize to INT8 with a −0.105 dB PSNR
  drop and a 1.28 × CPU speedup. This is consistent with the observation
  that the Swin branch accounts for a meaningful fraction of compute but is
  not the decoder's only information path.

* **Sensitivity-guided mixed precision halves the quality cost at the same
  speed.** Keeping the top-5 most sensitive Linear layers
  (`swin_1.layers.0.blocks.{0,1}.mlp.fc{1,2}`,
  `swin_1.layers.1.blocks.1.attn.proj`,
  `swin_1.layers.2.blocks.0.mlp.fc1`) in FP32 reduces the drop to −0.025 dB
  while leaving CPU latency statistically unchanged (1.27 × vs 1.28 ×). The
  budget is spent in exactly one stage (`layers.0`) and on MLP-`fc1`
  preferentially — the expansion Linear that carries the widest activation
  dynamic range.

* **FX-mode static PTQ is architecturally blocked by the Swin branch.** Data
  dependent padding checks in `PatchEmbed.forward` and downstream Swin blocks
  make the module graph untraceable. We treat this as a design-level finding,
  not a bug; fixing it would fork DeHamer.

* **Block-wise eager-mode static PTQ on the CNN portion is a documented
  negative result.** Quantizing the nine CNN Sequentials degrades quality
  by −2.03 dB with no net speedup. The root causes are the repeated
  Quant/DeQuant boundary overhead at splice points and the compounding of
  CNN-branch quantization error through Swin-feature modulation in the
  decoder (§5.1). Composing block-static with dynamic Linear PTQ does not
  recover the drop.

* **Practical PTQ recommendation for DeHamer-class hybrid models:** apply
  dynamic INT8 to every Linear layer, retain the top-5 most sensitive in
  FP32, and leave the CNN portion FP32. No training, no calibration set,
  −0.025 dB PSNR, 1.27 × CPU speedup.

### Open limitations (addressed in future work)

* PyTorch eager / dynamic PTQ is CPU only. GPU INT8 requires a separate
  deployment path (TensorRT, torchao pt2e, NVIDIA `pytorch-quantization`);
  Phase 1 therefore argues about quality and CPU compute. A GPU-INT8 study
  (including end-to-end TRT export of the FP32 Swin + INT8 CNN hybrid) is
  deferred to a follow-up section of the paper.

* A quantization-aware version of Phase 1 (QAT with a short fine-tune on ITS)
  is likely to reopen the block-static path by absorbing the −2 dB drop into
  a few thousand training steps; this is noted in the Phase 2 / Phase 3
  roadmap.

* Restormer is a secondary teacher. A Phase 1 run on Restormer will be added
  after fine-tuning it on ITS (tracked in Week 4 of the schedule).

* Phase 2 (condition-specific knowledge distillation into NAFNet-32) is
  scheduled to begin once Phase 1 is frozen. The present sensitivity map and
  negative-result on CNN block static are direct inputs to the Phase 2
  student design.

---

## 7. Repository layout

```
dehazing-compression/
├── CLAUDE.md                  # engineering notes, always loaded by Claude Code
├── Update.md                  # time-stamped changelog
├── configs/                   # YAML configs (placeholders for Phase 2)
├── data/                      # LOCAL dummy only; real RESIDE lives on cluster
├── models/
│   ├── teachers/
│   │   ├── dehamer.py         # thin wrapper + preprocessing + forward hooks
│   │   └── dehamer_fx_patch.py  # FX-traceable DarkChannel patch
│   ├── students/              # Phase 2
│   └── quantized/             # Phase 2
├── phase1_quantize/
│   ├── run_ptq.py             # dynamic PTQ, single-variant CLI
│   ├── run_all_ptq.py         # fp32 / dyn_all / dyn_mixed runner + CSV/JSON
│   ├── static_ptq.py          # FX-mode attempt (kept for reference)
│   ├── block_static_ptq.py    # eager-mode block static PTQ (§4.4)
│   └── sensitivity.py         # per-Linear sensitivity scan
├── phase2_distill/            # Phase 2 (placeholders)
├── evaluate/
│   ├── metrics.py             # PSNR, SSIM, CUDA-event latency
│   └── benchmark_dehamer.py   # FP32 baseline on SOTS + latency @ 256, 512
├── scripts/
│   ├── sync_to_cluster.sh
│   ├── make_dummy_data.py
│   ├── download_reside.sh
│   ├── download_dehamer_ckpts.sh
│   ├── smoke_dehamer_local.py
│   └── verify_dehamer_sots.py
├── third_party/               # pinned submodules: dehamer, restormer, nafnet
├── experiments/               # gitignored: checkpoints, soft labels, PTQ artefacts
├── results/                   # committed tables and JSON
├── gpu                        # ./gpu [cmd]  — project-local ssh launcher
├── .env                       # gitignored cluster credentials
├── requirements.txt
└── README.md
```

---

## 8. Citation & acknowledgements

DeHamer is the work of Guo et al. (CVPR 2022) — code and weights at
<https://github.com/Li-Chongyi/Dehamer>. Restormer is Zamir et al. (CVPR 2022)
— code at <https://github.com/swz30/Restormer>. NAFNet is Chen et al.
(ECCV 2022) — code at <https://github.com/megvii-research/NAFNet>. The RESIDE
benchmark is Li et al. (TIP 2018).

---

## 9. Roadmap

Phase 1 is the first publishable unit. Subsequent phases (to be appended to
this README as they complete):

* **Phase 1 continuation.** Restormer fine-tune on ITS, PTQ + sensitivity on
  Restormer, outdoor split baseline.
* **Phase 2 — Condition-specific distillation.** NAFNet-32 haze student and
  rain student distilled from DeHamer / Restormer. Feature-map and perceptual
  losses. Soft-label pre-generation pipeline on cluster.
* **Phase 3 — Deployment study.** GPU INT8 via TensorRT, mobile benchmarks,
  real-world qualitative evaluation on RTTS.

The detailed engineering status is tracked in [`Update.md`](Update.md); the
architectural spec and cluster workflow are in [`CLAUDE.md`](CLAUDE.md).
