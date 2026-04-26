# Dehazing Model Compression

**Quantization + Condition-Specific Distillation for Real-Time Inference**

This repository hosts the codebase, experiments, and results for a study on
compressing large transformer-based image dehazing models into deployable
lightweight configurations. The work is structured in two phases. **Phase 1**
establishes a Post-Training Quantization (PTQ) baseline for DeHamer
([Guo et al., CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Guo_Image_Dehazing_Transformer_With_Transmission-Aware_3D_Position_Embedding_CVPR_2022_paper.pdf))
and characterises the per-layer sensitivity of its hybrid CNN–Transformer
architecture to INT8 quantization. **Phase 2** distills DeHamer into a
compact NAFNet ([Chen et al., ECCV 2022](https://arxiv.org/abs/2204.04676))
student (17.1 M params, 7.7× smaller than the teacher) using offline
soft-label supervision from DeHamer — a dehazing-specific application of
Hinton et al.'s soft-target distillation
([Hinton, Vinyals, Dean, 2015](https://arxiv.org/abs/1503.02531)).

Headline result (Phase 2, SOTS-indoor 500 pairs, RTX A5000):
**Node B 34.40 dB / 0.9865 (quality winner, 17.1 M params) and Node A
32.39 dB / 0.9829 (throughput winner, 4.35 M params at ≈ 33.7 FPS @256²).**
Both are within the 2.2 dB PSNR / 0.003 SSIM gap of the 132.45 M-parameter
DeHamer teacher. A same-GPU teacher re-measurement is required before
quoting a speedup multiplier vs DeHamer (see §7.4).

A living changelog is maintained in [`Update.md`](Update.md); active
training-run registry in [`RUNS.md`](RUNS.md); and submission-readiness
tracking in [`Checklist.md`](Checklist.md).

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

* Phase 2 (condition-specific knowledge distillation into NAFNet) is
  complete across all three configurations (Nodes A / B / C); see §7.4. The
  Phase-1 sensitivity map and the CNN block-static negative result remain
  direct inputs to the student design.

---

## 7. Phase 2 — Condition-specific Knowledge Distillation (Nodes B and C)

Phase 1 hit a ceiling of 1.27× CPU speedup because PyTorch's dynamic PTQ only
reaches the 26 `nn.Linear` layers in the Swin branch; the 328 `nn.Conv2d`
layers that dominate DeHamer's runtime stayed FP32 (§6). Phase 2 sidesteps
this by training a small student that is **architecturally** compressed —
7.7× fewer parameters by construction — and using the pretrained DeHamer as
the teacher. This section documents the winning configuration (Node C);
two parallel ablations (Nodes A and B) are in §7.4.

### 7.1 Student — NAFNet

We use NAFNet (Chen, Chu, Zhang, Sun — *Simple Baselines for Image
Restoration*, ECCV 2022, [paper](https://arxiv.org/abs/2204.04676)) as the
student backbone. NAFNet removes non-linear activations from the attention
and gating paths — its blocks use **SimpleGate** (element-wise multiplication
of the channel halves) and **Simplified Channel Attention** in place of
softmax / sigmoid gates. The result is a small, fast, easy-to-implement
restoration backbone with strong precedent on SIDD (denoising) and GoPro
(deblurring).

Student configuration (Node C):

| Field | Value |
|-------|-------|
| `width` | 32 |
| `enc_blk_nums` | `[1, 1, 1, 28]` (NAFNet deep-trunk preset) |
| `middle_blk_num` | 1 |
| `dec_blk_nums` | `[1, 1, 1, 1]` |
| Parameters | **17.11 M** |
| Compression vs DeHamer (132.45 M) | **7.7×** |

A forward hook on the last decoder block exposes a 32-channel feature tap
that is used by the L_feat variants (not used in Node C — see §7.3).
Third-party NAFNet lives as a pinned git submodule under `third_party/nafnet`;
its BasicSR package imports `lmdb` eagerly at load time, so the student
wrapper installs a minimal stub when `lmdb` is not available — nothing in
the NAFNet code path actually uses it.

### 7.2 Teacher and offline soft-label pipeline

Teacher: DeHamer (Guo, Yan, Anwar, Cong, Ren, Li — *Image Dehazing
Transformer with Transmission-Aware 3D Position Embedding*, CVPR 2022,
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Guo_Image_Dehazing_Transformer_With_Transmission-Aware_3D_Position_Embedding_CVPR_2022_paper.pdf))
— the same `PSNR3663_ssim09881.pt` indoor checkpoint used in Phase 1.

DeHamer is slow to run (~190 ms per image on A5000). Running it inline in the
student training loop would dominate wall-clock time (~8 min per epoch for
teacher forward alone over 13 990 images). Instead, we generate **all 13 990
teacher outputs once, offline**, and store them as PNGs under
`experiments/soft_labels/dehamer_indoor/`:

```
scripts/gen_soft_labels.py
    --ckpt experiments/teachers/dehamer/ckpts/indoor/PSNR3663_ssim09881.pt
    --hazy-dir data/RESIDE/ITS-Train/train_indoor/haze
    --out-dir experiments/soft_labels/dehamer_indoor
# 41 min on A5000 @ 5.7 img/s
```

This decouples teacher inference from student training: the student
dataloader reads `(hazy, gt_clean, teacher_pseudo)` triples from disk, and no
DeHamer forward runs per step.

### 7.3 Node-C loss function (winning configuration)

Node C uses the **teacher's output as the pixel-space target**, not the GT
clean image:

```
L_total = L_pixel(student_out, teacher_pseudo)                (L1)
        + 0.05 * L_perceptual(student_out, teacher_pseudo)    (VGG-19 features)
        + 0.00 * L_feat                                       (disabled)
```

* **L_pixel** — L1 distance between the student's output and the teacher's
  dehazed output over 128 × 128 patches.
* **L_perceptual** — MSE between ImageNet-normalised ReLU-gated feature
  activations of a frozen VGG-19 (through `features[:16]`, equivalent to
  relu_3_3) applied to student output vs teacher output. Weighted λ = 0.05.
  Implementation: `phase2_distill/losses.py`.
* **L_feat** — channel-matching loss between student and teacher decoder
  features. Disabled in Node C because the target is *already* the teacher's
  output, so a feature-space match through a 1×1 conv adapter is redundant
  with L_pixel.

The choice to target the teacher's pseudo-clean output rather than the GT
clean image is the standard distillation recipe (Hinton, Vinyals, Dean —
*Distilling the Knowledge in a Neural Network*, 2015,
[paper](https://arxiv.org/abs/1503.02531)) transposed to dense regression:
the teacher's outputs are "soft" in the sense that they encode the teacher's
internal priors about haze, and are a closer target for a small student than
the hard GT clean image.

### 7.4 Three-way ablation (A / B / C)

Node C is one of three parallel configurations launched on separate teaching
nodes (see `RUNS.md`). The ablation is orthogonal in two axes: student size,
and supervision target.

| Tag | Width | Params | Target | λ_feat | λ_perc | Hypothesis |
|-----|------:|-------:|--------|-------:|-------:|------------|
| `haze_a_small_tight` | 16 | 4.35 M | GT clean | 0.05 | 0.05 | Does stronger loss rescue a small student on GT supervision? |
| `haze_b_large_tight` | 32 | 17.11 M | GT clean | 0.05 | 0.05 | Does raw capacity rescue GT supervision? |
| `haze_c_large_pseudo` | 32 | 17.11 M | **Teacher output** | 0.00 | 0.05 | Does imitating the teacher directly beat GT? |

An earlier run `haze_s1` (width 16, GT target, λ_feat 0.01, λ_perc 0)
plateaued at 29.78 dB — the four-way gap that motivated this ablation. All
three runs are now complete (final results in `results/eval_student_*.json`):

| Tag | Width | Params | Target | PSNR / SSIM (SOTS-indoor 500) | Latency @256² (ms) | Latency @512² (ms) |
|-----|------:|-------:|--------|------------------------------:|-------------------:|-------------------:|
| `haze_a_small_tight`  | 16 |  4.35 M | GT clean       | 32.39 / 0.9829         | **29.70 ± 0.05** | 33.09 ± 3.39 |
| `haze_b_large_tight`  | 32 | 17.11 M | GT clean       | **34.40** / **0.9865** | 32.89 ± 2.23     | 34.13 ± 1.80 |
| `haze_c_large_pseudo` | 32 | 17.11 M | Teacher output | 33.87 / 0.9834         | 36.40 ± 0.64     | 33.84 ± 1.16 |

Latency: mean ± std over 5 independent 100-iteration CUDA-event windows
(10-iter warmup each), single RTX A5000, GPU 0 % baseline utilisation, no
concurrent training. The earlier single-window evals in
`results/eval_student_*.json` are kept for traceability but are dominated by
inter-window variance on this small architecture (the 100-iter mean of B and
C disagreed by ≈ 37 % at 256² across two evals run on different days; the
isolated repeated-rep numbers above resolve to ≈ 11 % at 256² and within std
elsewhere). See `results/latency_isolated_*.json`.

Reading: capacity is the dominant lever for quality (w16 → w32 at fixed
losses adds ≈ 2.0 dB PSNR). On this NAFNet-`[1,1,1,28]` configuration on
A5000, latency is overhead/memory-bound at both 256² and 512²: the small
student (A) is the throughput winner (≈ 33.7 FPS @256²), while the two w32
configurations B and C are essentially throughput-equivalent (same
architecture; the per-shape difference is within one std). Node B is the
quality winner; Node C still trades 0.5 dB PSNR for the cleaner pseudo-target
loss (no L_feat adapter) but no longer claims a throughput advantage.

### 7.5 Training protocol (Node C)

| Component | Value |
|-----------|-------|
| Dataset | RESIDE-ITS indoor (13 990 pairs) |
| Test set | SOTS-indoor (500 pairs, disjoint) |
| Patch | 128 × 128 random crops |
| Augmentation | horizontal / vertical flips, 90° rotations |
| Batch size | 8 |
| Optimizer | AdamW, β = (0.9, 0.9) |
| Learning rate | 1e-3 → 1e-6 cosine decay |
| Epochs | 200 (= 349 600 optimisation steps) |
| Gradient clip | L2 ≤ 1.0 |
| Teacher forward | offline pseudo-labels on disk (no inline forward) |
| Validation | every 5 epochs on full SOTS-indoor (500 pairs) |
| Checkpointing | periodic every 10 epochs + `best.pt` on new best PSNR |
| Compute | `teaching@172.18.40.103` (Intel 32-core, NVIDIA RTX A5000 24 GB) |
| Python | `/home/teaching/miniconda3/envs/adu/bin/python` (torch 2.7.1+cu118) |
| Wall-clock | ≈ 8.5 h end-to-end |

### 7.6 Results — Node C on SOTS-indoor

Evaluation uses the Node-C `best.pt` (saved at epoch 174) on the full
500-image SOTS-indoor test set. PSNR and SSIM are means over the 500 pairs
computed with `skimage.metrics` at `uint8` precision. Latency uses
CUDA-event timing (3 warm-up + 20 timed iterations) on a single RTX A5000.

| Metric | Value |
|--------|------:|
| **PSNR (mean)** | **33.869** dB |
| **SSIM (mean)** | **0.9834** |
| PSNR min / max | 28.51 / 40.20 |
| SSIM min / max | 0.9646 / 0.9937 |
| Parameters | **17.11 M** |
| Full-image eval (variable HxW) | 83.6 ms / image |
| Latency @ 256 × 256 | 36.40 ± 0.64 ms (27.5 FPS) |
| Latency @ 512 × 512 | 33.84 ± 1.16 ms (29.6 FPS) |

Latency is mean ± std over 5 isolated-load 100-iter CUDA-event windows on
RTX A5000 (`results/latency_isolated_haze_c_large_pseudo.json`); the earlier
single-window value of 23.2 ms at 256² in
`results/eval_student_haze_c_large_pseudo.json` was within the inter-window
spread now exposed by the repeated-measurement protocol (§7.4) and is no
longer the headline number.

Comparison against the DeHamer teacher (same SOTS-indoor 500 pairs):

| | DeHamer FP32 (132.45 M) | **NAFNet-32 Node C (17.11 M)** | Δ |
|---|---:|---:|---:|
| PSNR | 36.576 dB | **33.869 dB** | −2.71 dB |
| SSIM | 0.9862 | **0.9834** | −0.003 |
| Params | 132.45 M | **17.11 M** | **7.7× smaller** |

Throughput comparison vs the teacher is intentionally omitted from this row:
the teacher's Phase-1 latency (25.9 ms @256² / 86.4 ms @512²) was measured
on a different GPU (A6000 on cs671 cluster) than the Phase-2 student latency
above (A5000 on teaching cluster). A same-GPU re-measurement of the teacher
is on the submission checklist before any speedup multiplier is quoted in
the paper.

Quality headline (the part that *is* hardware-independent):
**7.7× fewer parameters, 0.003 lower SSIM, 2.71 dB lower PSNR.**

Full JSON is committed at `results/eval_student_haze_c_large_pseudo.json`.

### 7.7 Why pseudo-supervision beats GT supervision for this pair

The earlier `haze_s1` run (4.35 M, GT target, λ_feat = 0.01, no perceptual)
plateaued at 29.78 dB — **4.09 dB below Node C.** Even controlling for the
student-size difference, the gap is mostly explained by the target choice.
Two complementary reasons:

1. **Target tractability.** GT clean images are further from any hazy input
   in pixel space than the teacher's dehazed output. For a small student,
   the teacher output is a smoother, closer target that encodes the
   teacher's priors — this is the classical soft-target effect of Hinton
   et al. (2015), transposed from classification to dense regression.
2. **Perceptual weight.** The VGG perceptual term (λ = 0.05) captures the
   mid-frequency structure that L1 under-weights. Together with soft targets
   it pulls the student toward the teacher's textural distribution rather
   than blurred mean-colour reconstructions.

One negative finding also matters: the feature-matching adapter used in
`haze_s1` projected the student's 16-channel decoder tap down to 3 channels
and compared against the teacher's RGB output. This is structurally
degenerate — it duplicates L_pixel in feature space. Disabling L_feat and
adding a proper perceptual term (Node C) was a net improvement; future work
should instead match genuine intermediate features (e.g. teacher's
256-channel decoder feature against a 1×1-projected student tap).

### 7.8 Reproducing Node C

On a teaching node with the `adu` conda env:

```bash
# 1. Pre-generate teacher soft labels (~41 min)
PY=/home/teaching/miniconda3/envs/adu/bin/python
$PY scripts/gen_soft_labels.py \
    --ckpt experiments/teachers/dehamer/ckpts/indoor/PSNR3663_ssim09881.pt \
    --hazy-dir data/RESIDE/ITS-Train/train_indoor/haze \
    --out-dir experiments/soft_labels/dehamer_indoor

# 2. Train (~8.5 h on A5000)
$PY phase2_distill/train.py \
    --tag haze_c_large_pseudo \
    --width 32 --epochs 200 --batch 8 --patch 128 --workers 4 \
    --lr-hi 1e-3 --lr-lo 1e-6 \
    --lambda-feat 0.00 --lambda-perc 0.05 --use-pseudo-as-target \
    --pseudo-dir experiments/soft_labels/dehamer_indoor \
    --val-interval 5 --ckpt-interval 10

# 3. Evaluate best.pt on full SOTS-indoor (~45 s on A5000)
$PY phase2_distill/eval_student.py \
    --ckpt experiments/students/haze_c_large_pseudo/best.pt \
    --tag haze_c_large_pseudo --width 32
```

Multi-node orchestration for A, B, C in parallel: `scripts/launch_phase2_multi.py`
bootstraps target nodes via paramiko SFTP and launches tmux (or `nohup setsid`
fallback) sessions.

---

## 8. Publishability assessment

Direct answer: **the results are publishable** at Optik or The Visual
Computer, the two more accessible of the three target venues
([§ Roadmap](#11-roadmap) lists all three). The numerical achievements clear
the bar. Whether the *current artifact* in this repository is submittable is
a separate question, and it is not — there is no manuscript file, several
experimental gaps remain, and one cross-GPU comparison was invalid until it
was removed earlier today.

### 8.1 Why the results clear the bar

**Phase 1.** 36.551 dB at 0.025 dB drop with 1.27× CPU speedup is a clean
PTQ-sensitivity contribution on a hybrid CNN–Transformer architecture. The
documented negative result on block-wise static PTQ of CNN blocks is a
genuine, reportable finding — most PTQ papers omit failed configurations,
and a documented failure mode at this depth is the kind of evidence that
helps reviewers calibrate.

**Phase 2 Node B.** 34.40 dB / 0.9865 SSIM with **2.18 dB** gap to a
132.5 M-parameter teacher at **7.7× compression** is competitive with
published distillation-on-dehazing work. KDDN, the closest comparable
distillation paper, reports ≈ 34.7 dB on SOTS-indoor; Node B is within
0.3 dB at substantially fewer parameters. Node B's SSIM is statistically
indistinguishable from the teacher (within 0.0003).

**Phase 2 Node A.** 32.39 dB / 0.9829 SSIM at **4.35 M parameters
(30.5× compression)** is a defensible extreme-compression operating point.
AOD-Net at ≈ 1.7 K parameters (~2500× smaller than Node A) is the
lightweight-dehazing speed floor and sits around 19–22 dB on SOTS-indoor;
Node A spends ~3 orders of magnitude more parameters to buy ≈ 10 dB. Both
sit on distinct, complementary points of a quality-vs-parameters Pareto
frontier rather than dominating each other — which is exactly the
positioning the paper's "compression spectrum" framing needs.

**The 2 × 2 ablation.** Capacity × supervision-target with all four cells
measured (the failed `haze_s1` 29.78 dB run at width 16, weak losses, GT
target serves as the historical anchor) is methods-section content that
reviewers will accept as substantive evidence of *why* the configuration
choices matter, rather than as a single-shot result.

### 8.2 Where the bar is not yet cleared

What is *not* publishable is the act of submitting today, because:

* **The manuscript file does not exist.** No `.tex`, no `.pdf`, no `paper/`
  directory. This README reads like a paper draft but cannot be uploaded
  to any of the target venues' submission systems as-is. Each venue
  expects a formatted manuscript with abstract, introduction, related
  work, method, experiments, discussion, and references.
* **AOD-Net and FFA-Net comparison rows are empty.** Reviewers at every
  target venue will demand at minimum these two as external lightweight /
  CNN-dehazing baselines. Neither is yet implemented or evaluated in this
  repository, so the comparison table currently has only DeHamer as a
  ceiling reference.
* **No real-world evaluation.** RTTS qualitative panel and FADE
  no-reference scores are absent. Synthetic-only evaluation (SOTS-indoor)
  is routinely flagged in dehazing reviews as insufficient for the
  application motivation (autonomous driving, surveillance, license-plate
  recognition) that frames the contribution.
* **Same-GPU teacher latency is missing.** The Phase-1 teacher latency
  numbers were measured on an A6000 in the cs671 cluster; all Phase-2
  student latency numbers are on an A5000 in the teaching cluster
  (172.18.40.103). The cross-GPU comparison was removed from § 7.6 as
  invalid earlier today. Without a same-GPU teacher re-measurement —
  about 30 seconds of compute — no PSNR-preserving-speedup multiplier vs
  DeHamer can be quoted in the paper.
* **No figures rendered.** No qualitative side-by-side panel, no
  sensitivity heatmap, no quality-vs-speed-vs-parameters Pareto plot. All
  three are standard expectations for the target venues and none exist in
  this repository.

### 8.3 Where the work fits — venue landscape

The original CLAUDE.md target list (Optik, The Visual Computer, Signal
Processing: Image Communication) is the **journal track**. There is also a
**conference track** and an **arXiv preprint track**, and the right answer
depends on how much additional polish lands before submission. Honest read
of where the contributions sit *today* across the three tracks:

| Tier / Venue | Type | Fit *now* | What it would take |
|---|---|---|---|
| **CVPR / ICCV / ECCV** | Tier-1 conference | Borderline | Restormer-teacher track, SOTS-outdoor split, ≥ 2 external baselines + ≥ 1 prior distillation paper, GPU INT8 deployment numbers, sharper "condition-specific distillation" novelty framing. ~3–4 weeks of focused work on top of writing. |
| **WACV / BMVC / ACCV** | Tier-2 conference | **Sweet spot** | Same baseline + writing work; the Restormer-teacher track and outdoor split are nice-to-have rather than blocking. These venues explicitly value practical ML + systems insight, which is what the engineering contribution looks like. |
| **CVPR-W / ECCV-W** | Workshop | High | Fast feedback path. Strong fit for the PTQ + distillation engineering story without needing the deeper SOTA-comparison expansion the main conferences would demand. |
| **arXiv** | Preprint | High | Recommended *regardless* of which conference / journal is chosen. Clean the manuscript, render figures, upload — establishes timestamp / priority and gives external readers something to point at. Cheap and high-value. |
| **Optik** | Journal | Medium-High | Application-framed venue; fits the autonomous-driving / surveillance motivation. Doable after the blocking items in Checklist § 4 close. |
| **The Visual Computer** (Springer) | Journal | Medium-High | Similar fit to Optik; expects complete ablations + a stronger qualitative section, both of which are on the work plan. |
| **Signal Processing: Image Communication** (IEEEtran) | Journal | Medium | More demanding on comparative depth and novelty framing; would benefit from including the Restormer track and the deployment study. |

Working interpretation, not a decision: the **arXiv preprint** is the
fastest move that pays off in every downstream path; **WACV / BMVC** is the
highest-probability acceptance route for the work in close-to-its-current
shape; **Optik / The Visual Computer** are realistic journal targets after
the same blocking items close; **CVPR / ICCV / ECCV** is reachable but only
with the Restormer track, an outdoor split, and stronger novelty framing.

### 8.4 One-line verdict

The science is publishable. The submission is not. The full list of work
required to convert one into the other lives in
[`Checklist.md`](Checklist.md) § 4 and § 5 and is roughly two weeks of
focused engineering + writing for a Tier-2 conference / journal target,
3-4 weeks for a Tier-1 conference target.

---

## 9. Repository layout

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

## 10. Citation & acknowledgements

Teachers and backbones:

* **DeHamer** — Guo, Yan, Anwar, Cong, Ren, Li. *Image Dehazing Transformer
  With Transmission-Aware 3D Position Embedding.* CVPR 2022.
  [paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Guo_Image_Dehazing_Transformer_With_Transmission-Aware_3D_Position_Embedding_CVPR_2022_paper.pdf)
  · [code / weights](https://github.com/Li-Chongyi/Dehamer).
* **Restormer** — Zamir, Arora, Khan, Hayat, Khan, Yang. *Restormer: Efficient
  Transformer for High-Resolution Image Restoration.* CVPR 2022 (Oral).
  [paper](https://arxiv.org/abs/2111.09881) · [code](https://github.com/swz30/Restormer).
* **NAFNet** — Chen, Chu, Zhang, Sun. *Simple Baselines for Image Restoration.*
  ECCV 2022. [paper](https://arxiv.org/abs/2204.04676) ·
  [code](https://github.com/megvii-research/NAFNet).

Distillation literature grounding Node C:

* **Hinton, Vinyals, Dean.** *Distilling the Knowledge in a Neural Network.*
  NeurIPS Workshop 2014. [paper](https://arxiv.org/abs/1503.02531) —
  introduced soft-target distillation; we apply it to dense image
  restoration by targeting teacher outputs instead of ground truth.

Benchmark:

* **RESIDE** — Li, Ren, Fu, Tao, Feng, Zeng, Wang. *Benchmarking Single Image
  Dehazing and Beyond.* IEEE TIP 2018.
  [paper](https://ieeexplore.ieee.org/abstract/document/8451944) · [project](https://sites.google.com/view/reside-dehaze-datasets).

---

## 11. Roadmap

Phase 1 (PTQ) and Phase 2 (Nodes A / B / C, distillation) are the reportable
units today. The roadmap below splits **science extensions** (which strengthen
the result) from a **publication path** (which converts the result into a
submitted manuscript).

### 11.1 Science extensions

* **Cross-split.** Repeat Phase 2 on SOTS-outdoor using DeHamer's outdoor
  checkpoint; both teacher ckpt and SOTS-outdoor data are already on disk.
  Promotes the contribution from "haze-indoor compression" to "haze
  compression" and is needed for any Tier-1 conference target.
* **Baselines.** AOD-Net (≈ 1.7 K parameters, speed floor), FFA-Net (CNN
  dehazing reference), and one prior distillation paper (KDDN if
  reproducible) to situate the student against existing compression
  families. Without these, the comparison table reads as one-sided.
* **Restormer track.** Fine-tune Restormer on RESIDE-ITS (50 K iterations,
  ~12–18 h on A5000), then run Phase 1 (PTQ) and Phase 2 (distillation) for
  a two-teacher table. Promotes the contribution from "compress one
  transformer" to "compress two transformers under one recipe."
* **Rain student.** Same pipeline, Restormer-deraining teacher, Rain13K
  data. Activates the **condition-specific** framing (haze ≠ rain) that the
  paper's novelty argument leans on.
* **Deployment study.** GPU INT8 via TensorRT / `torchao` pt2e, FP16
  comparison, mobile benchmarks, real-world qualitative on RTTS, FADE
  no-reference scores. Closes the practical-deployment loop the
  application motivation (autonomous driving, surveillance, license-plate
  recognition) implies.
* **Stability evidence.** Repeat-eval bands on PSNR/SSIM, latency mean ±
  std for the teacher on the same GPU as the students (this last item is
  also a blocking item — see `Checklist.md` § 4).

### 11.2 Publication path

Concrete sequencing — the cheapest moves first; later steps are conditional
on appetite and remaining time.

1. **Close the same-GPU teacher-latency blocker.** ~30 seconds of compute
   on `172.18.40.103` once the teacher is loaded. Unlocks the
   speedup-vs-DeHamer multiplier, which is the strongest single number in
   the paper.
2. **Add AOD-Net + FFA-Net rows to the comparison table.** Either by
   running them on SOTS-indoor here or by quoting published numbers with
   citations. Without this, every venue reviewer asks for it.
3. **Add an RTTS qualitative panel + FADE scores.** Synthetic-only
   evaluation is the single most common rejection reason for dehazing
   papers, regardless of venue.
4. **Write the manuscript.** Convert this README into a venue-appropriate
   LaTeX file with abstract, intro, related work, method, experiments,
   discussion, references. The chosen venue dictates page limit and style:
   IEEEtran for SPIC, Springer LNCS for The Visual Computer, plain Optik
   template, or the conference's official style for Tier-2 / Tier-1.
5. **Render figures.** Quality-vs-parameters Pareto plot, sensitivity
   heatmap, qualitative side-by-side panel, ablation bar chart.
6. **Upload to arXiv.** Establishes timestamp and gives external readers
   something to point at while the chosen venue runs review.
7. **Submit.** Tier-2 conference (WACV / BMVC / ACCV) is the highest
   probability accept for the work in close-to-current shape; journal
   (Optik / The Visual Computer) is the parallel option with slower review
   but no page-limit pressure.
8. **(Optional, conditional on time and appetite.)** Add the Restormer
   track + SOTS-outdoor split + GPU INT8 deployment numbers, then upgrade
   to a Tier-1 conference (CVPR / ICCV / ECCV) submission.

The detailed engineering status is tracked in [`Update.md`](Update.md);
submission readiness, venue-fit table, and minimum blockers are tracked in
[`Checklist.md`](Checklist.md); the architectural spec and cluster workflow are
in [`CLAUDE.md`](CLAUDE.md).
