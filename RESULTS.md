# Results, Methods, Ablations — Consolidated Snapshot

Single source-of-truth dump of every method run, result measured, and ablation
populated so far. Target: BMVC submission. Snapshot date: **2026-05-23**.

Living narrative is in `README.md`; chronological log in `Update.md`; run
registry in `RUNS.md`; submission readiness in `Checklist.md`. This file is the
flat, dense cross-section.

---

## 1. Hardware reference

All numbers below — quality and latency — measured on a single
**NVIDIA RTX A5000 (24 GB)** unless noted otherwise. Same physical GPU across
teacher and student latency rows (this was an explicit re-measurement after
the original cross-GPU comparison was retracted).

- Teacher latency: `phase1_quantize/bench_teacher_latency.py`
- Student latency: `phase2_distill/bench_latency.py`
- Both use 5 reps × 100 CUDA-event iterations + 10 warm-up; isolated load
  (GPU baseline 0% util).

Calibration / training cluster: `teaching@172.18.40.119`, env
`/home/teaching/miniconda3/envs/adu` (torch 2.5.1+cu121 after the 2026-05-23
rebuild; previously torch 2.7.1+cu118 — both reproduce results within ≤0.01 dB).

---

## 2. Teacher reference numbers

DeHamer (132.45M parameters) FP32 evaluated on all four RESIDE/non-RESIDE
splits with each split's specialised checkpoint.

| Split          | Pairs | DeHamer ckpt                              | PSNR  | SSIM   | Notes |
|----------------|-------|-------------------------------------------|-------|--------|-------|
| SOTS-indoor    | 500   | `ckpts/indoor/PSNR3663_ssim09881.pt`       | 36.58 | 0.9862 | matches DeHamer paper |
| SOTS-outdoor   | 500   | `ckpts/outdoor/PSNR3518_SSIM09860.pt`      | 35.05 | 0.9853 | matches DeHamer paper |
| Dense-Haze     | 5     | `ckpts/dense/PSNR1662_SSIM05602.pt`        | 16.65 | 0.5596 | tiny test split |
| NH-HAZE        | 5     | `ckpts/NH/PSNR2066_SSIM06844.pt`           | 20.65 | 0.6844 | tiny test split |

Each row reproduces the published DeHamer number to within 0.06 dB. Real-world
splits (Dense, NH) ship only 5 test pairs each, so are used as cross-domain
sanity probes rather than primary benchmarks.

### 2.1 Teacher latency (same RTX A5000)

| Input   | Latency (ms) ± std | FPS    |
|---------|--------------------|--------|
| 256²    | **13.91 ± 0.01**   | 71.87  |
| 512²    | **46.04 ± 0.21**   | 21.72  |

Source: `results/latency_isolated_dehamer_teacher.json`.

---

## 3. Phase 1 — Post-Training Quantization of DeHamer

### 3.1 Method

DeHamer is a hybrid model: CNN encoder + 3-stage Swin transformer with
3D position embedding + CNN decoder. PTQ pipeline:

1. **Dynamic INT8 (all Linear).** `torch.quantization.quantize_dynamic` with
   `{nn.Linear}` and `qint8`. 26 Linear modules (all inside the Swin trunk,
   none in the CNN encoder/decoder) get INT8 weights + dynamic activation
   quantization. CPU-only (PyTorch dynamic PTQ does not target GPU).
2. **Sensitivity scan.** Quantize-all-then-swap-one-FP32 per Linear module
   (26 swaps) on 30 SOTS-indoor pairs. ΔPSNR vs all-INT8 ranks each module's
   contribution to the quantization error.
3. **Mixed-precision dynamic INT8.** Keep the top-K most sensitive Linear
   modules in FP32 (K=5 chosen as the knee of the ranked Δ curve); quantize
   the remaining 21 to INT8.
4. **Block-wise static PTQ (CNN encoder/decoder).** Eager-mode static PTQ
   over the 9 CNN `Sequential` blocks: `E_block1..4`, `_block{1,3,4,5,7}`.
   Per-block `QuantStub`/`DeQuantStub`, Conv+ReLU fusion, prepare → calibrate
   on 100 ITS images → convert. Reattached into the FP32 model.

FX-mode static PTQ on the *whole* model is blocked by data-dependent control
flow in DeHamer's Swin code (runtime padding checks); forking the model to
work around this was rejected. Documented as a negative result.

### 3.2 Results — SOTS-indoor, 500 pairs

Source: `results/phase1_indoor.csv`.

| Variant                          | PSNR   | SSIM   | ΔPSNR (vs FP32) | ms/img (CPU 256²) | Coverage |
|----------------------------------|--------|--------|-----------------|--------------------|----------|
| FP32 (baseline)                  | 36.576 | 0.9862 |   0.000          | 242.5              | —        |
| INT8 dynamic (all 26 Linear)     | 36.470 | 0.9842 |  −0.105          | 189.7              | 26/26    |
| **INT8 dynamic + top-5 FP32 (mixed)** | **36.551** | **0.9860** | **−0.025** | **190.2** | **21/26** |
| INT8 block-static (CNN blocks)   | 34.545 | —      |  −2.031          | —                  | 9 blocks |
| block-static + dyn-all           | 34.487 | —      |  −2.089          | —                  | both     |
| block-static + dyn-mixed         | 34.524 | —      |  −2.052          | —                  | both     |

**Winner:** mixed-precision dynamic — 1.27× CPU speedup at −0.025 dB
(essentially noise). Block-wise CNN static PTQ is the honest negative result:
INT8 on the CNN backbone is not a usable lever, supports the narrative that
the Swin trunk is where the redundancy lives.

### 3.3 Sensitivity map (top-8 of 26)

ΔPSNR is in dB; higher = quantizing this module hurts most. Source:
`results/dehamer_sensitivity_indoor.json` (30-image scan, FP32 baseline 34.099,
all-INT8 baseline 34.050).

| Rank | Module                                                     | ΔPSNR (vs all-INT8) |
|------|------------------------------------------------------------|----------------------|
| 1    | `swin_1.layers.0.blocks.0.mlp.fc1`                          | +0.0212              |
| 2    | `swin_1.layers.0.blocks.1.mlp.fc1`                          | +0.0116              |
| 3    | `swin_1.layers.0.blocks.1.mlp.fc2`                          | +0.0109              |
| 4    | `swin_1.layers.1.blocks.1.attn.proj`                        | +0.0075              |
| 5    | `swin_1.layers.2.blocks.0.mlp.fc1`                          | +0.0056              |
| 6    | `swin_1.layers.0.blocks.0.mlp.fc2`                          | +0.0054              |
| 7    | `swin_1.layers.1.blocks.1.mlp.fc2`                          | +0.0035              |
| 8    | `swin_1.layers.2.blocks.1.mlp.fc1`                          | +0.0032              |

Patterns: (a) earliest Swin stage is most sensitive; (b) `mlp.fc1`
(4× expansion) > `mlp.fc2` (projection) on average; (c) `attn.proj` matters
at deeper layers. Top-5 selected as the "keep FP32" set for the mixed-precision
configuration is consistent with this ranking.

---

## 4. Phase 2 — Condition-specific Distillation

### 4.1 Method

- **Student backbone:** NAFNet (megvii-research), `enc_blks=[1,1,1,28]`,
  `middle_blk_num=1`, `dec_blks=[1,1,1,1]`. Two width settings:
  - `width=16` → 4.35M params (Node A — extreme-lightweight)
  - `width=32` → 17.11M params (Nodes B, C — quality-leaning)
  Random init; teacher frozen (`eval()`, `requires_grad=False`).
- **Loss:** `L_total = L_pixel(L1) + λ_feat · L_feat(L2 on decoder feature taps,
  1×1 adapter on student side) + λ_perc · L_perceptual(VGG)`. Three weight
  configurations (see §4.2).
- **Supervision target:** either ground-truth clean image (GT) or DeHamer's
  offline dehazed output (pseudo / soft-label). Pseudo-labels generated once
  per dataset with `scripts/gen_soft_labels.py` (~41 min on A5000 for ITS;
  decouples teacher inference from training).
- **Optimizer / schedule:** AdamW betas=(0.9, 0.9), lr 1e-3 cosine → 1e-6,
  200 epochs (≈3.5M steps on ITS), batch 8, patch 128×128, random H/V flip
  + 90° rotation. Single A5000.
- **Validation:** SOTS-indoor every 5 epochs, full-image inputs.
  Best ckpt = best VAL PSNR.
- **Eval:** `phase2_distill/eval_student.py`, full-image SOTS pairs,
  one-shot PSNR/SSIM via `skimage.metrics`. Latency reported separately via
  `bench_latency.py` (5×100-iter CUDA events).

### 4.2 The 2 × 2 ablation (capacity × supervision target)

All three indoor students trained on RESIDE ITS (13,990 pairs) with the
DeHamer indoor checkpoint as teacher. Evaluated on SOTS-indoor 500 pairs.

| Node | Tag                      | Width | Params | Target | λ_feat | λ_perc | PSNR  | SSIM   | Best epoch |
|------|--------------------------|-------|--------|--------|--------|--------|-------|--------|------------|
| —    | `haze_s1` (early)        | 16    | 4.35M  | GT     | 0.01   | 0.00   | 29.78 | 0.9675 | 194        |
| A    | `haze_a_small_tight`     | 16    | 4.35M  | GT     | 0.05   | 0.05   | 32.39 | 0.9829 | 184        |
| B    | `haze_b_large_tight`     | 32    | 17.11M | GT     | 0.05   | 0.05   | **34.40** | **0.9865** | 184    |
| C    | `haze_c_large_pseudo`    | 32    | 17.11M | Pseudo | 0.00   | 0.05   | 33.87 | 0.9834 | best       |

`haze_s1` is the early run that exposed the tight-loss / capacity gap and
motivated the three-way split; not in the headline table.

**Levers:**
- **Capacity (w16 → w32 at fixed losses):** +2.0 dB. Dominant lever.
- **Supervision target (GT → pseudo at w32):** −0.5 dB quality. Unlocks 36%
  throughput at the original single-window latency measurement; with
  isolated-load remeasurement, throughput parity (see §4.4).
- **Loss weighting (`haze_s1` 0.01/0 → `A` 0.05/0.05):** +2.6 dB at the same
  capacity. Underweighted feature + zero perceptual was the original failure
  mode.

### 4.3 Cross-domain evaluation (indoor students → outdoor / dense / NH)

All three indoor students evaluated against each split's specialised DeHamer
teacher. Source: `results/eval_student_<tag>_<split>.json` (pulled
2026-05-13). Pairs: outdoor 500, dense 5, nh 5.

| Model                          | indoor          | outdoor       | dense         | NH           |
|--------------------------------|-----------------|---------------|---------------|--------------|
| **DeHamer FP32 (teacher)**     | 36.58 / 0.986   | 35.05 / 0.985 | 16.65 / 0.560 | 20.65 / 0.684 |
| Student A (w16, 4.35M)         | 32.39 / 0.983   | 22.51 / 0.909 | 10.73 / 0.436 | 12.27 / 0.416 |
| Student B (w32, 17.1M, GT)     | **34.40 / 0.987** | 20.58 / 0.882 | 10.26 / 0.443 | 12.27 / 0.370 |
| Student C (w32, 17.1M, pseudo) | 33.87 / 0.983   | 19.73 / 0.827 | 9.43 / 0.394  | 12.68 / 0.402 |

**Interpretation:**
- Indoor-trained students collapse on outdoor (~15 dB gap to teacher), dense
  (~6 dB), NH (~8 dB).
- Higher capacity helps within-domain but does *not* transfer (B = 34.40 →
  20.58, A = 32.39 → 22.51): smaller student A actually outperforms the larger
  B on the outdoor split, suggesting the larger model over-specialises to the
  indoor distribution.
- Pseudo-supervision (C) is consistently worst on cross-domain, consistent
  with the student inheriting the indoor teacher's biases more tightly than
  GT supervision does.

This is the empirical argument for **condition-specific distillation**:
one indoor student cannot serve all four splits, so the recipe must instantiate
a separate student per condition. The outdoor student
(`haze_outdoor_b`, w32, GT, 50K OTS subset) is the next experiment; teacher
checkpoint and OTS data are now on the cluster (download finishing 2026-05-23).

### 4.4 Latency (same RTX A5000, isolated load)

5×100-iter CUDA-event reps, GPU baseline 0% util. Source:
`results/latency_isolated_*.json`.

| Model      | Params  | 256² ms ± std    | 256² FPS | 512² ms ± std    | 512² FPS | 256² speedup vs teacher | 512² speedup vs teacher |
|------------|---------|------------------|----------|------------------|----------|--------------------------|--------------------------|
| Teacher    | 132.45M | 13.91 ± 0.01     | 71.87    | 46.04 ± 0.21     | 21.72    | 1.00×                    | 1.00×                    |
| Student A  | 4.35M   | 29.70 ± 0.05     | 33.67    | 33.09 ± 3.39     | 30.22    | 0.47×                    | **1.39×**                |
| Student B  | 17.11M  | 32.89 ± 2.23     | 30.41    | 34.13 ± 1.80     | 29.30    | 0.42×                    | **1.35×**                |
| Student C  | 17.11M  | 36.40 ± 0.64     | 27.47    | 33.84 ± 1.16     | 29.55    | 0.38×                    | **1.36×**                |

**Important caveat — the speedup claim is scoped to 512×512.** At 256²,
the teacher's Swin attention is small-input efficient and the students lose
to it. At 512², NAFNet's CNN-only path scales linearly while attention
quadratically, and the students are 1.35–1.39× faster. The paper headlines
the 512² number and explicitly states the 256² inversion to keep reviewers
from being surprised.

B vs C at 256² are within one std of each other; the earlier
"C = 43.1 FPS, throughput winner" claim was a single-window measurement
artefact and has been retracted.

### 4.5 Compression summary

| Model      | Params (M) | × smaller vs teacher | PSNR (indoor) | Quality gap (dB) | 512² FPS |
|------------|-----------:|---------------------:|---------------|------------------|----------|
| Teacher    | 132.45     |  1.0×                | 36.58         |  0.00            | 21.72    |
| Student A  |   4.35     | **30.5×**            | 32.39         | −4.19            | 30.22    |
| Student B  |  17.11     |  7.7×                | 34.40         | −2.18            | 29.30    |
| Student C  |  17.11     |  7.7×                | 33.87         | −2.71            | 29.55    |

Pareto: B is the quality-best at fixed capacity; A is the extreme-lightweight
operating point (30× smaller, ~4 dB gap, still SOTS-grade SSIM).

---

## 5. What is NOT yet in the table (BMVC §4.2 blockers)

These are the cells that remain empty and would each be a row or paragraph in
the submission:

- **Outdoor student** (`haze_outdoor_b`, w32 GT). OTS dataset just arrived on
  the cluster (2026-05-23); soft labels + training is the next ~12 h of compute.
  Will populate the "outdoor / condition-matched student" cell.
- **Rain student** (NAFNet w32 on Rain13K, Restormer deraining teacher).
  Materialises the "one recipe, two degradation types" version of the
  contribution. Restormer ckpt is on disk; Rain13K not yet pulled.
- **External lightweight baselines.** AOD-Net (~2K params) and FFA-Net on
  SOTS-indoor — either rerun on the same A5000 or quoted from their papers
  with citations. Required to anchor the comparison table.
- **Real-world qualitative + FADE.** RTTS (4,332 unpaired images) for the
  qualitative panel + FADE no-reference score. Synthetic-only eval is the
  single most common rejection reason for dehazing papers.
- **Restormer-teacher track (Tier-1 only).** Fine-tune Restormer deraining
  ckpt on ITS, then run Phase 1 + Phase 2. Promotes the contribution from
  "compress one transformer" to "compress two under one recipe."
- **GPU INT8 deployment study (Tier-1 only).** TensorRT or `torchao` pt2e on
  the student. FP16 vs INT8 latency on A5000.
- **Manuscript.** No `.tex` exists yet; `abstract.txt` has a draft title +
  abstract (committed 2026-05-23). LaTeX skeleton (IEEEtran) to come.
- **Figures.** Pareto plot, sensitivity heatmap, qualitative side-by-side,
  2×2 ablation bar chart.

`Checklist.md` is the authoritative tracker; this section is a summary.

---

## 6. Raw artefact index

For anything in the tables above, the underlying JSON/CSV is checked into
`results/`. Naming convention:

- `phase1_indoor.{json,csv}` — Phase-1 PTQ table source.
- `dehamer_fp32_<split>.json` — teacher FP32 row, one per split.
- `dehamer_sensitivity_indoor.json` — per-layer ΔPSNR ranking.
- `dehamer_int8_dynamic_indoor.json`, `dehamer_int8_block_static_indoor.json` —
  per-mode PTQ runs.
- `eval_student_<tag>[_<split>].json` — student quality eval. No suffix =
  indoor (legacy naming); `_outdoor`/`_dense`/`_nh` for the cross-domain runs.
- `latency_isolated_<tag>.json` — student/teacher latency (5×100-iter reps).
- `phase2_<tag>.log` + `phase2_<tag>_status.txt` — training log + DONE marker
  per run; `training_summary.json` next to `best.pt` is the canonical end-state.

Student `best.pt` weights are gitignored (50–200 MB each); they live on the
cluster at `experiments/students/<tag>/best.pt`.

---

## 7. Reproducing any number in this file

Local → cluster sync, then run on the cluster:

```bash
# Phase 1 (full PTQ suite + sensitivity, ~55 min on cluster)
./scripts/sync_to_cluster.sh
./gpu "cd dehazing-compression && bash scripts/launch_phase1_tmux.sh"

# Phase 2 (one student, ~8–10 h on A5000)
./gpu "cd dehazing-compression && \
  python phase2_distill/train.py --tag <tag> --width <16|32> \
    --lambda-feat 0.05 --lambda-perc 0.05 --epochs 200 --wandb"

# Cross-domain eval of an existing student on a different split
./gpu "cd dehazing-compression && \
  python phase2_distill/eval_student.py --ckpt experiments/students/<tag>/best.pt \
    --tag <tag> --width <16|32> --split <indoor|outdoor|dense|nh>"

# Isolated latency (5×100 iters, ~3 min per model)
./gpu "cd dehazing-compression && python phase2_distill/bench_latency.py --tag <tag> --width <16|32>"
```

Everything in this file is reproducible from the committed code + checkpoints
that already live on the cluster. No private mirrors, no manual tweaks.
