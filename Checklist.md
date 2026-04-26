# Submission Readiness Checklist

Date: 2026-04-26
Scope: Phase-1 PTQ + Phase-2 distillation (haze)

## 1. Current Evidence Snapshot

| Item | Status | Key Evidence |
|---|---|---|
| Phase-1 PTQ baseline | Complete | `results/phase1_indoor.csv` |
| Teacher FP32 reference (SOTS-indoor) | Complete | 36.576 PSNR / 0.9862 SSIM |
| Phase-2 Node B (GT target, width 32) | Complete | 34.398 PSNR / 0.9865 SSIM, 31.6 FPS @256 |
| Phase-2 Node C (pseudo target, width 32) | Complete | 33.869 PSNR / 0.9834 SSIM, 43.1 FPS @256 |
| Phase-2 Node A (GT target, width 16) | Complete | 32.391 PSNR / 0.9829 SSIM, 33.0 FPS @256, 4.35M params |

## 2. Quantitative Quality Summary

Reference teacher (Phase-1 FP32 row):
- PSNR: 36.576
- SSIM: 0.9862

Node B vs teacher:
- PSNR gap: -2.18 dB
- SSIM gap: +0.0003
- Params: 17.11M vs 132.45M (~7.7x smaller)

Node C vs teacher:
- PSNR gap: -2.71 dB
- SSIM gap: -0.0027
- Params: 17.11M vs 132.45M (~7.7x smaller)

Node A vs teacher:
- PSNR gap: -4.19 dB
- SSIM gap: -0.0033
- Params: 4.35M vs 132.45M (~30.5x smaller)

Interpretation:
- Node B is the current quality winner.
- Node C is the current throughput winner.
- Node A is the extreme-compression operating point (30.5x smaller, ~4.2 dB gap).
- 2x2 ablation (capacity x supervision target) is now fully populated; capacity contributes ~+2.0 dB at fixed losses (w16 GT 32.39 -> w32 GT 34.40), supervision-target swap costs ~0.5 dB at fixed capacity (w32 GT 34.40 -> w32 pseudo 33.87) but unlocks ~36% throughput gain.

## 3. Venue-Wise Readiness

The original CLAUDE.md target list was journal-only. Broader venue landscape
— conferences, workshops, and preprint server — included for completeness.

### 3.1 Conferences

| Tier | Venue | Fit Now | Assessment |
|---|---|---|---|
| Tier-1 | CVPR / ICCV / ECCV | Borderline | Borderline only with the Restormer-teacher track, SOTS-outdoor split, >=2 external baselines + >=1 prior distillation paper, GPU INT8 deployment numbers, and sharper "condition-specific distillation" novelty framing. ~3-4 weeks of additional work on top of writing. |
| Tier-2 | **WACV / BMVC / ACCV** | **Sweet spot** | Highest-probability accept for the work in close-to-current shape. These venues explicitly value practical ML + systems insight, which matches the engineering contribution. Restormer track and outdoor split are nice-to-have rather than blocking. |
| Workshop | CVPR-W / ECCV-W | High | Fast feedback path. Strong fit for the PTQ + distillation engineering story without needing the deeper SOTA-comparison expansion the main conferences would demand. Consider as a fallback or as a parallel companion publication. |

### 3.2 Journals

| Venue | Fit Now | Assessment |
|---|---|---|
| Optik | Medium-High | Application-framed venue; fits the autonomous-driving / surveillance motivation. Doable after blocking items in §4 close. |
| The Visual Computer (Springer) | Medium-High | Similar fit to Optik; expects complete ablations + stronger qualitative section, both of which are on the work plan. |
| Signal Processing: Image Communication (IEEEtran) | Medium | More demanding on comparative depth and novelty framing; benefits significantly from the Restormer track and the deployment study. |

### 3.3 Preprint

| Venue | Fit Now | Assessment |
|---|---|---|
| **arXiv** | **High** | Recommended *regardless* of which conference / journal is chosen. Establishes timestamp / priority and gives external readers something to point at. Cheap and high-value. Upload after the manuscript reads cleanly and the latency multiplier is fixed (do not arXiv with the cross-GPU teacher-comparison error, since v1 is permanent record). |

## 4. Minimum Work Required Before Submission

Items are tagged by the *minimum* venue tier that requires them. Tier-1 work
implies the Tier-2 / journal / workshop work has also been done. Workshop-tier
items are the universal floor.

### 4.1 Universal floor (any venue, including workshops)

- [x] Finish Node A run and evaluate `best.pt` with `phase2_distill/eval_student.py`. *(Complete 2026-04-26: 32.391 / 0.9829.)*
- [x] Re-run latency for **students** under isolated GPU load. *(Complete 2026-04-26 via `phase2_distill/bench_latency.py`, 5x100-iter mean +/- std, A5000, GPU 0% baseline. JSONs in `results/latency_isolated_*.json`.)*
- [ ] **Re-measure the DeHamer teacher latency on the SAME RTX A5000 host** (172.18.40.103). The published Phase-1 teacher latency (25.9 ms @256 / 86.4 ms @512) was measured on cs671 A6000 and cannot be compared to the A5000 student numbers. Without this, no PSNR-preserving-speedup multiplier vs DeHamer can be quoted in the paper. ~30 seconds of compute.
- [ ] Finalize the full 2x2 ablation story (capacity x supervision target) in one consolidated table. *(All four cells measured; remaining work is to write the consolidated table into the paper, not to run.)*
- [ ] **Write the manuscript.** Convert README.md into a venue-appropriate LaTeX file (IEEEtran / Springer LNCS / Optik / conference style). Abstract, intro, related work, method, experiments, discussion, references. No `.tex` exists today.
- [ ] **Render figures.** Quality-vs-parameters Pareto plot, sensitivity heatmap, qualitative side-by-side panel, ablation bar chart.

### 4.2 Required for Tier-2 conferences (WACV / BMVC / ACCV) and journals (Optik / TVC / SPIC)

- [ ] Add at least 2 external baselines in one comparison table (**AOD-Net** and **FFA-Net** minimum). Either re-run on SOTS-indoor here or quote published numbers with citations.
- [ ] Add RTTS qualitative panel (and **FADE** no-reference scores if feasible) to strengthen real-world relevance. Synthetic-only evaluation is the single most common rejection reason for dehazing papers.
- [ ] Sharpen "condition-specific distillation" novelty framing in the introduction and contributions section. Currently reads as "applied known techniques carefully"; should read as a method.

### 4.3 Required for Tier-1 conferences (CVPR / ICCV / ECCV)

All of §4.2 above, plus:

- [ ] **Restormer-teacher track.** Fine-tune Restormer on RESIDE-ITS (50 K iterations, ~12-18 h on A5000), then run Phase 1 (PTQ) and Phase 2 (distillation). Promotes the contribution from "compress one transformer" to "compress two transformers under one recipe."
- [ ] **SOTS-outdoor split.** Repeat Phase 2 on SOTS-outdoor using DeHamer's outdoor checkpoint; both teacher ckpt and SOTS-outdoor data are already on disk.
- [ ] **One prior distillation-on-dehazing baseline** (KDDN if reproducible) in the comparison table — distinct from the lightweight CNN baselines in §4.2.
- [ ] **GPU INT8 deployment study.** TensorRT or `torchao` pt2e path on the student. FP16 vs INT8 latency. Closes the practical-deployment loop the application motivation implies.

### 4.4 Strongly recommended (lift acceptance probability across all tiers)

- [x] Confidence interval / repeated-run stability note for **latency**. *(Done for students 2026-04-26; mean +/- std reported in README §7.4. Teacher pending its same-GPU re-measurement.)*
- [ ] Repeat-eval stability note for PSNR/SSIM (5-seed retrains optional; per-image PSNR distribution figure cheaper).
- [ ] Short error-analysis section with 2-3 failure-case images for reviewer robustness concerns.
- [ ] Upload to arXiv before any conference / journal submission. Establishes timestamp; do *not* arXiv until the same-GPU teacher latency is in place (v1 is permanent record).

## 5. Suggested Submission Strategy

Tiered path. Step (1) is always first; the rest depend on appetite and time.

1. **Close §4.1 (universal floor) + §4.2 (Tier-2 / journal items).** Same-GPU teacher latency, AOD-Net + FFA-Net rows, RTTS qualitative + FADE, sharper novelty framing, manuscript file, figures. Roughly two weeks of focused engineering + writing.
2. **Upload v1 to arXiv.** Establishes timestamp / priority. Always do this before the formal submission.
3. **Submit to highest-probability venue first**: WACV / BMVC / ACCV (Tier-2 conference) for fast review *or* Optik / The Visual Computer (journal) for slower review without page-limit pressure. The two are roughly equivalent in expected acceptance probability; pick by the candidate's preference for review speed vs page budget.
4. **Conditional escalation.** If §4.3 (Tier-1 items) lands within ~3-4 weeks, prepare a CVPR / ICCV / ECCV version. The Restormer-teacher track and the GPU INT8 deployment study together do most of the work in moving the manuscript from "engineering paper" to "Tier-1 contribution."
5. **Workshop fallback.** If timing or scope tightens, a CVPR-W / ECCV-W workshop submission of the §4.1 + §4.2 material is a reasonable graceful-degradation path that still delivers a published result and external visibility.

## 6. Practical Verdict

The science is publishable. The submission is not. The cleanest first
move is steps 1-3 above (close the universal floor, add the Tier-2
external baselines and real-world eval, write the LaTeX, arXiv it,
submit to a Tier-2 venue or a journal). Tier-1 conference acceptance is
reachable but conditional on the Restormer track + outdoor split +
deployment study landing.
