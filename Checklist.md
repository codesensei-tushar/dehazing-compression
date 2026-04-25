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
| Phase-2 Node A (GT target, width 16) | In progress | resumed run on 172.18.40.103 (epoch ~109/200) |

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

Interpretation:
- Node B is the current quality winner.
- Node C is the current throughput winner.

## 3. Venue-Wise Readiness

| Venue | Readiness Now | Assessment |
|---|---|---|
| Optik | Medium-High | Likely publishable after closing missing ablation row and baseline table tightening. |
| The Visual Computer | Medium-High | Likely publishable with complete ablations + stronger qualitative section. |
| Signal Processing: Image Communication | Medium | Promising, but needs stronger comparative depth and clearer novelty framing for acceptance confidence. |

## 4. Minimum Work Required Before Submission

Blocking items (recommended before any journal submission):
- [ ] Finish Node A run and evaluate `best.pt` with `phase2_distill/eval_student.py`.
- [ ] Finalize the full 2x2 ablation story (capacity x supervision target) in one consolidated table.
- [ ] Re-run latency for teacher, Node B, and Node C under isolated GPU load (no concurrent training).
- [ ] Add at least 2 external baselines in one table (AOD-Net, FFA-Net minimum).
- [ ] Add RTTS qualitative panel (and FADE if feasible) to strengthen real-world relevance.

Strongly recommended (not strictly blocking):
- [ ] Add confidence interval or repeated-run stability note for latency and PSNR/SSIM.
- [ ] Add short error analysis examples (failure cases) for reviewer robustness concerns.
- [ ] Add one concise contribution figure: quality-speed-parameter tradeoff plot.

## 5. Suggested Submission Strategy

1. Preferred first target: Optik or The Visual Computer after blocking items are complete.
2. Then prepare a stricter version for Signal Processing: Image Communication with expanded baselines and stronger novelty framing.

## 6. Practical Verdict

Current results are strong and likely publishable, but not fully submission-ready yet because one ablation run is still incomplete and comparative coverage can be improved.
