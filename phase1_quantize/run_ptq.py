"""Phase-1 Post-Training Quantization for DeHamer.

Modes:
    dynamic : torch.quantization.quantize_dynamic on nn.Linear -> qint8 (CPU).
              Weights are quantized at load time; activations quantized per-forward.
              Easiest to run; only Linear layers get compressed (covers Swin
              attention QKV and FFN). Convs stay FP32.

    static  : (stub for Week 2b) FX-graph-mode static PTQ with fbgemm qconfig,
              calibrated on ITS images. Falls back with a clear error if the
              model can't be symbolically traced.

Comparison is CPU FP32 vs CPU INT8 (apples-to-apples), since dynamic/static
PyTorch PTQ is CPU-only. GPU INT8 is a separate story (TensorRT, Week 3+).

Run on cluster:
    python phase1_quantize/run_ptq.py --mode dynamic --split indoor --max-pairs 100
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

from evaluate.metrics import psnr, ssim
from models.teachers.dehamer import count_params, dehaze, load_dehamer, preprocess

ROOT = Path(__file__).resolve().parent.parent
CKPTS = {
    "indoor":  ROOT / "experiments/teachers/dehamer/ckpts/indoor/PSNR3663_ssim09881.pt",
    "outdoor": ROOT / "experiments/teachers/dehamer/ckpts/outdoor/PSNR3518_SSIM09860.pt",
}
SOTS_ROOT = ROOT / "data/RESIDE/SOTS-Test"


def pairs_for_split(split: str) -> list[tuple[Path, Path]]:
    sub = SOTS_ROOT / f"valid_{split}"
    pairs = []
    for hp in sorted((sub / "input").glob("*.png")):
        stem = hp.stem.split("_")[0]
        gp = (sub / "gt") / (stem + ".png")
        if gp.exists():
            pairs.append((hp, gp))
    return pairs


def apply_dynamic_quant(model: nn.Module) -> Tuple[nn.Module, Dict[str, int]]:
    """Dynamic INT8 for nn.Linear. Returns quantized model + coverage stats."""
    pre_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    pre_conv   = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))
    qmodel = torch.quantization.quantize_dynamic(
        model.cpu(), {nn.Linear}, dtype=torch.qint8
    )
    import torch.ao.nn.quantized.dynamic as qdyn
    quantized_linears = sum(1 for m in qmodel.modules() if isinstance(m, qdyn.Linear))
    return qmodel, {
        "linear_total": pre_linear,
        "conv_total": pre_conv,
        "linear_quantized": quantized_linears,
    }


@torch.no_grad()
def eval_model(model: nn.Module, pairs, device: str) -> Dict[str, float]:
    ps, ss = [], []
    t0 = time.perf_counter()
    for hp, gp in tqdm(pairs, desc=f"eval@{device}"):
        hazy = Image.open(hp).convert("RGB")
        gt = np.asarray(Image.open(gp).convert("RGB"))
        out = dehaze(model, hazy, device=device)
        h, w = out.shape[:2]
        ps.append(psnr(out, gt[:h, :w]))
        ss.append(ssim(out, gt[:h, :w]))
    elapsed = time.perf_counter() - t0
    return {
        "psnr_mean": float(np.mean(ps)),
        "ssim_mean": float(np.mean(ss)),
        "wall_s": elapsed,
        "ms_per_img": elapsed / len(pairs) * 1000.0,
        "n": len(pairs),
    }


@torch.no_grad()
def cpu_latency_ms(model: nn.Module, shape=(1, 3, 256, 256), warmup=3, iters=10) -> float:
    """CPU-only latency (small iters — this is slow)."""
    model.eval()
    x = torch.randn(*shape)
    for _ in range(warmup):
        _ = model(x)
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model(x)
    return (time.perf_counter() - t0) * 1000.0 / iters


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dynamic", "static"], default="dynamic")
    ap.add_argument("--split", choices=["indoor", "outdoor"], default="indoor")
    ap.add_argument("--max-pairs", type=int, default=100,
                    help="Cap eval set (CPU eval is slow). 0 = full 500.")
    ap.add_argument("--skip-latency", action="store_true",
                    help="Skip the synthetic-input CPU latency measurement.")
    ap.add_argument("--skip-fp32", action="store_true",
                    help="Skip the CPU FP32 reference run (assume results/dehamer_fp32_cpu_<split>.json already exists).")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.mode == "static":
        raise NotImplementedError("static mode is wired in task #12 — next step.")

    ckpt = CKPTS[args.split]
    assert ckpt.exists(), f"missing {ckpt}"

    pairs = pairs_for_split(args.split)
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]
    print(f"#pairs={len(pairs)} mode={args.mode} split={args.split}")

    # --- FP32 reference on CPU (apples-to-apples) ---
    print("\n--- FP32 (CPU) ---")
    fp32 = load_dehamer(ckpt_path=str(ckpt), device="cpu")
    n_params, m = count_params(fp32)
    print(f"params={n_params:,} ({m:.2f}M)")
    if args.skip_fp32:
        fp32_metrics = {}
    else:
        fp32_metrics = eval_model(fp32, pairs, device="cpu")
        print(f"FP32 CPU: PSNR {fp32_metrics['psnr_mean']:.3f}  SSIM {fp32_metrics['ssim_mean']:.4f}  {fp32_metrics['ms_per_img']:.1f} ms/img")

    lat_fp32 = None
    if not args.skip_latency:
        lat_fp32 = cpu_latency_ms(fp32)
        print(f"FP32 CPU synthetic 256²: {lat_fp32:.2f} ms  ({1000/lat_fp32:.2f} FPS)")

    # --- INT8 dynamic ---
    print("\n--- INT8 dynamic (CPU) ---")
    q, cov = apply_dynamic_quant(deepcopy(fp32))
    print(f"Linear total={cov['linear_total']}, quantized={cov['linear_quantized']}; Conv2d kept FP32={cov['conv_total']}")
    int8_metrics = eval_model(q, pairs, device="cpu")
    print(f"INT8 CPU: PSNR {int8_metrics['psnr_mean']:.3f}  SSIM {int8_metrics['ssim_mean']:.4f}  {int8_metrics['ms_per_img']:.1f} ms/img")

    lat_int8 = None
    if not args.skip_latency:
        lat_int8 = cpu_latency_ms(q)
        print(f"INT8 CPU synthetic 256²: {lat_int8:.2f} ms  ({1000/lat_int8:.2f} FPS)")

    # --- Delta ---
    if fp32_metrics:
        dpsnr = int8_metrics["psnr_mean"] - fp32_metrics["psnr_mean"]
        print(f"\nΔPSNR (INT8 - FP32) = {dpsnr:+.3f} dB")
    if lat_fp32 and lat_int8:
        print(f"Latency speedup      = {lat_fp32 / lat_int8:.2f}×")

    # --- Persist ---
    result = {
        "model": "DeHamer",
        "mode": args.mode,
        "device": "cpu",
        "split": args.split,
        "n_images": len(pairs),
        "n_params": n_params,
        "params_M": round(m, 2),
        "coverage": cov,
        "fp32": {**fp32_metrics, "latency_ms_256": lat_fp32},
        "int8": {**int8_metrics, "latency_ms_256": lat_int8},
    }
    out_path = Path(args.out) if args.out else ROOT / "results" / f"dehamer_int8_{args.mode}_{args.split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
