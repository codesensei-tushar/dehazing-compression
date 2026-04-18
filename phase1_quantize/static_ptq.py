"""Static Post-Training Quantization for DeHamer via FX graph mode.

Workflow:
  1. Load FP32 model on CPU, set eval().
  2. prepare_fx with fbgemm qconfig → inserts observers.
  3. Calibrate: forward N images (ITS preferred, SOTS fallback).
  4. convert_fx → replaces observers with quantized ops.
  5. Evaluate on SOTS-indoor + measure CPU synthetic-input latency.

Unlike dynamic PTQ, static quantizes BOTH Linear and Conv2d, which is the
relevant comparison for conv-heavy restoration models like DeHamer.

FX tracing may fail on DeHamer if the forward hits unsupported patterns.
In that case, error is surfaced; we then fall back to Eager-mode static PTQ
(separate script) or torchao.

Run on cluster:
    python phase1_quantize/static_ptq.py --calib-dir data/RESIDE/ITS/ITS/hazy \\
        --n-calib 200 --n-eval 100
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

from evaluate.metrics import psnr, ssim
from models.teachers.dehamer import count_params, dehaze, load_dehamer, preprocess
from models.teachers.dehamer_fx_patch import patch_instance  # class patch auto-applied on import

ROOT = Path(__file__).resolve().parent.parent
CKPTS = {
    "indoor":  ROOT / "experiments/teachers/dehamer/ckpts/indoor/PSNR3663_ssim09881.pt",
    "outdoor": ROOT / "experiments/teachers/dehamer/ckpts/outdoor/PSNR3518_SSIM09860.pt",
}
SOTS_ROOT = ROOT / "data/RESIDE/SOTS-Test"


def sots_pairs(split: str) -> list[tuple[Path, Path]]:
    sub = SOTS_ROOT / f"valid_{split}"
    out = []
    for hp in sorted((sub / "input").glob("*.png")):
        stem = hp.stem.split("_")[0]
        gp = (sub / "gt") / (stem + ".png")
        if gp.exists():
            out.append((hp, gp))
    return out


def find_calibration_images(calib_dir: Path | None, fallback_dir: Path, n: int) -> list[Path]:
    """Return a list of calibration image paths. Prefer calib_dir (ITS hazy); fall back."""
    for d in (calib_dir, fallback_dir):
        if d is None or not d.is_dir():
            continue
        files = sorted(d.glob("*.png")) + sorted(d.glob("*.jpg"))
        if files:
            if len(files) > n:
                rng = random.Random(0)
                files = rng.sample(files, n)
            return files
    raise FileNotFoundError(f"No calibration images in {calib_dir} or {fallback_dir}")


def apply_static_fx(model: nn.Module, calib_imgs: list[Path], backend: str = "fbgemm") -> nn.Module:
    """FX graph mode static PTQ.
    Returns a converted INT8 module (CPU)."""
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

    torch.backends.quantized.engine = backend
    qconfig_mapping = get_default_qconfig_mapping(backend)

    # Example input for tracing.
    example = preprocess(Image.open(calib_imgs[0]).convert("RGB"))

    model = model.cpu().eval()
    patch_instance(model)  # retrofit self.pad on existing DarkChannel instances
    prepared = prepare_fx(model, qconfig_mapping, example_inputs=(example,))

    with torch.no_grad():
        for p in tqdm(calib_imgs, desc="calibrate"):
            x = preprocess(Image.open(p).convert("RGB"))
            _ = prepared(x)

    return convert_fx(prepared)


@torch.no_grad()
def eval_on_sots(model: nn.Module, pairs) -> dict:
    ps, ss = [], []
    t0 = time.perf_counter()
    for hp, gp in tqdm(pairs, desc="eval@cpu"):
        hazy = Image.open(hp).convert("RGB")
        gt = np.asarray(Image.open(gp).convert("RGB"))
        out = dehaze(model, hazy, device="cpu")
        h, w = out.shape[:2]
        ps.append(psnr(out, gt[:h, :w]))
        ss.append(ssim(out, gt[:h, :w]))
    el = time.perf_counter() - t0
    return {"psnr_mean": float(np.mean(ps)), "ssim_mean": float(np.mean(ss)),
            "ms_per_img": el / len(pairs) * 1000.0, "n": len(pairs)}


@torch.no_grad()
def cpu_latency(model, shape=(1, 3, 256, 256), warmup=3, iters=10) -> float:
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
    ap.add_argument("--split", choices=["indoor", "outdoor"], default="indoor")
    ap.add_argument("--calib-dir", type=Path, default=None,
                    help="Dir of hazy images for calibration (prefer ITS hazy). Falls back to SOTS input.")
    ap.add_argument("--n-calib", type=int, default=200)
    ap.add_argument("--n-eval", type=int, default=100)
    ap.add_argument("--backend", choices=["fbgemm", "qnnpack"], default="fbgemm")
    ap.add_argument("--skip-latency", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ckpt = CKPTS[args.split]
    assert ckpt.exists(), f"missing {ckpt}"

    sots_fallback = SOTS_ROOT / f"valid_{args.split}" / "input"
    calib_imgs = find_calibration_images(args.calib_dir, sots_fallback, args.n_calib)
    print(f"calibration: {len(calib_imgs)} images from {calib_imgs[0].parent}")

    pairs = sots_pairs(args.split)[: args.n_eval] if args.n_eval else sots_pairs(args.split)
    print(f"eval: {len(pairs)} pairs from SOTS {args.split}")

    # FP32 reference
    print("\n--- FP32 (CPU) ---")
    fp32 = load_dehamer(ckpt_path=str(ckpt), device="cpu")
    n_params, m = count_params(fp32)
    print(f"params={n_params:,} ({m:.2f}M)")
    fp32_m = eval_on_sots(fp32, pairs)
    lat_fp32 = None if args.skip_latency else cpu_latency(fp32)
    print(f"FP32 CPU: PSNR {fp32_m['psnr_mean']:.3f}  SSIM {fp32_m['ssim_mean']:.4f}  {fp32_m['ms_per_img']:.1f} ms/img")
    if lat_fp32:
        print(f"FP32 CPU synthetic 256²: {lat_fp32:.2f} ms  ({1000/lat_fp32:.2f} FPS)")

    # Static INT8
    print(f"\n--- INT8 static-FX (CPU, backend={args.backend}) ---")
    try:
        q = apply_static_fx(deepcopy(fp32), calib_imgs, backend=args.backend)
    except Exception as e:
        print(f"FX tracing / conversion failed: {type(e).__name__}: {e}")
        print("Consider eager-mode static PTQ or torchao as a fallback.")
        raise
    int8_m = eval_on_sots(q, pairs)
    lat_int8 = None if args.skip_latency else cpu_latency(q)
    print(f"INT8 CPU: PSNR {int8_m['psnr_mean']:.3f}  SSIM {int8_m['ssim_mean']:.4f}  {int8_m['ms_per_img']:.1f} ms/img")
    if lat_int8:
        print(f"INT8 CPU synthetic 256²: {lat_int8:.2f} ms  ({1000/lat_int8:.2f} FPS)")

    # Delta
    dpsnr = int8_m["psnr_mean"] - fp32_m["psnr_mean"]
    print(f"\nΔPSNR (INT8 - FP32) = {dpsnr:+.3f} dB")
    if lat_fp32 and lat_int8:
        print(f"Latency speedup      = {lat_fp32 / lat_int8:.2f}×")

    # Persist
    result = {
        "model": "DeHamer",
        "mode": "static-fx",
        "backend": args.backend,
        "device": "cpu",
        "split": args.split,
        "n_calib": len(calib_imgs),
        "n_eval": len(pairs),
        "n_params": n_params,
        "params_M": round(m, 2),
        "fp32": {**fp32_m, "latency_ms_256": lat_fp32},
        "int8": {**int8_m, "latency_ms_256": lat_int8},
    }
    out_path = Path(args.out) if args.out else ROOT / "results" / f"dehamer_int8_static_{args.split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
