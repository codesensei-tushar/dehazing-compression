"""Phase-1 FP32 baseline for DeHamer on SOTS indoor.

Evaluates the pretrained indoor checkpoint on all 500 hazy/gt pairs, reports
mean PSNR/SSIM, and measures latency at 256x256 and 512x512 inputs.

Run on the cluster:
    CUDA_VISIBLE_DEVICES=1 python evaluate/benchmark_dehamer.py --split indoor
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from evaluate.metrics import latency_ms, psnr, ssim
from models.teachers.dehamer import count_params, dehaze, load_dehamer

ROOT = Path(__file__).resolve().parent.parent

CKPTS = {
    "indoor":  ROOT / "experiments/teachers/dehamer/ckpts/indoor/PSNR3663_ssim09881.pt",
    "outdoor": ROOT / "experiments/teachers/dehamer/ckpts/outdoor/PSNR3518_SSIM09860.pt",
}
SOTS_ROOT = ROOT / "data/RESIDE/SOTS-Test"


def pairs_for_split(split: str) -> list[tuple[Path, Path]]:
    sub = SOTS_ROOT / f"valid_{split}"
    hazy_dir = sub / "input"
    gt_dir = sub / "gt"
    pairs = []
    for hp in sorted(hazy_dir.glob("*.png")):
        # SOTS naming: "<id>_<k>.png" (10 hazy per GT indoor; outdoor varies)
        stem = hp.stem.split("_")[0]
        for ext in (".png", ".jpg"):
            gp = gt_dir / (stem + ext)
            if gp.exists():
                pairs.append((hp, gp))
                break
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["indoor", "outdoor"], default="indoor")
    ap.add_argument("--max-pairs", type=int, default=0, help="0 = all")
    ap.add_argument("--out", default=None, help="Output JSON path (default: results/<split>_fp32.json)")
    args = ap.parse_args()

    ckpt = CKPTS[args.split]
    assert ckpt.exists(), f"Missing checkpoint {ckpt}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} split={args.split}")

    model = load_dehamer(ckpt_path=str(ckpt), device=device)
    n_params, m = count_params(model)
    print(f"params={n_params:,} ({m:.2f}M)")

    pairs = pairs_for_split(args.split)
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]
    print(f"#pairs={len(pairs)}")

    psnrs: list[float] = []
    ssims: list[float] = []
    t_start = perf_counter()
    for hp, gp in tqdm(pairs, desc=f"DeHamer FP32 SOTS {args.split}"):
        hazy = Image.open(hp).convert("RGB")
        gt = np.asarray(Image.open(gp).convert("RGB"))
        out = dehaze(model, hazy, device=device)
        h, w = out.shape[:2]
        gt_c = gt[:h, :w]
        psnrs.append(psnr(out, gt_c))
        ssims.append(ssim(out, gt_c))
    elapsed = perf_counter() - t_start

    print("\n--- Quality ---")
    print(f"mean PSNR: {np.mean(psnrs):.3f}  (min {np.min(psnrs):.2f}, max {np.max(psnrs):.2f})")
    print(f"mean SSIM: {np.mean(ssims):.4f} (min {np.min(ssims):.4f}, max {np.max(ssims):.4f})")
    print(f"wall time: {elapsed:.1f}s over {len(pairs)} images ({elapsed/len(pairs)*1000:.1f} ms/img)")

    print("\n--- Synthetic-input latency ---")
    lat_256 = latency_ms(model, (1, 3, 256, 256), device=device)
    lat_512 = latency_ms(model, (1, 3, 512, 512), device=device)
    print(f"256x256: {lat_256:.2f} ms/img  ({1000/lat_256:.1f} FPS)")
    print(f"512x512: {lat_512:.2f} ms/img  ({1000/lat_512:.1f} FPS)")

    result = {
        "model": "DeHamer-FP32",
        "split": args.split,
        "ckpt": str(ckpt.relative_to(ROOT)),
        "n_params": n_params,
        "params_M": round(m, 2),
        "psnr_mean": float(np.mean(psnrs)),
        "ssim_mean": float(np.mean(ssims)),
        "n_images": len(pairs),
        "wall_time_s": elapsed,
        "latency_ms_256": lat_256,
        "latency_ms_512": lat_512,
        "fps_256": 1000.0 / lat_256,
        "fps_512": 1000.0 / lat_512,
    }
    out_path = Path(args.out) if args.out else ROOT / "results" / f"dehamer_fp32_{args.split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
