"""Evaluate a baseline dehazing model on a RESIDE SOTS split.

Shares the same eval loop, metrics, and data-loading logic as
phase2_distill/eval_student.py for a fair apples-to-apples comparison.

Usage (cluster):
    # AOD-Net on SOTS-indoor
    python evaluate/eval_baseline.py --model aodnet \
        --ckpt experiments/baselines/aodnet_indoor.pth \
        --tag aodnet --split indoor

    # FFA-Net on SOTS-indoor
    python evaluate/eval_baseline.py --model ffanet \
        --ckpt experiments/baselines/ffanet_indoor.pth \
        --tag ffanet --split indoor

    # SOTS-outdoor eval for any model
    python evaluate/eval_baseline.py --model ffanet \
        --ckpt experiments/baselines/ffanet_indoor.pth \
        --tag ffanet --split outdoor

Writes: results/eval_baseline_<tag>_<split>.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from evaluate.metrics import latency_ms, psnr, ssim
from models.baselines.aodnet import load_aodnet
from models.baselines.aodnet import count_params as _count
from models.baselines.ffanet import load_ffanet

ROOT = Path(__file__).resolve().parent.parent

_SPLIT_DEFAULTS = {
    "indoor":  ("data/RESIDE/SOTS-Test/valid_indoor/input",  "data/RESIDE/SOTS-Test/valid_indoor/gt"),
    "outdoor": ("data/RESIDE/SOTS-Test/valid_outdoor/input", "data/RESIDE/SOTS-Test/valid_outdoor/gt"),
}


def _load_pairs(split: str, hazy_dir: Path, gt_dir: Path):
    exts = ("*.png", "*.jpg")
    hazy_files = sorted(f for e in exts for f in hazy_dir.glob(e))
    pairs = []
    for hp in hazy_files:
        stem = hp.stem.split("_")[0]
        for ext in (".png", ".jpg"):
            gp = gt_dir / (stem + ext)
            if gp.exists():
                pairs.append((hp, gp))
                break
    return pairs


@torch.no_grad()
def eval_full(model, pairs, device):
    model.eval()
    ps, ss = [], []
    t0 = time.perf_counter()
    for hp, gp in tqdm(pairs, desc="eval", leave=False):
        hazy = to_tensor(Image.open(hp).convert("RGB")).unsqueeze(0).to(device)
        gt   = np.asarray(Image.open(gp).convert("RGB"))
        _, _, h, w = hazy.shape
        h8, w8 = (h // 8) * 8, (w // 8) * 8
        hazy = hazy[:, :, :h8, :w8]
        gt   = gt[:h8, :w8]
        out = model(hazy)
        if isinstance(out, (tuple, list)):
            out = out[0]
        out = out.clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0).cpu().numpy()
        out_u8 = (out * 255.0).astype(np.uint8)
        ps.append(psnr(out_u8, gt))
        ss.append(ssim(out_u8, gt))
    elapsed = time.perf_counter() - t0
    return {
        "psnr_mean":   float(np.mean(ps)),
        "ssim_mean":   float(np.mean(ss)),
        "psnr_min":    float(np.min(ps)),
        "psnr_max":    float(np.max(ps)),
        "ssim_min":    float(np.min(ss)),
        "ssim_max":    float(np.max(ss)),
        "n_images":    len(pairs),
        "wall_time_s": elapsed,
        "ms_per_img":  elapsed / max(1, len(pairs)) * 1000.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["aodnet", "ffanet"],
                    help="Which baseline to evaluate.")
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="Pretrained checkpoint. Omit to run random-init (sanity check only).")
    ap.add_argument("--tag", required=True, help="Short identifier for output filename.")
    ap.add_argument("--split", choices=["indoor", "outdoor"], default="indoor")
    ap.add_argument("--hazy-dir", type=Path, default=None)
    ap.add_argument("--gt-dir",   type=Path, default=None)
    ap.add_argument("--device",   default=None)
    ap.add_argument("--out",      type=Path, default=None)
    # FFA-Net options
    ap.add_argument("--ffa-gps",    type=int, default=3)
    ap.add_argument("--ffa-blocks", type=int, default=19)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"model={args.model}  split={args.split}  device={device}  ckpt={args.ckpt}")

    if args.model == "aodnet":
        model = load_aodnet(args.ckpt, device=device)
    else:
        model = load_ffanet(args.ckpt, device=device,
                            gps=args.ffa_gps, blocks=args.ffa_blocks)

    n_params, mM = _count(model)
    print(f"params: {n_params:,} ({mM:.3f}M)")

    hazy_rel, gt_rel = _SPLIT_DEFAULTS[args.split]
    hazy_dir = args.hazy_dir or (ROOT / hazy_rel)
    gt_dir   = args.gt_dir   or (ROOT / gt_rel)
    if not hazy_dir.exists():
        raise FileNotFoundError(f"hazy dir not found: {hazy_dir}")

    pairs = _load_pairs(args.split, hazy_dir, gt_dir)
    print(f"pairs: {len(pairs)}")
    if not pairs:
        raise RuntimeError(f"no pairs found — check paths")

    q = eval_full(model, pairs, device)
    print(f"\nPSNR {q['psnr_mean']:.3f}  SSIM {q['ssim_mean']:.4f}  "
          f"wall {q['wall_time_s']:.1f}s  {q['ms_per_img']:.1f} ms/img")

    lat_256 = latency_ms(model, (1, 3, 256, 256), device=device) if device == "cuda" else None
    lat_512 = latency_ms(model, (1, 3, 512, 512), device=device) if device == "cuda" else None
    if lat_256:
        print(f"256x256: {lat_256:.2f} ms  ({1000/lat_256:.1f} FPS)")
    if lat_512:
        print(f"512x512: {lat_512:.2f} ms  ({1000/lat_512:.1f} FPS)")

    out = {
        "model":      args.model,
        "tag":        args.tag,
        "split":      args.split,
        "ckpt":       str(args.ckpt) if args.ckpt else None,
        "params_M":   round(mM, 3),
        "device":     device,
        "gpu":        torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "eval":       q,
        "latency_ms_256": lat_256,
        "latency_ms_512": lat_512,
        "fps_256": (1000.0 / lat_256) if lat_256 else None,
        "fps_512": (1000.0 / lat_512) if lat_512 else None,
    }

    out_path = args.out or (ROOT / "results" / f"eval_baseline_{args.tag}_{args.split}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
