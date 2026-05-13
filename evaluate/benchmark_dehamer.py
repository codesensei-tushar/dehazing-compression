"""DeHamer FP32 baseline across all supported splits.

Evaluates the per-split pretrained checkpoint on all paired images, reports
mean PSNR/SSIM, and measures latency at 256x256 and 512x512 inputs.

Supported splits:
    indoor   — SOTS-indoor 500 pairs, indoor checkpoint
    outdoor  — SOTS-outdoor 500 pairs, outdoor checkpoint
    dense    — Dense-Haze 55 pairs, dense-haze checkpoint
    nh       — NH-HAZE 55 pairs, NH checkpoint

Run on the cluster:
    CUDA_VISIBLE_DEVICES=1 python evaluate/benchmark_dehamer.py --split indoor
    CUDA_VISIBLE_DEVICES=1 python evaluate/benchmark_dehamer.py --split outdoor
    CUDA_VISIBLE_DEVICES=1 python evaluate/benchmark_dehamer.py --split dense \\
        --hazy-dir data/Dense-Haze/hazy --gt-dir data/Dense-Haze/GT
    CUDA_VISIBLE_DEVICES=1 python evaluate/benchmark_dehamer.py --split nh \\
        --hazy-dir data/NH-HAZE/hazy --gt-dir data/NH-HAZE/GT
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

CKPTS: dict[str, Path] = {
    "indoor":  ROOT / "experiments/teachers/dehamer/ckpts/indoor/PSNR3663_ssim09881.pt",
    "outdoor": ROOT / "experiments/teachers/dehamer/ckpts/outdoor/PSNR3518_SSIM09860.pt",
    "dense":   ROOT / "experiments/teachers/dehamer/ckpts/dense/PSNR1662_SSIM05602.pt",
    "nh":      ROOT / "experiments/teachers/dehamer/ckpts/NH/PSNR2066_SSIM06844.pt",
}

# Default data dirs per split — override with --hazy-dir / --gt-dir.
_DATA_DIRS: dict[str, tuple[str, str]] = {
    "indoor":  ("data/RESIDE/SOTS-Test/valid_indoor/input",  "data/RESIDE/SOTS-Test/valid_indoor/gt"),
    "outdoor": ("data/RESIDE/SOTS-Test/valid_outdoor/input", "data/RESIDE/SOTS-Test/valid_outdoor/gt"),
    "dense":   ("data/Dense-Haze/hazy",                      "data/Dense-Haze/GT"),
    "nh":      ("data/NH-HAZE/hazy",                         "data/NH-HAZE/GT"),
}


def _load_pairs(split: str, hazy_dir: Path, gt_dir: Path) -> list[tuple[Path, Path]]:
    exts = ("*.png", "*.jpg")
    hazy_files = sorted(f for e in exts for f in hazy_dir.glob(e))
    pairs: list[tuple[Path, Path]] = []

    if split in ("indoor", "outdoor"):
        # SOTS convention: <id>_<k>_<beta>.png -> GT <id>.png
        for hp in hazy_files:
            stem = hp.stem.split("_")[0]
            for ext in (".png", ".jpg"):
                gp = gt_dir / (stem + ext)
                if gp.exists():
                    pairs.append((hp, gp))
                    break
    else:
        # Dense-Haze / NH-HAZE: same stem in hazy and GT dirs.
        for hp in hazy_files:
            for ext in (".png", ".jpg"):
                gp = gt_dir / (hp.stem + ext)
                if gp.exists():
                    pairs.append((hp, gp))
                    break
        if not pairs:
            gt_files = sorted(f for e in exts for f in gt_dir.glob(e))
            pairs = list(zip(hazy_files, gt_files))

    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["indoor", "outdoor", "dense", "nh"],
                    default="indoor")
    ap.add_argument("--hazy-dir", type=Path, default=None,
                    help="Override hazy dir (uses split default otherwise).")
    ap.add_argument("--gt-dir", type=Path, default=None,
                    help="Override GT dir (uses split default otherwise).")
    ap.add_argument("--max-pairs", type=int, default=0, help="0 = all")
    ap.add_argument("--out", default=None, help="Output JSON path.")
    args = ap.parse_args()

    ckpt = CKPTS[args.split]
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {ckpt}\n"
            f"Download DeHamer checkpoints and place at the path above."
        )

    hazy_rel, gt_rel = _DATA_DIRS[args.split]
    hazy_dir = args.hazy_dir or (ROOT / hazy_rel)
    gt_dir   = args.gt_dir   or (ROOT / gt_rel)
    if not hazy_dir.exists():
        raise FileNotFoundError(
            f"hazy dir not found: {hazy_dir}\n"
            f"Pass --hazy-dir to specify the correct path."
        )
    if not gt_dir.exists():
        raise FileNotFoundError(
            f"GT dir not found: {gt_dir}\n"
            f"Pass --gt-dir to specify the correct path."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  split={args.split}  ckpt={ckpt.name}")

    model = load_dehamer(ckpt_path=str(ckpt), device=device)
    n_params, m = count_params(model)
    print(f"params={n_params:,} ({m:.2f}M)")

    pairs = _load_pairs(args.split, hazy_dir, gt_dir)
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]
    print(f"#pairs={len(pairs)}")
    if not pairs:
        raise RuntimeError(f"no pairs found in {hazy_dir} / {gt_dir}")

    psnrs: list[float] = []
    ssims: list[float] = []
    t_start = perf_counter()
    for hp, gp in tqdm(pairs, desc=f"DeHamer FP32 [{args.split}]"):
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
        "psnr_min": float(np.min(psnrs)),
        "psnr_max": float(np.max(psnrs)),
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
