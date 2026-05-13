"""Evaluate a NAFNet student checkpoint on any RESIDE split.

Supported splits: indoor, outdoor, dense, nh.
Loads best.pt, runs on the full test set, records mean PSNR/SSIM +
CUDA-event latency at 256x256 and 512x512, writes a JSON summary.

Usage on cluster:
    # Original indoor eval (backward-compatible):
    python phase2_distill/eval_student.py \\
        --ckpt experiments/students/haze_b_large_tight/best.pt \\
        --tag haze_b_large_tight --width 32

    # Outdoor (cross-domain or dedicated outdoor student):
    python phase2_distill/eval_student.py \\
        --ckpt experiments/students/haze_b_large_tight/best.pt \\
        --tag haze_b_large_tight --width 32 --split outdoor

    # Dense-Haze (override paths if layout differs):
    python phase2_distill/eval_student.py \\
        --ckpt experiments/students/haze_b_large_tight/best.pt \\
        --tag haze_b_large_tight --width 32 --split dense \\
        --hazy-dir data/Dense-Haze/hazy --gt-dir data/Dense-Haze/GT
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
from models.students.nafnet_student import build_student, count_params

ROOT = Path(__file__).resolve().parent.parent

# Default data roots per split (override with --hazy-dir / --gt-dir).
_SPLIT_DEFAULTS: dict[str, tuple[str, str]] = {
    "indoor":  ("data/RESIDE/SOTS-Test/valid_indoor/input",  "data/RESIDE/SOTS-Test/valid_indoor/gt"),
    "outdoor": ("data/RESIDE/SOTS-Test/valid_outdoor/input", "data/RESIDE/SOTS-Test/valid_outdoor/gt"),
    "dense":   ("data/Dense-Haze/hazy",                      "data/Dense-Haze/GT"),
    "nh":      ("data/NH-HAZE/hazy",                         "data/NH-HAZE/GT"),
}


def _load_pairs(split: str, hazy_dir: Path, gt_dir: Path) -> list[tuple[Path, Path]]:
    """Return (hazy, gt) path pairs for the given split."""
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
        # Dense-Haze / NH-HAZE: same stem in both dirs.
        for hp in hazy_files:
            for ext in (".png", ".jpg"):
                gp = gt_dir / (hp.stem + ext)
                if gp.exists():
                    pairs.append((hp, gp))
                    break
        # Fallback: pair by sorted index.
        if not pairs:
            gt_files = sorted(f for e in exts for f in gt_dir.glob(e))
            pairs = list(zip(hazy_files, gt_files))

    return pairs


@torch.no_grad()
def eval_full(student, pairs: list[tuple[Path, Path]], device: str) -> dict:
    student.eval()
    ps, ss = [], []
    t0 = time.perf_counter()
    for hp, gp in tqdm(pairs, desc="eval", leave=False):
        hazy = to_tensor(Image.open(hp).convert("RGB")).unsqueeze(0)
        gt = np.asarray(Image.open(gp).convert("RGB"))
        _, _, h, w = hazy.shape
        h8, w8 = (h // 8) * 8, (w // 8) * 8
        hazy = hazy[:, :, :h8, :w8]
        gt = gt[:h8, :w8]
        out, _ = student(hazy.to(device))
        out = out.clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0).cpu().numpy()
        out_u8 = (out * 255.0).astype(np.uint8)
        ps.append(psnr(out_u8, gt))
        ss.append(ssim(out_u8, gt))
    elapsed = time.perf_counter() - t0
    return {
        "psnr_mean":  float(np.mean(ps)),
        "ssim_mean":  float(np.mean(ss)),
        "psnr_min":   float(np.min(ps)),
        "psnr_max":   float(np.max(ps)),
        "ssim_min":   float(np.min(ss)),
        "ssim_max":   float(np.max(ss)),
        "n_images":   len(pairs),
        "wall_time_s": elapsed,
        "ms_per_img":  elapsed / max(1, len(pairs)) * 1000.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--split", choices=["indoor", "outdoor", "dense", "nh"],
                    default="indoor",
                    help="Evaluation split. Default: indoor (backward-compatible).")
    ap.add_argument("--hazy-dir", type=Path, default=None,
                    help="Override hazy image directory (uses split default otherwise).")
    ap.add_argument("--gt-dir", type=Path, default=None,
                    help="Override GT directory (uses split default otherwise).")
    ap.add_argument("--device", default=None, help="cuda|cpu (default auto)")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output JSON path (default: results/eval_student_<tag>_<split>.json).")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  ckpt={args.ckpt}  width={args.width}  split={args.split}")

    # Resolve data dirs.
    hazy_rel, gt_rel = _SPLIT_DEFAULTS[args.split]
    hazy_dir = args.hazy_dir or (ROOT / hazy_rel)
    gt_dir   = args.gt_dir   or (ROOT / gt_rel)
    if not hazy_dir.exists():
        raise FileNotFoundError(f"hazy dir not found: {hazy_dir}\n"
                                f"Pass --hazy-dir to override or download the dataset first.")
    if not gt_dir.exists():
        raise FileNotFoundError(f"GT dir not found: {gt_dir}\n"
                                f"Pass --gt-dir to override or download the dataset first.")

    student = build_student(width=args.width).to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    student.load_state_dict(ck["student"])
    n_params, mM = count_params(student)
    print(f"student params: {n_params:,} ({mM:.2f}M)")

    pairs = _load_pairs(args.split, hazy_dir, gt_dir)
    print(f"pairs ({args.split}): {len(pairs)}")
    if not pairs:
        raise RuntimeError(f"no pairs found in {hazy_dir} / {gt_dir}")

    q = eval_full(student, pairs, device)
    print(f"\nPSNR {q['psnr_mean']:.3f}  SSIM {q['ssim_mean']:.4f}"
          f"   (min {q['psnr_min']:.2f}, max {q['psnr_max']:.2f})"
          f"   wall {q['wall_time_s']:.1f}s  {q['ms_per_img']:.1f} ms/img")

    lat_256 = latency_ms(student, (1, 3, 256, 256), device=device) if device == "cuda" else None
    lat_512 = latency_ms(student, (1, 3, 512, 512), device=device) if device == "cuda" else None
    if lat_256:
        print(f"256x256: {lat_256:.2f} ms/img  ({1000/lat_256:.1f} FPS)")
    if lat_512:
        print(f"512x512: {lat_512:.2f} ms/img  ({1000/lat_512:.1f} FPS)")

    cfg = ck.get("config", {})
    suffix = "" if args.split == "indoor" else f"_{args.split}"
    out = {
        "model": "NAFNet-student",
        "tag": args.tag,
        "split": args.split,
        "ckpt": str(args.ckpt),
        "best_psnr_train_reported": float(ck.get("best_psnr", -1.0)),
        "epoch_saved": int(ck.get("epoch", -1)),
        "n_params": n_params,
        "params_M": round(mM, 3),
        "device": device,
        "width": args.width,
        "train_config": {k: cfg.get(k) for k in
                         ("width", "lambda_feat", "lambda_perc",
                          "use_pseudo_as_target", "epochs", "batch", "patch")},
        "eval": q,
        "latency_ms_256": lat_256,
        "latency_ms_512": lat_512,
        "fps_256": (1000.0 / lat_256) if lat_256 else None,
        "fps_512": (1000.0 / lat_512) if lat_512 else None,
    }

    out_path = args.out or (ROOT / "results" / f"eval_student_{args.tag}{suffix}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
