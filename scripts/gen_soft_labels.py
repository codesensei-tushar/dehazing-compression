"""Pre-generate DeHamer teacher outputs for all ITS hazy images.

Saves dehazed PNGs under experiments/soft_labels/dehamer_<split>/ with the
same filename as the input hazy image. Reuses the DeHamer wrapper's preprocess
and dehaze helpers. Runs on GPU if available.

Why offline: DeHamer is slow (~190 ms/img GPU) and ITS has 13,990 hazy images.
Running it inside the student training loop would dominate wall-clock time
(~45 min per epoch for teacher forward alone). Pre-generating once (~45 min
total) and reading PNGs off disk during training decouples the two costs.

Usage (cluster):
    python scripts/gen_soft_labels.py \\
        --ckpt experiments/teachers/dehamer/ckpts/indoor/PSNR3663_ssim09881.pt \\
        --hazy-dir data/RESIDE/ITS-Train/train_indoor/haze \\
        --out-dir experiments/soft_labels/dehamer_indoor
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image
from tqdm import tqdm

from models.teachers.dehamer import dehaze, load_dehamer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--hazy-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max", type=int, default=0, help="0 = all")
    ap.add_argument("--skip-existing", action="store_true", default=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} ckpt={args.ckpt.name}")

    model = load_dehamer(ckpt_path=str(args.ckpt), device=device)

    images = sorted(list(args.hazy_dir.glob("*.png")) + list(args.hazy_dir.glob("*.jpg")))
    if args.max:
        images = images[: args.max]
    print(f"hazy images: {len(images)}  -> {args.out_dir}")

    done = skipped = 0
    t0 = time.perf_counter()
    for p in tqdm(images, desc="teacher"):
        out_path = args.out_dir / (p.stem + ".png")
        if args.skip_existing and out_path.exists():
            skipped += 1
            continue
        img = Image.open(p).convert("RGB")
        pseudo = dehaze(model, img, device=device)
        Image.fromarray(pseudo).save(out_path)
        done += 1

    elapsed = time.perf_counter() - t0
    rate = done / elapsed if done else 0.0
    print(f"\ndone={done}  skipped={skipped}  elapsed={elapsed:.1f}s  rate={rate:.1f} img/s")


if __name__ == "__main__":
    main()
