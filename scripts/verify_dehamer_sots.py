"""Load the pretrained DeHamer indoor checkpoint and run on one real SOTS indoor
hazy image. Verifies the wrapper works end-to-end with real weights + real data.

Saves the dehazed output next to the source under experiments/samples/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

from models.teachers.dehamer import count_params, dehaze, load_dehamer

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "experiments" / "teachers" / "dehamer" / "ckpts" / "indoor" / "PSNR3663_ssim09881.pt"
SOTS_HAZY = ROOT / "data" / "RESIDE" / "SOTS-Test" / "valid_indoor" / "input"
SOTS_GT = ROOT / "data" / "RESIDE" / "SOTS-Test" / "valid_indoor" / "gt"
OUT_DIR = ROOT / "experiments" / "samples"


def main() -> None:
    assert CKPT.exists(), f"Missing checkpoint {CKPT}"
    assert SOTS_HAZY.is_dir(), f"Missing SOTS indoor inputs at {SOTS_HAZY}"

    hazy_files = sorted(SOTS_HAZY.glob("*.png"))
    assert hazy_files, "no hazy .png files"
    hazy_path = hazy_files[0]
    # SOTS naming: "1400_1.png" -> gt "1400.png"
    gt_path = SOTS_GT / (hazy_path.stem.split("_")[0] + ".png")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    model = load_dehamer(ckpt_path=str(CKPT), device=device)
    n, m = count_params(model)
    print(f"params: {n:,} ({m:.2f}M)")

    hazy = Image.open(hazy_path).convert("RGB")
    gt = Image.open(gt_path).convert("RGB") if gt_path.exists() else None
    print(f"hazy: {hazy_path.name}, size={hazy.size}")
    if gt is not None:
        print(f"gt  : {gt_path.name}, size={gt.size}")

    out = dehaze(model, hazy, device=device)  # HxWx3 uint8
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"dehazed_{hazy_path.stem}.png"
    Image.fromarray(out).save(out_path)
    print(f"saved: {out_path}")

    if gt is not None:
        gt_arr = np.asarray(gt)
        # Crop both to match output (model cropped to multiples of 16)
        h, w = out.shape[:2]
        gt_arr_c = gt_arr[:h, :w]
        p = psnr_fn(gt_arr_c, out, data_range=255)
        s = ssim_fn(gt_arr_c, out, channel_axis=2, data_range=255)
        print(f"single-image PSNR={p:.2f} SSIM={s:.4f}")


if __name__ == "__main__":
    main()
