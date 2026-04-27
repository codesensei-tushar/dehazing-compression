"""Render before/after qualitative panels for the paper.

Picks N SOTS-indoor pairs, runs DeHamer (teacher) and the three NAFNet
students (Node A w16, Node B w32 GT, Node C w32 pseudo) on each, saves
individual model outputs and a horizontal strip per image.

Outputs land under ``results/qualitative/`` so they're committable
(small PNGs, separate from the large ``experiments/`` checkpoint tree).

Run on the cluster (GPU + data are there):
    python scripts/gen_qualitative_samples.py --n 6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import to_tensor

from models.students.nafnet_student import build_student
from models.teachers.dehamer import dehaze as dehaze_teacher
from models.teachers.dehamer import load_dehamer

ROOT = Path(__file__).resolve().parents[1]
SOTS = ROOT / "data/RESIDE/SOTS-Test/valid_indoor"
OUT  = ROOT / "results/qualitative"

STUDENTS = [
    ("nodeA_w16",  "haze_a_small_tight",  16),
    ("nodeB_w32",  "haze_b_large_tight",  32),
    ("nodeC_w32p", "haze_c_large_pseudo", 32),
]
TEACHER_CKPT = ROOT / "experiments/teachers/dehamer/ckpts/indoor/PSNR3663_ssim09881.pt"


def pick_pairs(n: int) -> list[tuple[Path, Path, str]]:
    pairs = []
    for hp in sorted((SOTS / "input").glob("*.png")):
        stem = hp.stem.split("_")[0]
        gp = (SOTS / "gt") / (stem + ".png")
        if gp.exists():
            pairs.append((hp, gp, hp.stem))
        if len(pairs) >= n:
            break
    return pairs


@torch.no_grad()
def run_student(model, img: Image.Image, device: str) -> np.ndarray:
    x = to_tensor(img).unsqueeze(0)
    _, _, h, w = x.shape
    h8, w8 = (h // 8) * 8, (w // 8) * 8
    x = x[:, :, :h8, :w8].to(device)
    out, _ = model(x)
    out = out.clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0).cpu().numpy()
    return (out * 255.0).astype(np.uint8)


def label_band(width: int, text: str, h: int = 22) -> Image.Image:
    band = Image.new("RGB", (width, h), (255, 255, 255))
    drw  = ImageDraw.Draw(band)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    drw.text((6, 3), text, fill=(0, 0, 0), font=font)
    return band


def hstack(arrays: list[np.ndarray], labels: list[str]) -> Image.Image:
    h = min(a.shape[0] for a in arrays)
    w = min(a.shape[1] for a in arrays)
    arrays = [a[:h, :w] for a in arrays]
    img = Image.fromarray(np.concatenate(arrays, axis=1))
    band = Image.new("RGB", (img.width, 22 * 1), (255, 255, 255))
    drw  = ImageDraw.Draw(band)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    for i, lbl in enumerate(labels):
        drw.text((i * w + 6, 3), lbl, fill=(0, 0, 0), font=font)
    out = Image.new("RGB", (img.width, img.height + band.height), (255, 255, 255))
    out.paste(band, (0, 0))
    out.paste(img, (0, band.height))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6, help="number of SOTS-indoor pairs to render")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)

    pairs = pick_pairs(args.n)
    print(f"selected {len(pairs)} pairs")

    teacher = load_dehamer(TEACHER_CKPT, device=device)
    print(f"teacher loaded from {TEACHER_CKPT.name}")

    students = []
    for label, tag, w in STUDENTS:
        s = build_student(width=w).to(device).eval()
        ck = torch.load(ROOT / f"experiments/students/{tag}/best.pt",
                        map_location=device, weights_only=False)
        s.load_state_dict(ck["student"])
        for p in s.parameters():
            p.requires_grad_(False)
        students.append((label, s))
        print(f"student {label}: {tag} loaded")

    for hp, gp, stem in pairs:
        d = OUT / stem
        d.mkdir(parents=True, exist_ok=True)
        hazy_img = Image.open(hp).convert("RGB")
        gt_img   = Image.open(gp).convert("RGB")
        hazy_arr = np.asarray(hazy_img)
        gt_arr   = np.asarray(gt_img)

        teacher_out = dehaze_teacher(teacher, hazy_img, device=device)

        student_outs = []
        for label, s in students:
            out = run_student(s, hazy_img, device)
            Image.fromarray(out).save(d / f"{label}.png")
            student_outs.append((label, out))

        Image.fromarray(hazy_arr).save(d / "hazy.png")
        Image.fromarray(gt_arr).save(d / "gt.png")
        Image.fromarray(teacher_out).save(d / "teacher_dehamer.png")

        labels = ["hazy", "GT", "DeHamer", "Node A (w16)", "Node B (w32 GT)", "Node C (w32 pseudo)"]
        arrays = [hazy_arr, gt_arr, teacher_out] + [o for _, o in student_outs]
        strip  = hstack(arrays, labels)
        strip.save(d / "strip.png")
        print(f"  {stem}: wrote 4 individual PNGs + strip.png")

    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()
