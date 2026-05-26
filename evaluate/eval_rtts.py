"""Real-world qualitative + no-reference eval on RTTS (4,322 unpaired hazy).

For each model (teacher + students), dehaze every RTTS image, save the output
PNGs under results/rtts_<tag>/, and compute mean NIQE + BRISQUE.

Writes:
    results/rtts_<tag>/<image_name>.png    — dehazed outputs (for figures)
    results/rtts_<tag>.json                — {model, ckpt, params_M, mean niqe/brisque, n_images, etc.}

Usage (cluster):
    python evaluate/eval_rtts.py --model teacher --tag dehamer_teacher \\
        --ckpt experiments/teachers/dehamer/ckpts/indoor/PSNR3663_ssim09881.pt
    python evaluate/eval_rtts.py --model student --width 32 --tag haze_b_large_tight \\
        --ckpt experiments/students/haze_b_large_tight/best.pt
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

from evaluate.fade import NoRefScorer

ROOT = Path(__file__).resolve().parent.parent


def list_rtts(rtts_dir: Path) -> list[Path]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.PNG")
    files = []
    for e in exts:
        files.extend(rtts_dir.glob(e))
        files.extend(rtts_dir.rglob(e))
    seen = set()
    out = []
    for p in sorted(files):
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


@torch.no_grad()
def dehaze_save(model, in_path: Path, out_path: Path, device: str,
                kind: str) -> None:
    if kind == "teacher":
        from models.teachers.dehamer import dehaze as dehaze_dehamer
        img = Image.open(in_path).convert("RGB")
        out = dehaze_dehamer(model, img, device=device)
        Image.fromarray(out).save(out_path)
        return

    # student / baseline: tensor in, tensor out
    img = Image.open(in_path).convert("RGB")
    t = to_tensor(img).unsqueeze(0).to(device)
    _, _, h, w = t.shape
    h8, w8 = (h // 8) * 8, (w // 8) * 8
    t = t[:, :, :h8, :w8]
    out = model(t)
    if isinstance(out, (tuple, list)):
        out = out[0]
    arr = (out.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["teacher", "student", "aodnet", "ffanet", "hazy_passthrough"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--width", type=int, default=32,
                    help="Student width (16 or 32). Ignored for non-student.")
    ap.add_argument("--rtts-dir", type=Path,
                    default=ROOT / "data/RTTS")
    ap.add_argument("--out-img-dir", type=Path, default=None)
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max", type=int, default=0, help="0=all")
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-niqe", action="store_true",
                    help="Skip NIQE/BRISQUE scoring (output PNGs only).")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"model={args.model}  tag={args.tag}  device={device}  ckpt={args.ckpt}")

    # Load model ────────────────────────────────────────────────────────────
    kind = args.model
    if kind == "teacher":
        from models.teachers.dehamer import load_dehamer
        if not args.ckpt:
            raise SystemExit("--ckpt required for teacher")
        model = load_dehamer(ckpt_path=str(args.ckpt), device=device)
    elif kind == "student":
        from models.students.nafnet_student import build_student
        if not args.ckpt:
            raise SystemExit("--ckpt required for student")
        model = build_student(width=args.width).to(device)
        sd = torch.load(args.ckpt, map_location=device, weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        model.load_state_dict(sd, strict=False)
        model.eval()
        kind = "student"
    elif kind == "aodnet":
        from models.baselines.aodnet import load_aodnet
        model = load_aodnet(args.ckpt, device=device)
        kind = "student"  # same tensor I/O
    elif kind == "ffanet":
        from models.baselines.ffanet import load_ffanet
        model = load_ffanet(args.ckpt, device=device)
        kind = "student"
    elif kind == "hazy_passthrough":
        model = None  # for sanity check (scores raw hazy images)
    else:
        raise SystemExit(f"unknown model kind: {args.model}")

    # RTTS images ───────────────────────────────────────────────────────────
    if not args.rtts_dir.exists():
        raise SystemExit(f"RTTS dir not found: {args.rtts_dir}\n"
                         f"Run scripts/download_rtts.sh first.")
    imgs = list_rtts(args.rtts_dir)
    if args.max:
        imgs = imgs[: args.max]
    if not imgs:
        raise SystemExit(f"no images found under {args.rtts_dir}")
    print(f"RTTS images: {len(imgs)}")

    # Output dir for dehazed PNGs ───────────────────────────────────────────
    out_img_dir = args.out_img_dir or (ROOT / "results" / f"rtts_{args.tag}")
    out_img_dir.mkdir(parents=True, exist_ok=True)

    # Dehaze pass ───────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    n_done = n_skipped = 0
    for p in tqdm(imgs, desc="dehaze"):
        out_path = out_img_dir / (p.stem + ".png")
        if args.skip_existing and out_path.exists():
            n_skipped += 1
            continue
        if model is None:
            Image.open(p).convert("RGB").save(out_path)
        else:
            dehaze_save(model, p, out_path, device, kind=(
                "teacher" if args.model == "teacher" else "tensor"
            ))
        n_done += 1
    elapsed = time.perf_counter() - t0
    print(f"\ndehaze: done={n_done}  skipped={n_skipped}  elapsed={elapsed:.1f}s")

    # No-reference scoring ──────────────────────────────────────────────────
    niqe_mean = brisque_mean = None
    if not args.no_niqe:
        print(f"scoring {len(imgs)} images with NIQE + BRISQUE…")
        scorer = NoRefScorer(device=device)
        ns, bs = [], []
        for p in tqdm(imgs, desc="score"):
            out_p = out_img_dir / (p.stem + ".png")
            if not out_p.exists():
                continue
            img = Image.open(out_p)
            s = scorer.score(img)
            ns.append(s["niqe"])
            bs.append(s["brisque"])
        niqe_mean = float(np.mean(ns)) if ns else None
        brisque_mean = float(np.mean(bs)) if bs else None
        print(f"NIQE mean    = {niqe_mean:.3f}")
        print(f"BRISQUE mean = {brisque_mean:.3f}")

    # Write JSON ────────────────────────────────────────────────────────────
    out_json = args.out_json or (ROOT / "results" / f"rtts_{args.tag}.json")
    out = {
        "model":     args.model,
        "tag":       args.tag,
        "ckpt":      str(args.ckpt) if args.ckpt else None,
        "device":    device,
        "gpu":       torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "n_images":  len(imgs),
        "out_dir":   str(out_img_dir),
        "wall_time_s": elapsed,
        "niqe_mean": niqe_mean,
        "brisque_mean": brisque_mean,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
