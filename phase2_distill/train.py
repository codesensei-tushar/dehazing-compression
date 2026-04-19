"""Phase-2 haze student training loop.

    Dataset : RESIDE ITS (13,990 hazy/clean pairs) + optional DeHamer pseudo-clean PNGs.
    Student : NAFNet (width=16, enc=[1,1,1,28], dec=[1,1,1,1], ~4.35M params).
    Teacher : offline (pseudo PNGs) — default.  Inline teacher is optional.
    Loss    : L_pixel(L1) + lambda_feat * L_feat(L2 on decoder taps, adapter).
    Optim   : AdamW betas=(0.9, 0.9), lr 1e-3 cosine -> 1e-6.
    Sched   : 200 epochs (or --epochs override).
    Input   : 128x128 random crops, random flips + 90° rotations.
    Batch   : 8 (fits ~10 GB VRAM).

Validation runs every `--val-interval` epochs on SOTS-indoor (500 pairs),
reports PSNR/SSIM on full-image inputs.

Checkpoints: experiments/students/<tag>/epoch_<N>.pt + best.pt.
Logs: wandb if available; otherwise stdout only.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.reside import ITSPairDataset, SOTSEvalDataset
from evaluate.metrics import psnr, ssim
from models.students.nafnet_student import FeatureAdapter, build_student, count_params
from phase2_distill.losses import DistillationLoss, DistillCfg

ROOT = Path(__file__).resolve().parent.parent


def cosine_lr(step: int, total: int, lr_hi: float, lr_lo: float) -> float:
    if total <= 0:
        return lr_hi
    frac = min(step / total, 1.0)
    return lr_lo + 0.5 * (lr_hi - lr_lo) * (1.0 + math.cos(math.pi * frac))


@torch.no_grad()
def validate(student: nn.Module, loader: DataLoader, device: str) -> dict:
    student.eval()
    psnrs, ssims = [], []
    for hazy, gt, _name in loader:
        hazy = hazy.to(device, non_blocking=True)
        out, _ = student(hazy)
        out = out.clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0).cpu().numpy()
        gt_np = gt.squeeze(0).permute(1, 2, 0).cpu().numpy()
        out_u8 = (out * 255.0).astype(np.uint8)
        gt_u8 = (gt_np * 255.0).astype(np.uint8)
        psnrs.append(psnr(out_u8, gt_u8))
        ssims.append(ssim(out_u8, gt_u8))
    return {"psnr": float(np.mean(psnrs)), "ssim": float(np.mean(ssims)),
            "psnr_min": float(np.min(psnrs)), "psnr_max": float(np.max(psnrs))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hazy-dir", type=Path, default=ROOT / "data/RESIDE/ITS-Train/train_indoor/haze")
    ap.add_argument("--clean-dir", type=Path, default=ROOT / "data/RESIDE/ITS-Train/train_indoor/clear_images")
    ap.add_argument("--pseudo-dir", type=Path, default=None,
                    help="DeHamer pseudo-clean PNGs (optional). Enables teacher-supervised L_pixel.")
    ap.add_argument("--sots-hazy", type=Path, default=ROOT / "data/RESIDE/SOTS-Test/valid_indoor/input")
    ap.add_argument("--sots-gt", type=Path, default=ROOT / "data/RESIDE/SOTS-Test/valid_indoor/gt")

    ap.add_argument("--tag", default="haze_s1")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr-hi", type=float, default=1e-3)
    ap.add_argument("--lr-lo", type=float, default=1e-6)
    ap.add_argument("--lambda-feat", type=float, default=0.01)
    ap.add_argument("--lambda-perc", type=float, default=0.0)
    ap.add_argument("--use-pseudo-as-target", action="store_true",
                    help="L_pixel against teacher pseudo-clean instead of GT.")
    ap.add_argument("--val-interval", type=int, default=5, help="Epochs between SOTS validation runs.")
    ap.add_argument("--ckpt-interval", type=int, default=10)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--resume", type=Path, default=None)
    args = ap.parse_args()

    ckpt_dir = ROOT / "experiments/students" / args.tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  tag={args.tag}")

    # --- Data ---
    train_ds = ITSPairDataset(
        hazy_dir=args.hazy_dir,
        clean_dir=args.clean_dir,
        pseudo_dir=args.pseudo_dir,
        patch_size=args.patch,
        augment=True,
    )
    print(f"train pairs: {len(train_ds)}")
    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
    )

    val_ds = SOTSEvalDataset(hazy_dir=args.sots_hazy, gt_dir=args.sots_gt)
    print(f"val pairs  : {len(val_ds)}")
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    # --- Model + adapter ---
    student = build_student().to(device)
    n, mM = count_params(student)
    print(f"student params: {n:,} ({mM:.2f}M)")

    # We match student decoder feature (tap_channels) with the pseudo teacher target's shape.
    # The teacher tap channel count is effectively 3 (its dehazed output), so the adapter
    # maps student 16ch -> 3ch to match a pixel-space target.
    feat_adapter = FeatureAdapter(student_ch=student.tap_channels, teacher_ch=3).to(device)

    criterion = DistillationLoss(DistillCfg(lambda_feat=args.lambda_feat,
                                            lambda_perc=args.lambda_perc)).to(device)

    optim = torch.optim.AdamW(
        list(student.parameters()) + list(feat_adapter.parameters()),
        lr=args.lr_hi, betas=(0.9, 0.9),
    )

    # --- Resume ---
    start_epoch = 0
    global_step = 0
    best_psnr = -1.0
    if args.resume and args.resume.exists():
        ck = torch.load(args.resume, map_location=device)
        student.load_state_dict(ck["student"])
        feat_adapter.load_state_dict(ck["adapter"])
        optim.load_state_dict(ck["optim"])
        start_epoch = ck.get("epoch", 0) + 1
        global_step = ck.get("global_step", 0)
        best_psnr = ck.get("best_psnr", -1.0)
        print(f"resumed from {args.resume}: epoch={start_epoch} step={global_step} best={best_psnr:.3f}")

    # --- Wandb (optional) ---
    wb = None
    if args.wandb:
        try:
            import wandb as wb_mod
            wb_mod.init(project="dehazing-compression", name=args.tag, config=vars(args))
            wb = wb_mod
        except Exception as e:
            print(f"wandb init failed ({e}); proceeding without it.")

    total_steps = args.epochs * len(train_loader)

    for epoch in range(start_epoch, args.epochs):
        student.train()
        t0 = time.perf_counter()
        pbar = tqdm(train_loader, desc=f"ep {epoch:03d}", leave=False)
        ep_losses = {"loss": 0.0, "l_pixel": 0.0, "l_feat": 0.0}
        for hazy, gt, pseudo in pbar:
            hazy = hazy.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            pseudo = pseudo.to(device, non_blocking=True)

            # cosine LR
            lr = cosine_lr(global_step, total_steps, args.lr_hi, args.lr_lo)
            for g in optim.param_groups:
                g["lr"] = lr

            target = pseudo if args.use_pseudo_as_target else gt
            out, feat = student(hazy)
            projected = feat_adapter(feat) if args.lambda_feat > 0 else None
            # "Teacher feature" here is the pseudo image itself, used as a pixel-resolution target.
            teacher_feat = pseudo if args.lambda_feat > 0 else None

            losses = criterion(out, target, projected, teacher_feat)
            optim.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(list(student.parameters()) + list(feat_adapter.parameters()), 1.0)
            optim.step()

            global_step += 1
            for k in ep_losses:
                ep_losses[k] += float(losses[k])
            pbar.set_postfix(loss=f"{float(losses['loss']):.4f}", lr=f"{lr:.2e}")

        n_batch = max(1, len(train_loader))
        for k in ep_losses:
            ep_losses[k] /= n_batch

        msg = (f"ep {epoch:03d}  loss {ep_losses['loss']:.4f}  "
               f"l_pix {ep_losses['l_pixel']:.4f}  l_feat {ep_losses['l_feat']:.4f}  "
               f"lr {lr:.2e}  ({time.perf_counter()-t0:.0f}s)")
        print(msg)
        if wb:
            wb.log({"train/" + k: v for k, v in ep_losses.items()} | {"train/lr": lr, "epoch": epoch})

        # Validate
        if (epoch + 1) % args.val_interval == 0 or epoch == args.epochs - 1:
            val = validate(student, val_loader, device)
            print(f"  VAL  PSNR {val['psnr']:.3f}  SSIM {val['ssim']:.4f}  "
                  f"(min {val['psnr_min']:.2f}, max {val['psnr_max']:.2f})")
            if wb:
                wb.log({"val/psnr": val["psnr"], "val/ssim": val["ssim"], "epoch": epoch})
            if val["psnr"] > best_psnr:
                best_psnr = val["psnr"]
                torch.save({"student": student.state_dict(),
                            "adapter": feat_adapter.state_dict(),
                            "optim": optim.state_dict(),
                            "epoch": epoch, "global_step": global_step,
                            "best_psnr": best_psnr, "config": vars(args)},
                           ckpt_dir / "best.pt")
                print(f"  -> new best, saved to {ckpt_dir / 'best.pt'}")

        # Periodic checkpoint
        if (epoch + 1) % args.ckpt_interval == 0:
            torch.save({"student": student.state_dict(),
                        "adapter": feat_adapter.state_dict(),
                        "optim": optim.state_dict(),
                        "epoch": epoch, "global_step": global_step,
                        "best_psnr": best_psnr, "config": vars(args)},
                       ckpt_dir / f"epoch_{epoch:03d}.pt")

    print(f"\nDone.  best VAL PSNR = {best_psnr:.3f}  (ckpt: {ckpt_dir / 'best.pt'})")
    # Persist a compact summary JSON
    (ckpt_dir / "training_summary.json").write_text(json.dumps({
        "tag": args.tag,
        "student_params_M": mM,
        "epochs_trained": args.epochs,
        "best_psnr": best_psnr,
        "config": vars(args),
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
