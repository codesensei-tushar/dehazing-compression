"""Sensitivity-driven distillation: pick teacher feature taps + per-tap loss
weight from the Phase-1 PTQ sensitivity ranking.

Motivation
----------
Phase 1 (PTQ) measures which Swin Linear modules in DeHamer are most
sensitive to INT8 quantization — i.e. which ones carry the most
information per parameter. Phase 2 (distillation) needs to choose which
teacher activations to match in the student. Currently these two choices
are independent (sensitivity ranking is used for mixed-precision PTQ only;
distillation uses an arbitrary single tap). This script ties them: the
per-stage sensitivity weight directly informs the per-stage feature-matching
weight in the distillation loss.

Concretely:
    1. Load `results/dehamer_sensitivity_indoor.json`.
    2. Aggregate per-stage sensitivity:
            w_s = sum(|Δ_PSNR| of all modules with ".layers.{s}." in name)
       normalised so Σ w_s = 1.
    3. Hook the DeHamer teacher at the end of each Swin stage (3 taps).
    4. Hook the NAFNet student at every decoder block (4 taps).
       Pick the 3 deepest student taps as counterpart to the 3 teacher stages.
    5. Per tap pair: bilinear-resample student tap to teacher's H,W, project
       channels via a 1x1 conv adapter, compute L2 to the teacher feature,
       scale by w_s and λ_feat.
    6. Train the student exactly as the baseline Node B configuration
       (w=32, GT target, 200 epochs, batch 8, patch 128).

Comparison axis (vs Node B):
    - Same student capacity, same supervision target, same losses except the
      L_feat now matches multiple teacher decoder stages, weighted by
      Phase-1 sensitivity.
    - If sensitivity-driven taps beat the arbitrary single tap, Phase 1 and
      Phase 2 are linked into one method instead of two stapled strategies.

Usage (cluster, on 172.18.40.113):
    python phase2_distill/train_sensitivity_taps.py \\
        --tag haze_b_sens --width 32 --epochs 200 --batch 8 --patch 128 \\
        --lambda-feat 0.05 --lambda-perc 0.05 --wandb
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.reside import ITSPairDataset, SOTSEvalDataset
from evaluate.metrics import psnr as psnr_fn, ssim as ssim_fn
from models.students.nafnet_student_multitap import (
    MultiTapAdapter,
    build_student_multitap,
    count_params,
)
from models.teachers.dehamer import load_dehamer

ROOT = Path(__file__).resolve().parent.parent


# ───────────────────────── sensitivity → per-stage weights ─────────────────

def stage_weights(sensitivity_json: Path, n_stages: int = 3) -> List[float]:
    """Aggregate per-Linear sensitivity into a vector of stage weights."""
    d = json.loads(sensitivity_json.read_text())
    sums = [0.0] * n_stages
    for entry in d["per_module"]:
        name = entry["module"]
        for s in range(n_stages):
            if f".layers.{s}." in name:
                sums[s] += abs(entry["delta_vs_baseline"])
                break
    total = sum(sums) or 1.0
    return [v / total for v in sums]


# ───────────────────────── teacher hooks ──────────────────────────────────

class TeacherStageTaps:
    """Register forward hooks on DeHamer's Swin stages (layers.0/1/2) and
    expose the post-stage feature maps as a list."""

    def __init__(self, dehamer_model: nn.Module) -> None:
        self.model = dehamer_model
        self.features: Dict[int, torch.Tensor] = {}
        self._handles = []
        # DeHamer's Swin trunk is at dehamer_model.swin_1 in the wrapper; the
        # raw model uses different attribute paths depending on the version.
        # We search by name to be robust.
        # Each "stage" is a BasicLayer at swin_1.layers.<s> with `blocks` inside.
        target_modules = []
        for name, mod in dehamer_model.named_modules():
            if name.endswith("swin_1.layers.0") or name.endswith("swin_1.layers.1") \
                    or name.endswith("swin_1.layers.2"):
                stage = int(name.split(".")[-1])
                target_modules.append((stage, name, mod))
        if len(target_modules) != 3:
            raise RuntimeError(
                f"expected 3 Swin stages, got {len(target_modules)}: "
                f"{[t[1] for t in target_modules]}"
            )
        for stage, _name, mod in target_modules:
            self._handles.append(mod.register_forward_hook(self._make_hook(stage)))

    def _make_hook(self, stage: int):
        def _hook(module, inputs, outputs):
            # BasicLayer typically returns (feat_map, ...) or just feat_map.
            feat = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
            self.features[stage] = feat
        return _hook

    def grab(self, n: int = 3) -> List[torch.Tensor]:
        feats = []
        for s in range(n):
            if s not in self.features:
                raise RuntimeError(f"teacher stage {s} feature never captured")
            feats.append(self.features[s])
        return feats

    def remove(self) -> None:
        for h in self._handles:
            h.remove()


def reshape_to_2d(feat: torch.Tensor) -> torch.Tensor:
    """Swin stages return (B, H*W, C). NAFNet decoders return (B, C, H, W).
    Reshape Swin to (B, C, H, W) by sqrt assumption (square inputs).
    """
    if feat.dim() == 4:
        return feat
    if feat.dim() == 3:
        B, N, C = feat.shape
        s = int(round(math.sqrt(N)))
        if s * s != N:
            # Padded — drop trailing tokens (DeHamer pads to multiples of window size).
            feat = feat[:, : s * s, :]
        return feat.transpose(1, 2).reshape(B, C, s, s).contiguous()
    raise ValueError(f"unexpected feature shape: {feat.shape}")


# ───────────────────────── loss ──────────────────────────────────────────

class MultiTapDistillLoss(nn.Module):
    def __init__(self, lambda_feat: float, lambda_perc: float,
                 stage_w: List[float]) -> None:
        super().__init__()
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()
        self.lambda_feat = lambda_feat
        self.lambda_perc = lambda_perc
        self.stage_w = stage_w
        if lambda_perc > 0:
            try:
                from torchvision.models import vgg16, VGG16_Weights
                vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].eval()
                for p in vgg.parameters():
                    p.requires_grad = False
                self.vgg = vgg
            except Exception:
                self.vgg = None
                print("warning: VGG load failed; perceptual loss disabled")
        else:
            self.vgg = None

    def perceptual(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.vgg is None:
            return torch.zeros((), device=x.device)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        xn = (x - mean) / std
        yn = (y - mean) / std
        return self.l2(self.vgg(xn), self.vgg(yn))

    def forward(self, student_out: torch.Tensor,
                target: torch.Tensor,
                student_feats_proj: List[torch.Tensor],
                teacher_feats: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        l_pix = self.l1(student_out, target)
        # multi-tap feature loss
        l_feat = torch.zeros((), device=student_out.device)
        for w, sf, tf in zip(self.stage_w, student_feats_proj, teacher_feats):
            tf2d = reshape_to_2d(tf)
            # Resample student to teacher H,W
            sf_r = F.interpolate(sf, size=tf2d.shape[-2:], mode="bilinear", align_corners=False)
            l_feat = l_feat + w * self.l2(sf_r, tf2d)
        l_perc = self.perceptual(student_out.clamp(0, 1), target.clamp(0, 1))
        total = l_pix + self.lambda_feat * l_feat + self.lambda_perc * l_perc
        return {"loss": total, "l_pixel": l_pix, "l_feat": l_feat, "l_perc": l_perc}


# ───────────────────────── train ─────────────────────────────────────────

def cosine_lr(step: int, total: int, lr_hi: float, lr_lo: float) -> float:
    if total <= 0:
        return lr_hi
    frac = min(step / total, 1.0)
    return lr_lo + 0.5 * (lr_hi - lr_lo) * (1.0 + math.cos(math.pi * frac))


@torch.no_grad()
def validate(student, loader, device):
    student.eval()
    ps, ss = [], []
    for hazy, gt, _name in loader:
        hazy = hazy.to(device, non_blocking=True)
        out, _ = student(hazy)
        out = out.clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0).cpu().numpy()
        gt_np = gt.squeeze(0).permute(1, 2, 0).cpu().numpy()
        out_u8 = (out * 255.0).astype(np.uint8)
        gt_u8 = (gt_np * 255.0).astype(np.uint8)
        ps.append(psnr_fn(out_u8, gt_u8))
        ss.append(ssim_fn(out_u8, gt_u8))
    student.train()
    return {"psnr": float(np.mean(ps)), "ssim": float(np.mean(ss))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="haze_b_sens")
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--hazy-dir", type=Path,
                    default=ROOT / "data/RESIDE/ITS-Train/train_indoor/haze")
    ap.add_argument("--clean-dir", type=Path,
                    default=ROOT / "data/RESIDE/ITS-Train/train_indoor/clear_images")
    ap.add_argument("--pseudo-dir", type=Path, default=None)
    ap.add_argument("--sots-hazy", type=Path,
                    default=ROOT / "data/RESIDE/SOTS-Test/valid_indoor/input")
    ap.add_argument("--sots-gt", type=Path,
                    default=ROOT / "data/RESIDE/SOTS-Test/valid_indoor/gt")
    ap.add_argument("--teacher-ckpt", type=Path,
                    default=ROOT / "experiments/teachers/dehamer/ckpts/indoor/PSNR3663_ssim09881.pt")
    ap.add_argument("--sensitivity-json", type=Path,
                    default=ROOT / "results/dehamer_sensitivity_indoor.json")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr-hi", type=float, default=1e-3)
    ap.add_argument("--lr-lo", type=float, default=1e-6)
    ap.add_argument("--lambda-feat", type=float, default=0.05)
    ap.add_argument("--lambda-perc", type=float, default=0.05)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--val-interval", type=int, default=5)
    ap.add_argument("--ckpt-interval", type=int, default=10)
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    ckpt_dir = ROOT / "experiments/students" / args.tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  tag={args.tag}  width={args.width}")

    # Sensitivity-derived stage weights
    sw = stage_weights(args.sensitivity_json, n_stages=3)
    print(f"per-stage weights (Phase-1 sensitivity-driven): {[round(x, 4) for x in sw]}")

    # Data
    train_ds = ITSPairDataset(
        hazy_dir=args.hazy_dir, clean_dir=args.clean_dir,
        pseudo_dir=args.pseudo_dir, patch_size=args.patch,
        augment=True, max_samples=args.max_samples,
    )
    print(f"train pairs: {len(train_ds)}")
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              drop_last=True)
    val_ds = SOTSEvalDataset(hazy_dir=args.sots_hazy, gt_dir=args.sots_gt)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    # Student
    student = build_student_multitap(width=args.width).to(device)
    n, mM = count_params(student)
    print(f"student params: {n:,} ({mM:.2f}M)")

    # Teacher
    print(f"loading teacher: {args.teacher_ckpt}")
    teacher = load_dehamer(ckpt_path=str(args.teacher_ckpt), device=device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    taps = TeacherStageTaps(teacher)
    print(f"hooked {len(taps._handles)} teacher Swin stages")

    # Discover teacher feature channels with a dry forward
    with torch.no_grad():
        dummy = torch.zeros(1, 3, args.patch, args.patch, device=device)
        teacher(dummy)
    teacher_feats_dry = taps.grab(3)
    teacher_ch = []
    for tf in teacher_feats_dry:
        tf2d = reshape_to_2d(tf)
        teacher_ch.append(tf2d.shape[1])
    print(f"teacher per-stage channels: {teacher_ch}")

    # Student taps: take 3 DEEPEST decoder taps (lowest → mid resolution),
    # matching the 3 Swin stage outputs (deep → shallow).
    student_ch_all = student.tap_channels  # [8w, 4w, 2w, w]
    student_ch = student_ch_all[:3]        # deepest 3
    print(f"student tap channels (deepest 3 decoder blocks): {student_ch}")

    adapter = MultiTapAdapter(student_ch=student_ch, teacher_ch=teacher_ch).to(device)
    criterion = MultiTapDistillLoss(args.lambda_feat, args.lambda_perc,
                                    stage_w=sw).to(device)
    optim = torch.optim.AdamW(
        list(student.parameters()) + list(adapter.parameters()),
        lr=args.lr_hi, betas=(0.9, 0.9),
    )

    # W&B optional
    wb = None
    if args.wandb:
        try:
            import wandb as wb_mod
            wb_mod.init(project="dehazing-compression",
                        name=args.tag, config={**vars(args), "stage_w": sw})
            wb = wb_mod
        except Exception as e:
            print(f"wandb init failed ({e}); proceeding without it.")

    total_steps = args.epochs * len(train_loader)
    global_step = 0
    best_psnr = -1.0

    status_path = ROOT / "results" / f"phase2_{args.tag}_status.txt"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(f"STARTED {time.strftime('%FT%T')} tag={args.tag}\n")

    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        ep = {"loss": 0.0, "l_pixel": 0.0, "l_feat": 0.0, "l_perc": 0.0}
        student.train()
        pbar = tqdm(train_loader, desc=f"ep {epoch:03d}", leave=False)
        for hazy, gt, _pseudo in pbar:
            hazy = hazy.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)

            lr = cosine_lr(global_step, total_steps, args.lr_hi, args.lr_lo)
            for g in optim.param_groups:
                g["lr"] = lr

            # Teacher forward to populate the hook buffers
            with torch.no_grad():
                _ = teacher(hazy)
                teacher_feats = taps.grab(3)

            # Student forward (multi-tap)
            out, student_feats_all = student(hazy)
            student_feats = student_feats_all[:3]                          # deepest 3
            student_feats_proj = adapter(student_feats)                    # → teacher_ch

            losses = criterion(out, gt, student_feats_proj, teacher_feats)
            optim.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                list(student.parameters()) + list(adapter.parameters()), 1.0)
            optim.step()

            for k in ep:
                ep[k] += float(losses[k].detach())
            pbar.set_postfix(loss=f"{losses['loss'].item():.4f}", lr=f"{lr:.2e}")
            global_step += 1

        n_steps = max(1, len(train_loader))
        ep_avg = {k: v / n_steps for k, v in ep.items()}
        print(f"ep {epoch:03d}  loss {ep_avg['loss']:.4f}  l_pix {ep_avg['l_pixel']:.4f}  "
              f"l_feat {ep_avg['l_feat']:.4f}  l_perc {ep_avg['l_perc']:.4f}  "
              f"{time.perf_counter() - t0:.1f}s", flush=True)
        if wb:
            wb.log({f"train/{k}": v for k, v in ep_avg.items()}, step=epoch)

        if (epoch + 1) % args.val_interval == 0 or epoch == args.epochs - 1:
            v = validate(student, val_loader, device)
            print(f"VAL  PSNR {v['psnr']:.3f}  SSIM {v['ssim']:.4f}", flush=True)
            if wb:
                wb.log({"val/psnr": v["psnr"], "val/ssim": v["ssim"]}, step=epoch)
            if v["psnr"] > best_psnr:
                best_psnr = v["psnr"]
                torch.save({
                    "student": student.state_dict(),
                    "adapter": adapter.state_dict(),
                    "epoch":   epoch,
                    "best_psnr": best_psnr,
                    "stage_w": sw,
                }, ckpt_dir / "best.pt")

        if (epoch + 1) % args.ckpt_interval == 0:
            torch.save({
                "student": student.state_dict(),
                "adapter": adapter.state_dict(),
                "optim":   optim.state_dict(),
                "epoch":   epoch,
                "global_step": global_step,
                "best_psnr": best_psnr,
                "stage_w": sw,
            }, ckpt_dir / f"epoch_{epoch:03d}.pt")

    summary = {
        "tag": args.tag, "epochs": args.epochs, "width": args.width,
        "lambda_feat": args.lambda_feat, "lambda_perc": args.lambda_perc,
        "best_psnr_val": best_psnr, "stage_w": sw,
        "teacher_ch": teacher_ch, "student_ch": student_ch,
    }
    (ckpt_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))

    with status_path.open("a") as f:
        f.write(f"DONE {time.strftime('%FT%T')} best_psnr={best_psnr:.3f}\n")
    print(f"Done. best VAL PSNR = {best_psnr:.3f}")


if __name__ == "__main__":
    main()
