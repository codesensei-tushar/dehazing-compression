"""Train FFA-Net (Qin et al., AAAI 2020) on RESIDE ITS.

Mirrors the AOD-Net trainer for an apples-to-apples comparison on the same
data split. FFA-Net (gps=3, blocks=19) is ~4.46M params; training to
convergence on ITS takes ~3-5 h on an A5000 at batch 1 / patch 240.

Usage (cluster):
    python evaluate/train_ffanet.py \\
        --its-hazy data/RESIDE/ITS-Train/train_indoor/haze \\
        --its-gt   data/RESIDE/ITS-Train/train_indoor/clear_images \\
        --out      experiments/baselines/ffanet_indoor.pth \\
        --epochs 100 --batch 1 --patch 240
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from tqdm import tqdm

from models.baselines.ffanet import FFANet, count_params

ROOT = Path(__file__).resolve().parent.parent


class ITSDataset(Dataset):
    def __init__(self, hazy_dir: Path, gt_dir: Path, patch: int = 240) -> None:
        self.patch = patch
        exts = ("*.png", "*.jpg")
        hazy_files = sorted(f for e in exts for f in hazy_dir.glob(e))
        self.pairs: list[tuple[Path, Path]] = []
        for hp in hazy_files:
            stem = hp.stem.split("_")[0]
            for ext in (".png", ".jpg"):
                gp = gt_dir / (stem + ext)
                if gp.exists():
                    self.pairs.append((hp, gp))
                    break
        print(f"[ITS] {len(self.pairs)} pairs")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        hp, gp = self.pairs[idx]
        hazy = Image.open(hp).convert("RGB")
        gt   = Image.open(gp).convert("RGB")
        w, h = hazy.size
        p = min(self.patch, h, w)
        i = random.randint(0, h - p)
        j = random.randint(0, w - p)
        hazy = TF.crop(hazy, i, j, p, p)
        gt   = TF.crop(gt,   i, j, p, p)
        if random.random() > 0.5:
            hazy, gt = TF.hflip(hazy), TF.hflip(gt)
        if random.random() > 0.5:
            hazy, gt = TF.vflip(hazy), TF.vflip(gt)
        return TF.to_tensor(hazy), TF.to_tensor(gt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--its-hazy", type=Path,
                    default=ROOT / "data/RESIDE/ITS-Train/train_indoor/haze")
    ap.add_argument("--its-gt",   type=Path,
                    default=ROOT / "data/RESIDE/ITS-Train/train_indoor/clear_images")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "experiments/baselines/ffanet_indoor.pth")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch",  type=int, default=1)
    ap.add_argument("--patch",  type=int, default=240)
    ap.add_argument("--lr",     type=float, default=1e-4)
    ap.add_argument("--gps",    type=int, default=3)
    ap.add_argument("--blocks", type=int, default=19)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  epochs={args.epochs}  batch={args.batch}  patch={args.patch}")

    dataset = ITSDataset(args.its_hazy, args.its_gt, patch=args.patch)
    loader  = DataLoader(dataset, batch_size=args.batch, shuffle=True,
                         num_workers=4, pin_memory=(device == "cuda"),
                         drop_last=True)

    model = FFANet(gps=args.gps, blocks=args.blocks).to(device)
    n, mM = count_params(model)
    print(f"FFA-Net params: {n:,} ({mM:.3f}M)")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.L1Loss()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for hazy, gt in tqdm(loader, desc=f"ep {epoch}/{args.epochs}", leave=False):
            hazy, gt = hazy.to(device), gt.to(device)
            out  = model(hazy)
            loss = criterion(out, gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()
        mean_loss = float(np.mean(losses))
        print(f"epoch {epoch:3d}  loss {mean_loss:.5f}  lr {scheduler.get_last_lr()[0]:.2e}", flush=True)
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(model.state_dict(), args.out)
            print(f"  saved best → {args.out}  (loss {best_loss:.5f})")

    print(f"\nbest loss: {best_loss:.5f}\nckpt: {args.out}")


if __name__ == "__main__":
    main()
