"""Train AOD-Net baseline on RESIDE ITS.

AOD-Net is tiny (~5K params), so training converges in ~30 min on A5000.
Self-training ensures identical data split and eval protocol as our student,
giving a fair apples-to-apples PSNR/SSIM entry in the comparison table.

After training, run eval_baseline.py to get the final PSNR/SSIM on SOTS-indoor.

Usage (cluster):
    python evaluate/train_aodnet.py \
        --its-hazy  data/RESIDE/ITS-Train/train_indoor/haze \
        --its-gt    data/RESIDE/ITS-Train/train_indoor/clear_images \
        --out       experiments/baselines/aodnet_indoor.pth \
        --epochs    200
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

from models.baselines.aodnet import AODNet

ROOT = Path(__file__).resolve().parent.parent


class ITSDataset(Dataset):
    def __init__(self, hazy_dir: Path, gt_dir: Path, patch: int = 240) -> None:
        self.patch = patch
        exts = ("*.png", "*.jpg")
        hazy_files = sorted(f for e in exts for f in hazy_dir.glob(e))
        self.pairs: list[tuple[Path, Path]] = []
        for hp in hazy_files:
            # ITS naming: <id>_<k>_<beta>.png → GT <id>.png
            stem = hp.stem.split("_")[0]
            for ext in (".png", ".jpg"):
                gp = gt_dir / (stem + ext)
                if gp.exists():
                    self.pairs.append((hp, gp))
                    break
        if not self.pairs:
            # Fallback: pair by sorted order (some ITS layouts differ)
            gt_files = sorted(f for e in exts for f in gt_dir.glob(e))
            self.pairs = list(zip(hazy_files, gt_files))
        print(f"[ITS] {len(self.pairs)} pairs")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        hp, gp = self.pairs[idx]
        hazy = Image.open(hp).convert("RGB")
        gt   = Image.open(gp).convert("RGB")
        # Random crop
        i, j, h, w = _rand_crop(hazy.size, self.patch)
        hazy = TF.crop(hazy, i, j, h, w)
        gt   = TF.crop(gt,   i, j, h, w)
        # Augmentation
        if random.random() > 0.5:
            hazy, gt = TF.hflip(hazy), TF.hflip(gt)
        if random.random() > 0.5:
            hazy, gt = TF.vflip(hazy), TF.vflip(gt)
        return TF.to_tensor(hazy), TF.to_tensor(gt)


def _rand_crop(img_size: tuple, patch: int) -> tuple:
    w, h = img_size
    p = min(patch, h, w)
    i = random.randint(0, h - p)
    j = random.randint(0, w - p)
    return i, j, p, p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--its-hazy", type=Path,
                    default=ROOT / "data/RESIDE/ITS-Train/train_indoor/haze")
    ap.add_argument("--its-gt",   type=Path,
                    default=ROOT / "data/RESIDE/ITS-Train/train_indoor/clear_images")
    ap.add_argument("--out",    type=Path,
                    default=ROOT / "experiments/baselines/aodnet_indoor.pth")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch",  type=int, default=8)
    ap.add_argument("--patch",  type=int, default=240)
    ap.add_argument("--lr",     type=float, default=1e-3)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  epochs={args.epochs}  batch={args.batch}")

    dataset = ITSDataset(args.its_hazy, args.its_gt, patch=args.patch)
    loader  = DataLoader(dataset, batch_size=args.batch, shuffle=True,
                         num_workers=4, pin_memory=(device == "cuda"), drop_last=True)

    model     = AODNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"AOD-Net params: {total_params:,} ({total_params/1e6:.4f}M)")

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
        if epoch % 10 == 0 or epoch == args.epochs:
            print(f"epoch {epoch:3d}  loss {np.mean(losses):.5f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"saved → {args.out}")


if __name__ == "__main__":
    main()
