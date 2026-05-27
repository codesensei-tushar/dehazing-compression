"""RESIDE paired datasets for Phase-2 distillation and evaluation.

Each training sample returns three tensors:
    hazy   : 3xHxW in [0,1]
    clean  : 3xHxW in [0,1]        (ground truth)
    pseudo : 3xHxW in [0,1]        (DeHamer teacher output, optional)

ITS/OTS filename convention:  <gt_id>_<k>_<beta>.png   e.g. "1400_1_0.2.png"
The GT image is <gt_id>.png in the clear_images/clear directory.

Dense-Haze / NH-HAZE convention: same stem in both hazy and GT dirs
(e.g. hazy/01.png <-> GT/01.png).
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


def _gt_stem(hazy_name: str) -> str:
    """'1400_1_0.2.png' -> '1400'  (ITS/OTS/SOTS naming convention)"""
    return hazy_name.split("_")[0]


class ITSPairDataset(Dataset):
    """ITS/OTS (haze) dataloader — random 128x128 crops with flips + 90° rotations.

    Works for both ITS (13,990 pairs) and OTS (313,950 pairs). Pass max_samples
    to randomly subsample OTS to a manageable training set (e.g. 50_000).
    """

    def __init__(
        self,
        hazy_dir: Path,
        clean_dir: Path,
        pseudo_dir: Optional[Path] = None,
        patch_size: int = 128,
        augment: bool = True,
        max_samples: int = 0,
    ) -> None:
        self.hazy_dir = Path(hazy_dir)
        self.clean_dir = Path(clean_dir)
        self.pseudo_dir = Path(pseudo_dir) if pseudo_dir else None
        self.patch = patch_size
        self.augment = augment

        hazy_files = sorted(self.hazy_dir.glob("*.png")) + sorted(self.hazy_dir.glob("*.jpg"))
        assert hazy_files, f"no hazy images in {self.hazy_dir}"

        # Keep only hazy items whose GT exists.
        index: list[tuple[Path, Path, Optional[Path]]] = []
        for h in hazy_files:
            stem = _gt_stem(h.name)
            gt = self.clean_dir / (stem + ".png")
            if not gt.exists():
                gt = self.clean_dir / (stem + ".jpg")
            if not gt.exists():
                continue
            ps = None
            if self.pseudo_dir is not None:
                pp = self.pseudo_dir / (h.stem + ".png")
                ps = pp if pp.exists() else None
            index.append((h, gt, ps))
        if not index:
            raise RuntimeError(
                f"no valid (hazy, gt) pairs found. hazy={self.hazy_dir} clean={self.clean_dir}"
            )

        if max_samples and max_samples < len(index):
            random.seed(42)
            index = random.sample(index, max_samples)
        self.index = index

    def __len__(self) -> int:
        return len(self.index)

    def _random_crop(self, *imgs: Image.Image) -> list[Image.Image]:
        w, h = imgs[0].size
        p = self.patch
        x = random.randint(0, max(0, w - p))
        y = random.randint(0, max(0, h - p))
        return [im.crop((x, y, x + p, y + p)) for im in imgs]

    def _augment(self, *imgs: Image.Image) -> list[Image.Image]:
        out = list(imgs)
        if random.random() < 0.5:
            out = [im.transpose(Image.FLIP_LEFT_RIGHT) for im in out]
        if random.random() < 0.5:
            out = [im.transpose(Image.FLIP_TOP_BOTTOM) for im in out]
        k = random.randint(0, 3)
        if k:
            out = [im.rotate(90 * k, resample=Image.BILINEAR, expand=True) for im in out]
        return out

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hp, gp, pp = self.index[i]
        hazy = Image.open(hp).convert("RGB")
        gt = Image.open(gp).convert("RGB")
        pseudo = Image.open(pp).convert("RGB") if pp is not None else gt

        hazy, gt, pseudo = self._random_crop(hazy, gt, pseudo)
        if self.augment:
            hazy, gt, pseudo = self._augment(hazy, gt, pseudo)

        return to_tensor(hazy), to_tensor(gt), to_tensor(pseudo)


class SOTSEvalDataset(Dataset):
    """Full-image SOTS test pairs for validation during training (no crop).

    Works for both SOTS-indoor and SOTS-outdoor — both use the ITS/OTS
    naming convention (<id>_<k>.png → GT <id>.png).
    """

    def __init__(self, hazy_dir: Path, gt_dir: Path) -> None:
        self.hazy_dir = Path(hazy_dir)
        self.gt_dir = Path(gt_dir)
        self.pairs: list[tuple[Path, Path]] = []
        hazy_files = sorted(
            list(self.hazy_dir.glob("*.png")) + list(self.hazy_dir.glob("*.jpg"))
        )
        for h in hazy_files:
            stem = _gt_stem(h.name)
            for ext in (".png", ".jpg"):
                gt = self.gt_dir / (stem + ext)
                if gt.exists():
                    self.pairs.append((h, gt))
                    break

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        hp, gp = self.pairs[i]
        hazy = to_tensor(Image.open(hp).convert("RGB"))
        gt = to_tensor(Image.open(gp).convert("RGB"))
        # Crop to multiples of 8 for NAFNet's internal padding alignment.
        _, h, w = hazy.shape
        h8, w8 = (h // 8) * 8, (w // 8) * 8
        hazy = hazy[:, :h8, :w8]
        gt = gt[:, :h8, :w8]
        return hazy, gt, hp.name


class SimplePairEvalDataset(Dataset):
    """Full-image eval dataset for Dense-Haze / NH-HAZE.

    These real-world datasets use same-stem naming: hazy/01.png <-> GT/01.png.
    Falls back to sorted-index pairing when stem matching yields no results.
    """

    def __init__(self, hazy_dir: Path, gt_dir: Path) -> None:
        self.pairs: list[tuple[Path, Path]] = []
        hazy_dir = Path(hazy_dir)
        gt_dir = Path(gt_dir)
        hazy_files = sorted(
            list(hazy_dir.glob("*.png")) + list(hazy_dir.glob("*.jpg"))
        )
        for hp in hazy_files:
            matched = False
            # 1. Exact same stem (Dense-Haze flat layout: 01.png <-> 01.png).
            for ext in (".png", ".jpg"):
                gp = gt_dir / (hp.stem + ext)
                if gp.exists():
                    self.pairs.append((hp, gp))
                    matched = True
                    break
            if matched:
                continue
            # 2. Strip _hazy suffix, append _GT (Dense-Haze DeHamer layout:
            #    01_hazy.png <-> 01_GT.png).
            if hp.stem.endswith("_hazy"):
                base = hp.stem[:-5]
                for ext in (".png", ".jpg"):
                    gp = gt_dir / (base + "_GT" + ext)
                    if gp.exists():
                        self.pairs.append((hp, gp))
                        break

        # Fallback: pair by sorted index if stem matching found nothing.
        if not self.pairs:
            gt_files = sorted(
                list(gt_dir.glob("*.png")) + list(gt_dir.glob("*.jpg"))
            )
            self.pairs = list(zip(hazy_files, gt_files))

        assert self.pairs, f"no pairs found: hazy={hazy_dir} gt={gt_dir}"

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        hp, gp = self.pairs[i]
        hazy = to_tensor(Image.open(hp).convert("RGB"))
        gt = to_tensor(Image.open(gp).convert("RGB"))
        _, h, w = hazy.shape
        h8, w8 = (h // 8) * 8, (w // 8) * 8
        hazy = hazy[:, :h8, :w8]
        gt = gt[:, :h8, :w8]
        return hazy, gt, hp.name
