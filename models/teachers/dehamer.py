"""Thin wrapper around third_party/dehamer so we can import the model, load a
pretrained checkpoint, preprocess hazy inputs, and (later) tap decoder features
for distillation.

DeHamer = Guo et al., CVPR 2022. Model class: `UNet_emb` in `swin_unet.py`.
Checkpoint format: saved from an `nn.DataParallel`-wrapped module, so keys are
prefixed with ``module.``.

Normalization (from their `val_data.py`):
    mean = (0.64, 0.60, 0.58)
    std  = (0.14, 0.15, 0.152)
Output is in [0, 1] (GT is ToTensor without normalization).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import Compose, Normalize, ToTensor

_ROOT = Path(__file__).resolve().parents[2]
_DEHAMER_SRC = _ROOT / "third_party" / "dehamer" / "src"
if str(_DEHAMER_SRC) not in sys.path:
    sys.path.insert(0, str(_DEHAMER_SRC))

from swin_unet import UNet_emb  # noqa: E402

DEHAMER_MEAN = (0.64, 0.60, 0.58)
DEHAMER_STD = (0.14, 0.15, 0.152)
_PREPROCESS = Compose([ToTensor(), Normalize(DEHAMER_MEAN, DEHAMER_STD)])


def _strip_module_prefix(state_dict: dict) -> dict:
    return {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }


def load_dehamer(
    ckpt_path: Optional[str | Path] = None,
    device: str | torch.device = "cpu",
    strict: bool = False,
) -> UNet_emb:
    """Build UNet_emb, optionally load a pretrained checkpoint, return eval model."""
    model = UNet_emb()
    if ckpt_path is not None:
        state = torch.load(str(ckpt_path), map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = _strip_module_prefix(state)
        missing, unexpected = model.load_state_dict(state, strict=strict)
        if missing:
            print(f"[dehamer] missing keys: {len(missing)} (first 3: {missing[:3]})")
        if unexpected:
            print(f"[dehamer] unexpected keys: {len(unexpected)} (first 3: {unexpected[:3]})")
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def preprocess(img: Image.Image | np.ndarray) -> torch.Tensor:
    """PIL image or HxWx3 uint8 ndarray -> 1x3xHxW normalized tensor, cropped to multiples of 16."""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    w, h = img.size
    h_c, w_c = h - (h % 16), w - (w % 16)
    if (h_c, w_c) != (h, w):
        img = img.crop((0, 0, w_c, h_c))
    return _PREPROCESS(img).unsqueeze(0)


@torch.no_grad()
def dehaze(model: nn.Module, img: Image.Image | np.ndarray, device: str | torch.device = "cpu") -> np.ndarray:
    """Convenience: preprocess -> forward -> HxWx3 uint8 ndarray in [0,255]."""
    x = preprocess(img).to(device)
    y = model(x).clamp(0.0, 1.0)[0].permute(1, 2, 0).cpu().numpy()
    return (y * 255.0).astype(np.uint8)


def count_params(model: nn.Module) -> Tuple[int, float]:
    n = sum(p.numel() for p in model.parameters())
    return n, n / 1e6
