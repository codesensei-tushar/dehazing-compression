"""FFA-Net — Qin et al., AAAI 2020.

Feature Fusion Attention Network for image dehazing.
Architecture reference: arxiv.org/abs/1911.07559
Official code:         github.com/zhilin007/FFA-Net

Each FA (Feature Attention) block applies:
    Channel Attention:  global avg-pool → FC → ReLU → FC → Sigmoid
    Pixel Attention:    1×1 conv → ReLU → 1×1 conv → Sigmoid
Both are multiplied element-wise into the residual branch.

Groups of FA blocks are stacked; their outputs are fused via a 1×1 conv
that concatenates all intermediate feature maps.

Default config (AAAI paper): gps=3, blocks=19 → 4.46 M params
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.fc(x).view(x.size(0), x.size(1), 1, 1)
        return x * attn


class PixelAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.conv(x)


class FABlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.act   = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.ca    = ChannelAttention(channels)
        self.pa    = PixelAttention(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.conv2(self.act(self.conv1(x)))
        res = self.ca(res)
        res = self.pa(res)
        return x + res


class FFANet(nn.Module):
    """FFA-Net with gps groups, each containing `blocks` FA blocks."""

    def __init__(self, gps: int = 3, blocks: int = 19, channels: int = 64) -> None:
        super().__init__()
        self.gps    = gps
        # Input projection
        self.head   = nn.Conv2d(3, channels, 3, padding=1, bias=True)
        # Groups of FA blocks
        self.groups = nn.ModuleList(
            [nn.Sequential(*[FABlock(channels) for _ in range(blocks)])
             for _ in range(gps)]
        )
        # Fusion: 1×1 conv over concatenated group outputs + input feat
        self.fuse   = nn.Sequential(
            nn.Conv2d(channels * (gps + 1), channels, 1, bias=True),
            nn.ReLU(inplace=True),
        )
        # Output projection
        self.tail   = nn.Conv2d(channels, 3, 3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.head(x)
        outs = [feat]
        f = feat
        for g in self.groups:
            f = g(f)
            outs.append(f)
        fused = self.fuse(torch.cat(outs, dim=1))
        return (self.tail(fused) + x).clamp(0.0, 1.0)


def load_ffanet(
    ckpt_path: str | Path | None = None,
    device: str | torch.device = "cpu",
    gps: int = 3,
    blocks: int = 19,
) -> FFANet:
    model = FFANet(gps=gps, blocks=blocks)
    if ckpt_path is not None:
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        if isinstance(state, dict):
            # Handle common checkpoint wrappers
            for key in ("state_dict", "model", "net"):
                if key in state:
                    state = state[key]
                    break
        # Strip DataParallel prefix
        state = {(k[len("module."):] if k.startswith("module.") else k): v
                 for k, v in state.items()}
        model.load_state_dict(state, strict=True)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def count_params(model: nn.Module) -> Tuple[int, float]:
    n = sum(p.numel() for p in model.parameters())
    return n, n / 1e6
