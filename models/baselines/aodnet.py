"""AOD-Net — Li et al., ICCV 2017.

End-to-end dehazing network that directly estimates the dehazed image
via the AOD (All-in-One Dehazing) reformulation:

    J = K(x) * x - K(x) + 1

where K(x) is a lightweight CNN that estimates a single intermediate
parameter (combining transmission map and atmospheric light estimation).

Architecture exactly as in the official Torch implementation
(github.com/Boyiliee/AOD-Net/blob/master/Torch/model.py):
    e_conv1: Conv2d(3, 3, 1×1)
    e_conv2: Conv2d(3, 3, 3×3, pad=1)
    e_conv3: Conv2d(6, 3, 5×5, pad=2)   — cat(x1, x2)
    e_conv4: Conv2d(6, 3, 7×7, pad=3)   — cat(x2, x3)
    e_conv5: Conv2d(12, 3, 3×3, pad=1)  — cat(x1, x2, x3, x4)
    output:  relu(K * x - K + 1)

Total params: ~4.5 K  (~0.005 M)
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn


class AODNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.e_conv1 = nn.Conv2d(3, 3, kernel_size=1, padding=0, bias=True)
        self.e_conv2 = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=True)
        self.e_conv3 = nn.Conv2d(6, 3, kernel_size=5, padding=2, bias=True)
        self.e_conv4 = nn.Conv2d(6, 3, kernel_size=7, padding=3, bias=True)
        self.e_conv5 = nn.Conv2d(12, 3, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(torch.cat([x1, x2], dim=1)))
        x4 = self.relu(self.e_conv4(torch.cat([x2, x3], dim=1)))
        k  = self.relu(self.e_conv5(torch.cat([x1, x2, x3, x4], dim=1)))
        return self.relu(k * x - k + 1)


def load_aodnet(
    ckpt_path: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> AODNet:
    model = AODNet()
    if ckpt_path is not None:
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def count_params(model: nn.Module) -> Tuple[int, float]:
    n = sum(p.numel() for p in model.parameters())
    return n, n / 1e6
