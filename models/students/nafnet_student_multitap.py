"""Variant of NAFNetStudent that exposes multiple decoder feature taps.

The base wrapper in nafnet_student.py taps only the LAST decoder block. For
sensitivity-driven distillation we want to match teacher activations at
multiple resolutions, so we hook every decoder block and return the full list.

Tap order: decoders[0] (deepest, lowest res) → decoders[-1] (shallowest, highest res).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from models.students.nafnet_student import (
    DEFAULT_CFG,
    NAFNet,
    count_params,  # noqa: F401  — re-exported for convenience
)


class NAFNetStudentMultiTap(nn.Module):
    """NAFNet student that exposes a feature tap per decoder block."""

    def __init__(self, cfg: Dict | None = None) -> None:
        super().__init__()
        cfg = dict(DEFAULT_CFG, **(cfg or {}))
        self.cfg = cfg
        self.net = NAFNet(**cfg)
        self._features: List[torch.Tensor | None] = [None] * len(self.net.decoders)
        for i, dec in enumerate(self.net.decoders):
            dec.register_forward_hook(self._make_hook(i))

    def _make_hook(self, i: int):
        def _hook(module, inputs, outputs):
            self._features[i] = outputs
        return _hook

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        out = self.net(x)
        feats = [f for f in self._features if f is not None]
        assert len(feats) == len(self.net.decoders), \
            f"expected {len(self.net.decoders)} taps, got {len(feats)}"
        return out, feats

    @property
    def tap_channels(self) -> List[int]:
        """Channels per decoder tap (deep → shallow)."""
        w = self.cfg["width"]
        # NAFNet decoder channels: deepest = width * 2^(n_stages-1), each next half'd
        # For 4 decoders with width w → [8w, 4w, 2w, w]
        n = len(self.net.decoders)
        return [w * (2 ** (n - 1 - i)) for i in range(n)]


def build_student_multitap(width: int = 32) -> NAFNetStudentMultiTap:
    cfg = dict(DEFAULT_CFG, width=width)
    return NAFNetStudentMultiTap(cfg)


class MultiTapAdapter(nn.Module):
    """One 1×1 conv adapter per tap, projecting student feature channels →
    teacher feature channels. Spatial alignment is left to the loss
    (bilinear interpolate to teacher's H×W).
    """

    def __init__(self, student_ch: List[int], teacher_ch: List[int]) -> None:
        super().__init__()
        assert len(student_ch) == len(teacher_ch)
        self.projs = nn.ModuleList([
            nn.Conv2d(s, t, kernel_size=1, bias=False)
            for s, t in zip(student_ch, teacher_ch)
        ])

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        return [self.projs[i](f) for i, f in enumerate(feats)]
