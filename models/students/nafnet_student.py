"""NAFNet student wrapper for Phase-2 condition-specific distillation.

Student config (default, after empirical param-count verification):
    width             = 16
    enc_blk_nums      = [1, 1, 1, 28]   # NAFNet deep-trunk architecture
    middle_blk_num    = 1
    dec_blk_nums      = [1, 1, 1, 1]
    -> 4.35M params, ~30x compression vs DeHamer (132.45M).

Note: the original project brief's ``width=32, [1,1,1,28], ~2.6M`` is
architecturally correct but the param count is wrong — that config is actually
17.11M. We scale the width down to 16 to recover a ~2-5M student budget while
preserving the deep-trunk architectural choice.

Random-init by default — the teacher supervises. We expose decoder feature taps
(the last decoder block's output) so the distillation loss can match them
against DeHamer decoder features via a learnable 1x1 conv adapter.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[2]
_NAFNET_SRC = _ROOT / "third_party" / "nafnet"
if str(_NAFNET_SRC) not in sys.path:
    sys.path.insert(0, str(_NAFNET_SRC))

# basicsr's package __init__ imports lmdb eagerly — stub it out if missing so
# we don't have to install a training-only dependency for inference/distillation.
try:
    import lmdb  # noqa: F401
except ImportError:
    import sys as _sys
    import types as _types
    _sys.modules["lmdb"] = _types.ModuleType("lmdb")

from basicsr.models.archs.NAFNet_arch import NAFNet  # noqa: E402

DEFAULT_CFG = dict(
    img_channel=3,
    width=16,
    middle_blk_num=1,
    enc_blk_nums=[1, 1, 1, 28],
    dec_blk_nums=[1, 1, 1, 1],
)


class NAFNetStudent(nn.Module):
    """NAFNet-32 with a forward hook on the last decoder block for feature taps."""

    def __init__(self, cfg: Dict | None = None) -> None:
        super().__init__()
        cfg = dict(DEFAULT_CFG, **(cfg or {}))
        self.cfg = cfg
        self.net = NAFNet(**cfg)
        self._feature: torch.Tensor | None = None
        # Tap the output of the LAST decoder block (highest-resolution, same channels as `width`).
        self.net.decoders[-1].register_forward_hook(self._capture_feature)

    def _capture_feature(self, module, inputs, outputs) -> None:  # noqa: D401
        self._feature = outputs

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.net(x)
        feat = self._feature
        assert feat is not None, "decoder hook did not fire"
        return out, feat

    @property
    def tap_channels(self) -> int:
        return self.cfg["width"]


def build_student(width: int = 16) -> NAFNetStudent:
    cfg = dict(DEFAULT_CFG, width=width)
    return NAFNetStudent(cfg)


def count_params(model: nn.Module) -> Tuple[int, float]:
    n = sum(p.numel() for p in model.parameters())
    return n, n / 1e6


class FeatureAdapter(nn.Module):
    """1x1 conv adapter to project student features to teacher feature channels."""

    def __init__(self, student_ch: int, teacher_ch: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(student_ch, teacher_ch, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
