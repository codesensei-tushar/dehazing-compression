"""FX-tracing compatibility patch for DeHamer.

FX symbolic tracing requires every submodule used in forward() to be a
registered attribute of its parent module. DeHamer's DarkChannel.forward
instantiates `nn.ReflectionPad2d(self.pad_size)` inline, which FX rejects
with: ``NameError: module is not installed as a submodule``.

This module monkey-patches DarkChannel so that:
  - ReflectionPad2d is created once in __init__ and stored as self.pad
  - forward reuses self.pad

Third-party code under third_party/dehamer/ is NOT edited.

Usage:
    from models.teachers.dehamer_fx_patch import apply_patch
    apply_patch()
"""
from __future__ import annotations

import torch.nn as nn

from models.teachers import dehamer  # ensures sys.path hack for third_party/dehamer/src is applied
import swin  # noqa: E402  (DeHamer's swin module, now importable)

_PATCHED = False


def apply_patch() -> None:
    """Idempotently replace DarkChannel to be FX-traceable."""
    global _PATCHED
    if _PATCHED:
        return

    _orig_init = swin.DarkChannel.__init__

    def patched_init(self, kernel_size: int = 15) -> None:
        _orig_init(self, kernel_size)
        # Register ReflectionPad2d as a submodule so FX can find it.
        self.pad = nn.ReflectionPad2d(self.pad_size)

    def patched_forward(self, x):
        H, W = x.size()[2], x.size()[3]
        x, _ = x.min(dim=1, keepdim=True)
        x = self.pad(x)
        x = self.unfold(x)
        x = x.unsqueeze(1)
        dark_map, _ = x.min(dim=2, keepdim=False)
        return dark_map.view(-1, 1, H, W)

    swin.DarkChannel.__init__ = patched_init
    swin.DarkChannel.forward = patched_forward
    _PATCHED = True


def patch_instance(model) -> None:
    """Add self.pad to any already-constructed DarkChannel instances in `model`."""
    for m in model.modules():
        if isinstance(m, swin.DarkChannel) and not hasattr(m, "pad"):
            m.pad = nn.ReflectionPad2d(m.pad_size)


# Auto-apply when imported.
apply_patch()
