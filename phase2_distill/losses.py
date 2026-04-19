"""Distillation losses for Phase-2 NAFNet student training.

Components:
    L_pixel      : L1 between student output and (GT or pseudo-clean).
    L_feat       : L2 between adapter(student decoder feature) and teacher feature.
                   Teacher features are loaded on demand or supplied by the caller.
    L_perceptual : optional VGG19 feature-match loss (off by default).

Defaults (from the project brief):
    lambda_feat  = 0.01
    lambda_perc  = 0.00   (bump to 0.05 only if outputs look blurry)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DistillCfg:
    lambda_feat: float = 0.01
    lambda_perc: float = 0.00


class DistillationLoss(nn.Module):
    """Weighted sum of L_pixel + lambda_feat * L_feat + lambda_perc * L_perceptual."""

    def __init__(self, cfg: DistillCfg = DistillCfg()) -> None:
        super().__init__()
        self.cfg = cfg
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()
        self._vgg: Optional[nn.Module] = None

    # Lazy VGG load so CPU-only checks don't pull in torchvision weights.
    def _get_vgg(self) -> nn.Module:
        if self._vgg is None:
            from torchvision.models import vgg19, VGG19_Weights
            net = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:16].eval()
            for p in net.parameters():
                p.requires_grad_(False)
            self._vgg = net
        return self._vgg

    def forward(
        self,
        student_out: torch.Tensor,
        target: torch.Tensor,
        student_feat: Optional[torch.Tensor] = None,
        teacher_feat: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        loss_pixel = self.l1(student_out, target)
        loss_feat = student_out.new_zeros(())
        loss_perc = student_out.new_zeros(())

        if self.cfg.lambda_feat > 0 and student_feat is not None and teacher_feat is not None:
            # If spatial sizes differ, bilinearly align student to teacher.
            if student_feat.shape[-2:] != teacher_feat.shape[-2:]:
                student_feat = F.interpolate(
                    student_feat, size=teacher_feat.shape[-2:],
                    mode="bilinear", align_corners=False,
                )
            loss_feat = self.l2(student_feat, teacher_feat.detach())

        if self.cfg.lambda_perc > 0:
            vgg = self._get_vgg().to(student_out.device)
            s = _imagenet_normalize(student_out.clamp(0.0, 1.0))
            t = _imagenet_normalize(target.clamp(0.0, 1.0))
            loss_perc = F.mse_loss(vgg(s), vgg(t).detach())

        total = loss_pixel + self.cfg.lambda_feat * loss_feat + self.cfg.lambda_perc * loss_perc
        return {
            "loss": total,
            "l_pixel": loss_pixel.detach(),
            "l_feat": loss_feat.detach(),
            "l_perc": loss_perc.detach(),
        }


_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _imagenet_normalize(x: torch.Tensor) -> torch.Tensor:
    mean = _MEAN.to(x.device, x.dtype)
    std = _STD.to(x.device, x.dtype)
    return (x - mean) / std
