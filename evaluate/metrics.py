"""Evaluation metrics shared by Phase 1 (PTQ) and Phase 2 (distillation).

PSNR / SSIM via scikit-image on uint8 ndarrays.
Latency via CUDA events, 100 runs with 10-run warmup.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as _psnr
from skimage.metrics import structural_similarity as _ssim


def psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(_psnr(gt, pred, data_range=255))


def ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(_ssim(gt, pred, channel_axis=2, data_range=255))


@torch.no_grad()
def latency_ms(
    model: torch.nn.Module,
    input_shape: Tuple[int, int, int, int] = (1, 3, 256, 256),
    device: str = "cuda",
    warmup: int = 10,
    iters: int = 100,
) -> float:
    """Return mean per-call latency in ms using CUDA events."""
    model.eval()
    x = torch.randn(*input_shape, device=device)
    if device == "cuda":
        for _ in range(warmup):
            _ = model(x)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            _ = model(x)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters
    import time
    for _ in range(warmup):
        _ = model(x)
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model(x)
    return (time.perf_counter() - t0) * 1000.0 / iters
