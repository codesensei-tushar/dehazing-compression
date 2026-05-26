"""No-reference haze quality metrics for RTTS.

For real-world unpaired evaluation we report:
  - NIQE   (Mittal et al. 2013) — natural-image statistics
  - BRISQUE (Mittal et al. 2012) — spatial-domain natural-image statistics

We had originally targeted FADE (Choi et al. 2015) per the CLAUDE.md spec, but
FADE has a parametric "clear-image" reference model that ships only as a MATLAB
artifact and is brittle under modern dehazing artifacts. NIQE / BRISQUE are
better-supported and more commonly reported in recent dehazing papers
(e.g. DehazeFormer, MixDehazeNet). Both are computed via `pyiqa` (pip install
pyiqa).

Lower NIQE = more natural / less hazy. Lower BRISQUE = better quality.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
from PIL import Image


def _to_tensor(img: Image.Image, device: str) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return t


class NoRefScorer:
    """Lazy-init wrapper over pyiqa metrics. Computes scores in batches."""

    def __init__(self, device: str = "cuda") -> None:
        try:
            import pyiqa  # type: ignore
        except ImportError as e:
            raise ImportError(
                "pyiqa is required for no-reference metrics. "
                "Install with: pip install pyiqa"
            ) from e
        self.device = device
        self._niqe = pyiqa.create_metric("niqe", device=device, as_loss=False)
        self._brisque = pyiqa.create_metric("brisque", device=device, as_loss=False)

    def score(self, img: Image.Image) -> Dict[str, float]:
        t = _to_tensor(img, self.device)
        n = float(self._niqe(t).item())
        b = float(self._brisque(t).item())
        return {"niqe": n, "brisque": b}

    def score_dir(self, png_dir: Path, glob: str = "*.png") -> Dict[str, list]:
        ns, bs = [], []
        for p in sorted(png_dir.glob(glob)):
            img = Image.open(p)
            s = self.score(img)
            ns.append(s["niqe"])
            bs.append(s["brisque"])
        return {"niqe": ns, "brisque": bs}
