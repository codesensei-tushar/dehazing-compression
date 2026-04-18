"""Local smoke test for the DeHamer wrapper.

No pretrained checkpoint is loaded (checkpoints live on the cluster). This
verifies:
  1. Submodule imports resolve (sys.path hack works).
  2. UNet_emb can be constructed.
  3. A forward pass on a dummy 256x256 image runs end-to-end.
  4. Output shape matches input shape.

Run: python scripts/smoke_dehamer_local.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from PIL import Image

from models.teachers.dehamer import count_params, dehaze, load_dehamer, preprocess

DUMMY = Path(__file__).resolve().parent.parent / "data" / "dummy" / "hazy" / "000.png"


def main() -> None:
    assert DUMMY.exists(), f"Run scripts/make_dummy_data.py first; missing {DUMMY}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    model = load_dehamer(ckpt_path=None, device=device)
    n, m = count_params(model)
    print(f"params: {n:,} ({m:.2f}M)")

    img = Image.open(DUMMY).convert("RGB")
    x = preprocess(img).to(device)
    print(f"input : {tuple(x.shape)}")

    with torch.no_grad():
        y = model(x)
    print(f"output: {tuple(y.shape)}, range=[{y.min().item():.3f}, {y.max().item():.3f}]")
    assert y.shape == x.shape, f"shape mismatch {y.shape} vs {x.shape}"

    out = dehaze(model, img, device=device)
    print(f"dehaze(): {out.shape}, dtype={out.dtype}")
    assert out.shape[2] == 3 and out.dtype == np.uint8

    print("SMOKE OK")


if __name__ == "__main__":
    main()
