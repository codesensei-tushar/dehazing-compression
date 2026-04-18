"""Generate tiny synthetic hazy/clean pairs for local smoke tests.

Output: data/dummy/clean/NNN.png and data/dummy/hazy/NNN.png (matched names).
Haze model: I(x) = J(x)*t(x) + A*(1 - t(x)), t = exp(-beta * depth).
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SIZE = 256
N_PAIRS = 8
OUT = Path(__file__).resolve().parent.parent / "data" / "dummy"


def make_clean(seed: int) -> np.ndarray:
    r = np.random.default_rng(seed)
    bg = tuple(int(v) for v in r.integers(40, 200, 3))
    img = Image.new("RGB", (SIZE, SIZE), bg)
    draw = ImageDraw.Draw(img)
    for _ in range(int(r.integers(5, 12))):
        x0, y0 = (int(v) for v in r.integers(0, SIZE - 40, 2))
        x1, y1 = x0 + int(r.integers(20, 80)), y0 + int(r.integers(20, 80))
        fill = tuple(int(v) for v in r.integers(0, 255, 3))
        if r.random() < 0.5:
            draw.rectangle([x0, y0, x1, y1], fill=fill)
        else:
            draw.ellipse([x0, y0, x1, y1], fill=fill)
    return np.asarray(img).astype(np.float32) / 255.0


def apply_haze(clean: np.ndarray, seed: int) -> np.ndarray:
    r = np.random.default_rng(seed + 1000)
    yy, _ = np.mgrid[0:SIZE, 0:SIZE] / SIZE
    depth = 0.3 + 0.7 * yy + 0.1 * r.normal(size=(SIZE, SIZE))
    depth = np.clip(depth, 0.0, 1.0)
    beta = float(r.uniform(1.0, 2.5))
    t = np.exp(-beta * depth)[..., None]
    A = float(r.uniform(0.75, 0.95))
    return np.clip(clean * t + A * (1.0 - t), 0.0, 1.0)


def main() -> None:
    (OUT / "clean").mkdir(parents=True, exist_ok=True)
    (OUT / "hazy").mkdir(parents=True, exist_ok=True)
    for i in range(N_PAIRS):
        clean = make_clean(i)
        hazy = apply_haze(clean, i)
        Image.fromarray((clean * 255).astype(np.uint8)).save(OUT / "clean" / f"{i:03d}.png")
        Image.fromarray((hazy * 255).astype(np.uint8)).save(OUT / "hazy" / f"{i:03d}.png")
        print(f"wrote pair {i:03d}")
    print(f"Done. {N_PAIRS} pairs in {OUT}")


if __name__ == "__main__":
    main()
