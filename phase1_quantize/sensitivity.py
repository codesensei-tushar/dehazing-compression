"""Per-module sensitivity analysis for DeHamer dynamic PTQ.

Baseline: dynamic INT8 on ALL Linear layers.
For each Linear module we toggle FP32, we measure PSNR *recovery* over the
all-INT8 baseline on a small SOTS-indoor subset.

Output: a CSV/JSON table of (module path, PSNR delta, FP32 kept count, quant count).

The top-K most sensitive modules define a mixed-precision configuration the
paper can use: keep those few FP32, quantize the rest to INT8.

This is CPU-only (PyTorch dynamic PTQ constraint).

Run on cluster:
    python phase1_quantize/sensitivity.py --n-eval 30 --top-k 10
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

from evaluate.metrics import psnr, ssim
from models.teachers.dehamer import count_params, dehaze, load_dehamer

ROOT = Path(__file__).resolve().parent.parent
CKPT_INDOOR = ROOT / "experiments/teachers/dehamer/ckpts/indoor/PSNR3663_ssim09881.pt"
SOTS_INDOOR = ROOT / "data/RESIDE/SOTS-Test/valid_indoor"


def sots_pairs(max_n: int) -> list[tuple[Path, Path]]:
    pairs = []
    for hp in sorted((SOTS_INDOOR / "input").glob("*.png")):
        stem = hp.stem.split("_")[0]
        gp = (SOTS_INDOOR / "gt") / (stem + ".png")
        if gp.exists():
            pairs.append((hp, gp))
            if len(pairs) >= max_n:
                break
    return pairs


def linear_module_paths(model: nn.Module) -> list[str]:
    return [name for name, m in model.named_modules() if isinstance(m, nn.Linear)]


def _set_submodule(root: nn.Module, path: str, new_module: nn.Module) -> None:
    """Replace the submodule at dotted `path` inside `root`."""
    parts = path.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def build_all_int8(fp32_model: nn.Module) -> nn.Module:
    """Dynamic INT8 on every Linear layer."""
    return torch.quantization.quantize_dynamic(
        deepcopy(fp32_model).cpu().eval(), {nn.Linear}, dtype=torch.qint8
    )


def build_int8_except(fp32_model: nn.Module, keep_fp32_paths: set[str]) -> nn.Module:
    """All-INT8, then restore FP32 Linear at each path in keep_fp32_paths."""
    q = build_all_int8(fp32_model)
    for path in keep_fp32_paths:
        fp32_linear = fp32_model.get_submodule(path)
        _set_submodule(q, path, deepcopy(fp32_linear).cpu().eval())
    return q


@torch.no_grad()
def eval_psnr(model: nn.Module, pairs) -> tuple[float, float]:
    ps, ss = [], []
    for hp, gp in pairs:
        hazy = Image.open(hp).convert("RGB")
        gt = np.asarray(Image.open(gp).convert("RGB"))
        out = dehaze(model, hazy, device="cpu")
        h, w = out.shape[:2]
        ps.append(psnr(out, gt[:h, :w]))
        ss.append(ssim(out, gt[:h, :w]))
    return float(np.mean(ps)), float(np.mean(ss))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eval", type=int, default=30, help="SOTS-indoor subset size per trial.")
    ap.add_argument("--top-k", type=int, default=10, help="Print top-K most sensitive modules.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    assert CKPT_INDOOR.exists(), f"missing {CKPT_INDOOR}"
    pairs = sots_pairs(args.n_eval)
    print(f"#pairs={len(pairs)}")

    fp32 = load_dehamer(ckpt_path=str(CKPT_INDOOR), device="cpu")
    n_params, mM = count_params(fp32)
    print(f"params={n_params:,} ({mM:.2f}M)")

    # FP32 reference on the same subset
    print("\n[ref] FP32...")
    t0 = time.perf_counter()
    psnr_fp32, ssim_fp32 = eval_psnr(fp32, pairs)
    print(f"[ref] FP32 PSNR={psnr_fp32:.3f} SSIM={ssim_fp32:.4f}  ({time.perf_counter()-t0:.1f}s)")

    # INT8-all baseline
    print("\n[baseline] INT8 on all Linear...")
    q_all = build_all_int8(fp32)
    t0 = time.perf_counter()
    psnr_int8, ssim_int8 = eval_psnr(q_all, pairs)
    print(f"[baseline] INT8 PSNR={psnr_int8:.3f} SSIM={ssim_int8:.4f}  ({time.perf_counter()-t0:.1f}s)")
    baseline_drop = psnr_fp32 - psnr_int8
    print(f"[baseline] drop vs FP32: {baseline_drop:+.3f} dB")

    # Per-Linear sensitivity
    paths = linear_module_paths(fp32)
    print(f"\nScanning {len(paths)} Linear modules (one at a time kept FP32)...")
    records: list[dict] = []
    for path in tqdm(paths, desc="sensitivity"):
        qm = build_int8_except(fp32, keep_fp32_paths={path})
        p, s = eval_psnr(qm, pairs)
        records.append({
            "module": path,
            "psnr": p,
            "ssim": s,
            "delta_vs_baseline": p - psnr_int8,
            "delta_vs_fp32": p - psnr_fp32,
        })

    # Sort by recovery (higher = more sensitive when quantized, biggest win when kept FP32)
    records.sort(key=lambda r: r["delta_vs_baseline"], reverse=True)

    # Report
    print(f"\nTop-{args.top_k} most sensitive modules (bigger delta_vs_baseline = worse to quantize):")
    print(f"{'delta_PSNR':>10}  {'PSNR':>7}  module")
    for r in records[: args.top_k]:
        print(f"{r['delta_vs_baseline']:+10.3f}  {r['psnr']:7.3f}  {r['module']}")

    # Persist
    out_path = Path(args.out) if args.out else ROOT / "results" / "dehamer_sensitivity_indoor.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "model": "DeHamer",
        "mode": "dynamic-sensitivity",
        "device": "cpu",
        "split": "indoor",
        "n_eval": len(pairs),
        "fp32_psnr": psnr_fp32,
        "fp32_ssim": ssim_fp32,
        "int8_all_psnr": psnr_int8,
        "int8_all_ssim": ssim_int8,
        "int8_all_drop_db": psnr_fp32 - psnr_int8,
        "per_module": records,
    }, indent=2))
    print(f"\nwrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
