"""Run all Phase-1 PTQ variants for DeHamer on full SOTS-indoor and persist
a single joined JSON + CSV for the paper.

Variants:
    fp32              : reference (CPU, apples-to-apples with quantized eval)
    int8_dyn_all      : dynamic INT8 on every nn.Linear (26 layers)
    int8_dyn_mixed_K  : dynamic INT8 on Linear except top-K most sensitive (FP32)
    int8_block_static : eager-mode static INT8 on the CNN encoder/decoder
                        Sequential blocks (E_block1..4, _block1..7), Swin FP32.
                        Added by a separate script; this runner only loads the
                        resulting state_dict when --with-block-static is set.
    int8_mixed_final  : int8_dyn_mixed_K composed with int8_block_static.

Always runs on CPU because PyTorch dynamic/eager PTQ is CPU-only. GPU INT8
(TensorRT, torchao) is orthogonal and deferred.

Output: one JSON per variant in results/, plus a combined results/phase1_indoor.csv.
"""
from __future__ import annotations

import argparse
import csv
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
RESULTS = ROOT / "results"


def sots_pairs(max_n: int = 0) -> list[tuple[Path, Path]]:
    pairs = []
    for hp in sorted((SOTS_INDOOR / "input").glob("*.png")):
        stem = hp.stem.split("_")[0]
        gp = (SOTS_INDOOR / "gt") / (stem + ".png")
        if gp.exists():
            pairs.append((hp, gp))
            if max_n and len(pairs) >= max_n:
                break
    return pairs


def _set_submodule(root: nn.Module, path: str, new_module: nn.Module) -> None:
    parts = path.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def build_all_int8(fp32_model: nn.Module) -> nn.Module:
    return torch.quantization.quantize_dynamic(
        deepcopy(fp32_model).cpu().eval(), {nn.Linear}, dtype=torch.qint8
    )


def build_int8_mixed(fp32_model: nn.Module, keep_fp32: set[str]) -> nn.Module:
    q = build_all_int8(fp32_model)
    for path in keep_fp32:
        fp32_linear = fp32_model.get_submodule(path)
        _set_submodule(q, path, deepcopy(fp32_linear).cpu().eval())
    return q


@torch.no_grad()
def eval_model(model: nn.Module, pairs) -> dict:
    ps, ss = [], []
    t0 = time.perf_counter()
    for hp, gp in tqdm(pairs, desc="eval", leave=False):
        hazy = Image.open(hp).convert("RGB")
        gt = np.asarray(Image.open(gp).convert("RGB"))
        out = dehaze(model, hazy, device="cpu")
        h, w = out.shape[:2]
        ps.append(psnr(out, gt[:h, :w]))
        ss.append(ssim(out, gt[:h, :w]))
    el = time.perf_counter() - t0
    return {
        "psnr_mean": float(np.mean(ps)),
        "ssim_mean": float(np.mean(ss)),
        "psnr_min": float(np.min(ps)),
        "psnr_max": float(np.max(ps)),
        "wall_s": el,
        "ms_per_img": el / len(pairs) * 1000.0,
        "n": len(pairs),
    }


@torch.no_grad()
def cpu_latency(model: nn.Module, shape=(1, 3, 256, 256), warmup=3, iters=20) -> float:
    model.eval()
    x = torch.randn(*shape)
    for _ in range(warmup):
        _ = model(x)
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model(x)
    return (time.perf_counter() - t0) * 1000.0 / iters


def load_sensitivity_top(k: int) -> list[str]:
    s = json.loads((RESULTS / "dehamer_sensitivity_indoor.json").read_text())
    records = sorted(s["per_module"], key=lambda r: r["delta_vs_baseline"], reverse=True)
    return [r["module"] for r in records[:k]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eval", type=int, default=0, help="0 = all 500")
    ap.add_argument("--top-k", type=int, default=5, help="Linear layers kept FP32 in mixed.")
    ap.add_argument("--skip-latency", action="store_true")
    ap.add_argument("--with-block-static", type=Path, default=None,
                    help="Path to .pt state_dict produced by block_static_ptq.py")
    ap.add_argument("--variants", nargs="+", default=["fp32", "dyn_all", "dyn_mixed"],
                    choices=["fp32", "dyn_all", "dyn_mixed", "block_static", "mixed_final"])
    args = ap.parse_args()

    assert CKPT_INDOOR.exists(), f"missing {CKPT_INDOOR}"
    pairs = sots_pairs(args.n_eval)
    print(f"#pairs={len(pairs)}  variants={args.variants}")

    fp32 = load_dehamer(ckpt_path=str(CKPT_INDOOR), device="cpu")
    n_params, mM = count_params(fp32)
    print(f"params={n_params:,} ({mM:.2f}M)")

    rows: list[dict] = []

    if "fp32" in args.variants:
        print("\n=== FP32 ===")
        m = eval_model(fp32, pairs)
        lat = None if args.skip_latency else cpu_latency(fp32)
        print(f"FP32 PSNR {m['psnr_mean']:.3f} SSIM {m['ssim_mean']:.4f}  {m['ms_per_img']:.1f} ms/img  synth-256² {lat}")
        rows.append({"variant": "fp32", **m, "latency_ms_256": lat, "quant_coverage": "-"})

    if "dyn_all" in args.variants:
        print("\n=== INT8 dynamic (all Linear) ===")
        q = build_all_int8(fp32)
        m = eval_model(q, pairs)
        lat = None if args.skip_latency else cpu_latency(q)
        print(f"INT8-dyn-all PSNR {m['psnr_mean']:.3f} SSIM {m['ssim_mean']:.4f}  {m['ms_per_img']:.1f} ms/img  synth-256² {lat}")
        rows.append({"variant": "int8_dyn_all", **m, "latency_ms_256": lat, "quant_coverage": "26/26 Linear"})

    if "dyn_mixed" in args.variants:
        keep = set(load_sensitivity_top(args.top_k))
        print(f"\n=== INT8 dynamic mixed (keep FP32: {len(keep)} sensitive Linear) ===")
        for k in sorted(keep):
            print(f"  keep FP32: {k}")
        q = build_int8_mixed(fp32, keep)
        m = eval_model(q, pairs)
        lat = None if args.skip_latency else cpu_latency(q)
        print(f"INT8-dyn-mixed PSNR {m['psnr_mean']:.3f} SSIM {m['ssim_mean']:.4f}  {m['ms_per_img']:.1f} ms/img  synth-256² {lat}")
        rows.append({"variant": f"int8_dyn_mixed_top{args.top_k}", **m, "latency_ms_256": lat,
                     "quant_coverage": f"{26 - len(keep)}/26 Linear"})

    # Block-static / mixed-final variants depend on a produced state-dict (handled later)

    # Persist CSV
    csv_path = RESULTS / "phase1_indoor.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["variant", "psnr_mean", "ssim_mean", "psnr_min", "psnr_max",
                  "ms_per_img", "latency_ms_256", "n", "quant_coverage"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})
    print(f"\nwrote {csv_path.relative_to(ROOT)}")

    out_json = RESULTS / "phase1_indoor.json"
    out_json.write_text(json.dumps({
        "model": "DeHamer",
        "split": "indoor",
        "ckpt": str(CKPT_INDOOR.relative_to(ROOT)),
        "n_params": n_params,
        "params_M": round(mM, 2),
        "top_k_fp32": args.top_k,
        "rows": rows,
    }, indent=2))
    print(f"wrote {out_json.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
