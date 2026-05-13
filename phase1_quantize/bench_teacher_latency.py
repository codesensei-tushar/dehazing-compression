"""Isolated-load GPU latency benchmark for the DeHamer teacher.

Mirrors phase2_distill/bench_latency.py exactly (5 outer reps × 100-iter
CUDA-event windows) so teacher and student numbers are directly comparable.

Usage (cluster):
    CUDA_VISIBLE_DEVICES=0 python phase1_quantize/bench_teacher_latency.py \
        --ckpt experiments/teachers/dehamer_indoor.pt

Writes: results/latency_isolated_dehamer_teacher.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, pstdev

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from evaluate.metrics import latency_ms
from models.teachers.dehamer import load_dehamer, count_params

ROOT = Path(__file__).resolve().parent.parent


def measure(model, shape, device, reps):
    samples = [latency_ms(model, shape, device=device) for _ in range(reps)]
    m = mean(samples)
    return {
        "samples_ms": [round(s, 4) for s in samples],
        "mean_ms": round(m, 4),
        "std_ms": round(pstdev(samples) if reps > 1 else 0.0, 4),
        "min_ms": round(min(samples), 4),
        "max_ms": round(max(samples), 4),
        "fps_mean": round(1000.0 / m, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="DeHamer checkpoint (.pt). Omit to benchmark random-init model.")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable — must run on GPU for meaningful latency.")
    device = "cuda"

    model = load_dehamer(ckpt_path=args.ckpt, device=device)
    _, mM = count_params(model)
    gpu_name = torch.cuda.get_device_name(0)
    print(f"DeHamer  params={mM:.2f}M  ckpt={args.ckpt or 'random-init'}")
    print(f"GPU: {gpu_name}   reps per shape: {args.reps}")

    results = {
        "tag": "dehamer_teacher",
        "params_M": round(mM, 3),
        "gpu": gpu_name,
        "reps": args.reps,
        "per_rep_iters": 100,
        "per_rep_warmup": 10,
        "shape_256": measure(model, (1, 3, 256, 256), device, args.reps),
        "shape_512": measure(model, (1, 3, 512, 512), device, args.reps),
    }
    print(f"  256x256  mean {results['shape_256']['mean_ms']} ms  "
          f"std {results['shape_256']['std_ms']} ms  ({results['shape_256']['fps_mean']} FPS)")
    print(f"  512x512  mean {results['shape_512']['mean_ms']} ms  "
          f"std {results['shape_512']['std_ms']} ms  ({results['shape_512']['fps_mean']} FPS)")

    out_path = args.out or (ROOT / "results" / "latency_isolated_dehamer_teacher.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
