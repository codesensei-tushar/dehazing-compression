"""Repeated isolated-load latency benchmark for a NAFNet student.

Each --reps call to evaluate.metrics.latency_ms already does 10-iter warmup +
100-iter CUDA-event timing window; this script wraps it in R outer reps to
report mean / std / min / max over R independent measurement windows. Quoting
mean +/- std lets the paper acknowledge inter-window variance instead of
treating a single 100-iter mean as the true latency.

Usage (cluster):
    python phase2_distill/bench_latency.py \
        --ckpt experiments/students/haze_b_large_tight/best.pt \
        --tag haze_b_large_tight --width 32 --reps 5
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
from models.students.nafnet_student import build_student, count_params

ROOT = Path(__file__).resolve().parent.parent


def measure(student, shape, device, reps):
    samples = [latency_ms(student, shape, device=device) for _ in range(reps)]
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
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable — isolated-load latency must be measured on GPU.")
    device = "cuda"

    student = build_student(width=args.width).to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    student.load_state_dict(ck["student"])
    n_params, mM = count_params(student)
    print(f"loaded {args.tag}: width={args.width}  params={mM:.2f}M  ckpt_epoch={ck.get('epoch', '?')}")

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}   reps per shape: {args.reps}")

    results = {
        "tag": args.tag,
        "width": args.width,
        "params_M": round(mM, 3),
        "gpu": gpu_name,
        "reps": args.reps,
        "per_rep_iters": 100,
        "per_rep_warmup": 10,
        "shape_256": measure(student, (1, 3, 256, 256), device, args.reps),
        "shape_512": measure(student, (1, 3, 512, 512), device, args.reps),
    }
    print(f"  256x256  mean {results['shape_256']['mean_ms']} ms  std {results['shape_256']['std_ms']} ms  "
          f"({results['shape_256']['fps_mean']} FPS)")
    print(f"  512x512  mean {results['shape_512']['mean_ms']} ms  std {results['shape_512']['std_ms']} ms  "
          f"({results['shape_512']['fps_mean']} FPS)")

    out_path = args.out or (ROOT / "results" / f"latency_isolated_{args.tag}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
