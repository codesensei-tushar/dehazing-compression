"""Block-wise eager-mode static PTQ for DeHamer's CNN encoder/decoder.

Strategy motivation:
    FX-graph-mode static PTQ on the full DeHamer is blocked by data-dependent
    control flow in the Swin branch (runtime padding checks in swin.py). Instead
    we quantize the Sequential CNN blocks individually in eager mode and splice
    them back into the model. Swin, DarkChannel, PPM, MSRB, InstanceNorms and
    conv adapters remain FP32.

Blocks targeted (all used in UNet_emb.forward):
    Encoder : E_block1, E_block2, E_block3, E_block4
    Decoder : _block1, _block3, _block4, _block5, _block7

Each block is a nn.Sequential of Conv2d/ReLU and either MaxPool2d (encoder) or
UpsamplingBilinear2d (decoder). We wrap with QuantStub/DeQuantStub, fuse every
adjacent (Conv2d, ReLU) pair, apply the fbgemm qconfig, calibrate with real
inputs captured via forward hooks on the FP32 model, and convert.

The final INT8 state is saved as a dict {block_name: converted_module_state_dict}
so it can be re-loaded and spliced into a fresh FP32 model elsewhere.

Run on cluster:
    python phase1_quantize/block_static_ptq.py \\
        --calib-dir data/RESIDE/ITS-Train/train_indoor/haze --n-calib 100
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

from evaluate.metrics import psnr, ssim
from models.teachers.dehamer import count_params, dehaze, load_dehamer, preprocess

ROOT = Path(__file__).resolve().parent.parent
CKPT_INDOOR = ROOT / "experiments/teachers/dehamer/ckpts/indoor/PSNR3663_ssim09881.pt"
SOTS_INDOOR = ROOT / "data/RESIDE/SOTS-Test/valid_indoor"

TARGET_BLOCKS = [
    "E_block1", "E_block2", "E_block3", "E_block4",
    "_block1", "_block3", "_block4", "_block5", "_block7",
]


class QuantizableBlock(nn.Module):
    """Wraps a nn.Sequential of conv/relu/pool/upsample with Quant/DeQuantStubs
    so it can be processed by eager-mode static PTQ in isolation."""

    def __init__(self, original_sequential: nn.Sequential) -> None:
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.seq = original_sequential  # keep original layer names; fusion uses indices
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.seq(x)
        x = self.dequant(x)
        return x

    def fuse_conv_relu(self) -> None:
        """Fuse every adjacent (Conv2d, ReLU) pair inside self.seq, in-place."""
        pairs: List[List[str]] = []
        modules = list(self.seq)
        i = 0
        while i < len(modules) - 1:
            if isinstance(modules[i], nn.Conv2d) and isinstance(modules[i + 1], nn.ReLU):
                pairs.append([f"seq.{i}", f"seq.{i + 1}"])
                i += 2
            else:
                i += 1
        if pairs:
            torch.ao.quantization.fuse_modules(self, pairs, inplace=True)


def capture_block_inputs(fp32_model: nn.Module, calib_imgs: List[Path],
                         block_names: List[str]) -> Dict[str, List[torch.Tensor]]:
    """Run FP32 model on calibration images and capture the first-position input
    to each named Sequential block via forward hooks. Returns {name: [tensors]}."""
    captured: Dict[str, List[torch.Tensor]] = {n: [] for n in block_names}
    handles = []
    for n in block_names:
        mod = fp32_model.get_submodule(n)
        def _hook(module, inputs, outputs, name=n):
            captured[name].append(inputs[0].detach().cpu())
        handles.append(mod.register_forward_hook(_hook))
    try:
        with torch.no_grad():
            for p in tqdm(calib_imgs, desc="capture"):
                x = preprocess(Image.open(p).convert("RGB"))
                _ = fp32_model(x)
    finally:
        for h in handles:
            h.remove()
    return captured


def quantize_block(original: nn.Sequential, calib_inputs: List[torch.Tensor],
                   backend: str = "fbgemm") -> nn.Module:
    """Eager-mode static PTQ on one Sequential block. Returns the converted module."""
    torch.backends.quantized.engine = backend
    block = QuantizableBlock(deepcopy(original).cpu().eval())
    block.fuse_conv_relu()
    block.qconfig = torch.ao.quantization.get_default_qconfig(backend)
    torch.ao.quantization.prepare(block, inplace=True)
    with torch.no_grad():
        for x in calib_inputs:
            _ = block(x)
    torch.ao.quantization.convert(block, inplace=True)
    return block


def splice_blocks_into_model(fp32_model: nn.Module, qblocks: Dict[str, nn.Module]) -> nn.Module:
    """Return a copy of fp32_model with each target block replaced by its quantized
    wrapper (which includes Quant/DeQuantStubs, so the I/O stays FP32)."""
    m = deepcopy(fp32_model).cpu().eval()
    for name, qb in qblocks.items():
        setattr(m, name, qb)
    return m


def find_calib_images(calib_dir: Path, fallback: Path, n: int) -> List[Path]:
    for d in (calib_dir, fallback):
        if d is None or not d.is_dir():
            continue
        files = sorted(d.glob("*.png")) + sorted(d.glob("*.jpg"))
        if files:
            if len(files) > n:
                files = random.Random(0).sample(files, n)
            return files
    raise FileNotFoundError("No calibration images found")


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
        "ms_per_img": el / len(pairs) * 1000.0,
        "n": len(pairs),
    }


@torch.no_grad()
def cpu_latency(model: nn.Module, shape=(1, 3, 256, 256), warmup=3, iters=10) -> float:
    model.eval()
    x = torch.randn(*shape)
    for _ in range(warmup):
        _ = model(x)
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model(x)
    return (time.perf_counter() - t0) * 1000.0 / iters


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-dir", type=Path,
                    default=ROOT / "data/RESIDE/ITS-Train/train_indoor/haze")
    ap.add_argument("--n-calib", type=int, default=100)
    ap.add_argument("--n-eval", type=int, default=0, help="0 = all 500")
    ap.add_argument("--backend", choices=["fbgemm", "qnnpack"], default="fbgemm")
    ap.add_argument("--save-state", type=Path,
                    default=ROOT / "experiments/ptq/dehamer_block_static_indoor.pt")
    ap.add_argument("--skip-latency", action="store_true")
    args = ap.parse_args()

    # Pairs
    pairs = []
    for hp in sorted((SOTS_INDOOR / "input").glob("*.png")):
        stem = hp.stem.split("_")[0]
        gp = (SOTS_INDOOR / "gt") / (stem + ".png")
        if gp.exists():
            pairs.append((hp, gp))
            if args.n_eval and len(pairs) >= args.n_eval:
                break
    print(f"#pairs={len(pairs)}")

    fp32 = load_dehamer(ckpt_path=str(CKPT_INDOOR), device="cpu")
    n_params, mM = count_params(fp32)
    print(f"params={n_params:,} ({mM:.2f}M)")

    # FP32 reference on CPU
    print("\n--- FP32 (CPU) ---")
    fp32_m = eval_model(fp32, pairs)
    lat_fp32 = None if args.skip_latency else cpu_latency(fp32)
    print(f"FP32 PSNR {fp32_m['psnr_mean']:.3f} SSIM {fp32_m['ssim_mean']:.4f}"
          f"  {fp32_m['ms_per_img']:.1f} ms/img  synth-256² {lat_fp32}")

    # Calibration capture
    sots_fallback = SOTS_INDOOR / "input"
    calib_imgs = find_calib_images(args.calib_dir, sots_fallback, args.n_calib)
    print(f"\ncalibration set: {len(calib_imgs)} images from {calib_imgs[0].parent.name}")
    captured = capture_block_inputs(fp32, calib_imgs, TARGET_BLOCKS)
    for n in TARGET_BLOCKS:
        print(f"  {n}: {len(captured[n])} inputs, shape {tuple(captured[n][0].shape) if captured[n] else None}")

    # Quantize each block
    print(f"\nQuantizing {len(TARGET_BLOCKS)} blocks (backend={args.backend}) ...")
    qblocks: Dict[str, nn.Module] = {}
    for name in TARGET_BLOCKS:
        original = fp32.get_submodule(name)
        try:
            qb = quantize_block(original, captured[name], backend=args.backend)
            qblocks[name] = qb
            print(f"  {name}: OK")
        except Exception as e:
            print(f"  {name}: FAILED ({type(e).__name__}: {e}); keeping FP32")

    # Splice into a fresh copy of FP32
    spliced = splice_blocks_into_model(fp32, qblocks)

    # Evaluate
    print("\n--- INT8 block-static (CPU) ---")
    int8_m = eval_model(spliced, pairs)
    lat_int8 = None if args.skip_latency else cpu_latency(spliced)
    print(f"INT8 PSNR {int8_m['psnr_mean']:.3f} SSIM {int8_m['ssim_mean']:.4f}"
          f"  {int8_m['ms_per_img']:.1f} ms/img  synth-256² {lat_int8}")

    d = int8_m["psnr_mean"] - fp32_m["psnr_mean"]
    print(f"\nΔPSNR = {d:+.3f} dB")
    if lat_fp32 and lat_int8:
        print(f"Speedup = {lat_fp32 / lat_int8:.2f}×")

    # Save spliced state for reuse in run_all_ptq.py
    args.save_state.parent.mkdir(parents=True, exist_ok=True)
    # Save the full spliced model state + qblock module (pickled together).
    torch.save({"state_dict": spliced.state_dict(),
                "qblocks": {k: v for k, v in qblocks.items()}},
               args.save_state, _use_new_zipfile_serialization=True)
    print(f"wrote {args.save_state.relative_to(ROOT)}")

    # Persist JSON
    out_json = ROOT / "results" / "dehamer_int8_block_static_indoor.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({
        "model": "DeHamer",
        "mode": "block-static",
        "backend": args.backend,
        "device": "cpu",
        "split": "indoor",
        "n_calib": len(calib_imgs),
        "n_eval": len(pairs),
        "n_params": n_params,
        "params_M": round(mM, 2),
        "blocks_quantized": list(qblocks.keys()),
        "fp32": {**fp32_m, "latency_ms_256": lat_fp32},
        "int8": {**int8_m, "latency_ms_256": lat_int8},
    }, indent=2))
    print(f"wrote {out_json.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
