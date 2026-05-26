#!/usr/bin/env python3
"""BMVC-targeted figures:

  1. pareto_with_baselines.png   — PSNR vs params, with lightweight CNN baselines
  2. sensitivity_heatmap.png     — per-layer ΔPSNR for all 26 Swin Linear modules
  3. cross_domain_bars.png       — 4 models × 4 splits PSNR bars

Numbers for AOD-Net / FFA-Net / GridDehazeNet / gUNet / MixDehazeNet /
DehazeFormer are quoted from their publications (citations in README). Update
once we have own-machine baseline runs.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"
FIGS.mkdir(exist_ok=True)

# Light theme — BMVC submissions are easier to scan on white
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

C_OURS = {
    "teacher": "#7C3AED",
    "B": "#10B981",
    "C": "#3B82F6",
    "A": "#F59E0B",
}
C_BASELINE = "#9CA3AF"


# ───────────────────────── 1. Pareto with baselines ─────────────────────────

def pareto_with_baselines():
    """Quality vs params on SOTS-indoor.

    Our points come from local JSONs; baseline points are quoted from the
    cited papers (mark as quoted via a small "[cite]" tag).
    """
    fig, ax = plt.subplots(figsize=(9.5, 6.0))

    # Quoted baselines on SOTS-indoor (PSNR / Params(M))
    # Sources are pinned in the README citations section.
    baselines = [
        ("AOD-Net",       0.0018, 19.06, "Li et al. 2017"),
        ("GridDehazeNet", 0.96,   32.16, "Liu et al. 2019"),
        ("FFA-Net",       4.68,   36.39, "Qin et al. 2020"),
        ("MSBDN-DFF",     31.35,  33.79, "Dong et al. 2020"),
        ("AECR-Net",      2.61,   37.17, "Wu et al. 2021"),
        ("gUNet-T",       0.65,   38.62, "Song et al. 2023"),
        ("DehazeFormer-S",25.40,  40.05, "Song et al. 2023"),
        ("MixDehazeNet-L",7.32,   42.62, "Lu et al. 2023"),
    ]

    # Our points (measured)
    ours = [
        ("DeHamer (teacher)",        132.45, 36.576, "ours / re-eval", "teacher", "D", 140),
        ("Student B (w32, GT)",      17.11,  34.398, "ours",           "B",       "s", 110),
        ("Student C (w32, pseudo)",  17.11,  33.869, "ours",           "C",       "^", 110),
        ("Student A (w16, GT)",      4.35,   32.391, "ours",           "A",       "o", 110),
    ]

    # Baselines
    for name, p, psnr, _src in baselines:
        ax.scatter(p, psnr, c=C_BASELINE, s=70, marker="x",
                   linewidths=1.8, zorder=4)
        ax.annotate(name, (p, psnr), fontsize=8, color="#374151",
                    xytext=(6, 4), textcoords="offset points")

    # Ours
    for name, p, psnr, _src, key, marker, sz in ours:
        ax.scatter(p, psnr, c=C_OURS[key], s=sz, marker=marker, zorder=6,
                   edgecolors="white", linewidths=1.2)
        ax.annotate(name, (p, psnr), fontsize=9, color=C_OURS[key],
                    fontweight="bold",
                    xytext=(8, -10), textcoords="offset points")

    # Compression annotations on the two flagship students
    ax.annotate("", xy=(17.11, 34.398), xytext=(132.45, 36.576),
                arrowprops=dict(arrowstyle="->", color="#7C3AED", alpha=0.4,
                                connectionstyle="arc3,rad=-0.25"))
    ax.text(45, 35.6, "7.7× smaller\n−2.2 dB", color="#7C3AED", fontsize=9,
            ha="center", alpha=0.85, fontweight="bold")

    ax.annotate("", xy=(4.35, 32.391), xytext=(132.45, 36.576),
                arrowprops=dict(arrowstyle="->", color="#F59E0B", alpha=0.4,
                                connectionstyle="arc3,rad=-0.45"))
    ax.text(15, 31.2, "30.5× smaller\n−4.2 dB", color="#F59E0B", fontsize=9,
            ha="center", alpha=0.85, fontweight="bold")

    ax.set_xscale("log")
    ax.set_xlabel("Parameters (M)", fontsize=12)
    ax.set_ylabel("PSNR (dB) — SOTS-indoor", fontsize=12)
    ax.set_title("Quality vs. Parameters on SOTS-indoor\n"
                 "(× = quoted from cited paper; coloured = this work)",
                 fontsize=12)
    ax.set_xlim(0.001, 250)
    ax.set_ylim(18, 44)

    # Legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="D", color="w", markerfacecolor=C_OURS["teacher"],
               markersize=10, label="Teacher (DeHamer)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=C_OURS["B"],
               markersize=10, label="Student B"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=C_OURS["C"],
               markersize=10, label="Student C"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_OURS["A"],
               markersize=10, label="Student A"),
        Line2D([0], [0], marker="x", color=C_BASELINE, markersize=10,
               linewidth=0, label="Published baseline"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.95)

    fig.tight_layout()
    out = FIGS / "pareto_with_baselines.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# ──────────────────────── 2. Sensitivity heatmap ────────────────────────────

def sensitivity_heatmap():
    """Heatmap of per-layer Δ in PSNR (vs all-INT8 baseline) for every
    Linear module in DeHamer's Swin trunk.

    Layout: rows = (stage, block) pair; columns = component
    (attn.qkv, attn.proj, mlp.fc1, mlp.fc2).
    """
    with open(RESULTS / "dehamer_sensitivity_indoor.json") as f:
        d = json.load(f)
    per = d["per_module"]
    fp32 = d["fp32_psnr"]
    int8_all = d["int8_all_psnr"]

    # Parse each entry's path:
    #   swin_1.layers.<s>.blocks.<b>.<component>   (24 modules)
    #   swin_1.layers.<s>.downsample.reduction     (2 modules — show as extra rows)
    rows = {}
    for entry in per:
        name = entry["module"]
        parts = name.split(".")
        stage = int(parts[2])
        if parts[3] == "downsample":
            comp = "downsample.reduction"
            row_key = (stage, "down")
        else:
            block = int(parts[4])
            comp = ".".join(parts[5:])
            row_key = (stage, f"blk{block}")
        rows.setdefault(row_key, {})[comp] = entry["delta_vs_baseline"]

    comp_order = ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2", "downsample.reduction"]

    def sort_key(rk):
        stage, label = rk
        order = {"blk0": 0, "blk1": 1, "down": 2}
        return (stage, order.get(label, 3))

    rk = sorted(rows.keys(), key=sort_key)
    M = np.full((len(rk), len(comp_order)), np.nan)
    for ri, key in enumerate(rk):
        for ci, comp in enumerate(comp_order):
            if comp in rows[key]:
                M[ri, ci] = rows[key][comp] * 1000.0  # mdB for readability

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    vmax = np.nanmax(np.abs(M))
    im = ax.imshow(M, cmap="RdYlGn_r", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(comp_order)))
    ax.set_xticklabels(comp_order, fontsize=10)
    ax.set_yticks(range(len(rk)))
    def row_label(rk_entry):
        s, lbl = rk_entry
        if lbl == "down":
            return f"stage {s} · downsample"
        return f"stage {s} · {lbl}"
    ax.set_yticklabels([row_label(k) for k in rk], fontsize=9)
    ax.set_title(
        f"PTQ sensitivity — ΔPSNR when this Linear is kept FP32\n"
        f"(FP32 baseline {fp32:.3f} dB · all-INT8 baseline {int8_all:.3f} dB · "
        f"30 SOTS-indoor pairs)",
        fontsize=11,
    )
    # Annotate cells with values
    for ri in range(M.shape[0]):
        for ci in range(M.shape[1]):
            v = M[ri, ci]
            if np.isnan(v):
                ax.text(ci, ri, "—", ha="center", va="center",
                        color="#9CA3AF", fontsize=9)
            else:
                ax.text(ci, ri, f"{v:+.1f}", ha="center", va="center",
                        color="black" if abs(v) < 0.6 * vmax else "white",
                        fontsize=8.5, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Δ PSNR (mdB) vs all-INT8 baseline", fontsize=10)

    fig.tight_layout()
    out = FIGS / "sensitivity_heatmap.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# ──────────────────────── 3. Cross-domain bars ─────────────────────────────

def cross_domain_bars():
    """4 splits × {teacher, A, B, C} PSNR bars — visualises the case for
    condition-specific students.
    """
    splits = ["indoor", "outdoor", "dense", "nh"]
    nice = {"indoor": "SOTS-indoor", "outdoor": "SOTS-outdoor",
            "dense": "Dense-Haze", "nh": "NH-HAZE"}
    psnrs = {"teacher": [], "A": [], "B": [], "C": []}

    # Teacher
    for s in splits:
        psnrs["teacher"].append(json.load(open(RESULTS / f"dehamer_fp32_{s}.json"))["psnr_mean"])
    # Students
    tags = {"A": "haze_a_small_tight", "B": "haze_b_large_tight",
            "C": "haze_c_large_pseudo"}
    for k, tag in tags.items():
        for s in splits:
            suffix = "" if s == "indoor" else f"_{s}"
            d = json.load(open(RESULTS / f"eval_student_{tag}{suffix}.json"))
            psnrs[k].append(d["eval"]["psnr_mean"])

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(splits))
    w = 0.2

    series = [
        ("Teacher", psnrs["teacher"], C_OURS["teacher"]),
        ("B (w32 GT)", psnrs["B"], C_OURS["B"]),
        ("C (w32 pseudo)", psnrs["C"], C_OURS["C"]),
        ("A (w16 GT)", psnrs["A"], C_OURS["A"]),
    ]
    for i, (name, vals, c) in enumerate(series):
        offset = (i - 1.5) * w
        bars = ax.bar(x + offset, vals, w, label=name, color=c, zorder=3,
                      edgecolor="white", linewidth=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2.0, v + 0.5, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=8, color=c, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([nice[s] for s in splits], fontsize=11)
    ax.set_ylabel("PSNR (dB)", fontsize=12)
    ax.set_title("Cross-domain evaluation — indoor-trained students vs split-specific teachers\n"
                 "(motivation for condition-specific students)", fontsize=12)
    ax.set_ylim(0, 42)
    ax.legend(loc="upper right", fontsize=10, ncol=2)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    fig.tight_layout()
    out = FIGS / "cross_domain_bars.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


if __name__ == "__main__":
    print("generating BMVC figures…")
    pareto_with_baselines()
    sensitivity_heatmap()
    cross_domain_bars()
    print("done.")
