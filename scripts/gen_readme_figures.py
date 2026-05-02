#!/usr/bin/env python3
"""Generate all figures for README.md: training curves, qualitative strips,
sensitivity bar chart, and Pareto plot.

Outputs go to results/figures/ as PNGs.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"
FIGS.mkdir(exist_ok=True)

# ── Colour palette ──────────────────────────────────────────────────────────
C_TEACHER = "#8B5CF6"   # purple
C_A       = "#F59E0B"   # amber
C_B       = "#10B981"   # emerald
C_C       = "#3B82F6"   # blue
C_S1      = "#EF4444"   # red (haze_s1 historical)
BG_DARK   = "#1E1E2E"
FG_LIGHT  = "#CDD6F4"
GRID_CLR  = "#45475A"


def _style():
    plt.rcParams.update({
        "figure.facecolor": BG_DARK,
        "axes.facecolor": BG_DARK,
        "axes.edgecolor": GRID_CLR,
        "axes.labelcolor": FG_LIGHT,
        "text.color": FG_LIGHT,
        "xtick.color": FG_LIGHT,
        "ytick.color": FG_LIGHT,
        "grid.color": GRID_CLR,
        "grid.alpha": 0.4,
        "font.family": "sans-serif",
        "font.size": 11,
        "legend.facecolor": "#313244",
        "legend.edgecolor": GRID_CLR,
        "legend.labelcolor": FG_LIGHT,
    })


# ── 1. Parse training logs ─────────────────────────────────────────────────

def parse_log(path: Path):
    """Parse clean or raw log for epoch losses and validation PSNR/SSIM."""
    text = path.read_text(errors="replace")
    epochs, losses, val_epochs, val_psnrs, val_ssims = [], [], [], [], []

    for line in text.split("\n"):
        # Training loss line:  ep 042  loss 0.0582  l_pix 0.0325  ...
        m = re.match(r"ep\s+(\d+)\s+loss\s+([\d.]+)\s+l_pix\s+([\d.]+)", line.strip())
        if m:
            epochs.append(int(m.group(1)))
            losses.append(float(m.group(2)))

        # VAL line:  VAL  PSNR 33.869  SSIM 0.9834  (min 28.51, max 40.20)
        m = re.search(r"VAL\s+PSNR\s+([\d.]+)\s+SSIM\s+([\d.]+)", line)
        if m:
            # The VAL line follows the last training epoch line
            val_ep = epochs[-1] if epochs else 0
            val_epochs.append(val_ep)
            val_psnrs.append(float(m.group(1)))
            val_ssims.append(float(m.group(2)))

    return {
        "epochs": np.array(epochs),
        "losses": np.array(losses),
        "val_epochs": np.array(val_epochs),
        "val_psnrs": np.array(val_psnrs),
        "val_ssims": np.array(val_ssims),
    }


# ── 2. Training curves figure ──────────────────────────────────────────────

def plot_training_curves():
    _style()

    # Parse all three clean logs (they have the cleanest data)
    log_c = parse_log(RESULTS / "phase2_haze_c_large_pseudo.clean.log")
    log_a = parse_log(RESULTS / "phase2_haze_a_small_tight.clean.log")
    log_b = parse_log(RESULTS / "phase2_haze_b_large_tight.clean.log")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # ── Panel 1: Training loss ──
    ax = axes[0]
    ax.plot(log_a["epochs"], log_a["losses"], color=C_A, alpha=0.8, linewidth=1.2,
            label="Node A (w16, GT)")
    ax.plot(log_b["epochs"], log_b["losses"], color=C_B, alpha=0.8, linewidth=1.2,
            label="Node B (w32, GT)")
    ax.plot(log_c["epochs"], log_c["losses"], color=C_C, alpha=0.8, linewidth=1.2,
            label="Node C (w32, pseudo)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss (total)")
    ax.set_title("Training Loss Convergence", fontweight="bold", fontsize=13)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, linestyle="--")
    ax.set_xlim(0, 200)

    # ── Panel 2: Validation PSNR ──
    ax = axes[1]
    ax.plot(log_a["val_epochs"], log_a["val_psnrs"], "o-", color=C_A, markersize=3,
            linewidth=1.5, label="Node A (w16, GT)")
    ax.plot(log_b["val_epochs"], log_b["val_psnrs"], "s-", color=C_B, markersize=3,
            linewidth=1.5, label="Node B (w32, GT)")
    ax.plot(log_c["val_epochs"], log_c["val_psnrs"], "^-", color=C_C, markersize=3,
            linewidth=1.5, label="Node C (w32, pseudo)")

    # Teacher reference line
    ax.axhline(y=36.576, color=C_TEACHER, linestyle="--", linewidth=1.5,
               alpha=0.7, label="DeHamer teacher (36.58)")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Validation PSNR on SOTS-indoor", fontweight="bold", fontsize=13)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, linestyle="--")
    ax.set_xlim(0, 200)

    # ── Panel 3: Validation SSIM ──
    ax = axes[2]
    ax.plot(log_a["val_epochs"], log_a["val_ssims"], "o-", color=C_A, markersize=3,
            linewidth=1.5, label="Node A (w16, GT)")
    ax.plot(log_b["val_epochs"], log_b["val_ssims"], "s-", color=C_B, markersize=3,
            linewidth=1.5, label="Node B (w32, GT)")
    ax.plot(log_c["val_epochs"], log_c["val_ssims"], "^-", color=C_C, markersize=3,
            linewidth=1.5, label="Node C (w32, pseudo)")
    ax.axhline(y=0.9862, color=C_TEACHER, linestyle="--", linewidth=1.5,
               alpha=0.7, label="DeHamer teacher (0.986)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("SSIM")
    ax.set_title("Validation SSIM on SOTS-indoor", fontweight="bold", fontsize=13)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, linestyle="--")
    ax.set_xlim(0, 200)

    fig.suptitle("Phase 2 — Student Training Dynamics (200 epochs, RESIDE-ITS)",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGS / "training_curves.png", dpi=180, bbox_inches="tight",
                facecolor=BG_DARK)
    plt.close(fig)
    print(f"  ✓ training_curves.png")


# ── 3. Sensitivity bar chart ───────────────────────────────────────────────

def plot_sensitivity():
    _style()
    with open(RESULTS / "dehamer_sensitivity_indoor.json") as f:
        data = json.load(f)

    modules = data["per_module"][:15]  # top-15
    names = [m["module"].replace("swin_1.", "").replace("blocks.", "b") for m in modules]
    deltas = [m["delta_vs_baseline"] * 1000 for m in modules]  # milli-dB for readability

    colors = [C_B if d > 0 else C_A for d in deltas]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    bars = ax.barh(range(len(names)), deltas, color=colors, edgecolor="none", height=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9, fontfamily="monospace")
    ax.invert_yaxis()
    ax.set_xlabel("Δ_recovery (milli-dB)", fontsize=11)
    ax.set_title("Phase 1 — Per-Linear Sensitivity Map (top-15 of 26 modules)",
                 fontweight="bold", fontsize=13)
    ax.axvline(0, color=FG_LIGHT, linewidth=0.5, alpha=0.5)
    ax.grid(True, axis="x", linestyle="--")

    # Annotate top-5
    for i in range(5):
        ax.get_yticklabels()[i].set_fontweight("bold")
        ax.get_yticklabels()[i].set_color("#F9E2AF")

    # Legend
    ax.text(0.98, 0.02, "■ Green = positive recovery (more sensitive)\n"
            "■ Amber = negative (reverting hurts more)",
            transform=ax.transAxes, fontsize=8, va="bottom", ha="right",
            color=FG_LIGHT, alpha=0.7)

    fig.tight_layout()
    fig.savefig(FIGS / "sensitivity_barplot.png", dpi=180, bbox_inches="tight",
                facecolor=BG_DARK)
    plt.close(fig)
    print(f"  ✓ sensitivity_barplot.png")


# ── 4. Pareto plot: PSNR vs Parameters ──────────────────────────────────────

def plot_pareto():
    _style()
    fig, ax = plt.subplots(figsize=(9, 6))

    # Data points: (params_M, psnr, label, color, marker)
    points = [
        (132.45, 36.576, "DeHamer\n(teacher)", C_TEACHER, "D", 120),
        (17.11,  34.40,  "Node B\n(w32, GT)", C_B, "s", 100),
        (17.11,  33.87,  "Node C\n(w32, pseudo)", C_C, "^", 100),
        (4.35,   32.39,  "Node A\n(w16, GT)", C_A, "o", 100),
        (4.35,   29.78,  "haze_s1\n(w16, weak)", C_S1, "x", 80),
    ]

    for params, psnr, label, color, marker, size in points:
        ax.scatter(params, psnr, c=color, s=size, marker=marker, zorder=5,
                   edgecolors="white", linewidths=0.8)
        # offset label
        xoff = 4 if params > 10 else 0.3
        yoff = 0.3
        if "s1" in label:
            yoff = -0.5
        ax.annotate(label, (params, psnr), fontsize=8.5,
                    xytext=(xoff, yoff), textcoords="offset points",
                    ha="left", va="bottom", color=color, fontweight="bold")

    # Compression annotations
    ax.annotate("7.7× smaller", xy=(17.11, 34.40), xytext=(50, 34.8),
                fontsize=8, color=FG_LIGHT, alpha=0.6,
                arrowprops=dict(arrowstyle="->", color=FG_LIGHT, alpha=0.4))
    ax.annotate("30.5× smaller", xy=(4.35, 32.39), xytext=(20, 32.0),
                fontsize=8, color=FG_LIGHT, alpha=0.6,
                arrowprops=dict(arrowstyle="->", color=FG_LIGHT, alpha=0.4))

    ax.set_xlabel("Parameters (M)", fontsize=12)
    ax.set_ylabel("PSNR (dB) on SOTS-indoor", fontsize=12)
    ax.set_title("Quality vs. Model Size — Compression Pareto Frontier",
                 fontweight="bold", fontsize=14)
    ax.set_xscale("log")
    ax.set_xticks([4, 10, 20, 50, 100, 130])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.grid(True, linestyle="--")
    ax.set_ylim(28, 38)

    fig.tight_layout()
    fig.savefig(FIGS / "pareto_psnr_params.png", dpi=180, bbox_inches="tight",
                facecolor=BG_DARK)
    plt.close(fig)
    print(f"  ✓ pareto_psnr_params.png")


# ── 5. Phase 1 results bar chart ────────────────────────────────────────────

def plot_phase1_bars():
    _style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    variants = [
        "FP32\n(reference)",
        "INT8 dyn\n(all Linear)",
        "INT8 dyn\n(mixed top-5)",
        "Block-static\n(CNN only)",
        "Block + dyn\n(all)",
        "Block + dyn\n(mixed)",
    ]
    psnrs = [36.576, 36.470, 36.551, 34.545, 34.487, 34.524]
    speedups = [1.00, 1.28, 1.27, 1.22, 1.11, 1.10]
    colors = ["#6C7086", "#89B4FA", "#A6E3A1", "#F38BA8", "#F9E2AF", "#FAB387"]

    # PSNR bars
    bars = ax1.bar(range(len(variants)), psnrs, color=colors, edgecolor="none", width=0.7)
    ax1.set_xticks(range(len(variants)))
    ax1.set_xticklabels(variants, fontsize=8)
    ax1.set_ylabel("PSNR (dB)")
    ax1.set_title("Phase 1 — Quality Comparison", fontweight="bold", fontsize=13)
    ax1.set_ylim(33.5, 37.0)
    ax1.grid(True, axis="y", linestyle="--")
    for bar, val in zip(bars, psnrs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=8, color=FG_LIGHT)

    # Speedup bars
    bars2 = ax2.bar(range(len(variants)), speedups, color=colors, edgecolor="none", width=0.7)
    ax2.set_xticks(range(len(variants)))
    ax2.set_xticklabels(variants, fontsize=8)
    ax2.set_ylabel("CPU Speedup (×)")
    ax2.set_title("Phase 1 — CPU Speedup @ 256²", fontweight="bold", fontsize=13)
    ax2.set_ylim(0, 1.5)
    ax2.grid(True, axis="y", linestyle="--")
    for bar, val in zip(bars2, speedups):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{val:.2f}×", ha="center", va="bottom", fontsize=8, color=FG_LIGHT)

    fig.suptitle("Phase 1 — Post-Training Quantization of DeHamer (SOTS-indoor, 500 pairs)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGS / "phase1_bars.png", dpi=180, bbox_inches="tight",
                facecolor=BG_DARK)
    plt.close(fig)
    print(f"  ✓ phase1_bars.png")


# ── 6. Qualitative comparison strips ────────────────────────────────────────

def plot_qualitative():
    _style()
    qual_dir = RESULTS / "qualitative"
    scenes = sorted([d.name for d in qual_dir.iterdir() if d.is_dir()])

    # Pick 3 representative scenes
    chosen = scenes[:3] if len(scenes) >= 3 else scenes

    labels = ["Hazy Input", "Ground Truth", "DeHamer\n(teacher)", "Node A\n(w16, 4.35M)",
              "Node B\n(w32, 17.1M)", "Node C\n(w32, pseudo)"]
    files = ["hazy.png", "gt.png", "teacher_dehamer.png", "nodeA_w16.png",
             "nodeB_w32.png", "nodeC_w32p.png"]

    n_scenes = len(chosen)
    n_cols = len(files)
    fig, axes = plt.subplots(n_scenes, n_cols, figsize=(n_cols * 3.2, n_scenes * 3.2))
    if n_scenes == 1:
        axes = axes[np.newaxis, :]

    for row, scene in enumerate(chosen):
        for col, (fname, label) in enumerate(zip(files, labels)):
            ax = axes[row, col]
            img_path = qual_dir / scene / fname
            if img_path.exists():
                img = mpimg.imread(str(img_path))
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, color=FG_LIGHT)
            ax.axis("off")
            if row == 0:
                ax.set_title(label, fontsize=10, fontweight="bold", pad=8)
            if col == 0:
                ax.set_ylabel(f"Scene {scene}", fontsize=10, fontweight="bold",
                              rotation=0, labelpad=60, va="center")

    fig.suptitle("Qualitative Comparison — SOTS-indoor Test Scenes",
                 fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(FIGS / "qualitative_comparison.png", dpi=150, bbox_inches="tight",
                facecolor=BG_DARK)
    plt.close(fig)
    print(f"  ✓ qualitative_comparison.png ({n_scenes} scenes)")


# ── 7. Ablation heatmap (2x2 + anchor) ──────────────────────────────────────

def plot_ablation():
    _style()
    fig, ax = plt.subplots(figsize=(7, 5))

    # 2x2 grid: rows = capacity, cols = target
    data = np.array([
        [29.78, 32.39],  # w16: haze_s1 (weak), Node A (tight)
        [33.87, 34.40],  # w32: Node C (pseudo), Node B (GT)
    ])
    labels_grid = np.array([
        ["haze_s1\n29.78 dB\n(weak losses)", "Node A\n32.39 dB"],
        ["Node C\n33.87 dB", "Node B\n34.40 dB\n(WINNER)"],
    ])

    im = ax.imshow(data, cmap="YlGn", vmin=28, vmax=36, aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pseudo (teacher output)", "GT clean"], fontsize=11)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Width 16\n(4.35M params)", "Width 32\n(17.11M params)"], fontsize=11)
    ax.set_xlabel("Supervision Target", fontsize=12, labelpad=10)
    ax.set_ylabel("Student Capacity", fontsize=12, labelpad=10)

    for i in range(2):
        for j in range(2):
            text_color = "black" if data[i, j] > 32 else "white"
            ax.text(j, i, labels_grid[i, j], ha="center", va="center",
                    fontsize=10, fontweight="bold", color=text_color)

    ax.set_title("Phase 2 — Capacity × Target Ablation (PSNR on SOTS-indoor)",
                 fontweight="bold", fontsize=13, pad=15)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="PSNR (dB)")
    cbar.ax.yaxis.label.set_color(FG_LIGHT)
    cbar.ax.tick_params(colors=FG_LIGHT)

    fig.tight_layout()
    fig.savefig(FIGS / "ablation_heatmap.png", dpi=180, bbox_inches="tight",
                facecolor=BG_DARK)
    plt.close(fig)
    print(f"  ✓ ablation_heatmap.png")


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating README figures...")
    plot_training_curves()
    plot_sensitivity()
    plot_pareto()
    plot_phase1_bars()
    plot_qualitative()
    plot_ablation()
    print(f"\nAll figures saved to {FIGS}/")
