"""
Figure 6: Computational cost comparison — M-ACAM vs baselines (Section 5.12).

Baseline training protocol (manuscript): same splits, augmentation, hardware (RTX 3060),
optimizer (AdamW for Transformers, Adam for CNNs per papers), stopping (patience=20).

This script plots five bar charts from the paper values:
  Params (M), FLOPs (G), Train time (h), Memory (MB), FPS.

Usage:
  python -m m_acam.plot_figure6_computational_cost --out checkpoints/figures/figure6_computational_cost.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Manuscript colors: U-Net green, M-ACAM orange, Swin/Trans light blue tones
COLORS: Dict[str, str] = {
    "U-Net": "#2ca02c",
    "M-ACAM": "#ff7f0e",
    "Swin-Unet": "#a6cee3",
    "TransUNet": "#6baed6",
}

# Default x-axis order (Params, Train, Memory, FPS)
ORDER_DEFAULT: List[str] = ["U-Net", "M-ACAM", "Swin-Unet", "TransUNet"]
# FLOPs panel uses different left-to-right order in the manuscript figure
ORDER_FLOPS: List[str] = ["M-ACAM", "U-Net", "Swin-Unet", "TransUNet"]

DATA: Dict[str, Tuple[List[str], List[float]]] = {
    "Params (M)": (ORDER_DEFAULT, [23.1, 24.8, 41.3, 105.3]),
    "FLOPs (G)": (ORDER_FLOPS, [31.2, 54.3, 82.4, 118.6]),
    "Train time (h)": (ORDER_DEFAULT, [4.2, 18.1, 19.3, 28.7]),
    "Memory (MB)": (ORDER_DEFAULT, [982.0, 1638.0, 2150.0, 3277.0]),
    "FPS": (ORDER_DEFAULT, [87.0, 38.0, 26.0, 19.0]),
}

CAPTION = "Figure 6: Computational Cost Comparison: Proposed M-ACAM vs Baselines"
NOTE = "M-ACAM achieves 2.6x efficiency vs. Swin-Unet while improving Dice by +1.3%."


def plot_figure6(
    out_path: Path,
    dpi: int = 150,
    figsize: Tuple[float, float] = (14.0, 3.2),
) -> None:
    titles = list(DATA.keys())
    n = len(titles)
    fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)
    ax_flat = axes[0]

    for ax, title in zip(ax_flat, titles):
        models, values = DATA[title]
        x = np.arange(len(models))
        colors = [COLORS[m] for m in models]
        bars = ax.bar(x, values, color=colors, edgecolor="0.25", linewidth=0.6, width=0.72)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=25, ha="right", fontsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.set_axisbelow(True)
        # Value labels on bars (compact)
        for b, v in zip(bars, values):
            h = b.get_height()
            lbl = f"{v:.1f}" if title != "Memory (MB)" else f"{int(round(v))}"
            if title == "FPS":
                lbl = f"{int(round(v))}"
            ax.text(
                b.get_x() + b.get_width() / 2.0,
                h,
                lbl,
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=0,
            )

    fig.suptitle(CAPTION, fontsize=11, y=0.99)
    fig.text(0.5, 0.02, NOTE, ha="center", fontsize=9, style="italic")
    fig.tight_layout(rect=[0, 0.10, 1, 0.93])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Figure 6 computational cost bar charts")
    p.add_argument(
        "--out",
        type=str,
        default="checkpoints/figures/figure6_computational_cost.png",
        help="Output image path (.png or .pdf)",
    )
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    plot_figure6(Path(args.out), dpi=args.dpi)
    print(f"Wrote {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
