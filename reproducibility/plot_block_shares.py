#!/usr/bin/env python3
"""Plot where parameter movement goes within R4 for the Q3 learning-rate settings.

Each bar shows one setting's share of squared parameter displacement per
trainable block, in U-Net order (E4 | E5 | B | D5 | D4), with UMD forgetting
annotated on the right. The input is the per-seed CSV written by
``summarize_questions.py`` (default: the reported reference results).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


HERE = Path(__file__).resolve().parent
BLOCK_ORDER = ["E4", "E5", "B", "D5", "D4"]
COLORS = {"E4": "#F59449", "D4": "#F59449", "E5": "#B8C2CC", "D5": "#B8C2CC", "B": "#6591C0"}
LABEL_INK = {"E4": "#1F2328", "D4": "#1F2328", "E5": "#1F2328", "D5": "#1F2328", "B": "#1F2328"}
ROW_LABELS = {
    "Shared LR 0.01": "Shared, 0.01",
    "Uniform LR 0.005": "Uniform, 0.005",
    "Uniform LR 0.0025": "Uniform, 0.0025",
    "Scale-aware": "Scale-aware",
}
INK = "#1F2328"
MUTED = "#57606A"


def read_rows(path: Path):
    rows = []
    with path.open() as handle:
        for row in csv.DictReader(handle):
            shares = dict(item.split("=") for item in row["block_shares_percent"].split(";"))
            rows.append({
                "label": ROW_LABELS.get(row["row"], row["row"]),
                "shares": {block: float(shares[block]) for block in BLOCK_ORDER},
                "forgetting": float(row["old_task_forgetting"]),
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE / "reported" / "Q3_uniform_lr_control_per_seed.csv")
    parser.add_argument("--output", type=Path, default=Path("outputs/figures/r4_block_shares.pdf"))
    args = parser.parse_args()

    rows = read_rows(args.input)
    plt.rcParams.update({"font.size": 7, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
                         "xtick.color": MUTED, "ytick.color": INK, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.4, 1.75))

    for y, row in enumerate(rows):
        left = 0.0
        for block in BLOCK_ORDER:
            width = row["shares"][block]
            ax.barh(y, width, left=left, height=0.66, color=COLORS[block],
                    edgecolor="white", linewidth=1.0)
            if width >= 9:
                ax.text(left + width / 2, y, f"{block}\n{width:.0f}%", ha="center", va="center",
                        fontsize=5.6, color=LABEL_INK[block], linespacing=0.95)
            elif width >= 5:  # narrow segment: value only; colour and position identify the block
                ax.text(left + width / 2, y, f"{width:.0f}%", ha="center", va="center",
                        fontsize=5.0, color=LABEL_INK[block])
            left += width
        ax.text(103, y, f"{row['forgetting']:.3f}", ha="left", va="center", fontsize=6.5, color=INK)

    ax.text(103, -0.75, "UMD\nforgetting", ha="left", va="bottom", fontsize=6, color=MUTED, linespacing=0.95)
    ax.set_yticks(range(len(rows)), [row["label"] for row in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Share of squared parameter displacement in $\\mathcal{R}_4$ (%)", color=INK)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=2, width=0.6)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    handles = [Patch(facecolor=COLORS["E4"], label="Outer pair ($E_4$, $D_4$)"),
               Patch(facecolor=COLORS["E5"], label="$E_5$, $D_5$"),
               Patch(facecolor=COLORS["B"], label="Bottleneck ($B$)")]
    ax.legend(handles=handles, ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.45, 1.0),
              fontsize=6, handlelength=1.0, handleheight=0.8, columnspacing=1.0, borderaxespad=0.3)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", pad_inches=0.02, metadata={"Author": ""})
    fig.savefig(args.output.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved {args.output} and {args.output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
