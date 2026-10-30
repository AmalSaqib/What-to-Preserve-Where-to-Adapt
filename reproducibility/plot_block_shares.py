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


HERE = Path(__file__).resolve().parent
BLOCK_ORDER = ["E4", "E5", "B", "D5", "D4"]
COLORS = {"E4": "#F59449", "D4": "#F59449", "E5": "#B8C2CC", "D5": "#B8C2CC", "B": "#6591C0"}
LABEL_INK = {"E4": "#1F2328", "D4": "#1F2328", "E5": "#1F2328", "D5": "#1F2328", "B": "#1F2328"}
ROW_LABELS = {
    "Shared LR 0.01": "Shared\n0.01",
    "Uniform LR 0.005": "Uniform\n0.005",
    "Uniform LR 0.0025": "Uniform\n0.0025",
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
                         "xtick.color": MUTED, "ytick.color": INK, "ytick.labelsize": 6.5, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.4, 1.75))

    for y, row in enumerate(rows):
        left = 0.0
        for block in BLOCK_ORDER:
            width = row["shares"][block]
            ax.barh(y, width, left=left, height=0.6, color=COLORS[block],
                    edgecolor="white", linewidth=1.0)
            label = f"{block}\n{width:.0f}%"
            if width >= 5:
                ax.text(left + width / 2, y, label, ha="center", va="center",
                        fontsize=5.6 if width >= 9 else 4.8, color=LABEL_INK[block], linespacing=0.95)
            else:  # too narrow for text: call out in the gap above the bar
                ax.annotate(f"{block} {width:.0f}%", xy=(left + width / 2, y - 0.3),
                            xytext=(left + width / 2 + 3, y - 0.5), ha="left", va="center",
                            fontsize=5.0, color=INK,
                            arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.5,
                                            shrinkA=0, shrinkB=0))
            left += width
        ax.text(103, y, f"{row['forgetting']:.3f}", ha="left", va="center", fontsize=6.5, color=INK)

    ax.text(103, -0.55, "UMD\nforgetting", ha="left", va="bottom", fontsize=6, color=MUTED, linespacing=0.95)
    ax.set_yticks(range(len(rows)), [row["label"] for row in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Share of squared parameter displacement in $\\mathcal{R}_4$ (%)", color=INK)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=2, width=0.6)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", pad_inches=0.02, metadata={"Author": ""})
    fig.savefig(args.output.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved {args.output} and {args.output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
