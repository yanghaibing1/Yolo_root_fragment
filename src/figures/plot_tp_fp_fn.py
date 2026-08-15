"""Plot aggregate lesion-level TP, FP, and FN counts."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.figures.common import COLORS, save, style


def main():
    """Load per-image counts and save the aggregate bar chart."""
    p = argparse.ArgumentParser(
        description="Plot observed TP/FP/FN counts (not a classification confusion matrix)."
    )
    p.add_argument("--per-image", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    d = pd.read_csv(a.per_image)
    vals = [d.tp.sum(), d.fp.sum(), d.fn.sum()]
    style()
    fig, ax = plt.subplots(figsize=(5, 4.5))
    bars = ax.bar(["TP", "FP", "FN"], vals, color=[COLORS[2], COLORS[0], COLORS[1]])
    ax.bar_label(bars)
    ax.set_ylabel("Observed lesion count")
    save(fig, a.output)


if __name__ == "__main__":
    main()
