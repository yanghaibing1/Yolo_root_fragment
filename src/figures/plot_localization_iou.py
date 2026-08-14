import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.figures.common import COLORS, save, style


def main():
    p = argparse.ArgumentParser(
        description="Plot matched-lesion IoU distribution from real test matches."
    )
    p.add_argument("--matched-lesions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    d = pd.read_csv(a.matched_lesions)
    style()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.hist(d.iou, bins=20, range=(0, 1), color=COLORS[1], edgecolor="white")
    ax.axvline(0.5, color=COLORS[0], ls="--")
    ax.axvline(0.75, color=COLORS[2], ls="--")
    ax.set(xlabel="Matched-box IoU", ylabel="Lesion count", xlim=(0, 1))
    save(fig, a.output)


if __name__ == "__main__":
    main()
