import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.figures.common import COLORS, save, style


def main():
    p = argparse.ArgumentParser(description="Plot real validation threshold curves.")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    d = pd.read_csv(a.input)
    style()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for col, label, c in [
        ("precision", "Precision", COLORS[3]),
        ("sensitivity", "Sensitivity", COLORS[0]),
        ("f1", "F1", COLORS[2]),
    ]:
        ax.plot(d.threshold, d[col], label=label, color=c)
    best = d.loc[d.f1.idxmax()]
    ax.axvline(
        best.threshold,
        color="gray",
        ls="--",
        label=f"Best F1 threshold={best.threshold:.3f}",
    )
    ax.set(xlabel="Confidence threshold", ylabel="Score", ylim=(0, 1.05))
    ax.legend()
    save(fig, a.output)


if __name__ == "__main__":
    main()
