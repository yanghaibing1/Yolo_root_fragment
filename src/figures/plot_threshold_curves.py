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
    use_means = "mean_f1" in d
    plot_data = d.drop_duplicates("threshold") if use_means else d
    for col, label, c in [
        ("mean_precision" if use_means else "precision", "Precision", COLORS[3]),
        ("mean_sensitivity" if use_means else "sensitivity", "Sensitivity", COLORS[0]),
        ("mean_f1" if use_means else "f1", "F1", COLORS[2]),
    ]:
        ax.plot(plot_data.threshold, plot_data[col], label=label, color=c)
    metric = "mean_f1" if use_means else "f1"
    candidates = d[d[metric] == d[metric].max()]
    best = candidates.loc[candidates.threshold.idxmax()]
    ax.axvline(
        best.threshold,
        color="gray",
        ls="--",
        label=f"Best mean F1 threshold={best.threshold:.2f}",
    )
    ax.set(xlabel="Confidence threshold", ylabel="Score", ylim=(0, 1.05))
    ax.legend()
    save(fig, a.output)


if __name__ == "__main__":
    main()
