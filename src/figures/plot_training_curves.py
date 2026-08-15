"""Plot training metrics exported by Ultralytics."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.figures.common import COLORS, save, style


def main():
    """Load epoch metrics and save the training-curve figure."""
    p = argparse.ArgumentParser(
        description="Plot real Ultralytics Precision/Sensitivity/F1/mAP50 by epoch."
    )
    p.add_argument("--results-csv", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    d = pd.read_csv(a.results_csv)
    d.columns = d.columns.str.strip()
    pr = d["metrics/precision(B)"]
    se = d["metrics/recall(B)"]
    f1 = 2 * pr * se / (pr + se).replace(0, np.nan)
    ep = d["epoch"]
    style()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for y, label, c in [
        (pr, "Precision", COLORS[3]),
        (se, "Sensitivity", COLORS[0]),
        (f1, "F1", COLORS[2]),
        (d["metrics/mAP50(B)"], "mAP50", COLORS[1]),
    ]:
        ax.plot(ep, y, label=label, color=c)
    i = int(np.nanargmax(f1))
    ax.scatter(ep.iloc[i], f1.iloc[i], color=COLORS[2])
    ax.annotate(
        f"Best F1={f1.iloc[i]:.3f}",
        (ep.iloc[i], f1.iloc[i]),
        xytext=(8, 8),
        textcoords="offset points",
    )
    ax.set(xlabel="Epoch", ylabel="Score", ylim=(0, 1.05))
    ax.legend()
    save(fig, a.output)


if __name__ == "__main__":
    main()
