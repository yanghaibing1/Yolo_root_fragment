import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.figures.common import COLORS, save, style


def envelope(d):
    x = np.sort(d.fp_per_image.unique())
    y = np.array([d.loc[d.fp_per_image == v, "sensitivity"].max() for v in x])
    return x, np.maximum.accumulate(y)


def main():
    p = argparse.ArgumentParser(
        description="Plot four real-model FROC curves and export sensitivity at 0.5/1.0 FP/image."
    )
    p.add_argument("--inputs", type=Path, nargs=4, required=True)
    p.add_argument(
        "--labels", nargs=4, default=["YOLOv8n", "YOLOv10n", "YOLO11n", "YOLO12n"]
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--operating-points-csv", type=Path, required=True)
    a = p.parse_args()
    style()
    fig, ax = plt.subplots(figsize=(6, 5))
    rows = []
    for path, label, color in zip(a.inputs, a.labels, COLORS):
        data = pd.read_csv(path)
        if "descriptive_only" not in data or not data.descriptive_only.astype(bool).all():
            raise ValueError(f"{path} must be marked descriptive_only.")
        x, y = envelope(data)
        ax.plot(x, y, label=label, color=color)
        for target in [0.5, 1.0]:
            rows.append(
                {
                    "model": label,
                    "fp_per_image": target,
                    "sensitivity": float(
                        np.interp(target, x, y, left=y[0], right=y[-1])
                    ),
                }
            )
    ax.set(
        xlabel="False positives per image",
        ylabel="Sensitivity",
        ylim=(0, 1.05),
        xlim=(0, None),
    )
    ax.legend()
    a.operating_points_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(a.operating_points_csv, index=False)
    save(fig, a.output)


if __name__ == "__main__":
    main()
