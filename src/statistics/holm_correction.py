import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser(
        description="Holm correction for pairwise model comparisons."
    )
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="CSV containing model_a, model_b, p_value.",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--alpha", type=float, default=0.05)
    a = p.parse_args()
    df = pd.read_csv(a.input)
    order = np.argsort(df.p_value.to_numpy())
    m = len(df)
    adj = np.empty(m)
    running = 0
    for rank, idx in enumerate(order):
        running = max(running, min(1, (m - rank) * df.p_value.iloc[idx]))
        adj[idx] = running
    df["p_holm"] = adj
    df["reject_holm"] = df.p_holm <= a.alpha
    a.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.output, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
