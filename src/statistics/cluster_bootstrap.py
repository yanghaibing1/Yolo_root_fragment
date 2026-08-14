import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def metrics(d):
    tp, fp, fn = d[["tp", "fp", "fn"]].sum()
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    return {
        "precision": p,
        "sensitivity": r,
        "f1": 2 * p * r / (p + r) if p + r else 0,
        "fp_per_image": fp / len(d),
    }


def main():
    p = argparse.ArgumentParser(
        description="Paired cluster bootstrap CI by patient (or image)."
    )
    p.add_argument("--per-image", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--cluster", choices=["patient_id", "image_path"], default="patient_id"
    )
    p.add_argument("--iterations", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    df = pd.read_csv(a.per_image, dtype={"patient_id": str})
    groups = list(df.groupby(a.cluster))
    rng = np.random.default_rng(a.seed)
    draws = []
    for _ in range(a.iterations):
        sample = pd.concat(
            [groups[i][1] for i in rng.integers(0, len(groups), len(groups))],
            ignore_index=True,
        )
        draws.append(metrics(sample))
    point = metrics(df)
    result = {
        k: {
            "estimate": point[k],
            "ci95": [
                float(np.percentile([x[k] for x in draws], 2.5)),
                float(np.percentile([x[k] for x in draws], 97.5)),
            ],
        }
        for k in point
    }
    result["cluster_unit"] = a.cluster
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
