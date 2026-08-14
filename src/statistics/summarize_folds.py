import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser(
        description="Summarize real fold summary JSON files as mean and SD."
    )
    p.add_argument("--inputs", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    keys = ["precision", "sensitivity", "f1", "fp_per_image"]
    rows = []
    for path in a.inputs:
        d = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"fold": path.stem, **{k: d[k] for k in keys}})
    df = pd.DataFrame(rows)
    out = pd.DataFrame(
        {
            "metric": keys,
            "mean": [df[k].mean() for k in keys],
            "sd": [df[k].std(ddof=1) for k in keys],
            "n_folds": len(df),
        }
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.output, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
