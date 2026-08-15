import argparse
import json
from pathlib import Path

import pandas as pd


KEYS = ["precision", "sensitivity", "f1", "fp_per_image"]


def summarize(paths):
    rows = []
    model = None
    seen = set()
    for path in paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        required = {"model", "outer_fold", "evaluation_split", *KEYS}
        missing = required - set(data)
        if missing:
            raise ValueError(f"{path} missing fields: {sorted(missing)}")
        if data["evaluation_split"] != "outer_test":
            raise ValueError(f"{path} is not an outer_test evaluation.")
        if model is None:
            model = str(data["model"])
        elif str(data["model"]) != model:
            raise ValueError("All fold summaries must belong to the same model.")
        fold = int(data["outer_fold"])
        if fold in seen:
            raise ValueError(f"Duplicate outer_fold {fold}.")
        seen.add(fold)
        rows.append({"outer_fold": fold, **{key: data[key] for key in KEYS}})
    if seen != {0, 1, 2, 3, 4}:
        raise ValueError(f"Outer folds must be exactly 0-4; got {sorted(seen)}.")
    df = pd.DataFrame(rows)
    return pd.DataFrame(
        {
            "model": model,
            "metric": KEYS,
            "mean": [df[key].mean() for key in KEYS],
            "sample_sd": [df[key].std(ddof=1) for key in KEYS],
            "n_folds": 5,
        }
    )


def main():
    p = argparse.ArgumentParser(
        description="Strictly summarize outer-test folds 0-4 as mean and sample SD."
    )
    p.add_argument("--inputs", type=Path, nargs=5, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    out = summarize(a.inputs)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.output, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
