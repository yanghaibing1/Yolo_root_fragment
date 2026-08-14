import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = {"image_path", "label_path", "patient_id"}


def main():
    p = argparse.ArgumentParser(description="Patient-level train/val/test split.")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--train-ratio", type=float, default=0.60)
    p.add_argument("--val-ratio", type=float, default=0.20)
    p.add_argument("--test-ratio", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    if not np.isclose(a.train_ratio + a.val_ratio + a.test_ratio, 1.0):
        raise ValueError("Ratios must sum to 1.")
    df = pd.read_csv(a.manifest, dtype={"patient_id": str})
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    patients = df.patient_id.drop_duplicates().to_numpy()
    rng = np.random.default_rng(a.seed)
    rng.shuffle(patients)
    n = len(patients)
    n_train = int(round(n * a.train_ratio))
    n_val = int(round(n * a.val_ratio))
    mapping = {x: "train" for x in patients[:n_train]}
    mapping.update({x: "val" for x in patients[n_train : n_train + n_val]})
    mapping.update({x: "test" for x in patients[n_train + n_val :]})
    df["split"] = df.patient_id.map(mapping)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.output, index=False)
    print(df.groupby("split")["patient_id"].agg(images="size", patients="nunique"))


if __name__ == "__main__":
    main()
