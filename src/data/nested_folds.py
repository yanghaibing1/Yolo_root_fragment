import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser(
        description="Generate patient-level nested 5-fold manifests."
    )
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument(
        "--val-ratio-within-remainder",
        type=float,
        default=0.25,
        help="0.25 yields 60/20/20 overall with five outer folds.",
    )
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    df = pd.read_csv(a.manifest, dtype={"patient_id": str})
    required = {"image_path", "label_path", "patient_id"}
    if required - set(df):
        raise ValueError(f"Missing columns: {sorted(required-set(df))}")
    patients = df.patient_id.drop_duplicates().to_numpy()
    np.random.default_rng(a.seed).shuffle(patients)
    outer = np.array_split(patients, a.folds)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    for fold, test_ids in enumerate(outer):
        rest = np.concatenate([x for i, x in enumerate(outer) if i != fold]).copy()
        np.random.default_rng(a.seed + fold + 1).shuffle(rest)
        n_val = int(round(len(rest) * a.val_ratio_within_remainder))
        val_ids, train_ids = set(rest[:n_val]), set(rest[n_val:])
        test_ids = set(test_ids)
        out = df.copy()
        out["outer_fold"] = fold
        out["split"] = out.patient_id.map(
            lambda x: "test" if x in test_ids else ("val" if x in val_ids else "train")
        )
        out.to_csv(a.output_dir / f"fold_{fold}.csv", index=False)
    print(f"Wrote {a.folds} folds. Select thresholds only on each fold's val rows.")


if __name__ == "__main__":
    main()
