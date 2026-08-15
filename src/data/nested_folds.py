import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED = {"image_path", "label_path", "patient_id"}


def generate_nested_folds(df, seed=42, outer_folds=5, inner_folds=4):
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if outer_folds != 5 or inner_folds != 4:
        raise ValueError("The protocol requires exactly 5 outer folds and 4 inner folds.")
    patients = df.patient_id.astype(str).drop_duplicates().to_numpy()
    if len(patients) < outer_folds:
        raise ValueError("At least five patients are required.")
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)
    outer_parts = np.array_split(patients, outer_folds)
    outputs = []
    for outer_fold, test_array in enumerate(outer_parts):
        test_ids = set(test_array)
        training = np.concatenate(
            [part for i, part in enumerate(outer_parts) if i != outer_fold]
        ).copy()
        inner_rng = np.random.default_rng(seed + outer_fold + 1)
        inner_rng.shuffle(training)
        inner_parts = np.array_split(training, inner_folds)
        inner_map = {
            patient: inner_fold
            for inner_fold, part in enumerate(inner_parts)
            for patient in part
        }
        out = df.copy()
        out["patient_id"] = out.patient_id.astype(str)
        out["outer_fold"] = outer_fold
        out["split"] = out.patient_id.map(
            lambda patient: "outer_test" if patient in test_ids else "outer_train"
        )
        out["inner_fold"] = out.patient_id.map(inner_map).astype("Int64")
        outputs.append(out)
    return outputs


def main():
    p = argparse.ArgumentParser(
        description="Generate patient-level outer 5-fold and nested inner 4-fold manifests."
    )
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--folds", type=int, default=5, help="Must be 5.")
    p.add_argument("--inner-folds", type=int, default=4, help="Must be 4.")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    df = pd.read_csv(a.manifest, dtype={"patient_id": str})
    outputs = generate_nested_folds(df, a.seed, a.folds, a.inner_folds)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    for outer_fold, out in enumerate(outputs):
        out.to_csv(a.output_dir / f"outer_fold_{outer_fold}.csv", index=False)
    print(
        "Wrote five outer manifests. For each outer fold, train four models: "
        "inner validation k is outer_train rows with inner_fold=k; outer_test stays isolated."
    )


if __name__ == "__main__":
    main()
