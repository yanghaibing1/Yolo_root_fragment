import argparse
import json
from pathlib import Path

import pandas as pd

from src.evaluation.evaluate_heldout_test import validate_identity
from src.evaluation.select_operating_point import fold_curve


def descriptive_sweep(df, locked):
    validate_identity(df, locked)
    curves = fold_curve(df, float(locked["matching_iou"]))
    curves.insert(0, "outer_fold", int(locked["outer_fold"]))
    curves.insert(0, "model", str(locked["model"]))
    curves["descriptive_only"] = True
    curves["locked_threshold"] = float(locked["confidence_threshold"])
    curves["threshold_zero_definition"] = (
        "all captured predictions; see prediction CSV capture_floor"
    )
    return curves


def main():
    p = argparse.ArgumentParser(
        description="Generate descriptive-only outer-test threshold sweep/FROC data without selecting a threshold."
    )
    p.add_argument("--test-predictions", type=Path, required=True)
    p.add_argument("--threshold-json", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, required=True)
    a = p.parse_args()
    before = a.threshold_json.read_bytes()
    locked = json.loads(before.decode("utf-8"))
    curves = descriptive_sweep(pd.read_csv(a.test_predictions), locked)
    a.output_csv.parent.mkdir(parents=True, exist_ok=True)
    curves.to_csv(a.output_csv, index=False)
    if a.threshold_json.read_bytes() != before:
        raise AssertionError("Locked threshold JSON was modified.")
    print(f"{a.output_csv} (descriptive_only=true; locked threshold unchanged)")


if __name__ == "__main__":
    main()
