import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import boxes, match, safe, scores


THRESHOLDS = np.round(np.arange(101) / 100, 2)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fold_curve(df, matching_iou):
    rows = []
    for threshold in THRESHOLDS:
        tp = fp = fn = 0
        for _, row in df.iterrows():
            pb = boxes(row.pred_boxes)
            sc = scores(row.pred_scores)
            gb = boxes(row.gt_boxes)
            pairs, fold_fp, fold_fn = match(pb[sc >= threshold], gb, matching_iou)
            tp += len(pairs)
            fp += fold_fp
            fn += fold_fn
        precision = safe(tp, tp + fp)
        sensitivity = safe(tp, tp + fn)
        rows.append(
            {
                "threshold": float(threshold),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "sensitivity": sensitivity,
                "f1": safe(2 * tp, 2 * tp + fp + fn),
                "fp_per_image": safe(fp, len(df)),
            }
        )
    return pd.DataFrame(rows)


def select_threshold(prediction_paths, model, outer_fold, manifest_path, matching_iou=0.5):
    if len(prediction_paths) != 4:
        raise ValueError("Exactly four inner-validation prediction CSVs are required.")
    curves = []
    seen = set()
    fold_records = []
    for path in prediction_paths:
        df = pd.read_csv(path)
        required = {"model", "outer_fold", "inner_fold", "split"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing metadata columns: {sorted(missing)}")
        if set(df.model.astype(str)) != {str(model)}:
            raise ValueError(f"{path} model does not match {model}.")
        if set(df.outer_fold.astype(int)) != {int(outer_fold)}:
            raise ValueError(f"{path} outer_fold does not match {outer_fold}.")
        if set(df.split.astype(str)) != {"inner_val"}:
            raise ValueError(f"{path} must contain split=inner_val only.")
        inner_values = set(df.inner_fold.astype(int))
        if len(inner_values) != 1:
            raise ValueError(f"{path} must contain exactly one inner_fold.")
        inner_fold = inner_values.pop()
        if inner_fold in seen:
            raise ValueError(f"Duplicate inner_fold {inner_fold}.")
        seen.add(inner_fold)
        curve = fold_curve(df, matching_iou)
        curve["inner_fold"] = inner_fold
        curves.append(curve)
        fold_records.append({"inner_fold": inner_fold, "n_validation_images": len(df)})
    if seen != {0, 1, 2, 3}:
        raise ValueError(f"Inner folds must be exactly 0-3; got {sorted(seen)}.")
    all_curves = pd.concat(curves, ignore_index=True)
    means = (
        all_curves.groupby("threshold", as_index=False)[
            ["precision", "sensitivity", "f1", "fp_per_image"]
        ]
        .mean()
        .rename(columns={
            "precision": "mean_precision",
            "sensitivity": "mean_sensitivity",
            "f1": "mean_f1",
            "fp_per_image": "mean_fp_per_image",
        })
    )
    best = means.sort_values(
        ["mean_f1", "threshold"], ascending=[False, False]
    ).iloc[0]
    selected = float(best.threshold)
    for record in fold_records:
        row = all_curves[
            (all_curves.inner_fold == record["inner_fold"])
            & (all_curves.threshold == selected)
        ].iloc[0]
        record.update(
            {
                "f1_at_selected_threshold": float(row.f1),
                "precision_at_selected_threshold": float(row.precision),
                "sensitivity_at_selected_threshold": float(row.sensitivity),
                "tp": int(row.tp),
                "fp": int(row.fp),
                "fn": int(row.fn),
            }
        )
    manifest_path = Path(manifest_path)
    payload = {
        "model": str(model),
        "outer_fold": int(outer_fold),
        "manifest": {
            "source": str(manifest_path.resolve()),
            "sha256": file_sha256(manifest_path),
        },
        "confidence_threshold": selected,
        "threshold_grid": {"start": 0.0, "stop": 1.0, "step": 0.01, "count": 101},
        "selection_metric": "arithmetic mean of four inner-validation lesion-level F1 values",
        "tie_break": "higher confidence threshold",
        "selection_split": "inner_validation_only",
        "matching_iou": float(matching_iou),
        "inner_folds": sorted(fold_records, key=lambda item: item["inner_fold"]),
        "selected_mean_f1": float(best.mean_f1),
    }
    return payload, all_curves.merge(means, on="threshold", how="left")


def main():
    p = argparse.ArgumentParser(
        description="Select a locked threshold by mean F1 across four inner validation folds."
    )
    p.add_argument("--validation-predictions", type=Path, nargs=4, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--outer-fold", type=int, choices=range(5), required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--curves-csv", type=Path, required=True)
    p.add_argument("--matching-iou", type=float, default=0.5)
    a = p.parse_args()
    payload, curves = select_threshold(
        a.validation_predictions, a.model, a.outer_fold, a.manifest, a.matching_iou
    )
    a.output_json.parent.mkdir(parents=True, exist_ok=True)
    a.curves_csv.parent.mkdir(parents=True, exist_ok=True)
    a.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    curves.to_csv(a.curves_csv, index=False)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
