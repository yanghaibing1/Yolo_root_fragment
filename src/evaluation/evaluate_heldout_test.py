"""Evaluate outer-test predictions using an inner-validation-locked threshold."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import boxes, match, safe, scores


def validate_identity(df, locked):
    required = {"model", "outer_fold", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Test predictions missing metadata columns: {sorted(missing)}")
    models = set(df.model.astype(str))
    folds = set(df.outer_fold.astype(int))
    splits = set(df.split.astype(str))
    if models != {str(locked["model"])}:
        raise ValueError("Threshold JSON model does not match test predictions.")
    if folds != {int(locked["outer_fold"])}:
        raise ValueError("Threshold JSON outer_fold does not match test predictions.")
    if splits != {"outer_test"}:
        raise ValueError("Input must contain outer_test rows only.")


def evaluate(df, locked, matching_iou=None):
    validate_identity(df, locked)
    threshold = float(locked["confidence_threshold"])
    locked_iou = float(locked["matching_iou"])
    if matching_iou is not None and float(matching_iou) != locked_iou:
        raise ValueError("--matching-iou cannot override the locked JSON matching_iou.")
    per = []
    matched = []
    for _, row in df.iterrows():
        pb, sc, gb = boxes(row.pred_boxes), scores(row.pred_scores), boxes(row.gt_boxes)
        keep = sc >= threshold
        pb, sc = pb[keep], sc[keep]
        pairs, fp, fn = match(pb, gb, locked_iou)
        for pi, gi, iou in pairs:
            matched.append(
                {
                    "image_path": row.image_path,
                    "patient_id": row.get("patient_id", ""),
                    "pred_index": pi,
                    "gt_index": gi,
                    "confidence": sc[pi],
                    "iou": iou,
                }
            )
        per.append(
            {
                "image_path": row.image_path,
                "patient_id": row.get("patient_id", ""),
                "tp": len(pairs),
                "fp": fp,
                "fn": fn,
                "gt_count": len(gb),
                "pred_count": len(pb),
                "image_success": int(fn == 0),
            }
        )
    out = pd.DataFrame(per)
    matched_df = pd.DataFrame(
        matched,
        columns=["image_path", "patient_id", "pred_index", "gt_index", "confidence", "iou"],
    )
    tp, fp, fn = int(out.tp.sum()), int(out.fp.sum()), int(out.fn.sum())
    gt = int(out.gt_count.sum())
    if tp + fn != gt:
        raise AssertionError("Invariant failed: TP+FN != total GT boxes")
    ious = matched_df.iou.to_numpy() if len(matched_df) else np.array([])
    q = np.percentile(ious, [25, 75]) if len(ious) else [None, None]
    summary = {
        "model": str(locked["model"]),
        "outer_fold": int(locked["outer_fold"]),
        "evaluation_split": "outer_test",
        "threshold": threshold,
        "threshold_source": "locked inner-validation JSON",
        "matching_iou": locked_iou,
        "n_images": len(out),
        "total_gt_boxes": gt,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": safe(tp, tp + fp),
        "sensitivity": safe(tp, tp + fn),
        "f1": safe(2 * tp, 2 * tp + fp + fn),
        "fp_per_image": safe(fp, len(out)),
        "localization_iou": {
            "mean": float(np.mean(ious)) if len(ious) else None,
            "sd": float(np.std(ious, ddof=1)) if len(ious) > 1 else None,
            "median": float(np.median(ious)) if len(ious) else None,
            "iqr": float(q[1] - q[0]) if len(ious) else None,
            "proportion_ge_0.50": float(np.mean(ious >= 0.5)) if len(ious) else None,
            "proportion_ge_0.75": float(np.mean(ious >= 0.75)) if len(ious) else None,
        },
    }
    return out, matched_df, summary


def main():
    p = argparse.ArgumentParser(
        description="Evaluate isolated outer-test predictions at the locked inner-validation threshold."
    )
    p.add_argument("--test-predictions", type=Path, required=True)
    p.add_argument("--threshold-json", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--matching-iou", type=float, default=None)
    p.add_argument("--ap-summary-json", type=Path)
    a = p.parse_args()
    df = pd.read_csv(a.test_predictions)
    locked = json.loads(a.threshold_json.read_text(encoding="utf-8"))
    out, matched, summary = evaluate(df, locked, a.matching_iou)
    if a.ap_summary_json:
        summary["ultralytics_heldout_val_api_ap"] = json.loads(
            a.ap_summary_json.read_text(encoding="utf-8")
        )
        summary["ap_source"] = "Ultralytics outer-test val API; separate from locked-threshold lesion matching"
    a.output_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.output_dir / "per_image.csv", index=False)
    matched.to_csv(a.output_dir / "matched_lesions.csv", index=False)
    (a.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
