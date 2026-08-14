"""Evaluate held-out predictions using a validation-locked threshold."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import boxes, match, safe, scores


def main():
    p = argparse.ArgumentParser(
        description="Evaluate held-out test predictions at a locked validation threshold."
    )
    p.add_argument("--test-predictions", type=Path, required=True)
    p.add_argument("--threshold-json", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--matching-iou", type=float, default=None)
    p.add_argument(
        "--ap-summary-json",
        type=Path,
        help="Optional AP50/AP75/mAP50-95 exported by Ultralytics held-out val API.",
    )
    a = p.parse_args()
    df = pd.read_csv(a.test_predictions)
    if "split" in df and set(df.split) != {"test"}:
        raise ValueError("Input must contain held-out test rows only (split=test).")
    locked = json.loads(a.threshold_json.read_text(encoding="utf-8"))
    thr = float(locked["confidence_threshold"])
    miou = float(a.matching_iou or locked["matching_iou"])
    per = []
    matched = []
    for _, r in df.iterrows():
        pb, sc, gb = boxes(r.pred_boxes), scores(r.pred_scores), boxes(r.gt_boxes)
        keep = sc >= thr
        pb = pb[keep]
        sc = sc[keep]
        pairs, fp, fn = match(pb, gb, miou)
        for pi, gi, iou in pairs:
            matched.append(
                {
                    "image_path": r.image_path,
                    "patient_id": r.get("patient_id", ""),
                    "pred_index": pi,
                    "gt_index": gi,
                    "confidence": sc[pi],
                    "iou": iou,
                }
            )
        per.append(
            {
                "image_path": r.image_path,
                "patient_id": r.get("patient_id", ""),
                "tp": len(pairs),
                "fp": fp,
                "fn": fn,
                "gt_count": len(gb),
                "pred_count": len(pb),
                "image_success": int(fn == 0),
            }
        )
    out = pd.DataFrame(per)
    m = pd.DataFrame(
        matched,
        columns=[
            "image_path",
            "patient_id",
            "pred_index",
            "gt_index",
            "confidence",
            "iou",
        ],
    )
    tp = int(out.tp.sum())
    fp = int(out.fp.sum())
    fn = int(out.fn.sum())
    gt = int(out.gt_count.sum())
    if tp + fn != gt:
        raise AssertionError("Invariant failed: TP+FN != total GT boxes")
    ious = m.iou.to_numpy() if len(m) else np.array([])
    q = np.percentile(ious, [25, 75]) if len(ious) else [None, None]
    summary = {
        "threshold": thr,
        "threshold_source": "locked validation JSON",
        "matching_iou": miou,
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
    if a.ap_summary_json:
        summary["ultralytics_heldout_val_api_ap"] = json.loads(
            a.ap_summary_json.read_text(encoding="utf-8")
        )
        summary["ap_source"] = (
            "Ultralytics held-out val API; separate from locked-threshold lesion matching"
        )
    a.output_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.output_dir / "per_image.csv", index=False)
    m.to_csv(a.output_dir / "matched_lesions.csv", index=False)
    (a.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
