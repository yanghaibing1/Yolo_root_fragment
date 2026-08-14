import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import boxes, match, safe, scores


def main():
    p = argparse.ArgumentParser(
        description="Select confidence threshold using validation predictions only."
    )
    p.add_argument("--validation-predictions", type=Path, required=True)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--curves-csv", type=Path, required=True)
    p.add_argument("--matching-iou", type=float, default=0.5)
    p.add_argument("--steps", type=int, default=1001)
    a = p.parse_args()
    df = pd.read_csv(a.validation_predictions)
    if "split" in df and set(df.split) != {"val"}:
        raise ValueError("Input must contain validation rows only (split=val).")
    rows = []
    for threshold in np.linspace(0, 1, a.steps):
        tp = fp = fn = 0
        for _, r in df.iterrows():
            pb, sc, gb = boxes(r.pred_boxes), scores(r.pred_scores), boxes(r.gt_boxes)
            pairs, fpi, fni = match(pb[sc >= threshold], gb, a.matching_iou)
            tp += len(pairs)
            fp += fpi
            fn += fni
        precision = safe(tp, tp + fp)
        sensitivity = safe(tp, tp + fn)
        f1 = safe(2 * precision * sensitivity, precision + sensitivity)
        rows.append(
            {
                "threshold": threshold,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "sensitivity": sensitivity,
                "f1": f1,
                "fp_per_image": safe(fp, len(df)),
            }
        )
    curves = pd.DataFrame(rows)
    best = (
        curves.sort_values(["f1", "threshold"], ascending=[False, False])
        .iloc[0]
        .to_dict()
    )
    payload = {
        "confidence_threshold": best["threshold"],
        "selection_metric": "lesion-level F1",
        "selection_split": "validation",
        "matching_iou": a.matching_iou,
        "n_validation_images": len(df),
    }
    a.output_json.parent.mkdir(parents=True, exist_ok=True)
    a.curves_csv.parent.mkdir(parents=True, exist_ok=True)
    a.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    curves.to_csv(a.curves_csv, index=False)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
