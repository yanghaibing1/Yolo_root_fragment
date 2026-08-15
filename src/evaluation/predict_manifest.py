import argparse
import json
from pathlib import Path

import pandas as pd


def yolo_gt(path, w, h):
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        _, x, y, bw, bh, *_ = map(float, line.split())
        out.append(
            [(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h]
        )
    return out


def main():
    p = argparse.ArgumentParser(
        description="Run Ultralytics inference and export boxes/scores with fold identity."
    )
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--split", choices=["inner_val", "outer_test"], required=True)
    p.add_argument("--outer-fold", type=int, choices=range(5), required=True)
    p.add_argument("--inner-fold", type=int, choices=range(4))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--model-family",
        choices=["yolov8", "yolov10", "yolo11", "yolo12"],
        required=True,
    )
    p.add_argument("--model-id", help="Stable model identifier; defaults to model-family.")
    p.add_argument("--yolov10-mode", choices=["end2end", "nms-free"], default="end2end")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--capture-floor",
        type=float,
        default=0.0,
        help="Use 0 when supported; otherwise pass the smallest accepted value. Threshold 0 then means all captured predictions.",
    )
    a = p.parse_args()
    if a.split == "inner_val" and a.inner_fold is None:
        p.error("--inner-fold is required for inner_val.")
    if a.split == "outer_test" and a.inner_fold is not None:
        p.error("--inner-fold must not be supplied for outer_test.")
    if not 0 <= a.capture_floor <= 1:
        p.error("--capture-floor must be within [0, 1].")
    from ultralytics import YOLO

    df = pd.read_csv(a.manifest, dtype={"patient_id": str})
    if set(df.outer_fold.astype(int)) != {a.outer_fold}:
        raise ValueError("Manifest outer_fold does not match --outer-fold.")
    if a.split == "outer_test":
        df = df[df.split == "outer_test"].reset_index(drop=True)
    else:
        df = df[
            (df.split == "outer_train") & (df.inner_fold.astype("Int64") == a.inner_fold)
        ].reset_index(drop=True)
    model = YOLO(str(a.weights))
    model_id = a.model_id or a.model_family
    results = model.predict(
        source=df.image_path.tolist(),
        imgsz=a.imgsz,
        conf=a.capture_floor,
        device=a.device,
        verbose=False,
    )
    rows = []
    for (_, row), result in zip(df.iterrows(), results):
        h, w = result.orig_shape
        b = [] if result.boxes is None else result.boxes.xyxy.cpu().numpy().tolist()
        s = [] if result.boxes is None else result.boxes.conf.cpu().numpy().tolist()
        rows.append(
            {
                "image_path": row.image_path,
                "patient_id": row.patient_id,
                "model": model_id,
                "outer_fold": a.outer_fold,
                "inner_fold": a.inner_fold if a.split == "inner_val" else "",
                "split": a.split,
                "gt_boxes": json.dumps(yolo_gt(row.label_path, w, h)),
                "pred_boxes": json.dumps(b),
                "pred_scores": json.dumps(s),
                "capture_floor": a.capture_floor,
                "threshold_zero_definition": "all predictions captured at or above capture_floor",
                "inference_mode": (
                    a.yolov10_mode if a.model_family == "yolov10" else "official_ultralytics_default"
                ),
                "nms_iou": "official_default_not_matching_iou",
            }
        )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(a.output, index=False)
    print(f"{a.output} (capture_floor={a.capture_floor}; threshold 0 uses all captured predictions)")


if __name__ == "__main__":
    main()
