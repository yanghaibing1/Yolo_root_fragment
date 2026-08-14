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
        description="Run Ultralytics inference and export real boxes/scores for evaluation."
    )
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--split", choices=["val", "test"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--model-family",
        choices=["yolov8", "yolov10", "yolo11", "yolo12"],
        required=True,
    )
    p.add_argument(
        "--yolov10-mode",
        choices=["end2end", "nms-free"],
        default="end2end",
        help="Recorded explicitly; no traditional NMS is forced.",
    )
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="cpu")
    p.add_argument("--min-confidence", type=float, default=0.001)
    a = p.parse_args()
    from ultralytics import YOLO

    df = pd.read_csv(a.manifest)
    df = df[df.split == a.split].reset_index(drop=True)
    model = YOLO(str(a.weights))
    rows = []
    # Keep the official YOLOv10 end-to-end/NMS-free behavior; do not override NMS IoU.
    results = model.predict(
        source=df.image_path.tolist(),
        imgsz=a.imgsz,
        conf=a.min_confidence,
        device=a.device,
        verbose=False,
    )
    for (_, r), result in zip(df.iterrows(), results):
        h, w = result.orig_shape
        b = [] if result.boxes is None else result.boxes.xyxy.cpu().numpy().tolist()
        s = [] if result.boxes is None else result.boxes.conf.cpu().numpy().tolist()
        rows.append(
            {
                "image_path": r.image_path,
                "patient_id": r.patient_id,
                "split": a.split,
                "gt_boxes": json.dumps(yolo_gt(r.label_path, w, h)),
                "pred_boxes": json.dumps(b),
                "pred_scores": json.dumps(s),
                "inference_mode": (
                    a.yolov10_mode
                    if a.model_family == "yolov10"
                    else "official_ultralytics_default"
                ),
                "nms_iou": "official_default_not_matching_iou",
            }
        )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(a.output, index=False)
    print(a.output)


if __name__ == "__main__":
    main()
