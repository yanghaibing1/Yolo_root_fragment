import argparse
import json
from pathlib import Path


def parser_for(model_name):
    p = argparse.ArgumentParser(description=f"Train {model_name} with Ultralytics.")
    p.add_argument("--model", default=model_name)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--name", default=Path(model_name).stem)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--optimizer", default="AdamW")
    p.add_argument("--lr0", type=float, default=0.008)
    p.add_argument("--lrf", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=float, default=5)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument("--cos-lr", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--degrees", type=float, default=0)
    p.add_argument("--translate", type=float, default=0.1)
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--shear", type=float, default=0)
    p.add_argument("--perspective", type=float, default=0)
    p.add_argument("--fliplr", type=float, default=0.5)
    p.add_argument("--flipud", type=float, default=0)
    p.add_argument("--hsv-h", type=float, default=0.015)
    p.add_argument("--hsv-s", type=float, default=0.7)
    p.add_argument("--hsv-v", type=float, default=0.4)
    p.add_argument("--mosaic", type=float, default=1)
    p.add_argument("--mixup", type=float, default=0)
    return p


def train(model_name):
    a = parser_for(model_name).parse_args()
    from ultralytics import YOLO

    kwargs = vars(a).copy()
    model = kwargs.pop("model")
    data = str(kwargs.pop("data"))
    project = str(kwargs["project"])
    run_dir = Path(project) / kwargs["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "invocation_args.json").write_text(
        json.dumps(
            {**vars(a), "data": str(a.data), "project": str(a.project)}, indent=2
        ),
        encoding="utf-8",
    )
    YOLO(model).train(data=data, **kwargs)
