import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

MODES = ("raw", "bf", "he", "bf_he")


def read_image(path):
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


def write_image(path, image):
    ok, data = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise IOError(f"Cannot encode {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data.tofile(str(path))


def he(image):
    ycc = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ycc[:, :, 0] = cv2.equalizeHist(ycc[:, :, 0])
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


def main():
    p = argparse.ArgumentParser(
        description="Build deterministic raw/BF/HE/BF+HE ablation data."
    )
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--mode", choices=MODES, required=True)
    p.add_argument("--d", type=int, default=7)
    p.add_argument("--sigma-color", type=float, default=60)
    p.add_argument("--sigma-space", type=float, default=60)
    p.add_argument(
        "--alpha",
        type=float,
        default=0.40,
        help="Fixed HE weight in I_mix=alpha*I_he+(1-alpha)*I_bf.",
    )
    a = p.parse_args()
    if not 0 <= a.alpha <= 1:
        raise ValueError("alpha must be in [0,1]")
    df = pd.read_csv(a.manifest)
    rows = []
    for i, row in df.iterrows():
        src, label = Path(row.image_path), Path(row.label_path)
        split = row.get("split", "all")
        image = read_image(src)
        if image is None:
            raise IOError(f"Cannot read {src}")
        bf = cv2.bilateralFilter(image, a.d, a.sigma_color, a.sigma_space)
        out = {
            "raw": image,
            "bf": bf,
            "he": he(image),
            "bf_he": cv2.addWeighted(he(image), a.alpha, bf, 1 - a.alpha, 0),
        }[a.mode]
        dst_img = (
            a.output_root / a.mode / split / "images" / f"{i:06d}{src.suffix.lower()}"
        )
        dst_lbl = a.output_root / a.mode / split / "labels" / f"{i:06d}.txt"
        write_image(dst_img, out)
        dst_lbl.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(label, dst_lbl)
        rows.append(
            {
                **row.to_dict(),
                "output_image_path": str(dst_img),
                "output_label_path": str(dst_lbl),
                "mode": a.mode,
                "d": a.d,
                "sigmaColor": a.sigma_color,
                "sigmaSpace": a.sigma_space,
                "alpha": a.alpha,
            }
        )
    out_manifest = a.output_root / a.mode / "preprocessing_manifest.csv"
    pd.DataFrame(rows).to_csv(out_manifest, index=False)
    (a.output_root / a.mode / "preprocessing_parameters.json").write_text(
        json.dumps(vars(a), default=str, indent=2), encoding="utf-8"
    )
    print(out_manifest)


if __name__ == "__main__":
    main()
