import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def collisions(df, key):
    grouped = df.groupby(key)["split"].agg(lambda x: sorted(set(x)))
    return {str(k): v for k, v in grouped.items() if len(v) > 1}


def main():
    p = argparse.ArgumentParser(
        description="Audit split leakage by patient, SHA256, and filename."
    )
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    df = pd.read_csv(a.manifest, dtype={"patient_id": str})
    required = {"image_path", "patient_id", "split"}
    if required - set(df):
        raise ValueError(f"Missing columns: {sorted(required-set(df))}")
    missing_files = [x for x in df.image_path if not Path(x).is_file()]
    df["filename"] = df.image_path.map(lambda x: Path(x).name.lower())
    df["sha256"] = df.image_path.map(sha256)
    report = {
        "patient_id_cross_split": collisions(df, "patient_id"),
        "sha256_cross_split": collisions(df, "sha256"),
        "filename_cross_split": collisions(df, "filename"),
        "missing_files": missing_files,
    }
    report["leakage_found"] = any(report[k] for k in report if k != "leakage_found")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(1 if report["leakage_found"] else 0)


if __name__ == "__main__":
    main()
