import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest


def main():
    p = argparse.ArgumentParser(
        description="Exact McNemar test on paired image-level success/failure."
    )
    p.add_argument("--model-a", type=Path, required=True)
    p.add_argument("--model-b", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--success-column",
        default="image_success",
        help="Default success means no missed GT lesion (FN=0) at each model's locked threshold.",
    )
    a = p.parse_args()
    x = pd.read_csv(a.model_a)[["image_path", a.success_column]].rename(
        columns={a.success_column: "a"}
    )
    y = pd.read_csv(a.model_b)[["image_path", a.success_column]].rename(
        columns={a.success_column: "b"}
    )
    d = x.merge(y, on="image_path", validate="one_to_one")
    b = int(((d.a == 1) & (d.b == 0)).sum())
    c = int(((d.a == 0) & (d.b == 1)).sum())
    pv = float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0
    out = {
        "definition": f"success={a.success_column}; default is FN=0 at locked threshold",
        "n_pairs": len(d),
        "a_success_b_failure": b,
        "a_failure_b_success": c,
        "exact_two_sided_p": pv,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
