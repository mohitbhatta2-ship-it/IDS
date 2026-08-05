#!/usr/bin/env python3
"""
Check that each per-attack CSV actually predicts as its own class.

Runs every file in per_attack/ through the same ml.predict_batch() the batch
page calls, and writes VERIFICATION.csv next to them. A generated dataset is
only useful if it has been scored -- a file that quietly predicts something
else is worse than no file at all.

Usage:
    python sample_data/verify_per_attack_datasets.py            # default model
    python sample_data/verify_per_attack_datasets.py mlp        # a named model
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "webapp_django"))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ids_web.settings")

warnings.filterwarnings("ignore")

import django  # noqa: E402

django.setup()

import pandas as pd  # noqa: E402

from predictor import ml  # noqa: E402

OUT = Path(__file__).resolve().parent / "per_attack"


def main(model_key: str = ml.DEFAULT_MODEL) -> int:
    files = sorted(p for p in OUT.glob("*.csv") if p.name != "MANIFEST.csv"
                   and p.name != "VERIFICATION.csv")
    if not files:
        print(f"No CSVs in {OUT} -- run build_per_attack_datasets.py first.")
        return 1

    rows = []
    for path in files:
        df = pd.read_csv(path)
        truth = df["Label"].iloc[0]
        result = ml.predict_batch(df.copy(), model_key)
        frame = result["frame"]
        top = max(result["distribution"], key=lambda d: d["count"])
        correct = float((frame["Predicted Class"] == truth).mean())

        rows.append(
            {
                "file": path.name,
                "true_class": truth,
                "rows": result["rows"],
                "top_prediction": top["label"],
                "top_share_pct": round(top["share_pct"], 1),
                "correct_pct": round(100 * correct, 1),
                "mean_confidence": round(float(frame["Confidence"].mean()), 3),
                "distinct_predictions": len(result["distribution"]),
                "reads_as_own_class": top["label"] == truth,
                # What it is confused with, most common first.
                "confusion": "; ".join(
                    f"{d['label']} {d['share_pct']:.1f}%"
                    for d in sorted(result["distribution"], key=lambda x: -x["count"])
                    if d["label"] != truth
                ) or "-",
            }
        )

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "VERIFICATION.csv", index=False)

    # The unlabelled twins must predict identically. Label is never a model
    # input, so any difference here would mean ground truth is leaking into the
    # features -- which would make every score on this page meaningless.
    identical = _check_no_labels_match(files, model_key)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 34)
    print(f"Model: {ml.MODEL_REGISTRY[model_key]['name']}\n")
    print(
        table[
            ["file", "rows", "top_prediction", "top_share_pct",
             "correct_pct", "mean_confidence"]
        ].to_string(index=False)
    )

    ok = int(table["reads_as_own_class"].sum())
    print(f"\n{ok}/{len(table)} files predict as their own class.")
    failed = table.loc[~table["reads_as_own_class"], "file"].tolist()
    if failed:
        print("Does not: " + ", ".join(failed))

    print(
        f"no_labels/ twins predict identically: {identical}/{len(files)} files"
        + ("  (no label leakage)" if identical == len(files) else "  ** MISMATCH **")
    )
    return 0 if identical == len(files) else 2


def _check_no_labels_match(files: list[Path], model_key: str) -> int:
    """Count files whose unlabelled copy yields byte-identical predictions."""
    same = 0
    for path in files:
        twin = OUT / "no_labels" / path.name
        if not twin.is_file():
            continue
        a = ml.predict_batch(pd.read_csv(path), model_key)["frame"]
        b = ml.predict_batch(pd.read_csv(twin), model_key)["frame"]
        if (
            a["Predicted Class"].equals(b["Predicted Class"])
            and a["Confidence"].equals(b["Confidence"])
        ):
            same += 1
    return same


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else ml.DEFAULT_MODEL))
