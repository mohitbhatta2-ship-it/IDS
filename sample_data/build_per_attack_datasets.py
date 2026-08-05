#!/usr/bin/env python3
"""
Build one ready-to-upload CSV per traffic class.

Feed any one of these to the batch page and the prediction should come back
dominated by that single class -- a per-attack demo, rather than the mixed
sample in test_sample_with_labels.csv.

Where the rows come from
------------------------
Held-out test rows first. Those are from ``test_selected.parquet``, which no
model ever trained on, so a correct prediction on them means something.

Six classes do not have 500 held-out rows to give (SQL Injection has 17). For
those the file is topped up from ``balanced_train_selected.parquet``. Rows the
model trained on are easier for it, so a file with a top-up flatters the model
slightly; the exact split is written into MANIFEST.csv and the README so the
number is never quoted without that context. Nothing is duplicated or invented
-- a file is short rather than padded, which is why SQL Injection has 87 rows.

CTGAN synthetic rows were deliberately not used. They were generated to augment
training, so scoring the model on them would be circular.

Usage:
    python sample_data/build_per_attack_datasets.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "webapp_data" / "Processed_Data"
OUT = Path(__file__).resolve().parent / "per_attack"
# Same rows, Label column stripped -- what traffic of unknown origin looks like.
OUT_NO_LABELS = OUT / "no_labels"

TARGET_ROWS = 500
SEED = 42

# Written by notebook 01; maps the encoded Label back to the class name.
LABEL_MAP = dict(
    pd.read_csv(DATA / "label_mapping.csv")[["Encoded", "Class"]].values
)


def slug(name: str) -> str:
    """A filename that survives every OS: lowercase, no spaces or punctuation."""
    s = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return s


def main() -> int:
    if not (DATA / "test_selected.parquet").is_file():
        print(f"Missing {DATA/'test_selected.parquet'} -- run `git lfs pull`.")
        return 1

    test = pd.read_parquet(DATA / "test_selected.parquet")
    train = pd.read_parquet(DATA / "balanced_train_selected.parquet")

    features = [c for c in test.columns if c != "Label"]
    OUT.mkdir(parents=True, exist_ok=True)
    OUT_NO_LABELS.mkdir(parents=True, exist_ok=True)

    rows = []
    for code, name in sorted(LABEL_MAP.items()):
        pool_test = test[test["Label"] == code]
        take_test = min(TARGET_ROWS, len(pool_test))
        picked = [pool_test.sample(take_test, random_state=SEED)]

        # Only reach into the training set when the held-out set cannot fill the
        # file on its own.
        short = TARGET_ROWS - take_test
        take_train = 0
        if short > 0:
            pool_train = train[train["Label"] == code]
            take_train = min(short, len(pool_train))
            if take_train:
                picked.append(pool_train.sample(take_train, random_state=SEED))

        frame = pd.concat(picked, ignore_index=True)
        # Shuffle so the topped-up training rows are not all at the bottom.
        frame = frame.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

        # Spell the label out. The app accepts either form, but a human opening
        # the CSV should not have to decode a 13.
        frame["Label"] = name
        frame = frame[features + ["Label"]]

        path = OUT / f"{slug(name)}.csv"
        frame.to_csv(path, index=False)

        # The identical rows without the ground-truth column. Predictions are
        # unaffected -- Label is never a model input -- so this pair also serves
        # as the check that no label leaks into the features.
        path_nl = OUT_NO_LABELS / f"{slug(name)}.csv"
        frame.drop(columns=["Label"]).to_csv(path_nl, index=False)

        rows.append(
            {
                "file": path.name,
                "class": name,
                "rows": len(frame),
                "from_test": take_test,
                "from_train": take_train,
                "test_pool": len(pool_test),
                "kb": round(path.stat().st_size / 1024, 1),
                "kb_no_labels": round(path_nl.stat().st_size / 1024, 1),
            }
        )
        print(f"{path.name:<32} {len(frame):>4} rows  "
              f"({take_test} held-out + {take_train} train)")

    manifest = pd.DataFrame(rows)
    manifest.to_csv(OUT / "MANIFEST.csv", index=False)
    print(f"\n{len(rows)} labelled files written to {OUT}")
    print(f"{len(rows)} unlabelled copies written to {OUT_NO_LABELS}")
    print(f"total rows: {manifest['rows'].sum():,} (per set)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
