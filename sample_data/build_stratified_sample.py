"""
Build a stratified sample from test_selected.parquet.

The sample preserves the natural class proportions of the held-out test set.

Usage:
    python sample_data/build_stratified_sample.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "webapp_data" / "Processed_Data"
OUT = Path(__file__).resolve().parent

DEFAULT_ROWS = 5_000
SEED = 42


def main() -> int:

    parquet_path = DATA / "test_selected.parquet"

    if not parquet_path.is_file():
        print(f"Missing {parquet_path}.")
        return 1

    label_map_path = DATA / "label_mapping.csv"

    if not label_map_path.is_file():
        print(f"Missing {label_map_path}")
        return 1

    label_map = dict(
        pd.read_csv(label_map_path)[["Encoded", "Class"]].values
    )

    test = pd.read_parquet(parquet_path)
    total = len(test)

    # Use the entire test set if it contains fewer than DEFAULT_ROWS.
    rows = min(DEFAULT_ROWS, total)

    if rows < DEFAULT_ROWS:
        print(
            f"Requested {DEFAULT_ROWS:,} rows but test set only has "
            f"{total:,}. Using {total:,}."
        )

    # Stratified sample: each class contributes proportionally
    # to its share in the test set.
    parts = []

    for label_code, group in test.groupby("Label"):

        n_take = max(
            1,
            round(len(group) / total * rows)
        )

        parts.append(
            group.sample(
                n=n_take,
                random_state=SEED
            )
        )

    sample = pd.concat(
        parts,
        ignore_index=True
    )

    # Shuffle so classes are not grouped together.
    sample = sample.sample(
        frac=1.0,
        random_state=SEED
    ).reset_index(drop=True)

    # Convert encoded labels to human-readable class names.
    sample["Label"] = sample["Label"].map(label_map)

    n = len(sample)

    stem = f"test_stratified_{n}"

    labelled_path = OUT / f"{stem}_with_labels.csv"
    no_labels_path = OUT / f"{stem}_no_labels.csv"

    # Save labelled version.
    sample.to_csv(
        labelled_path,
        index=False
    )

    # Save version without Label.
    sample.drop(
        columns=["Label"]
    ).to_csv(
        no_labels_path,
        index=False
    )

    # Print class distribution.
    dist = (
        sample["Label"]
        .value_counts()
        .sort_values(ascending=False)
    )

    print(
        f"\nStratified sample — {n:,} rows "
        f"from {parquet_path.name}"
    )

    print(f"  {labelled_path.name}")
    print(f"  {no_labels_path.name}\n")

    print(
        f"{'Class':<30} {'Rows':>6}  {'Share':>6}"
    )

    print("-" * 46)

    for cls, count in dist.items():

        print(
            f"{cls:<30} "
            f"{count:>6}  "
            f"{count / n * 100:>5.1f}%"
        )

    print("-" * 46)

    print(
        f"{'Total':<30} "
        f"{n:>6}  "
        f"100.0%"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())