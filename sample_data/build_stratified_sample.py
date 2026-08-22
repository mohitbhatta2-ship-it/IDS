#!/usr/bin/env python3
"""
Build a stratified sample from test_selected.parquet.

Unlike test_sample_with_labels.csv (which forces ~300 rows per class),
this sample preserves the natural class proportions of the held-out test set.
That means accuracy on it matches the model's reported ~98% figure because
Benign dominates just as it does in real traffic.

Usage:
    python sample_data/build_stratified_sample.py
    python sample_data/build_stratified_sample.py --rows 5000
    python sample_data/build_stratified_sample.py --rows 5000 --no-labels

Output files (written next to this script):
    test_stratified_<N>_with_labels.csv
    test_stratified_<N>_no_labels.csv  (if --no-labels is passed, or always)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "webapp_data" / "Processed_Data"
OUT = Path(__file__).resolve().parent

DEFAULT_ROWS = 5_000
SEED = 42


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS,
                        help=f"Total rows in the sample (default: {DEFAULT_ROWS})")
    parser.add_argument("--no-labels", action="store_true",
                        help="Also write a label-free copy (always written alongside the labelled one)")
    args = parser.parse_args()

    parquet_path = DATA / "test_selected.parquet"
    if not parquet_path.is_file():
        print(f"Missing {parquet_path} -- run `git lfs pull`.")
        return 1

    label_map_path = DATA / "label_mapping.csv"
    if not label_map_path.is_file():
        print(f"Missing {label_map_path} -- run `git lfs pull`.")
        return 1

    label_map = dict(pd.read_csv(label_map_path)[["Encoded", "Class"]].values)

    test = pd.read_parquet(parquet_path)
    total = len(test)

    if args.rows > total:
        print(f"Requested {args.rows:,} rows but test set only has {total:,}. "
              f"Using {total:,}.")
        args.rows = total

    # Stratified sample: each class contributes proportionally to its share in the test set.
    # Uses a plain loop rather than groupby().apply() to avoid pandas 2.2+ behaviour where
    # the grouping column is excluded from the group passed to the lambda.
    parts = []
    for label_code, group in test.groupby("Label"):
        n_take = max(1, round(len(group) / total * args.rows))
        parts.append(group.sample(n=n_take, random_state=SEED))
    sample = pd.concat(parts, ignore_index=True)

    # Shuffle so classes aren't grouped together.
    sample = sample.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    # Decode integer labels to human-readable class names.
    sample["Label"] = sample["Label"].map(label_map)

    n = len(sample)
    stem = f"test_stratified_{n}"

    labelled_path = OUT / f"{stem}_with_labels.csv"
    no_labels_path = OUT / f"{stem}_no_labels.csv"

    sample.to_csv(labelled_path, index=False)
    sample.drop(columns=["Label"]).to_csv(no_labels_path, index=False)

    # Print a summary of what was written.
    dist = sample["Label"].value_counts().sort_values(ascending=False)
    print(f"\nStratified sample — {n:,} rows from {parquet_path.name}")
    print(f"  {labelled_path.name}")
    print(f"  {no_labels_path.name}\n")
    print(f"{'Class':<30} {'Rows':>6}  {'Share':>6}")
    print("-" * 46)
    for cls, count in dist.items():
        print(f"{cls:<30} {count:>6}  {count/n*100:>5.1f}%")
    print("-" * 46)
    print(f"{'Total':<30} {n:>6}  100.0%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
