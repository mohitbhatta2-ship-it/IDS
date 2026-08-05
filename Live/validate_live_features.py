"""
Feature parity check: are the features we compute from live packets in the same
range as the ones the models were trained on?

This matters more than it looks. The models learned from features produced by
CICFlowMeter; feature_calculator.py is a hand-written reimplementation of 30 of
them. If any definition has drifted, the model receives out-of-distribution input
and its predictions are meaningless *even when nothing raises an error*. That
failure is silent, which is exactly why it needs an explicit check.

Usage
-----
    python validate_live_features.py                     # uses live_flows.csv
    python validate_live_features.py --live other.csv

Run sniff_test.py first to produce the live CSV.
"""

import argparse
import sys

import pandas as pd

import config

# A live value outside the training range is not automatically wrong -- real
# traffic can be more extreme than the dataset -- but a large fraction outside
# means the feature is probably computed differently, not just unusual.
SUSPICIOUS_SHARE = 0.20


def load_training():
    parquet = config.PROCESSED_DATA_PATH / "balanced_train_selected.parquet"
    if not parquet.is_file():
        raise FileNotFoundError(
            f"Training data not found: {parquet}\n"
            "It is stored in Git LFS -- run `git lfs install && git lfs pull`."
        )
    train = pd.read_parquet(parquet)
    return train.drop(columns=["Label"], errors="ignore")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare live features against training features.")
    parser.add_argument("--live", default=str(config.CSV_FILE), help="CSV produced by sniff_test.py")
    args = parser.parse_args(argv)

    train = load_training()

    try:
        live = pd.read_csv(args.live)
    except FileNotFoundError:
        print(f"No live capture found at {args.live}.", file=sys.stderr)
        print("Run sniff_test.py first to generate it.", file=sys.stderr)
        return 1

    print(f"Training : {train.shape[0]:,} rows x {train.shape[1]} features")
    print(f"Live     : {live.shape[0]:,} rows x {live.shape[1]} columns")
    print()

    missing = [c for c in train.columns if c not in live.columns]
    if missing:
        print(f"MISSING from the live capture ({len(missing)}):")
        for c in missing:
            print(f"  - {c}")
        print()

    shared = [c for c in train.columns if c in live.columns]
    if not shared:
        print("No overlapping feature columns -- nothing to compare.", file=sys.stderr)
        return 1

    rows = []
    for col in shared:
        t = pd.to_numeric(train[col], errors="coerce").dropna()
        l = pd.to_numeric(live[col], errors="coerce").dropna()
        if l.empty:
            continue

        lo, hi = t.min(), t.max()
        outside = ((l < lo) | (l > hi)).mean()

        rows.append(
            {
                "Feature": col,
                "Train Min": lo,
                "Train Max": hi,
                "Live Min": l.min(),
                "Live Max": l.max(),
                "Train Median": t.median(),
                "Live Median": l.median(),
                "Out of range": outside,
            }
        )

    report = pd.DataFrame(rows).sort_values("Out of range", ascending=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.float_format", lambda v: f"{v:,.3f}")

    print("Feature comparison (worst offenders first)\n")
    print(report.to_string(index=False))

    flagged = report[report["Out of range"] > SUSPICIOUS_SHARE]
    print()
    if len(flagged):
        print(f"{len(flagged)} feature(s) have more than {SUSPICIOUS_SHARE:.0%} of live values "
              "outside the training range:")
        for _, r in flagged.iterrows():
            print(f"  - {r['Feature']:<22} {r['Out of range']:.0%} outside")
        print(
            "\nCheck these against the CICFlowMeter definitions before trusting any\n"
            "live prediction. Known candidates: 'Fwd Header Len' (transport vs IP\n"
            "header), 'Fwd Seg Size Min' (header vs payload), and the flow timeout,\n"
            "which shifts Flow Duration and every IAT feature."
        )
    else:
        print("No feature has a suspicious share of out-of-range values.")

    out = config.DATA_ROOT / "Results" / "Tables" / "live_feature_parity.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out, index=False)
    print(f"\nSaved -> {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
