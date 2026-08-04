"""Classify the first row of a captured CSV. A quick smoke test of the model path."""

import sys

import pandas as pd

import config
from live_predictor import predict
from preprocess_live import preprocess_live

path = sys.argv[1] if len(sys.argv) > 1 else str(config.CSV_FILE)

try:
    live = pd.read_csv(path)
except FileNotFoundError:
    raise SystemExit(f"No capture at {path}. Run sniff_test.py first.")

if live.empty:
    raise SystemExit(f"{path} has no rows.")

row = preprocess_live(live.iloc[0].to_dict())

print(config.describe())
print()
for label, prob in predict(row):
    print(f"  {label:<28} {prob:.2%}")
