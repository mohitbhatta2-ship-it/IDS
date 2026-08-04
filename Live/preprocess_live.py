import pickle

import pandas as pd

import config

with open(config.FEATURES_FILE, "rb") as f:
    FEATURE_ORDER = pickle.load(f)


def preprocess_live(feature_dict):
    """Turn a feature dict into a one-row frame with the exact training columns."""

    df = pd.DataFrame([feature_dict])

    # Ensure every expected feature exists
    for col in FEATURE_ORDER:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns exactly as used during training
    df = df[FEATURE_ORDER]

    return df
