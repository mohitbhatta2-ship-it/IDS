import pandas as pd
import pickle

with open(r"M:\IDS\c_filesnew\Processed_Data\selected_features.pkl", "rb") as f:
    FEATURE_ORDER = pickle.load(f)


def preprocess_live(feature_dict):

    df = pd.DataFrame([feature_dict])

    # Ensure every expected feature exists
    for col in FEATURE_ORDER:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns exactly as used during training
    df = df[FEATURE_ORDER]

    return df