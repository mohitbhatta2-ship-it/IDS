import joblib
import pandas as pd

# Load model
MODEL_PATH = r"M:\IDS\c_filesnew\Results\Models\RandomForest_Tuned.pkl"
model = joblib.load(MODEL_PATH)

# Load label mapping
mapping = pd.read_csv(
    r"M:\IDS\c_filesnew\Processed_Data\label_mapping.csv"
)

# Build dictionary: {0: "Benign", 1: "Bot", ...}
label_map = dict(zip(mapping["Encoded"], mapping["Class"]))

print("Loaded label map:")
print(label_map)

def predict(df):
    probabilities = model.predict_proba(df)[0]

    top3 = sorted(
        zip(model.classes_, probabilities),
        key=lambda x: x[1],
        reverse=True
    )[:3]

    results = [
        (label_map[int(cls)], float(prob))
        for cls, prob in top3
    ]

    return results



