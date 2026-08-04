import pandas as pd

train = pd.read_csv("train_selected.csv")
live = pd.read_csv("live_flows.csv")

print("Training shape :", train.shape)
print("Live shape     :", live.shape)

print("\nFeature comparison\n")

for col in train.columns:

    if col not in live.columns:
        print(f"{col:<25} Missing")
        continue

    print(
        f"{col:<25}"
        f"Train Min={train[col].min():>12.3f}"
        f" Train Max={train[col].max():>12.3f}"
        f" Live Min={live[col].min():>12.3f}"
        f" Live Max={live[col].max():>12.3f}"
    )