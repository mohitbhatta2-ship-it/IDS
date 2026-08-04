import pandas as pd
from preprocess_live import preprocess_live
from live_predictor import predict

live = pd.read_csv("live_flows.csv")

row = preprocess_live(live.iloc[0].to_dict())

print(predict(row))
