import time

from feature_calculator import calculate_features
from csv_writer import append_flow
from preprocess_live import preprocess_live
from preprocess_live import preprocess_live
from live_predictor import predict
import json
from datetime import datetime
from pathlib import Path




FLOW_TIMEOUT = 30      # seconds


def cleanup_flows(flows):

    now = time.time()

    expired = []

    for key, flow in flows.items():

        idle = now - flow.last_seen

        if idle > FLOW_TIMEOUT:

            features = calculate_features(flow)

            processed = preprocess_live(features)

            prediction, confidence = predict(processed)

            print("=" * 70)
            print(f"Flow : {flow.src_ip}:{flow.src_port} -> {flow.dst_ip}:{flow.dst_port}")
            print(f"Protocol : {flow.protocol}")
            print(f"Duration : {features['Flow Duration']}")
            print(f"Packets : {flow.total_packets}")
            print(f"Prediction : {prediction}")
            print(f"Confidence : {confidence:.2%}")
            print("=" * 70)

            append_flow(processed.iloc[0].to_dict())

            

            expired.append(key)

    for key in expired:

        del flows[key]