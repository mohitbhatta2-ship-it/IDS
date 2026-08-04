"""
Expires finished flows, classifies them and writes them out.

Two things were wrong here before:

  * ``prediction, confidence = predict(processed)`` unpacked a three-element
    list into two names, so the very first expired flow raised
    ``ValueError: too many values to unpack``. The live pipeline could never
    have produced a single prediction.
  * expiry compared ``time.time()`` against packet timestamps. That is roughly
    right while sniffing live and completely wrong when replaying a stored
    capture, where it expires every flow immediately. Expiry now runs on the
    capture clock, supplied by the caller.
"""

import config
from csv_writer import append_flow
from feature_calculator import calculate_features
from live_predictor import predict
from preprocess_live import preprocess_live


def classify(flow):
    """Turn one finished flow into (features, processed frame, ranked predictions)."""
    features = calculate_features(flow)
    processed = preprocess_live(features)
    ranked = predict(processed)
    return features, processed, ranked


def report(flow, features, ranked):
    label, confidence = ranked[0]

    print("=" * 70)
    print(f"Flow       : {flow.src_ip}:{flow.src_port} -> {flow.dst_ip}:{flow.dst_port}")
    print(f"Protocol   : {flow.protocol}")
    print(f"Duration   : {features['Flow Duration']:.0f} us")
    print(f"Packets    : {flow.total_packets}")
    print(f"Prediction : {label}")
    print(f"Confidence : {confidence:.2%}")

    if len(ranked) > 1:
        alternatives = ", ".join(f"{name} {prob:.1%}" for name, prob in ranked[1:])
        print(f"Also       : {alternatives}")

    print("=" * 70)


def expire_flow(flow, quiet=False):
    """Classify a single flow and append it to the output CSV."""
    features, processed, ranked = classify(flow)

    if not quiet:
        report(flow, features, ranked)

    row = processed.iloc[0].to_dict()
    row["Predicted Class"], row["Confidence"] = ranked[0]
    append_flow(row)

    return ranked[0]


def cleanup_flows(flows, now, quiet=False):
    """
    Expire every flow idle for longer than the timeout.

    `now` is the current time on the CAPTURE clock -- the timestamp of the packet
    just processed -- so this behaves identically live and when replaying a file.
    """
    expired = [
        key
        for key, flow in flows.items()
        if flow.last_seen is not None
        and (now - float(flow.last_seen)) > config.FLOW_TIMEOUT
    ]

    for key in expired:
        expire_flow(flows.pop(key), quiet=quiet)

    return len(expired)


def flush_all(flows, quiet=False):
    """
    Classify everything still open.

    Needed at the end of a capture: without it, any flow that never went idle
    for the full timeout is silently dropped -- which for a short capture can be
    all of them.
    """
    count = 0
    for key in list(flows.keys()):
        expire_flow(flows.pop(key), quiet=quiet)
        count += 1
    return count
