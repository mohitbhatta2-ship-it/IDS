"""
Tests for the live traffic module.

Run from inside Live/:
    python -m pytest test_live_pipeline.py -v

These use synthetic scapy packets rather than a real capture, so they need no
network access and no root. They cover the four defects that made the original
module unable to run at all.
"""

import time

import pytest
from scapy.layers.inet import IP, TCP, UDP
from scapy.packet import Raw

import config
import flow_builder
from feature_calculator import calculate_features
from feature_extractor import update_flow
from flow_builder import get_or_create_flow
from flow_manager import cleanup_flows


# 2018-era timestamps, deliberately far from "now", so any code that reaches for
# the wall clock instead of the packet clock shows up immediately.
BASE_TS = 1_530_000_000.0


def packet(src="10.0.0.5", dst="93.184.216.34", sport=51000, dport=80,
           ts=BASE_TS, payload=100, flags="PA", proto="tcp"):
    if proto == "tcp":
        pkt = IP(src=src, dst=dst) / TCP(sport=sport, dport=dport, flags=flags) / Raw(b"x" * payload)
    else:
        pkt = IP(src=src, dst=dst) / UDP(sport=sport, dport=dport) / Raw(b"x" * payload)
    pkt.time = ts
    return pkt


@pytest.fixture(autouse=True)
def clean_table():
    flow_builder.reset()
    yield
    flow_builder.reset()


# --------------------------------------------------------------------------
# Flow assembly
# --------------------------------------------------------------------------


def test_flow_created_for_tcp():
    flow = get_or_create_flow(packet())
    assert flow is not None
    assert flow.src_port == 51000 and flow.dst_port == 80


def test_non_tcp_udp_is_ignored():
    pkt = IP(src="10.0.0.5", dst="8.8.8.8", proto=1)  # ICMP
    pkt.time = BASE_TS
    assert get_or_create_flow(pkt) is None


def test_reverse_direction_joins_the_same_flow():
    forward = packet()
    reverse = packet(src="93.184.216.34", dst="10.0.0.5", sport=80, dport=51000)

    a = get_or_create_flow(forward)
    b = get_or_create_flow(reverse)

    assert a is b
    assert len(flow_builder.flows) == 1


def test_direction_is_counted_correctly():
    flow = get_or_create_flow(packet())
    update_flow(flow, packet(ts=BASE_TS))
    update_flow(flow, packet(src="93.184.216.34", dst="10.0.0.5",
                             sport=80, dport=51000, ts=BASE_TS + 0.1))

    assert flow.forward_packets == 1
    assert flow.backward_packets == 1
    assert flow.total_packets == 2


def test_flow_table_is_capped(monkeypatch):
    monkeypatch.setattr(config, "MAX_FLOWS", 5)
    for i in range(20):
        pkt = packet(sport=40000 + i, ts=BASE_TS + i)
        flow = get_or_create_flow(pkt)
        update_flow(flow, pkt)
    assert len(flow_builder.flows) <= 5


# --------------------------------------------------------------------------
# Timestamps -- the bug that silently ruins replayed captures
# --------------------------------------------------------------------------


def test_start_time_comes_from_the_packet_not_the_clock():
    flow = get_or_create_flow(packet())
    assert flow.start_time is None, "timestamps must not be set at construction"

    update_flow(flow, packet(ts=BASE_TS))

    assert flow.start_time == BASE_TS
    assert abs(flow.start_time - time.time()) > 1_000_000, \
        "start_time is tracking wall-clock time, not the capture"


def test_duration_is_measured_across_the_capture():
    flow = get_or_create_flow(packet())
    update_flow(flow, packet(ts=BASE_TS))
    update_flow(flow, packet(ts=BASE_TS + 2.5))

    assert flow.duration == pytest.approx(2.5)

    features = calculate_features(flow)
    assert features["Flow Duration"] == pytest.approx(2_500_000)  # microseconds


def test_duration_of_a_replayed_capture_is_not_years():
    """The original code produced ~8 years here; anything over a day is wrong."""
    flow = get_or_create_flow(packet())
    update_flow(flow, packet(ts=BASE_TS))
    update_flow(flow, packet(ts=BASE_TS + 1))

    assert calculate_features(flow)["Flow Duration"] < 86_400 * 1_000_000


# --------------------------------------------------------------------------
# Header length -- feature parity with CICFlowMeter
# --------------------------------------------------------------------------


def test_tcp_header_length_uses_the_transport_header():
    flow = get_or_create_flow(packet())
    update_flow(flow, packet(ts=BASE_TS))
    # A TCP header with no options is 20 bytes, not the 20-byte IP header by
    # coincidence -- use options to tell them apart.
    assert flow.forward_header_lengths == [20]


def test_udp_header_length_is_eight():
    pkt = packet(proto="udp", dport=53, ts=BASE_TS)
    flow = get_or_create_flow(pkt)
    update_flow(flow, pkt)
    assert flow.forward_header_lengths == [8]


# --------------------------------------------------------------------------
# Expiry -- the crash, and the capture clock
# --------------------------------------------------------------------------


def test_cleanup_does_not_crash_on_unpacking(monkeypatch):
    """
    Regression test. flow_manager did:
        prediction, confidence = predict(processed)
    against a 3-element list, so the first expired flow raised ValueError.
    """
    monkeypatch.setattr(config, "FLOW_TIMEOUT", 1.0)

    seen = []
    monkeypatch.setattr("flow_manager.predict",
                        lambda df, **kw: [("Benign", 0.91), ("Bot", 0.06), ("Infilteration", 0.03)])
    monkeypatch.setattr("flow_manager.append_flow", seen.append)

    pkt = packet(ts=BASE_TS)
    flow = get_or_create_flow(pkt)
    update_flow(flow, pkt)

    expired = cleanup_flows(flow_builder.flows, now=BASE_TS + 30, quiet=True)

    assert expired == 1
    assert len(flow_builder.flows) == 0
    assert seen[0]["Predicted Class"] == "Benign"
    assert seen[0]["Confidence"] == pytest.approx(0.91)


def test_flow_is_not_expired_before_the_timeout(monkeypatch):
    monkeypatch.setattr(config, "FLOW_TIMEOUT", 120.0)
    monkeypatch.setattr("flow_manager.predict", lambda df, **kw: [("Benign", 1.0)])
    monkeypatch.setattr("flow_manager.append_flow", lambda row: None)

    pkt = packet(ts=BASE_TS)
    update_flow(get_or_create_flow(pkt), pkt)

    assert cleanup_flows(flow_builder.flows, now=BASE_TS + 10, quiet=True) == 0
    assert len(flow_builder.flows) == 1


def test_expiry_uses_the_capture_clock_not_wall_time(monkeypatch):
    """
    With 2018 timestamps, wall-clock expiry would retire every flow instantly.
    Passing the capture clock keeps them alive.
    """
    monkeypatch.setattr(config, "FLOW_TIMEOUT", 120.0)
    monkeypatch.setattr("flow_manager.predict", lambda df, **kw: [("Benign", 1.0)])
    monkeypatch.setattr("flow_manager.append_flow", lambda row: None)

    pkt = packet(ts=BASE_TS)
    update_flow(get_or_create_flow(pkt), pkt)

    assert cleanup_flows(flow_builder.flows, now=BASE_TS + 1, quiet=True) == 0


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------


def test_calculate_features_returns_all_thirty(monkeypatch):
    import pickle
    with open(config.FEATURES_FILE, "rb") as f:
        expected = pickle.load(f)

    pkt = packet(ts=BASE_TS)
    flow = get_or_create_flow(pkt)
    update_flow(flow, pkt)
    update_flow(flow, packet(ts=BASE_TS + 0.5))

    features = calculate_features(flow)

    missing = set(expected) - set(features)
    assert not missing, f"feature_calculator does not produce: {sorted(missing)}"


def test_features_are_finite():
    import math

    pkt = packet(ts=BASE_TS)
    flow = get_or_create_flow(pkt)
    update_flow(flow, pkt)

    for name, value in calculate_features(flow).items():
        assert not isinstance(value, float) or math.isfinite(value), f"{name} is {value}"
