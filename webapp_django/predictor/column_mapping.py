"""
Maps columns from other CICFlowMeter-derived datasets onto our 30 feature names.

CIC-IDS2017, the raw CSE-CIC-IDS2018 CSVs and our processed parquet all describe
the *same* CICFlowMeter measurements, but spell them differently:

    Destination Port            ->  Dst Port
    Total Length of Fwd Packets ->  TotLen Fwd Pkts
    Init_Win_bytes_forward      ->  Init Fwd Win Byts

Several of the published CSVs also carry a leading space in every header, which
is a notorious source of "column not found" errors. Without this module an
upload from any of those sources is rejected outright even though the data is
perfectly usable.

The matching is deliberately conservative: names are normalised and compared
against an explicit alias table. Nothing is guessed by fuzzy similarity, because
silently mapping the wrong column would produce confident nonsense.
"""

from __future__ import annotations

import re

import pandas as pd

# Canonical name -> alternative spellings seen in the published datasets.
ALIASES: dict[str, list[str]] = {
    "Dst Port": ["Destination Port", "dst_port", "dest port"],
    "Flow Duration": ["Flow Duration", "flow_duration"],
    "Init Fwd Win Byts": [
        "Init_Win_bytes_forward", "Init Win bytes forward",
        "init_fwd_win_bytes", "Init Fwd Win Bytes",
    ],
    "Fwd Header Len": ["Fwd Header Length", "fwd_header_length"],
    "Flow Pkts/s": ["Flow Packets/s", "flow_packets_s", "Flow Packets per second"],
    "Fwd Pkts/s": ["Fwd Packets/s", "fwd_packets_s"],

    "Flow IAT Mean": ["Flow IAT Mean"],
    "Flow IAT Max": ["Flow IAT Max"],
    "Flow IAT Min": ["Flow IAT Min"],
    "Fwd IAT Tot": ["Fwd IAT Total", "fwd_iat_total"],
    "Fwd IAT Mean": ["Fwd IAT Mean"],
    "Fwd IAT Max": ["Fwd IAT Max"],
    "Fwd IAT Min": ["Fwd IAT Min"],

    "TotLen Fwd Pkts": ["Total Length of Fwd Packets", "Total Length of Fwd Packet"],
    "TotLen Bwd Pkts": ["Total Length of Bwd Packets", "Total Length of Bwd Packet"],
    "Subflow Fwd Byts": ["Subflow Fwd Bytes"],
    "Subflow Bwd Byts": ["Subflow Bwd Bytes"],

    "Fwd Pkt Len Mean": ["Fwd Packet Length Mean"],
    "Fwd Pkt Len Max": ["Fwd Packet Length Max"],
    "Fwd Pkt Len Std": ["Fwd Packet Length Std"],
    "Bwd Pkt Len Mean": ["Bwd Packet Length Mean"],
    "Bwd Pkt Len Max": ["Bwd Packet Length Max"],

    "Fwd Seg Size Avg": ["Avg Fwd Segment Size", "Average Fwd Segment Size"],
    "Bwd Seg Size Avg": ["Avg Bwd Segment Size", "Average Bwd Segment Size"],
    "Fwd Seg Size Min": ["min_seg_size_forward", "Min Seg Size Forward"],

    "Pkt Len Mean": ["Packet Length Mean"],
    "Pkt Len Max": ["Max Packet Length", "Packet Length Max"],
    "Pkt Len Std": ["Packet Length Std"],
    "Pkt Len Var": ["Packet Length Variance"],
    "Pkt Size Avg": ["Average Packet Size", "Avg Packet Size"],
}

# The ground-truth column, however it is spelled.
LABEL_ALIASES = ["Label", "label", "Attack", "attack", "Class", "class", "Category"]


def normalise(name: str) -> str:
    """
    Reduce a column name to a comparable form.

    Handles the leading/trailing whitespace that the published CSVs are full of,
    collapses separators, and lowercases -- so "  Fwd Packet Length Max" and
    "fwd_packet_length_max" both land on the same key.
    """
    n = str(name).strip().lower()
    n = re.sub(r"[\s_\-]+", " ", n)
    n = n.replace("/s", " per s")
    n = re.sub(r"[^a-z0-9 ]", "", n)
    return n.strip()


def _build_lookup(features: list[str]) -> dict[str, str]:
    """normalised name -> canonical feature name."""
    lookup: dict[str, str] = {}
    for canonical in features:
        lookup[normalise(canonical)] = canonical
        for alias in ALIASES.get(canonical, []):
            lookup[normalise(alias)] = canonical
    return lookup


def map_columns(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, dict]:
    """
    Rename an uploaded frame's columns onto the canonical feature names.

    Returns the renamed frame and a report describing what happened, so the UI
    can be honest about which columns were matched by an alias rather than
    given exactly.
    """
    lookup = _build_lookup(features)

    renames: dict[str, str] = {}
    aliased: list[tuple[str, str]] = []
    exact: list[str] = []

    for column in df.columns:
        key = normalise(column)
        canonical = lookup.get(key)
        if canonical is None or canonical in renames.values():
            continue

        renames[column] = canonical
        if column == canonical:
            exact.append(canonical)
        else:
            aliased.append((column, canonical))

    out = df.rename(columns=renames)

    # Ground truth, however it was spelled.
    label_found = None
    if "Label" not in out.columns:
        label_lookup = {normalise(a): a for a in LABEL_ALIASES}
        for column in out.columns:
            if normalise(column) in label_lookup:
                out = out.rename(columns={column: "Label"})
                label_found = column
                break

    matched = [f for f in features if f in out.columns]
    missing = [f for f in features if f not in out.columns]

    report = {
        "total_required": len(features),
        "matched": len(matched),
        "exact": sorted(exact),
        "aliased": sorted(aliased, key=lambda p: p[1]),
        "missing": missing,
        "label_renamed_from": label_found,
        "unrecognised": [c for c in df.columns if c not in renames and normalise(c) not in
                         {normalise(a) for a in LABEL_ALIASES}],
    }

    return out, report
