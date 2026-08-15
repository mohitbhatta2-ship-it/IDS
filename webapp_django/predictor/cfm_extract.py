"""
Experimental CICFlowMeter extraction path (analysis only; nothing production).

Purpose: extract the real PCAPs with an *independent* CICFlowMeter implementation
instead of the project's custom `Live/feature_calculator.py`, so we can tell
model-generalisation failure apart from custom-extractor mismatch.

Provenance / honesty
--------------------
CSE-CIC-IDS2018 was NOT extracted by this project — its features are pre-computed
CSVs the dataset authors produced with the **Java CICFlowMeter V3**. That tool is
not present here and its native (jnetpcap) dependencies are impractical to build
in this sandbox. This module therefore uses the **Python `cicflowmeter` port
(hieulw, 0.2.0)** — a genuine, widely used CICFlowMeter reimplementation, but NOT
byte-identical to the Java original. This is an explicit, documented substitution,
not a silent one; any residual difference between this port and the Java tool is a
stated caveat.

scapy 2.7 broke the port's `AsyncSniffer(session=FlowSession)` glue (it no longer
delivers packets to the session), so this module drives the port's **own
`FlowSession` feature code directly** from a filterless `PcapReader`. Only the
packet-delivery plumbing is replaced; the feature computation is the port's.

Nothing here imports into or changes ml.py / live_capture.py / pcap_validation.py
/ Dataset Testing, and it writes no production artifacts.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import ml

# ---------------------------------------------------------------------------
# Mapping: 30 ml.FEATURES  ->  cicflowmeter column (verified by definition below)
# ---------------------------------------------------------------------------

CFM_TO_ML: dict[str, str] = {
    "Init Fwd Win Byts": "init_fwd_win_byts",
    "Fwd IAT Tot": "fwd_iat_tot",
    "Fwd IAT Max": "fwd_iat_max",
    "Dst Port": "dst_port",
    "Fwd IAT Mean": "fwd_iat_mean",
    "Fwd Header Len": "fwd_header_len",
    "Flow IAT Max": "flow_iat_max",
    "Flow Duration": "flow_duration",
    "Fwd Pkts/s": "fwd_pkts_s",
    "TotLen Fwd Pkts": "totlen_fwd_pkts",
    "Subflow Fwd Byts": "subflow_fwd_byts",
    "Fwd Pkt Len Mean": "fwd_pkt_len_mean",
    "Fwd Seg Size Avg": "fwd_seg_size_avg",
    "Fwd Pkt Len Max": "fwd_pkt_len_max",
    "Flow IAT Mean": "flow_iat_mean",
    "Flow Pkts/s": "flow_pkts_s",
    "Pkt Len Max": "pkt_len_max",
    "Pkt Len Mean": "pkt_len_mean",
    "Fwd IAT Min": "fwd_iat_min",
    "Bwd Pkt Len Mean": "bwd_pkt_len_mean",
    "Pkt Size Avg": "pkt_size_avg",
    "Bwd Seg Size Avg": "bwd_seg_size_avg",
    "Bwd Pkt Len Max": "bwd_pkt_len_max",
    "TotLen Bwd Pkts": "totlen_bwd_pkts",
    "Subflow Bwd Byts": "subflow_bwd_byts",
    "Pkt Len Var": "pkt_len_var",
    "Pkt Len Std": "pkt_len_std",
    "Flow IAT Min": "flow_iat_min",
    "Fwd Seg Size Min": "fwd_seg_size_min",
    "Fwd Pkt Len Std": "fwd_pkt_len_std",
}

# Definition notes — same NAME does not guarantee same DEFINITION. Verified by
# inspecting the port's output values against the Live extractor and CIC.
DEFINITION_NOTES = {
    "units": "Flow Duration and all IAT features are MICROSECONDS in both the CIC "
             "dataset and this cicflowmeter port (verified: an ~18 s real flow reads "
             "~18.7e6). No unit conversion is applied.",
    "Fwd Pkt Len / TotLen": "The port's packet-length features include L3/L4 header "
             "bytes, whereas the Live extractor and CIC use the L4 payload. This is a "
             "real definition difference (reported), not corrected here.",
    "flow_segmentation": "The port cuts flows on an activity/idle timeout and does not "
             "create the short post-FIN residual flows the Live engine does, so the "
             "flow COUNT per capture differs.",
    "Fwd Seg Size Min / Init Fwd Win Byts": "Both agree with the Live extractor (20 and "
             "8192 for real FTP) and both differ from CIC (40 and 26883) — the SHAP-"
             "critical features are NOT moved toward CIC by switching extractor.",
}

CFM_COLUMNS_TOTAL = 82  # the port emits 82 columns; we use 30.


def cicflowmeter_version() -> str:
    try:
        import importlib.metadata as m
        return "cicflowmeter " + m.version("cicflowmeter")
    except Exception:  # noqa: BLE001
        return "cicflowmeter (unknown version)"


# ---------------------------------------------------------------------------
# Run the port's feature code directly (packet delivery replaced; features intact)
# ---------------------------------------------------------------------------


def run_cicflowmeter(pcap_path) -> pd.DataFrame:
    """
    Extract flows from a pcap with the cicflowmeter port and return its RAW output
    (all columns). Raises on unreadable input; returns an empty frame if the pcap
    yields no TCP/UDP flows.
    """
    from scapy.utils import PcapReader
    from cicflowmeter.flow_session import FlowSession

    pcap_path = Path(pcap_path)
    if not pcap_path.is_file():
        raise FileNotFoundError(f"pcap not found: {pcap_path}")

    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    tmp.close()
    try:
        FlowSession.output_mode = "csv"
        FlowSession.output = tmp.name
        FlowSession.fields = None
        FlowSession.verbose = False
        session = FlowSession()
        for pkt in PcapReader(str(pcap_path)):
            if "IP" in pkt and ("TCP" in pkt or "UDP" in pkt):
                session.on_packet_received(pkt)
        # Flush all open flows + close the writer. (toPacketList() would do this but
        # then calls a scapy super() method removed in 2.7, so we call the two steps.)
        session.garbage_collect(None)
        del session.output_writer

        if os.path.getsize(tmp.name) == 0:
            return pd.DataFrame()
        return pd.read_csv(tmp.name)
    finally:
        os.unlink(tmp.name)


# ---------------------------------------------------------------------------
# Map to the 30 model features — exact order, finite, no zero-fill
# ---------------------------------------------------------------------------


@dataclass
class MappedFlows:
    X: pd.DataFrame                 # exactly ml.FEATURES (order), all finite
    raw: pd.DataFrame               # full cicflowmeter output for the valid rows
    invalid: list = field(default_factory=list)   # rows dropped, with the reason


def map_to_ml_features(cfm_raw: pd.DataFrame) -> MappedFlows:
    """
    Select + rename the 30 features, verify finiteness, and DROP (never zero-fill)
    any flow with a missing/non-finite feature — recording why.
    """
    if cfm_raw.empty:
        return MappedFlows(X=pd.DataFrame(columns=ml.FEATURES), raw=cfm_raw)

    missing_cols = [f for f, c in CFM_TO_ML.items() if c not in cfm_raw.columns]
    if missing_cols:
        raise ValueError(f"cicflowmeter output is missing columns for: {missing_cols}")

    renamed = cfm_raw.rename(columns={v: k for k, v in CFM_TO_ML.items()})
    sub = renamed[list(ml.FEATURES)].apply(pd.to_numeric, errors="coerce")

    keep_mask = np.isfinite(sub.to_numpy()).all(axis=1)
    invalid = []
    for idx in sub.index[~keep_mask]:
        bad = [f for f in ml.FEATURES if not np.isfinite(pd.to_numeric(sub.loc[idx, f], errors="coerce"))]
        invalid.append({"row": int(idx), "non_finite_features": bad})

    X = sub[keep_mask].reset_index(drop=True)[list(ml.FEATURES)]  # exact order
    raw_valid = cfm_raw[keep_mask].reset_index(drop=True)
    return MappedFlows(X=X, raw=raw_valid, invalid=invalid)


def extract_real_pcaps(pcap_dir_label_pairs) -> dict:
    """
    Run the whole real-PCAP set through cicflowmeter. Ground-truth label comes only
    from the folder (passed in), never from a model. Returns per-pcap frames + a
    combined labelled frame + validity stats.
    """
    per_pcap = []
    frames = []
    for pcap, label in pcap_dir_label_pairs:
        raw = run_cicflowmeter(pcap)
        mapped = map_to_ml_features(raw)
        rec = {"pcap": str(pcap), "label": label, "cfm_flows": int(len(raw)),
               "valid_flows": int(len(mapped.X)), "invalid_flows": len(mapped.invalid),
               "invalid": mapped.invalid}
        per_pcap.append(rec)
        if not mapped.X.empty:
            f = mapped.X.copy()
            f["Label"] = label
            f["pcap"] = Path(pcap).name
            # carry a few descriptive (non-model) cicflowmeter fields for §9
            for extra in ("tot_fwd_pkts", "tot_bwd_pkts", "fwd_pkt_len_min",
                          "bwd_pkt_len_min", "flow_byts_s"):
                if extra in mapped.raw.columns:
                    f[extra] = mapped.raw[extra].to_numpy()
            frames.append(f)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=ml.FEATURES)
    return {"per_pcap": per_pcap, "combined": combined,
            "total_flows": int(sum(p["valid_flows"] for p in per_pcap)),
            "total_invalid": int(sum(p["invalid_flows"] for p in per_pcap))}
