"""
CICFlowMeter vs custom-extractor experiment (analysis only; production frozen).

Central question: is the real-PCAP FTP-BruteForce failure caused by
  (A) model generalisation failure, or
  (B) a mismatch in *our* custom Live feature extractor?

Method: run the EXACT same real PCAPs through an independent CICFlowMeter
implementation (see ``cfm_extract``) and score them with the UNCHANGED production
model, then compare three evaluation paths:

  A. CIC-IDS2018 held-out    — production model on the CIC test parquet.
  B. Real PCAP / custom      — the frozen ``validation/results/flows.csv`` baseline
                               (our Live extractor). Read-only; never regenerated.
  C. Real PCAP / CICFlowMeter — the same PCAPs re-extracted with cicflowmeter,
                               scored by the same production model.

If C substantially beats B, the failure is extraction (B); if C is also poor,
the failure is model generalisation (A). Nothing here retrains, and nothing here
writes to ml.py / live_capture.py / pcap_validation.py or the frozen baseline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import ml, cfm_extract as cf

FTP = "FTP-BruteForce"
BENIGN = "Benign"

# §9 feature-distribution comparison. Some of these are NOT among the 30 model
# features (Total Fwd/Bwd Packets, Flow Bytes/s, Fwd/Bwd Pkt Len Min); for those,
# CIC (which we only have as the 30-feature parquet) has no column, and the frozen
# custom flows.csv may not carry them either. Availability is reported per source,
# never fabricated.
DIST_FEATURES = [
    "Flow Duration", "TotLen Fwd Pkts", "TotLen Bwd Pkts", "Flow Pkts/s",
    "Fwd Pkt Len Max", "Fwd Pkt Len Mean", "Fwd Pkt Len Std",
    "Bwd Pkt Len Max", "Bwd Pkt Len Mean",
    "Fwd Seg Size Min", "Init Fwd Win Byts", "Dst Port",
    # not in the 30 — descriptive columns only, compared where available:
    "Total Fwd Packets", "Total Bwd Packets", "Flow Bytes/s",
    "Fwd Pkt Len Min", "Bwd Pkt Len Min",
]

# where each §9 feature lives in each source ("" == not available there)
_CUSTOM_COL = {  # columns in the frozen validation/results/flows.csv
    "Total Fwd Packets": "FwdPackets", "Total Bwd Packets": "BwdPackets",
}
_CFM_EXTRA = {  # descriptive (non-model) cols carried from cicflowmeter raw output
    "Total Fwd Packets": "tot_fwd_pkts", "Total Bwd Packets": "tot_bwd_pkts",
    "Flow Bytes/s": "flow_byts_s",
    "Fwd Pkt Len Min": "fwd_pkt_len_min", "Bwd Pkt Len Min": "bwd_pkt_len_min",
}


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "sample_data" / "real_pcap").is_dir():
            return parent
    raise RuntimeError("repo root with sample_data/real_pcap not found")


# ---------------------------------------------------------------------------
# Prediction + metrics (self-contained; does not touch predict_batch internals)
# ---------------------------------------------------------------------------


def _decode(codes) -> np.ndarray:
    dec = {int(k): v for k, v in ml.LABELS.items()}
    return np.array([dec.get(int(c), str(c)) for c in codes])


def predict_production(X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Score X (exactly ml.FEATURES) with the UNCHANGED production model."""
    model, scaler = ml._load(ml.DEFAULT_MODEL)
    Xp = ml._prepare(X[list(ml.FEATURES)], scaler)
    pred = _decode(model.predict(Xp))
    conf = model.predict_proba(Xp).max(axis=1)
    return pred, conf


def metrics(truth_names, pred_names, conf=None) -> dict:
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, classification_report, confusion_matrix)
    truth = np.array([str(t) for t in truth_names])
    pred = np.array([str(p) for p in pred_names])
    labels = sorted(set(truth) | set(pred))
    rep = classification_report(truth, pred, labels=labels, output_dict=True, zero_division=0)
    per_class = [{
        "class": c, "precision": rep[c]["precision"], "recall": rep[c]["recall"],
        "f1": rep[c]["f1-score"], "support": int(rep[c]["support"]),
    } for c in labels if c in rep and rep[c]["support"] > 0]
    cm = confusion_matrix(truth, pred, labels=labels)
    out = {
        "n": int(len(truth)),
        "accuracy": float(accuracy_score(truth, pred)),
        "macro_precision": float(precision_score(truth, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(truth, pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, pred, average="weighted", zero_division=0)),
        "ftp_recall": (float(recall_score(truth, pred, labels=[FTP], average="micro", zero_division=0))
                       if FTP in truth else None),
        "benign_recall": (float(recall_score(truth, pred, labels=[BENIGN], average="micro", zero_division=0))
                          if BENIGN in truth else None),
        "per_class": per_class,
        "confusion": {"labels": labels, "matrix": cm.tolist()},
        "prediction_distribution": {k: int(v) for k, v in pd.Series(pred).value_counts().items()},
    }
    if conf is not None and len(conf):
        out["confidence"] = {"mean": float(np.mean(conf)), "median": float(np.median(conf))}
    return out


# ---------------------------------------------------------------------------
# The three paths
# ---------------------------------------------------------------------------


def path_a_cic_heldout() -> dict:
    """A — production model on the CIC held-out test parquet."""
    test = pd.read_parquet(ml.DATA_ROOT / "Processed_Data" / "test_selected.parquet")
    X = test[list(ml.FEATURES)].reset_index(drop=True)
    truth = _decode(test["Label"].to_numpy())
    pred, conf = predict_production(X)
    m = metrics(truth, pred, conf)
    m["source"] = "CIC-IDS2018 test_selected.parquet (production model)"
    return m


def path_b_custom() -> dict:
    """
    B — the frozen custom-extractor real-PCAP result. Read the committed
    validation/results/flows.csv (Live extractor + production model); do NOT
    regenerate it. Returns metrics + the per-flow frame for §9.
    """
    csv = repo_root() / "validation" / "results" / "flows.csv"
    df = pd.read_csv(csv)
    truth = df["Label"].to_numpy()
    pred = df["Predicted Class"].to_numpy()
    conf = df["Confidence"].to_numpy() if "Confidence" in df.columns else None
    m = metrics(truth, pred, conf)
    m["source"] = f"frozen {csv.relative_to(repo_root())} (custom Live extractor)"
    return m, df


def path_c_cicflowmeter() -> dict:
    """C — the same PCAPs, cicflowmeter-extracted, scored by the production model."""
    from . import pcap_validation as pv
    root = repo_root() / "sample_data" / "real_pcap"
    pairs = list(pv.iter_labelled_pcaps(root))
    ext = cf.extract_real_pcaps(pairs)
    combined = ext["combined"]
    if combined.empty:
        raise RuntimeError("cicflowmeter produced no flows for any real PCAP")
    pred, conf = predict_production(combined)
    m = metrics(combined["Label"].to_numpy(), pred, conf)
    m["source"] = f"real PCAPs via {cf.cicflowmeter_version()} (production model)"
    m["extraction"] = ext
    combined = combined.copy()
    combined["Predicted Class"] = pred
    combined["Confidence"] = conf
    return m, combined


# ---------------------------------------------------------------------------
# §8 FTP recall  &  §9/§10 feature distribution (custom vs CICFlowMeter vs CIC)
# ---------------------------------------------------------------------------


def _median_from(df: pd.DataFrame, col: str):
    if col not in df.columns:
        return None
    v = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(v.median()) if len(v) else None


def feature_distribution(custom_df: pd.DataFrame, cfm_df: pd.DataFrame) -> pd.DataFrame:
    """
    §9 — median of each listed feature for real FTP flows under the custom
    extractor (B) and cicflowmeter (C), against the CIC FTP training median.
    §10 — for features available in all three, whether cicflowmeter is CLOSER to
    CIC than the custom extractor (|C-CIC| < |B-CIC|).
    Unavailable values are reported as None, never fabricated.
    """
    train = pd.read_parquet(ml.DATA_ROOT / "Processed_Data" / "balanced_train_selected.parquet")
    enc = {v: k for k, v in ml.LABELS.items()}[FTP]
    cic_ftp = train[train["Label"] == enc]

    custom_ftp = custom_df[custom_df["Label"] == FTP]
    cfm_ftp = cfm_df[cfm_df["Label"] == FTP]

    rows = []
    for feat in DIST_FEATURES:
        cic = float(cic_ftp[feat].median()) if feat in cic_ftp.columns else None

        b_col = feat if feat in custom_ftp.columns else _CUSTOM_COL.get(feat, "")
        b = _median_from(custom_ftp, b_col) if b_col else None

        c_col = feat if feat in cfm_ftp.columns else _CFM_EXTRA.get(feat, "")
        c = _median_from(cfm_ftp, c_col) if c_col else None

        closer = None
        if None not in (cic, b, c):
            db, dc = abs(b - cic), abs(c - cic)
            closer = "cicflowmeter" if dc < db else ("custom" if db < dc else "tie")
        rows.append({
            "feature": feat,
            "in_model_features": feat in ml.FEATURES,
            "cic_ftp_median": cic,
            "custom_real_ftp_median": b,
            "cicflowmeter_real_ftp_median": c,
            "cfm_closer_to_cic": closer,
        })
    return pd.DataFrame(rows)


def summarise_closer(dist: pd.DataFrame) -> dict:
    """§10 verdict: does cicflowmeter move real FTP toward CIC on comparable feats?"""
    comparable = dist[dist["cfm_closer_to_cic"].notna()]
    counts = comparable["cfm_closer_to_cic"].value_counts().to_dict()
    return {
        "comparable_features": int(len(comparable)),
        "cicflowmeter_closer": int(counts.get("cicflowmeter", 0)),
        "custom_closer": int(counts.get("custom", 0)),
        "tie": int(counts.get("tie", 0)),
    }
