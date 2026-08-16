"""
Final robustness analysis of the frozen Candidate 2 (ANALYSIS ONLY).

Diagnoses WHY Candidate 2 behaves as it does on the FROZEN independent test set
(0.754 FTP recall / 0.282 benign FP), using only the existing frozen models, the
existing extraction pipeline, and the existing SHAP infrastructure. Nothing is
retrained, no threshold is changed, no heuristic is added, no test case is
removed, and the independent labels are never altered.

Data separation (strict):
  * TRAINING distribution   = CIC balanced_train (+ v1/v2 real, used only where a
    training/eval distribution is explicitly being *compared against*).
  * v2 VALIDATION distribution = validation/realistic_pcaps_v2 (prior LOCO eval).
  * INDEPENDENT test        = validation/independent_real_pcaps (FROZEN; used for
    evaluation/diagnosis only, never for any fitting decision).

Every function here is read-only with respect to models and data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import ml, independent_eval as ie

FTP = "FTP-BruteForce"
BENIGN = "Benign"


def repo_root() -> Path:
    return ie.repo_root()


def indep_dir() -> Path:
    return ie.test_dir()


# ---------------------------------------------------------------------------
# Build the per-flow analysis frame (independent test), frozen predictions
# ---------------------------------------------------------------------------


def _capture_id_from_pcap(pcap_name: str) -> str:
    parts = pcap_name.split("_")
    return "_".join(parts[:2])          # e.g. benign_01_custom_x.pcap -> benign_01


def _manifest() -> pd.DataFrame:
    m = pd.read_csv(indep_dir() / "MANIFEST.csv")
    # mode lives in the per-capture metadata JSON detail
    modes = {}
    for cap_id in m["capture_id"]:
        meta = indep_dir() / "metadata" / f"{cap_id}.json"
        if meta.is_file():
            modes[cap_id] = json.loads(meta.read_text()).get("detail", {}).get("mode", "unknown")
    m["mode"] = m["capture_id"].map(modes).fillna("unknown")
    # normalise server implementation vs configuration
    m["server_impl"] = np.where(m["server"].str.contains("custom"), "raw-socket", "pyftpdlib")
    m["server_config"] = m["server"].str.replace(r"\(.*\)", "", regex=True)
    return m


def build_frame() -> pd.DataFrame:
    """
    One row per INDEPENDENT-test flow with 30 features, folder label, capture
    metadata (scenario/client/server/mode/env), and BOTH frozen models'
    predictions + confidence. Read-only.
    """
    flows = ie.extract_test_flows()
    if flows.invalid:
        raise RuntimeError(f"unexpected invalid flows: {flows.invalid[:3]}")
    df = flows.df.copy()
    df["capture_id"] = df["capture"].map(_capture_id_from_pcap)

    man = _manifest().set_index("capture_id")
    for col in ("scenario", "client", "server", "server_impl", "server_config",
                "environment", "connection_pattern", "mode"):
        df[col] = df["capture_id"].map(man[col])

    (prod, pscaler), (cand, cscaler) = ie.load_frozen_models()
    df["prod_pred"], df["prod_conf"] = ie.predict(prod, pscaler, df)
    df["cand_pred"], df["cand_conf"] = ie.predict(cand, cscaler, df)
    # correctness / error-group tags for Candidate 2
    df["cand_correct"] = df["cand_pred"] == df["Label"]
    df["error_group"] = np.select(
        [(df["Label"] == FTP) & (df["cand_pred"] == FTP),
         (df["Label"] == FTP) & (df["cand_pred"] != FTP),
         (df["Label"] == BENIGN) & (df["cand_pred"] == BENIGN),
         (df["Label"] == BENIGN) & (df["cand_pred"] != BENIGN)],
        ["TP_FTP", "FN_FTP", "TP_Benign", "FP_Benign"], default="other")
    return df


# ---------------------------------------------------------------------------
# Grouped metric helpers
# ---------------------------------------------------------------------------


def _fp_rate(sub) -> float:
    benign = sub[sub["Label"] == BENIGN]
    if len(benign) == 0:
        return np.nan
    return float((benign["cand_pred"] != BENIGN).mean())


def _recall(sub, cls) -> float:
    m = sub[sub["Label"] == cls]
    if len(m) == 0:
        return np.nan
    return float((m["cand_pred"] == cls).mean())


def _macro_f1(sub) -> float:
    from sklearn.metrics import f1_score
    if sub.empty:
        return np.nan
    return float(f1_score(sub["Label"], sub["cand_pred"], average="macro", zero_division=0))


def group_metrics(df, by) -> pd.DataFrame:
    rows = []
    for key, sub in df.groupby(by):
        rows.append({
            **({by: key} if isinstance(by, str) else dict(zip(by, key))),
            "flows": int(len(sub)),
            "benign_support": int((sub["Label"] == BENIGN).sum()),
            "ftp_support": int((sub["Label"] == FTP).sum()),
            "ftp_recall": _recall(sub, FTP),
            "benign_recall": _recall(sub, BENIGN),
            "benign_fp_rate": _fp_rate(sub),
            "macro_f1": _macro_f1(sub),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Analysis 1 — benign FP breakdown (per benign capture)
# ---------------------------------------------------------------------------


def benign_fp_breakdown(df) -> pd.DataFrame:
    rows = []
    for cap, sub in df[df["Label"] == BENIGN].groupby("capture"):
        r = sub.iloc[0]
        fp = int((sub["cand_pred"] != BENIGN).sum())
        rows.append({
            "pcap": cap, "capture_id": r["capture_id"], "scenario": r["scenario"],
            "client": r["client"], "server": r["server"], "server_impl": r["server_impl"],
            "environment": r["environment"], "mode": r["mode"],
            "connection_pattern": r["connection_pattern"],
            "n_flows": int(len(sub)), "benign_flows": int(len(sub)),
            "correct_benign": int(len(sub) - fp), "false_positives": fp,
            "benign_recall": round(1 - fp / len(sub), 4),
            "mean_confidence": round(float(sub["cand_conf"].mean()), 4),
            "median_confidence": round(float(sub["cand_conf"].median()), 4),
        })
    return pd.DataFrame(rows).sort_values("false_positives", ascending=False)


# ---------------------------------------------------------------------------
# Analysis 2 — per-scenario metrics
# ---------------------------------------------------------------------------


def scenario_metrics(df) -> pd.DataFrame:
    rows = []
    for (scenario, cls), sub in df.groupby(["scenario", "Label"]):
        correct = int((sub["cand_pred"] == cls).sum())
        rows.append({
            "scenario": scenario, "class": cls,
            "capture_count": int(sub["capture"].nunique()), "flow_count": int(len(sub)),
            "correct": correct, "incorrect": int(len(sub) - correct),
            "recall": round(correct / len(sub), 4),
            "mean_confidence": round(float(sub["cand_conf"].mean()), 4),
        })
    return pd.DataFrame(rows).sort_values(["class", "recall"])


# ---------------------------------------------------------------------------
# Analysis 5 — feature distributions across the three data roles
# ---------------------------------------------------------------------------

FOCUS_FEATURES = [
    "Fwd Seg Size Min", "Init Fwd Win Byts", "Dst Port", "Flow Duration",
    "Flow Pkts/s", "TotLen Fwd Pkts", "TotLen Bwd Pkts", "Fwd Header Len",
    "Fwd Pkts/s", "Fwd IAT Min", "Flow IAT Min", "Pkt Len Mean",
]


def feature_distributions(indep_df) -> pd.DataFrame:
    # training distribution: CIC FTP
    train = pd.read_parquet(ml.DATA_ROOT / "Processed_Data" / "balanced_train_selected.parquet")
    enc = {v: k for k, v in ml.LABELS.items()}
    cic_ftp = train[train["Label"] == enc[FTP]]
    # v2 validation distribution: real FTP flows from v2
    from . import retraining_v2 as r2
    v2 = r2.extract_real_flows_v2().df
    v2_ftp = v2[v2["Label"] == FTP]
    # independent test distribution
    ind_ftp = indep_df[indep_df["Label"] == FTP]
    ind_ben = indep_df[indep_df["Label"] == BENIGN]

    def stat(frame, feat):
        if feat not in frame.columns or len(frame) == 0:
            return (None, None, None)
        v = pd.to_numeric(frame[feat], errors="coerce").dropna()
        return (float(v.median()), float(v.quantile(0.25)), float(v.quantile(0.75)))

    rows = []
    for feat in FOCUS_FEATURES:
        c = stat(cic_ftp, feat); v = stat(v2_ftp, feat)
        i = stat(ind_ftp, feat); b = stat(ind_ben, feat)
        rows.append({
            "feature": feat,
            "train_cic_ftp_median": c[0], "train_cic_ftp_q25": c[1], "train_cic_ftp_q75": c[2],
            "v2_real_ftp_median": v[0], "v2_real_ftp_q25": v[1], "v2_real_ftp_q75": v[2],
            "indep_ftp_median": i[0], "indep_ftp_q25": i[1], "indep_ftp_q75": i[2],
            "indep_benign_median": b[0], "indep_benign_q25": b[1], "indep_benign_q75": b[2],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Analysis 6 — SHAP by error group (Candidate 2)
# ---------------------------------------------------------------------------


def shap_error_groups(df, top_n=8) -> tuple[pd.DataFrame, dict]:
    try:
        import shap
    except Exception:  # noqa: BLE001
        return pd.DataFrame(), {"available": False}
    (_prod, _ps), (cand, _cs) = ie.load_frozen_models()
    explainer = shap.TreeExplainer(cand)
    ftp_class_idx = list(cand.classes_).index({v: k for k, v in ml.LABELS.items()}[FTP])

    rows, tops = [], {}
    for grp in ("TP_FTP", "FN_FTP", "TP_Benign", "FP_Benign"):
        sub = df[df["error_group"] == grp]
        if len(sub) == 0:
            tops[grp] = []
            continue
        X = sub[list(ml.FEATURES)]
        sv = np.asarray(explainer.shap_values(X))          # (n, features, classes)
        # mean |SHAP| across samples, toward the FTP class (what pushes to FTP)
        mean_abs_all = np.abs(sv).mean(axis=(0, 2))
        mean_ftp = sv[:, :, ftp_class_idx].mean(axis=0)    # signed contribution to FTP
        order = np.argsort(mean_abs_all)[::-1]
        top_feats = [ml.FEATURES[j] for j in order[:top_n]]
        tops[grp] = top_feats
        for rank, j in enumerate(order[:top_n], 1):
            rows.append({"group": grp, "rank": rank, "feature": ml.FEATURES[j],
                         "mean_abs_shap": float(mean_abs_all[j]),
                         "signed_shap_toward_ftp": float(mean_ftp[j]),
                         "n_flows": int(len(sub))})
    return pd.DataFrame(rows), {"available": True, "top_features": tops}


# ---------------------------------------------------------------------------
# Analysis 7 — confidence by error group
# ---------------------------------------------------------------------------


def confidence_error_groups(df) -> pd.DataFrame:
    rows = []
    for grp in ("TP_FTP", "FN_FTP", "TP_Benign", "FP_Benign"):
        c = df[df["error_group"] == grp]["cand_conf"]
        if len(c) == 0:
            continue
        rows.append({
            "group": grp, "count": int(len(c)),
            "mean": round(float(c.mean()), 4), "median": round(float(c.median()), 4),
            "p10": round(float(c.quantile(0.10)), 4), "p25": round(float(c.quantile(0.25)), 4),
            "p75": round(float(c.quantile(0.75)), 4), "p90": round(float(c.quantile(0.90)), 4),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Analysis 8 — capture-level robustness
# ---------------------------------------------------------------------------


def capture_metrics(df) -> tuple[pd.DataFrame, dict]:
    rows = []
    for cap, sub in df.groupby("capture"):
        r = sub.iloc[0]
        correct = int((sub["cand_pred"] == sub["Label"]).sum())
        rows.append({
            "capture": cap, "capture_id": r["capture_id"], "scenario": r["scenario"],
            "class": r["Label"], "flow_count": int(len(sub)),
            "correct": correct, "incorrect": int(len(sub) - correct),
            "recall": round(correct / len(sub), 4),
            "confidence": round(float(sub["cand_conf"].mean()), 4),
        })
    cdf = pd.DataFrame(rows).sort_values(["class", "recall"])
    ftp = cdf[cdf["class"] == FTP]
    ben = cdf[cdf["class"] == BENIGN]
    summary = {
        "n_ftp_captures": int(len(ftp)), "n_benign_captures": int(len(ben)),
        "pct_ftp_recall_ge_0.5": round(float((ftp["recall"] >= 0.5).mean()), 4),
        "pct_ftp_recall_ge_0.75": round(float((ftp["recall"] >= 0.75).mean()), 4),
        "pct_ftp_recall_ge_0.9": round(float((ftp["recall"] >= 0.9).mean()), 4),
        "pct_benign_captures_zero_fp": round(float((ben["recall"] >= 1.0).mean()), 4),
    }
    return cdf, summary
