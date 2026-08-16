"""
INDEPENDENT real-PCAP evaluation of the two FROZEN models (analysis only).

Evaluates the CURRENT PRODUCTION model and the frozen Candidate 2 (best unweighted
CIC+real model from retraining-v2) on the brand-new independent test corpus in
``validation/independent_real_pcaps/``. Nothing is retrained, no threshold is
changed, no preprocessing is altered, and no candidate is re-selected after seeing
results. Extraction uses the EXISTING ``pcap_validation`` pipeline only.

The independent set is used ONLY to evaluate the already-frozen models.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import ml, pcap_validation as pv, live_capture

FTP = "FTP-BruteForce"
BENIGN = "Benign"


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "validation" / "independent_real_pcaps").is_dir():
            return p
    for p in here.parents:
        if (p / ".git").is_dir():
            return p
    raise RuntimeError("repo root not found")


def test_dir() -> Path:
    return repo_root() / "validation" / "independent_real_pcaps"


def candidate2_dir() -> Path:
    return repo_root() / "validation" / "models" / "realistic_pcap_candidate_v2"


def candidate2_file() -> Path:
    """The exact Candidate 2 artifact = best (unweighted) per retraining-v2 metadata."""
    verdict = json.loads((repo_root() / "validation" / "results" / "retraining_v2"
                          / "final_verdict.json").read_text())
    best = verdict["best_candidate"]                # e.g. "cic+real_none"
    tag = best.split("_")[-1]                        # "none"
    return candidate2_dir() / f"candidate2_{tag}.pkl"


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def production_model_path() -> Path:
    return ml.MODELS_DIR / ml.MODEL_REGISTRY[ml.DEFAULT_MODEL]["file"]


# ---------------------------------------------------------------------------
# Test flows -- existing extraction pipeline, capture-tagged, no zero-fill
# ---------------------------------------------------------------------------


@dataclass
class TestFlows:
    df: pd.DataFrame            # ml.FEATURES + Label + capture + flow_uid
    invalid: list

    @property
    def captures(self):
        return sorted(self.df["capture"].unique())


def extract_test_flows() -> TestFlows:
    live_capture._ensure_live_on_path()
    root = test_dir()
    rows, invalid = [], []
    for folder, label in (("benign", BENIGN), ("ftp_bruteforce", FTP)):
        for pcap in sorted((root / folder).glob("*.pcap")):
            for i, fl in enumerate(pv.replay_pcap(pcap)):
                problem = pv._feature_problem(fl["features"])
                if problem:
                    invalid.append({"capture": pcap.name, "flow_index": i, "reason": problem})
                    continue
                row = {f: float(fl["features"][f]) for f in ml.FEATURES}
                row["Label"] = label
                row["capture"] = pcap.name
                row["flow_uid"] = f"{pcap.name}#{i}"
                rows.append(row)
    return TestFlows(df=pd.DataFrame(rows), invalid=invalid)


# ---------------------------------------------------------------------------
# Prediction (frozen models; no threshold change, no preprocessing change)
# ---------------------------------------------------------------------------


def _decode(codes) -> np.ndarray:
    dec = {int(k): v for k, v in ml.LABELS.items()}
    return np.array([dec.get(int(c), str(c)) for c in codes])


def predict(model, scaler, X: pd.DataFrame):
    Xp = ml._prepare(X[list(ml.FEATURES)], scaler) if scaler is not None else X[list(ml.FEATURES)]
    pred = _decode(model.predict(Xp))
    conf = model.predict_proba(Xp).max(axis=1)
    return pred, conf


def load_frozen_models():
    import joblib
    production, scaler = ml._load(ml.DEFAULT_MODEL)
    candidate = joblib.load(candidate2_file())
    return (production, scaler), (candidate, None)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def metrics(truth, pred, conf=None) -> dict:
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, classification_report, confusion_matrix)
    truth = np.array([str(t) for t in truth]); pred = np.array([str(p) for p in pred])
    labels = sorted(set(truth) | set(pred))
    rep = classification_report(truth, pred, labels=labels, output_dict=True, zero_division=0)
    per_class = {c: {"precision": rep[c]["precision"], "recall": rep[c]["recall"],
                     "f1": rep[c]["f1-score"], "support": int(rep[c]["support"])}
                 for c in labels if c in rep}
    cm = confusion_matrix(truth, pred, labels=labels)
    out = {
        "n": int(len(truth)),
        "accuracy": float(accuracy_score(truth, pred)),
        "macro_precision": float(precision_score(truth, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(truth, pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, pred, average="weighted", zero_division=0)),
        "benign_precision": per_class.get(BENIGN, {}).get("precision"),
        "benign_recall": per_class.get(BENIGN, {}).get("recall"),
        "ftp_precision": per_class.get(FTP, {}).get("precision"),
        "ftp_recall": per_class.get(FTP, {}).get("recall"),
        "per_class": per_class,
        "confusion": {"labels": labels, "matrix": cm.tolist()},
        "prediction_distribution": {k: int(v) for k, v in pd.Series(pred).value_counts().items()},
    }
    # benign false-positive rate = fraction of Benign flows predicted non-Benign
    benign_mask = truth == BENIGN
    out["benign_fp_rate"] = float((pred[benign_mask] != BENIGN).mean()) if benign_mask.any() else None
    if conf is not None and len(conf):
        out["confidence"] = {
            "mean": float(np.mean(conf)), "median": float(np.median(conf)),
            "p10": float(np.percentile(conf, 10)), "p90": float(np.percentile(conf, 90)),
            "min": float(np.min(conf)), "max": float(np.max(conf)),
        }
    return out


def per_capture(model, scaler, df: pd.DataFrame) -> list:
    rows = []
    for cap in sorted(df["capture"].unique()):
        held = df[df["capture"] == cap]
        pred, conf = predict(model, scaler, held)
        truth = held["Label"].to_numpy()
        correct = int((pred == truth).sum())
        rows.append({
            "capture": cap, "label": held["Label"].iloc[0], "n_flows": int(len(held)),
            "correct": correct, "incorrect": int(len(held) - correct),
            "recall": float((pred == truth).mean()),
            "predictions": json.dumps({k: int(v) for k, v in pd.Series(pred).value_counts().items()}),
            "confidence_mean": round(float(conf.mean()), 4),
            "confidence_max": round(float(conf.max()), 4),
            "confidence_min": round(float(conf.min()), 4),
        })
    return rows


# ---------------------------------------------------------------------------
# Capture-level bootstrap CIs (honest about small n)
# ---------------------------------------------------------------------------


def bootstrap_cis(df, model, scaler, n_boot=2000, seed=42) -> dict:
    """
    Capture-level bootstrap: resample captures with replacement, pool their flows,
    recompute metrics. Percentile 95% CIs. Respects within-capture correlation.
    """
    rng = np.random.default_rng(seed)
    caps = sorted(df["capture"].unique())
    by_cap = {c: df[df["capture"] == c] for c in caps}
    acc, ftp_rec, ben_rec = [], [], []
    for _ in range(n_boot):
        pick = rng.choice(caps, size=len(caps), replace=True)
        sample = pd.concat([by_cap[c] for c in pick], ignore_index=True)
        pred, _ = predict(model, scaler, sample)
        truth = sample["Label"].to_numpy()
        acc.append(float((pred == truth).mean()))
        fm = truth == FTP; bm = truth == BENIGN
        ftp_rec.append(float((pred[fm] == FTP).mean()) if fm.any() else np.nan)
        ben_rec.append(float((pred[bm] == BENIGN).mean()) if bm.any() else np.nan)

    def ci(arr):
        a = np.array(arr); a = a[~np.isnan(a)]
        return {"mean": float(a.mean()), "lo95": float(np.percentile(a, 2.5)),
                "hi95": float(np.percentile(a, 97.5))}
    return {"n_captures": len(caps), "n_flows": int(len(df)), "n_boot": n_boot,
            "accuracy": ci(acc), "ftp_recall": ci(ftp_rec), "benign_recall": ci(ben_rec)}


# ---------------------------------------------------------------------------
# Confidence analysis
# ---------------------------------------------------------------------------


def confidence_analysis(df, model, scaler, hi=0.90, lo=0.60) -> dict:
    pred, conf = predict(model, scaler, df)
    truth = df["Label"].to_numpy()
    wrong = pred != truth
    return {
        "n": int(len(df)),
        "confidently_wrong": int(((wrong) & (conf >= hi)).sum()),
        "confidently_wrong_ftp_as_benign": int(((truth == FTP) & (pred == BENIGN) & (conf >= hi)).sum()),
        "confidently_wrong_benign_as_ftp": int(((truth == BENIGN) & (pred == FTP) & (conf >= hi)).sum()),
        "low_confidence_lt_%.2f" % lo: int((conf < lo).sum()),
        "mean_conf_correct": float(conf[~wrong].mean()) if (~wrong).any() else None,
        "mean_conf_wrong": float(conf[wrong].mean()) if wrong.any() else None,
        "hi_threshold": hi, "low_threshold": lo,
    }
