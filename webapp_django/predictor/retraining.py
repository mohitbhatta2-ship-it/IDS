"""
Realistic-PCAP retraining EXPERIMENT (candidate only — baseline never touched).

Question: does adding realistically captured FTP traffic to the CIC training
distribution improve the model's real-world FTP-BruteForce generalisation,
without regressing CIC performance?

Design principles (see docs/realistic-pcap-retraining.md):
  * The baseline saved model is frozen. Candidates are new HGB models trained
    with the SAME tuned hyperparameters (HistGradientBoosting_Tuned_best_params)
    and the SAME 30 ``ml.FEATURES`` — so the only variable is the training data.
    (A from-scratch retrain on CIC alone reproduces the baseline exactly, which
    makes this a clean A/B.)
  * Real flows come only from the existing validated Live-Capture extraction
    (`pcap_validation.replay_pcap`); no second feature implementation, no
    zero-filling — a flow missing any of the 30 finite features is reported and
    excluded, never silently trained on.
  * Ground truth is the PCAP folder label; the baseline's predictions are never
    used as labels.
  * Leakage control: evaluation is capture-level (leave-one-capture-out). Flows
    from one PCAP never appear in both train and test.

Nothing here imports into or changes ml.py / live_capture.py / pcap_validation.py
/ Dataset Testing / the web app, and nothing overwrites webapp_data/Results/Models.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import ml, pcap_validation as pv

SEED = 42
FTP = "FTP-BruteForce"
BENIGN = "Benign"

# Real FTP flows are ~31 against CIC's ~7,968 FTP rows, so unweighted they are
# numerically negligible. The weighted strategy upweights each real FTP flow so
# the real FTP mass in training ~ the CIC FTP class support. This is a documented,
# reproducible sample-weight (NOT row duplication).
FTP_UPWEIGHT = 250.0


def _name_to_encoded() -> dict[str, int]:
    return {v: k for k, v in ml.LABELS.items()}


def _encoded_to_name() -> dict[int, str]:
    return {int(k): v for k, v in ml.LABELS.items()}


def hgb_params() -> dict:
    """Tuned HGB hyperparameters, mapped to sklearn kwargs (hgb_ prefix stripped)."""
    p = json.loads((ml.MODELS_DIR / "HistGradientBoosting_Tuned_best_params.json").read_text())
    return {k[4:]: v for k, v in p.items() if k.startswith("hgb_")}


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "sample_data" / "real_pcap").is_dir():
            return parent
    raise RuntimeError("repo root with sample_data/real_pcap not found")


# ---------------------------------------------------------------------------
# Real flows — reuse the validated extraction; no zero-fill, capture-tagged
# ---------------------------------------------------------------------------


@dataclass
class RealFlows:
    df: pd.DataFrame                       # ml.FEATURES + Label + capture
    invalid: list = field(default_factory=list)

    @property
    def captures(self) -> list[str]:
        return sorted(self.df["capture"].unique())


def extract_real_flows() -> RealFlows:
    """Every real flow with its 30 finite features, ground-truth label and capture id."""
    root = repo_root() / "sample_data" / "real_pcap"
    rows, invalid = [], []
    for pcap, label in pv.iter_labelled_pcaps(root):
        for fl in pv.replay_pcap(pcap):
            problem = pv._feature_problem(fl["features"])
            if problem:
                invalid.append({"capture": pcap.name, "reason": problem})
                continue
            row = {f: float(fl["features"][f]) for f in ml.FEATURES}
            row["Label"] = label
            row["capture"] = pcap.name
            rows.append(row)
    df = pd.DataFrame(rows)
    return RealFlows(df=df, invalid=invalid)


def load_cic() -> tuple[pd.DataFrame, pd.Series]:
    train = pd.read_parquet(ml.DATA_ROOT / "Processed_Data" / "balanced_train_selected.parquet")
    return train[ml.FEATURES].reset_index(drop=True), train["Label"].reset_index(drop=True)


def load_cic_test() -> tuple[pd.DataFrame, pd.Series]:
    test = pd.read_parquet(ml.DATA_ROOT / "Processed_Data" / "test_selected.parquet")
    return test[ml.FEATURES].reset_index(drop=True), test["Label"].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def assemble_training(cic_X, cic_y, real_df, exclude_captures=(), ftp_weight=1.0):
    """
    Training set = CIC + real flows (minus excluded captures), with sample weights.
    Returns X, y (encoded), sample_weight, and a manifest of what went in.
    """
    enc = _name_to_encoded()
    real = real_df[~real_df["capture"].isin(exclude_captures)].copy()

    X = pd.concat([cic_X, real[ml.FEATURES]], ignore_index=True)
    y = pd.concat([cic_y, real["Label"].map(enc)], ignore_index=True).astype(int)

    w = np.ones(len(X), dtype=float)
    # real FTP rows sit at the tail; upweight only those.
    real_is_ftp = (real["Label"] == FTP).to_numpy()
    tail = np.zeros(len(X), dtype=bool)
    tail[len(cic_X):] = True
    ftp_tail = tail.copy()
    ftp_tail[len(cic_X):] = real_is_ftp
    w[ftp_tail] = ftp_weight

    manifest = {
        "cic_rows": int(len(cic_X)),
        "real_rows": int(len(real)),
        "real_ftp_rows": int(real_is_ftp.sum()),
        "real_benign_rows": int((~real_is_ftp & (real["Label"] == BENIGN).to_numpy()).sum()),
        "excluded_captures": list(exclude_captures),
        "included_captures": sorted(real["capture"].unique()),
        "ftp_weight": ftp_weight,
        "real_ftp_total_weight": float(w[ftp_tail].sum()),
    }
    return X, y, w, manifest


def train(X, y, sample_weight):
    from sklearn.ensemble import HistGradientBoostingClassifier
    model = HistGradientBoostingClassifier(**hgb_params(), random_state=SEED)
    model.fit(X, y, sample_weight=sample_weight)
    return model


# ---------------------------------------------------------------------------
# Evaluation (decoded to class names; full metric set)
# ---------------------------------------------------------------------------


def evaluate(model, X, y_true_names) -> dict:
    """Full metrics for a model on X with ground-truth class NAMES."""
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, classification_report, confusion_matrix)
    dec = _encoded_to_name()
    pred = np.array([dec.get(int(c), str(c)) for c in model.predict(X)])
    truth = np.array([str(t) for t in y_true_names])
    proba = model.predict_proba(X)
    conf = proba.max(axis=1)

    labels = sorted(set(truth) | set(pred))
    rep = classification_report(truth, pred, labels=labels, output_dict=True, zero_division=0)
    per_class = [{
        "class": c, "precision": rep[c]["precision"], "recall": rep[c]["recall"],
        "f1": rep[c]["f1-score"], "support": int(rep[c]["support"]),
    } for c in labels if c in rep and rep[c]["support"] > 0]

    cm = confusion_matrix(truth, pred, labels=labels)
    return {
        "n": int(len(truth)),
        "accuracy": accuracy_score(truth, pred),
        "macro_precision": precision_score(truth, pred, average="macro", zero_division=0),
        "macro_recall": recall_score(truth, pred, average="macro", zero_division=0),
        "macro_f1": f1_score(truth, pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(truth, pred, average="weighted", zero_division=0),
        "ftp_recall": recall_score(truth, pred, labels=[FTP], average="micro", zero_division=0)
                      if FTP in truth else None,
        "benign_recall": recall_score(truth, pred, labels=[BENIGN], average="micro", zero_division=0)
                         if BENIGN in truth else None,
        "per_class": per_class,
        "confusion": {"labels": labels, "matrix": cm.tolist()},
        "prediction_distribution": {k: int(v) for k, v in pd.Series(pred).value_counts().items()},
        "confidence": {"mean": float(conf.mean()), "median": float(np.median(conf))},
    }


# ---------------------------------------------------------------------------
# Leave-one-capture-out — the uncontaminated real-PCAP estimate
# ---------------------------------------------------------------------------


def leave_one_capture_out(cic_X, cic_y, real: RealFlows, ftp_weight=1.0) -> dict:
    """
    For each real capture: train on CIC + all OTHER real captures, test on the
    held-out capture. No PCAP appears in both train and test. Returns per-fold
    results and the pooled held-out prediction set.
    """
    folds = []
    pooled_truth, pooled_pred = [], []
    dec = _encoded_to_name()
    for cap in real.captures:
        held = real.df[real.df["capture"] == cap]
        X, y, w, manifest = assemble_training(cic_X, cic_y, real.df,
                                              exclude_captures=(cap,), ftp_weight=ftp_weight)
        model = train(X, y, w)
        pred = np.array([dec.get(int(c), str(c)) for c in model.predict(held[ml.FEATURES])])
        truth = held["Label"].to_numpy()
        pooled_truth += list(truth); pooled_pred += list(pred)
        folds.append({
            "held_out_capture": cap,
            "true_label": held["Label"].iloc[0],
            "n_flows": int(len(held)),
            "correct": int((pred == truth).sum()),
            "recall": float((pred == truth).mean()),
            "train_manifest": manifest,
            "predictions": {k: int(v) for k, v in pd.Series(pred).value_counts().items()},
        })
    pooled = _pooled_metrics(pooled_truth, pooled_pred)
    return {"ftp_weight": ftp_weight, "folds": folds, "pooled": pooled}


def _pooled_metrics(truth, pred) -> dict:
    truth = np.array(truth); pred = np.array(pred)
    out = {"n": int(len(truth)), "accuracy": float((truth == pred).mean())}
    for cls in (FTP, BENIGN):
        mask = truth == cls
        out[f"{cls}_support"] = int(mask.sum())
        out[f"{cls}_recall"] = float((pred[mask] == cls).mean()) if mask.any() else None
    return out


# ---------------------------------------------------------------------------
# Saving the candidate artifact (separate path; never overwrites production)
# ---------------------------------------------------------------------------


def candidate_dir() -> Path:
    return repo_root() / "validation" / "models" / "realistic_pcap_candidate"


def save_candidate(model, metadata: dict) -> dict:
    import joblib
    out = candidate_dir()
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "model.pkl")
    # The HGB pipeline uses no scaler (MODEL_REGISTRY scaled=False), so no
    # scaler.pkl is written; metadata records scaler=None explicitly.
    metadata = {**metadata, "scaler": None}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    return {"model": str(out / "model.pkl"), "metadata": str(out / "metadata.json")}
