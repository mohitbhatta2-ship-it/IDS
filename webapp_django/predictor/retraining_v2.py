"""
Realistic-PCAP retraining EXPERIMENT v2 (candidate only; production frozen).

Question: does adding the 83 diversified real FTP/benign captures
(``validation/realistic_pcaps_v2/``) to the CIC training distribution improve the
model's real-world FTP-BruteForce generalisation, without regressing CIC?

Design (see docs/realistic-pcap-retraining-v2.md):
  * The production saved model is never touched. Candidates are new HGB models
    trained with the SAME tuned hyperparameters and the SAME 30 ``ml.FEATURES``
    (seed 42), so only the training data changes.
  * Real flows come ONLY from the existing ``pcap_validation`` extraction (no
    second feature implementation, no zero-fill); a flow missing any of the 30
    finite features is reported and excluded. Ground truth is the PCAP folder.
  * Leakage control: capture-level leave-one-capture-out (LOCO). Flows from one
    PCAP never appear in both train and test.
  * Compute: the saved candidates and the CIC-regression numbers use the FULL CIC
    train set (so CIC held-out is the honest headline). The 83-fold LOCO real-PCAP
    estimate uses a FIXED stratified CIC subsample (``CIC_LOCO_SAMPLE``, seed 42)
    for tractability, with a matched subsample-only control so the "did real data
    regress CIC on the same base" comparison is apples-to-apples. This caveat is
    documented, not hidden.

Nothing here imports into or changes ml.py / live_capture.py / pcap_validation.py
(used read-only) / Dataset Testing / the web app, and nothing overwrites
webapp_data/Results/Models or any frozen v1 result. Reuses the v1 ``retraining``
module's model/eval primitives without modifying them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import ml, retraining as rt, pcap_validation as pv, live_capture

SEED = 42
FTP = "FTP-BruteForce"
BENIGN = "Benign"

# real-flow sample weights (applied to ALL real rows, both classes).
#   none     -> real FTP mass (671) stays ~8% of CIC FTP (7968)
#   moderate -> x6  ~ 0.5x CIC FTP mass
#   strong   -> x24 ~ 2x  CIC FTP mass
WEIGHTS = {"none": 1.0, "moderate": 6.0, "strong": 24.0}

# fixed CIC subsample size for the (expensive) 83-fold LOCO folds
CIC_LOCO_SAMPLE = 30000


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "validation" / "realistic_pcaps_v2").is_dir():
            return parent
    raise RuntimeError("repo root with validation/realistic_pcaps_v2 not found")


def v2_capture_dir() -> Path:
    return repo_root() / "validation" / "realistic_pcaps_v2"


# ---------------------------------------------------------------------------
# Real flows -- existing extraction pipeline, capture-tagged, no zero-fill
# ---------------------------------------------------------------------------


@dataclass
class RealFlows:
    df: pd.DataFrame                 # ml.FEATURES + Label + capture + flow_uid
    invalid: list

    @property
    def captures(self) -> list[str]:
        return sorted(self.df["capture"].unique())


def extract_real_flows_v2() -> RealFlows:
    """Every v2 real flow with its 30 finite features, folder label and capture id."""
    live_capture._ensure_live_on_path()
    root = v2_capture_dir()
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
                row["flow_uid"] = f"{pcap.name}#{i}"     # every flow traces to its PCAP
                rows.append(row)
    return RealFlows(df=pd.DataFrame(rows), invalid=invalid)


def stratified_cic_subsample(cic_X, cic_y, n, seed=SEED):
    df = cic_X.copy()
    df["__y"] = cic_y.values
    frac = min(1.0, n / len(df))
    parts = [g.sample(max(1, int(round(len(g) * frac))), random_state=seed)
             for _, g in df.groupby("__y")]
    sub = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return sub[list(ml.FEATURES)].reset_index(drop=True), sub["__y"].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Assemble (weight ALL real rows) + train/evaluate reuse v1 primitives
# ---------------------------------------------------------------------------


def assemble_v2(cic_X, cic_y, real_df, exclude_captures=(), real_weight=1.0):
    enc = rt._name_to_encoded()
    real = real_df[~real_df["capture"].isin(exclude_captures)].copy()
    X = pd.concat([cic_X, real[ml.FEATURES]], ignore_index=True)
    y = pd.concat([cic_y.reset_index(drop=True), real["Label"].map(enc)], ignore_index=True).astype(int)
    w = np.ones(len(X), dtype=float)
    w[len(cic_X):] = real_weight                    # weight ALL real rows
    manifest = {
        "cic_rows": int(len(cic_X)), "real_rows": int(len(real)),
        "real_ftp_rows": int((real["Label"] == FTP).sum()),
        "real_benign_rows": int((real["Label"] == BENIGN).sum()),
        "excluded_captures": list(exclude_captures),
        "included_captures": sorted(real["capture"].unique()),
        "real_weight": real_weight,
        "real_total_weight": float(w[len(cic_X):].sum()),
    }
    return X, y, w, manifest


train = rt.train
evaluate = rt.evaluate


# ---------------------------------------------------------------------------
# Leave-one-capture-out (capture-level; no PCAP in both train and test)
# ---------------------------------------------------------------------------


def leave_one_capture_out_v2(cic_X, cic_y, real: RealFlows, real_weight=1.0) -> dict:
    folds = []
    pooled_truth, pooled_pred = [], []
    dec = rt._encoded_to_name()
    for cap in real.captures:
        held = real.df[real.df["capture"] == cap]
        X, y, w, manifest = assemble_v2(cic_X, cic_y, real.df,
                                        exclude_captures=(cap,), real_weight=real_weight)
        # leakage assertion: held-out capture must not be in training captures
        assert cap not in manifest["included_captures"], f"leak: {cap} in train"
        model = train(X, y, w)
        pred = np.array([dec.get(int(c), str(c)) for c in model.predict(held[ml.FEATURES])])
        truth = held["Label"].to_numpy()
        proba = model.predict_proba(held[ml.FEATURES]).max(axis=1)
        pooled_truth += list(truth); pooled_pred += list(pred)
        held_flow_uids = set(held["flow_uid"])
        train_flow_uids = set(real.df[~real.df["capture"].isin((cap,))]["flow_uid"])
        folds.append({
            "held_out_capture": cap, "true_label": held["Label"].iloc[0],
            "n_flows": int(len(held)), "correct": int((pred == truth).sum()),
            "recall": float((pred == truth).mean()),
            "predictions": {k: int(v) for k, v in pd.Series(pred).value_counts().items()},
            "confidence_mean": float(proba.mean()),
            "n_train_captures": len(manifest["included_captures"]),
            # leakage guarantees, recorded per fold:
            "held_out_in_train_captures": cap in manifest["included_captures"],
            "train_test_flow_overlap": len(held_flow_uids & train_flow_uids),
        })
    pooled = _pooled_metrics(pooled_truth, pooled_pred)
    return {"real_weight": real_weight, "folds": folds, "pooled": pooled}


def _pooled_metrics(truth, pred) -> dict:
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                 recall_score, confusion_matrix)
    truth = np.array(truth); pred = np.array(pred)
    labels = sorted(set(truth) | set(pred))
    out = {
        "n": int(len(truth)),
        "accuracy": float(accuracy_score(truth, pred)),
        "macro_precision": float(precision_score(truth, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(truth, pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, pred, average="weighted", zero_division=0)),
        "confusion": {"labels": labels, "matrix": confusion_matrix(truth, pred, labels=labels).tolist()},
    }
    for cls in (FTP, BENIGN):
        mask = truth == cls
        out[f"{cls}_support"] = int(mask.sum())
        out[f"{cls}_recall"] = float((pred[mask] == cls).mean()) if mask.any() else None
    return out


# ---------------------------------------------------------------------------
# Candidate artifacts -- separate path, never overwrites production
# ---------------------------------------------------------------------------


def candidate_dir_v2() -> Path:
    return repo_root() / "validation" / "models" / "realistic_pcap_candidate_v2"


def save_candidate_v2(model, metadata: dict, name: str) -> dict:
    import joblib
    out = candidate_dir_v2()
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / f"{name}.pkl"
    meta_path = out / f"{name}.metadata.json"
    joblib.dump(model, model_path)
    metadata = {**metadata, "scaler": None}
    meta_path.write_text(json.dumps(metadata, indent=2, default=str))
    return {"model": str(model_path), "metadata": str(meta_path)}
