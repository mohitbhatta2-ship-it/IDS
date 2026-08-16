"""
Balanced real-PCAP retraining EXPERIMENT (candidate only; production frozen).

Question: can a LARGER, class-BALANCED set of real FTP-BruteForce + real Benign
flows (v1 + v2 + targeted benign) added to CIC produce a model that detects real
FTP attacks while keeping benign false positives acceptable -- evaluated on the
FROZEN independent 36-PCAP test set?

Real training flows (approved corpora only):
  * FTP-BruteForce: v1 + v2 ftp_bruteforce captures.
  * Benign: v1 + v2 benign captures + the 41 targeted benign captures.
The independent 36-PCAP set is NEVER used for training/weighting/selection.

Candidates (same tuned HGB family/seed, same 30-feature `pcap_validation`
extraction, no threshold/heuristic change):
  1. CIC only control                 (must reproduce the baseline)
  2. CIC + real, unweighted
  3. CIC + real, moderate FTP weighting
  4. CIC + real, balanced benign/FTP weighting (equal real-class mass)

Selection uses CIC held-out + v2 LOCO only; the independent test is evaluated once,
after selection. Candidate artifacts go to
``validation/models/balanced-real-candidate/`` -- never webapp_data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import ml, retraining as rt, retraining_v2 as r2, retraining_targeted as tgt, \
    pcap_validation as pv, live_capture

SEED = 42
FTP = "FTP-BruteForce"
BENIGN = "Benign"


def repo_root() -> Path:
    return r2.repo_root()


def candidate_dir() -> Path:
    return repo_root() / "validation" / "models" / "balanced-real-candidate"


# ---------------------------------------------------------------------------
# Real training flows -- existing extraction, capture-tagged, source-tagged
# ---------------------------------------------------------------------------


def _extract_dir(root: Path, source: str, rows: list, invalid: list):
    for folder, label in (("benign", BENIGN), ("ftp_bruteforce", FTP)):
        d = root / folder
        if not d.is_dir():
            continue
        for pcap in sorted(d.glob("*.pcap")):
            for i, fl in enumerate(pv.replay_pcap(pcap)):
                problem = pv._feature_problem(fl["features"])
                if problem:
                    invalid.append({"source": source, "capture": pcap.name, "flow_index": i, "reason": problem})
                    continue
                row = {f: float(fl["features"][f]) for f in ml.FEATURES}
                row["Label"] = label
                row["capture"] = pcap.name
                row["flow_uid"] = f"{source}:{pcap.name}#{i}"
                row["source"] = source
                rows.append(row)


def combined_real_flows() -> r2.RealFlows:
    """v1 + v2 + targeted benign real flows (approved training corpora; NOT independent)."""
    live_capture._ensure_live_on_path()
    root = repo_root() / "validation"
    rows, invalid = [], []
    _extract_dir(root / "realistic_pcaps", "v1", rows, invalid)
    _extract_dir(root / "realistic_pcaps_v2", "v2", rows, invalid)
    _extract_dir(root / "targeted_benign_pcaps", "targeted", rows, invalid)
    return r2.RealFlows(df=pd.DataFrame(rows), invalid=invalid)


def v2_captures(real: r2.RealFlows) -> list[str]:
    return sorted(real.df[real.df["source"] == "v2"]["capture"].unique())


# ---------------------------------------------------------------------------
# Weighting strategies
# ---------------------------------------------------------------------------


def class_weights(real_df, strategy: str) -> tuple[float, float]:
    """Return (ftp_weight, benign_weight) for a strategy, given the real flows."""
    n_ftp = int((real_df["Label"] == FTP).sum())
    n_ben = int((real_df["Label"] == BENIGN).sum())
    if strategy == "unweighted":
        return 1.0, 1.0
    if strategy == "moderate_ftp":
        return 5.0, 1.0
    if strategy == "balanced":
        # equal real-class total mass: ftp_w*n_ftp == ben_w*n_ben. Anchor ftp_w=3
        # (a modest real boost vs CIC) and scale benign to match FTP mass.
        ftp_w = 3.0
        ben_w = ftp_w * (n_ftp / n_ben) if n_ben else ftp_w
        return ftp_w, ben_w
    raise ValueError(strategy)


STRATEGIES = ["unweighted", "moderate_ftp", "balanced"]


# ---------------------------------------------------------------------------
# Assemble + LOCO
# ---------------------------------------------------------------------------


def assemble(cic_X, cic_y, real_df, exclude_captures=(), ftp_weight=1.0, benign_weight=1.0):
    enc = rt._name_to_encoded()
    real = real_df[~real_df["capture"].isin(exclude_captures)].copy()
    X = pd.concat([cic_X, real[ml.FEATURES]], ignore_index=True)
    y = pd.concat([cic_y.reset_index(drop=True), real["Label"].map(enc)], ignore_index=True).astype(int)
    w = np.ones(len(X), dtype=float)
    tail = np.zeros(len(X), dtype=bool)
    tail[len(cic_X):] = True
    is_ftp = np.zeros(len(X), dtype=bool)
    is_ftp[len(cic_X):] = (real["Label"] == FTP).to_numpy()
    is_ben = np.zeros(len(X), dtype=bool)
    is_ben[len(cic_X):] = (real["Label"] == BENIGN).to_numpy()
    w[is_ftp] = ftp_weight
    w[is_ben] = benign_weight
    manifest = {
        "cic_rows": int(len(cic_X)), "real_rows": int(len(real)),
        "real_ftp_rows": int(is_ftp.sum()), "real_benign_rows": int(is_ben.sum()),
        "ftp_weight": ftp_weight, "benign_weight": benign_weight,
        "real_ftp_total_weight": float(w[is_ftp].sum()),
        "real_benign_total_weight": float(w[is_ben].sum()),
        "excluded_captures": list(exclude_captures),
        "sources": sorted(real["source"].unique().tolist()),
    }
    return X, y, w, manifest


train = rt.train
evaluate = rt.evaluate


def leave_one_capture_out(cic_X, cic_y, real: r2.RealFlows, held_captures, ftp_weight, benign_weight) -> dict:
    """LOCO over the given (v2) captures; v1 + targeted stay in training."""
    folds, pooled_truth, pooled_pred = [], [], []
    dec = rt._encoded_to_name()
    for cap in held_captures:
        held = real.df[real.df["capture"] == cap]
        X, y, w, manifest = assemble(cic_X, cic_y, real.df, exclude_captures=(cap,),
                                     ftp_weight=ftp_weight, benign_weight=benign_weight)
        train_uids = set(real.df[~real.df["capture"].isin((cap,))]["flow_uid"])
        assert set(held["flow_uid"]).isdisjoint(train_uids), f"leak {cap}"
        model = train(X, y, w)
        pred = np.array([dec.get(int(c), str(c)) for c in model.predict(held[ml.FEATURES])])
        truth = held["Label"].to_numpy()
        pooled_truth += list(truth); pooled_pred += list(pred)
        folds.append({"held_out_capture": cap, "true_label": held["Label"].iloc[0],
                      "n_flows": int(len(held)), "correct": int((pred == truth).sum()),
                      "recall": float((pred == truth).mean()),
                      "predictions": {k: int(v) for k, v in pd.Series(pred).value_counts().items()}})
    return {"ftp_weight": ftp_weight, "benign_weight": benign_weight,
            "folds": folds, "pooled": r2._pooled_metrics(pooled_truth, pooled_pred)}


def save_candidate(model, metadata, name) -> dict:
    import joblib
    out = candidate_dir()
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / f"{name}.pkl")
    (out / f"{name}.metadata.json").write_text(json.dumps({**metadata, "scaler": None}, indent=2, default=str))
    return {"model": str(out / f"{name}.pkl"), "metadata": str(out / f"{name}.metadata.json")}
