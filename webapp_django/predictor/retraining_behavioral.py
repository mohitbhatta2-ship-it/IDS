"""
FTP-behavioural-feature retraining EXPERIMENT (candidate only; production frozen).

Tests whether AUGMENTING the existing 30 CIC features with the experimental FTP
control-channel behavioural features (``ftp_behavioral.BEHAV_FEATURES``) lets a
model break the FTP-recall vs benign-FP trade-off that re-weighting the 30 features
alone could not.

Feature set = the 30 ``ml.FEATURES`` (from the unchanged ``pcap_validation``
pipeline) + 15 behavioural features (from the PCAP's FTP control channel), attached
per-capture. CIC flows have no PCAP, so their behavioural features are **NaN**
(genuinely not computable -- HGB handles NaN natively; this is not zero-fill).
Real flows carry their capture's measured behaviour.

Real training corpora: v1 + v2 + targeted benign. The independent 36-PCAP set is
never used for training/feature-selection/tuning (hash-guarded elsewhere). Capture
level LOCO. Candidate artifacts go to
``validation/models/ftp-behavioral-candidate/`` -- never webapp_data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import ml, retraining as rt, retraining_v2 as r2, pcap_validation as pv, \
    live_capture, ftp_behavioral as fb

SEED = 42
FTP = "FTP-BruteForce"
BENIGN = "Benign"

FEATURES_AUG = list(ml.FEATURES) + list(fb.BEHAV_FEATURES)     # 30 + 15


def repo_root() -> Path:
    return r2.repo_root()


def candidate_dir() -> Path:
    return repo_root() / "validation" / "models" / "ftp-behavioral-candidate"


# ---------------------------------------------------------------------------
# Real flows: 30 pipeline features + per-capture behavioural features
# ---------------------------------------------------------------------------


def _extract_dir(root: Path, source: str, rows: list, invalid: list):
    for folder, label in (("benign", BENIGN), ("ftp_bruteforce", FTP)):
        d = root / folder
        if not d.is_dir():
            continue
        for pcap in sorted(d.glob("*.pcap")):
            behav = fb.behavioural_features_for_pcap(pcap)     # per capture
            for i, fl in enumerate(pv.replay_pcap(pcap)):
                problem = pv._feature_problem(fl["features"])
                if problem:
                    invalid.append({"source": source, "capture": pcap.name, "reason": problem})
                    continue
                row = {f: float(fl["features"][f]) for f in ml.FEATURES}
                row.update({bf: float(behav[bf]) for bf in fb.BEHAV_FEATURES})
                row["Label"] = label
                row["capture"] = pcap.name
                row["flow_uid"] = f"{source}:{pcap.name}#{i}"
                row["source"] = source
                rows.append(row)


def extract_real_augmented(sources=("v1", "v2", "targeted")) -> r2.RealFlows:
    live_capture._ensure_live_on_path()
    base = repo_root() / "validation"
    dmap = {"v1": "realistic_pcaps", "v2": "realistic_pcaps_v2",
            "targeted": "targeted_benign_pcaps", "independent": "independent_real_pcaps",
            "robustness": "robustness_pcaps", "robust_train": "robust_train_pcaps"}
    rows, invalid = [], []
    for src in sources:
        _extract_dir(base / dmap[src], src, rows, invalid)
    return r2.RealFlows(df=pd.DataFrame(rows), invalid=invalid)


def load_cic_augmented():
    """CIC 30 features with NaN behavioural columns (no PCAP -> genuinely missing)."""
    cic_X, cic_y = rt.load_cic()
    aug = cic_X.copy()
    for bf in fb.BEHAV_FEATURES:
        aug[bf] = np.nan
    return aug[FEATURES_AUG], cic_y


def load_cic_test_augmented():
    X, y = rt.load_cic_test()
    aug = X.copy()
    for bf in fb.BEHAV_FEATURES:
        aug[bf] = np.nan
    return aug[FEATURES_AUG], y


def stratified_cic_subsample_aug(cic_Xaug, cic_y, n):
    """Subsample CIC (via the 30 features) and re-attach NaN behavioural columns."""
    sub_X30, sub_y = r2.stratified_cic_subsample(cic_Xaug[list(ml.FEATURES)], cic_y, n)
    aug = sub_X30.copy()
    for bf in fb.BEHAV_FEATURES:
        aug[bf] = np.nan
    return aug[FEATURES_AUG], sub_y


def v2_captures(real: r2.RealFlows) -> list[str]:
    return sorted(real.df[real.df["source"] == "v2"]["capture"].unique())


# ---------------------------------------------------------------------------
# Weighting + assemble + LOCO (per-class weights, augmented feature set)
# ---------------------------------------------------------------------------


def class_weights(real_df, strategy: str) -> tuple[float, float]:
    n_ftp = int((real_df["Label"] == FTP).sum())
    n_ben = int((real_df["Label"] == BENIGN).sum())
    if strategy == "unweighted":
        return 1.0, 1.0
    if strategy == "balanced":
        ftp_w = 3.0
        return ftp_w, (ftp_w * (n_ftp / n_ben) if n_ben else ftp_w)
    raise ValueError(strategy)


STRATEGIES = ["unweighted", "balanced"]


def assemble(cic_Xaug, cic_y, real_df, exclude_captures=(), ftp_weight=1.0, benign_weight=1.0):
    enc = rt._name_to_encoded()
    real = real_df[~real_df["capture"].isin(exclude_captures)].copy()
    X = pd.concat([cic_Xaug[FEATURES_AUG], real[FEATURES_AUG]], ignore_index=True)
    y = pd.concat([cic_y.reset_index(drop=True), real["Label"].map(enc)], ignore_index=True).astype(int)
    w = np.ones(len(X), dtype=float)
    is_ftp = np.zeros(len(X), dtype=bool); is_ftp[len(cic_Xaug):] = (real["Label"] == FTP).to_numpy()
    is_ben = np.zeros(len(X), dtype=bool); is_ben[len(cic_Xaug):] = (real["Label"] == BENIGN).to_numpy()
    w[is_ftp] = ftp_weight; w[is_ben] = benign_weight
    manifest = {"cic_rows": int(len(cic_Xaug)), "real_rows": int(len(real)),
                "real_ftp_rows": int(is_ftp.sum()), "real_benign_rows": int(is_ben.sum()),
                "ftp_weight": ftp_weight, "benign_weight": benign_weight,
                "n_features": len(FEATURES_AUG), "sources": sorted(real["source"].unique().tolist())}
    return X, y, w, manifest


def train(X, y, sample_weight):
    from sklearn.ensemble import HistGradientBoostingClassifier
    model = HistGradientBoostingClassifier(**rt.hgb_params(), random_state=SEED)
    model.fit(X, y, sample_weight=sample_weight)          # HGB handles NaN natively
    return model


def evaluate(model, X, y_true_names):
    return rt.evaluate(model, X, y_true_names)


def leave_one_capture_out(cic_Xaug, cic_y, real: r2.RealFlows, held_captures, ftp_weight, benign_weight) -> dict:
    folds, pooled_truth, pooled_pred = [], [], []
    dec = rt._encoded_to_name()
    for cap in held_captures:
        held = real.df[real.df["capture"] == cap]
        X, y, w, manifest = assemble(cic_Xaug, cic_y, real.df, exclude_captures=(cap,),
                                     ftp_weight=ftp_weight, benign_weight=benign_weight)
        train_uids = set(real.df[~real.df["capture"].isin((cap,))]["flow_uid"])
        assert set(held["flow_uid"]).isdisjoint(train_uids), f"leak {cap}"
        model = train(X, y, w)
        pred = np.array([dec.get(int(c), str(c)) for c in model.predict(held[FEATURES_AUG])])
        truth = held["Label"].to_numpy()
        pooled_truth += list(truth); pooled_pred += list(pred)
        folds.append({"held_out_capture": cap, "true_label": held["Label"].iloc[0],
                      "n_flows": int(len(held)), "recall": float((pred == truth).mean()),
                      "predictions": {k: int(v) for k, v in pd.Series(pred).value_counts().items()}})
    return {"ftp_weight": ftp_weight, "benign_weight": benign_weight,
            "folds": folds, "pooled": r2._pooled_metrics(pooled_truth, pooled_pred)}


def predict_names(model, X_aug):
    dec = rt._encoded_to_name()
    return np.array([dec.get(int(c), str(c)) for c in model.predict(X_aug)]), model.predict_proba(X_aug).max(axis=1)


def save_candidate(model, metadata, name) -> dict:
    import joblib
    out = candidate_dir(); out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / f"{name}.pkl")
    (out / f"{name}.metadata.json").write_text(json.dumps({**metadata, "scaler": None,
                                               "features": FEATURES_AUG}, indent=2, default=str))
    return {"model": str(out / f"{name}.pkl")}
