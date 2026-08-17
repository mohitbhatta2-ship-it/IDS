"""
Cross-session retraining helpers (candidate only; production frozen).

Feature set = 30 ``ml.FEATURES`` + 15 ``ftp_behavioral.BEHAV_FEATURES`` + 13
``ftp_cross_session.CROSS_FEATURES`` (58 total). The 30 packet features and 15 behavioural
features are PRESERVED unchanged. CIC flows have no PCAP, so their 15+13 application-layer
features are NaN (genuinely missing; HGB handles NaN; never zero-filled).

Reuses ``retraining_behavioral`` (rbh) for CIC loading / weighting plumbing. Candidate
artifacts go under ``validation/models/`` -- never webapp_data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import ml, retraining as rt, retraining_v2 as r2, pcap_validation as pv, \
    live_capture, ftp_behavioral as fb, ftp_cross_session as fcs, retraining_behavioral as rbh

SEED = 42
FTP, BENIGN = "FTP-BruteForce", "Benign"
APP_FEATURES = list(fb.BEHAV_FEATURES) + list(fcs.CROSS_FEATURES)          # 15 + 13
FEATURES_AUG_CROSS = list(ml.FEATURES) + APP_FEATURES                      # 30 + 28 = 58


def repo_root() -> Path:
    return rbh.repo_root()


def candidate_dir() -> Path:
    return repo_root() / "validation" / "models" / "ftp-cross-session"


def _dmap():
    return {"v1": "realistic_pcaps", "v2": "realistic_pcaps_v2", "targeted": "targeted_benign_pcaps",
            "robust_train": "robust_train_pcaps", "benign_failed_login": "benign_failed_login_pcaps",
            "cross_session": "cross_session_pcaps", "independent_ftp_val": "independent_ftp_validation_pcaps"}


def _extract_dir(root: Path, source: str, rows: list, invalid: list):
    for folder, label in (("benign", BENIGN), ("ftp_bruteforce", FTP)):
        d = root / folder
        if not d.is_dir():
            continue
        for pcap in sorted(d.glob("*.pcap")):
            behav = fb.behavioural_features_for_pcap(pcap)
            cross = fcs.cross_session_features_for_pcap(pcap)
            for i, fl in enumerate(pv.replay_pcap(pcap)):
                if pv._feature_problem(fl["features"]):
                    invalid.append({"source": source, "capture": pcap.name}); continue
                row = {f: float(fl["features"][f]) for f in ml.FEATURES}
                row.update({bfeat: float(behav[bfeat]) for bfeat in fb.BEHAV_FEATURES})
                row.update({xf: float(cross[xf]) for xf in fcs.CROSS_FEATURES})     # may be NaN (kept)
                row["Label"] = label; row["capture"] = pcap.name
                row["flow_uid"] = f"{source}:{pcap.name}#{i}"; row["source"] = source
                rows.append(row)


def extract_real_cross(sources) -> r2.RealFlows:
    live_capture._ensure_live_on_path()
    base = repo_root() / "validation"; dmap = _dmap()
    rows, invalid = [], []
    for src in sources:
        _extract_dir(base / dmap[src], src, rows, invalid)
    return r2.RealFlows(df=pd.DataFrame(rows), invalid=invalid)


def load_cic_cross():
    cic_X, cic_y = rt.load_cic()
    aug = cic_X.copy()
    for f in APP_FEATURES:
        aug[f] = np.nan
    return aug[FEATURES_AUG_CROSS], cic_y


def load_cic_test_cross():
    X, y = rt.load_cic_test()
    aug = X.copy()
    for f in APP_FEATURES:
        aug[f] = np.nan
    return aug[FEATURES_AUG_CROSS], y


def stratified_cic_subsample_cross(cic_Xaug, cic_y, n):
    sub_X30, sub_y = r2.stratified_cic_subsample(cic_Xaug[list(ml.FEATURES)], cic_y, n)
    aug = sub_X30.copy()
    for f in APP_FEATURES:
        aug[f] = np.nan
    return aug[FEATURES_AUG_CROSS], sub_y


def train_cols(cic_Xaug, cic_y, real_df, cols, ftp_weight=1.0, benign_weight=1.0):
    enc = rt._name_to_encoded()
    X = pd.concat([cic_Xaug[cols], real_df[cols]], ignore_index=True)
    y = pd.concat([cic_y.reset_index(drop=True), real_df["Label"].map(enc)], ignore_index=True).astype(int)
    w = np.ones(len(X), dtype=float)
    w[len(cic_Xaug):] = np.where(real_df["Label"].to_numpy() == FTP, ftp_weight, benign_weight)
    return train(X, y, w)


def train(X, y, sample_weight):
    from sklearn.ensemble import HistGradientBoostingClassifier
    model = HistGradientBoostingClassifier(**rt.hgb_params(), random_state=SEED)
    model.fit(X, y, sample_weight=sample_weight)
    return model


def leave_one_capture_out_cols(cic_X, cic_y, real: r2.RealFlows, held_captures, cols) -> dict:
    enc = rt._name_to_encoded(); dec = rt._encoded_to_name()
    pooled_t, pooled_p = [], []
    for cap in held_captures:
        held = real.df[real.df["capture"] == cap]
        tr = real.df[real.df["capture"] != cap]
        X = pd.concat([cic_X[cols], tr[cols]], ignore_index=True)
        y = pd.concat([cic_y.reset_index(drop=True), tr["Label"].map(enc)], ignore_index=True).astype(int)
        model = train(X, y, np.ones(len(X)))
        pred = np.array([dec.get(int(c), str(c)) for c in model.predict(held[cols])])
        pooled_t += list(held["Label"].to_numpy()); pooled_p += list(pred)
    return r2._pooled_metrics(pooled_t, pooled_p)


def save_candidate(model, metadata, name) -> dict:
    import json
    import joblib
    out = candidate_dir(); out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / f"{name}.pkl")
    (out / f"{name}.metadata.json").write_text(json.dumps({**metadata, "scaler": None,
                                               "features": FEATURES_AUG_CROSS}, indent=2, default=str))
    return {"model": str(out / f"{name}.pkl")}
