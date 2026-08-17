"""
Per-connection + cross-session retraining helpers (candidate only; production frozen).

Feature set builds on the C3 baseline (30 packet + 13 behavioural, i.e. the 15 minus
``ftp_failed_logins`` / ``ftp_failed_login_ratio``) and ADDS the 11 per-connection features
and the 13 cross-session features:

    FEATURES_PC = F43 (43) + PC_FEATURES (11) + CROSS_FEATURES (13) = 67

Per-connection features give the single-session attack signal; cross-session features give
multi-session context; together session count alone cannot decide the label. The 30 packet
features and their order are preserved. CIC flows have NaN for all application-layer
features (never zero-filled; HGB handles NaN). Reuses ``retraining_behavioral`` (rbh).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import ml, retraining as rt, retraining_v2 as r2, pcap_validation as pv, \
    live_capture, ftp_behavioral as fb, ftp_cross_session as fcs, ftp_per_connection as fpc, \
    retraining_behavioral as rbh

SEED = 42
FTP, BENIGN = "FTP-BruteForce", "Benign"
DROP = ("ftp_failed_logins", "ftp_failed_login_ratio")
F43 = [f for f in rbh.FEATURES_AUG if f not in DROP]                       # 30 packet + 13 behavioural
BEHAV_13 = [f for f in fb.BEHAV_FEATURES if f not in DROP]
# CIC NaN columns = ALL 15 behavioural + 11 per-connection + 13 cross-session, so that any
# frozen model (C3 43 / cross 58 / candidate 67) can be sliced from the same CIC frame.
APP_FEATURES = list(fb.BEHAV_FEATURES) + list(fpc.PC_FEATURES) + list(fcs.CROSS_FEATURES)  # 15 + 11 + 13
FEATURES_PC = F43 + list(fpc.PC_FEATURES) + list(fcs.CROSS_FEATURES)        # 67
# handy sub-sets for ablation
FEATURES_NO_CROSS = F43 + list(fpc.PC_FEATURES)                             # per-connection only (no cross)
FEATURES_NO_PC = F43 + list(fcs.CROSS_FEATURES)                            # cross only (no per-connection)


def repo_root() -> Path:
    return rbh.repo_root()


def candidate_dir() -> Path:
    return repo_root() / "validation" / "models" / "ftp-per-connection"


def _dmap():
    return {"v1": "realistic_pcaps", "v2": "realistic_pcaps_v2", "targeted": "targeted_benign_pcaps",
            "robust_train": "robust_train_pcaps", "benign_failed_login": "benign_failed_login_pcaps",
            "cross_session": "cross_session_pcaps", "per_connection": "per_connection_pcaps",
            "independent_ftp_val": "independent_ftp_validation_pcaps",
            "independent_ftp_val2": "independent_ftp_validation2_pcaps",
            "independent_ftp_val3": "independent_ftp_validation3_pcaps"}


def _extract_dir(root: Path, source: str, rows: list, invalid: list):
    for folder, label in (("benign", BENIGN), ("ftp_bruteforce", FTP)):
        d = root / folder
        if not d.is_dir():
            continue
        for pcap in sorted(d.glob("*.pcap")):
            behav = fb.behavioural_features_for_pcap(pcap)
            pc = fpc.per_connection_features_for_pcap(pcap)
            cross = fcs.cross_session_features_for_pcap(pcap)
            for i, fl in enumerate(pv.replay_pcap(pcap)):
                if pv._feature_problem(fl["features"]):
                    invalid.append({"source": source, "capture": pcap.name}); continue
                row = {f: float(fl["features"][f]) for f in ml.FEATURES}
                # carry ALL 15 behavioural (so a frozen 45/58-feature model can also be sliced),
                # even though FEATURES_PC uses only the 13 (C3 drops failed_logins/ratio)
                row.update({bfeat: float(behav[bfeat]) for bfeat in fb.BEHAV_FEATURES})
                row.update({pf: float(pc[pf]) for pf in fpc.PC_FEATURES})
                row.update({xf: float(cross[xf]) for xf in fcs.CROSS_FEATURES})
                row["Label"] = label; row["capture"] = pcap.name
                row["flow_uid"] = f"{source}:{pcap.name}#{i}"; row["source"] = source
                rows.append(row)


def extract_real_pc(sources) -> r2.RealFlows:
    live_capture._ensure_live_on_path()
    base = repo_root() / "validation"; dmap = _dmap()
    rows, invalid = [], []
    for src in sources:
        _extract_dir(base / dmap[src], src, rows, invalid)
    return r2.RealFlows(df=pd.DataFrame(rows), invalid=invalid)


def load_cic_pc():
    cic_X, cic_y = rt.load_cic(); aug = cic_X.copy()
    for f in APP_FEATURES:
        aug[f] = np.nan
    return aug[list(ml.FEATURES) + APP_FEATURES], cic_y


def load_cic_test_pc():
    X, y = rt.load_cic_test(); aug = X.copy()
    for f in APP_FEATURES:
        aug[f] = np.nan
    return aug[list(ml.FEATURES) + APP_FEATURES], y


def stratified_cic_subsample_pc(cic_Xaug, cic_y, n):
    sub_X30, sub_y = r2.stratified_cic_subsample(cic_Xaug[list(ml.FEATURES)], cic_y, n)
    aug = sub_X30.copy()
    for f in APP_FEATURES:
        aug[f] = np.nan
    return aug[list(ml.FEATURES) + APP_FEATURES], sub_y


def train(X, y, sample_weight):
    from sklearn.ensemble import HistGradientBoostingClassifier
    model = HistGradientBoostingClassifier(**rt.hgb_params(), random_state=SEED)
    model.fit(X, y, sample_weight=sample_weight)
    return model


def train_cols(cic_Xaug, cic_y, real_df, cols, ftp_weight=1.0, benign_weight=1.0):
    enc = rt._name_to_encoded()
    X = pd.concat([cic_Xaug[cols], real_df[cols]], ignore_index=True)
    y = pd.concat([cic_y.reset_index(drop=True), real_df["Label"].map(enc)], ignore_index=True).astype(int)
    w = np.ones(len(X), dtype=float)
    w[len(cic_Xaug):] = np.where(real_df["Label"].to_numpy() == FTP, ftp_weight, benign_weight)
    return train(X, y, w)


def leave_one_capture_out_cols(cic_X, cic_y, real: r2.RealFlows, held_captures, cols) -> dict:
    enc = rt._name_to_encoded(); dec = rt._encoded_to_name()
    pooled_t, pooled_p = [], []
    for cap in held_captures:
        held = real.df[real.df["capture"] == cap]; tr = real.df[real.df["capture"] != cap]
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
    (out / f"{name}.metadata.json").write_text(json.dumps({**metadata, "scaler": None, "features": FEATURES_PC}, indent=2, default=str))
    return {"model": str(out / f"{name}.pkl")}
