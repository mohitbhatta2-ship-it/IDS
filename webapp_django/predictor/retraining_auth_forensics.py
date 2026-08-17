"""
Auth-forensics retraining helpers (candidate only; production frozen).

Feature set builds on the per-connection candidate (FEATURES_PC = 67: 30 packet + 13
behavioural + 11 per-connection + 13 cross-session) and ADDS the 10 authentication-forensics
features (edit-distance of failed passwords to the successful one, distinct-password counts,
inter-attempt timing):

    FEATURES_AF = FEATURES_PC (67) + FORENSIC_FEATURES (10) = 77

The forensic features target the one case per-connection/cross features cannot separate: a
benign user who mistypes (typos -- edit-distance-CLOSE to the correct password) then succeeds,
vs an attacker who fails-then-succeeds with DICTIONARY guesses (edit-distance-FAR). The 30
packet features and their order are preserved. CIC flows have NaN for all application-layer
features (never zero-filled; HGB handles NaN). Genuinely undefined forensic measurements
(e.g. distance-to-success when there was no success) remain NaN -- never zero-filled.
Reuses ``retraining_per_connection`` (rpc) machinery.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import ml, retraining as rt, retraining_v2 as r2, pcap_validation as pv, \
    live_capture, ftp_behavioral as fb, ftp_cross_session as fcs, ftp_per_connection as fpc, \
    ftp_auth_forensics as faf, retraining_per_connection as rpc

SEED = 42
FTP, BENIGN = "FTP-BruteForce", "Benign"
F43 = rpc.F43
# CIC NaN columns = ALL 15 behavioural + 11 per-connection + 13 cross-session + 10 forensic,
# so that any frozen model (C3 43 / cross 58 / per-connection 67 / auth-forensics 77) can be
# sliced from the same CIC frame.
APP_FEATURES = list(rpc.APP_FEATURES) + list(faf.FORENSIC_FEATURES)         # 39 + 10 = 49
FEATURES_AF = rpc.FEATURES_PC + list(faf.FORENSIC_FEATURES)                 # 67 + 10 = 77
# ablation sub-sets
FEATURES_NO_FORENSIC = rpc.FEATURES_PC                                      # per-connection candidate (67)
FEATURES_NO_EDITDIST = [f for f in FEATURES_AF if f not in (
    "ftpaf_min_editdist_fail_to_success", "ftpaf_mean_editdist_fail_to_success",
    "ftpaf_mean_editdist_consecutive")]                                    # forensic minus edit-distance


def repo_root() -> Path:
    return rpc.repo_root()


def candidate_dir() -> Path:
    return repo_root() / "validation" / "models" / "ftp-auth-forensics"


def _dmap():
    d = dict(rpc._dmap())
    d["auth_forensics"] = "auth_forensics_pcaps"
    d["independent_ftp_val4"] = "independent_ftp_validation4_pcaps"
    return d


def _extract_dir(root: Path, source: str, rows: list, invalid: list):
    for folder, label in (("benign", BENIGN), ("ftp_bruteforce", FTP)):
        d = root / folder
        if not d.is_dir():
            continue
        for pcap in sorted(d.glob("*.pcap")):
            behav = fb.behavioural_features_for_pcap(pcap)
            pc = fpc.per_connection_features_for_pcap(pcap)
            cross = fcs.cross_session_features_for_pcap(pcap)
            forensic = faf.forensic_features_for_pcap(pcap)
            for i, fl in enumerate(pv.replay_pcap(pcap)):
                if pv._feature_problem(fl["features"]):
                    invalid.append({"source": source, "capture": pcap.name}); continue
                row = {f: float(fl["features"][f]) for f in ml.FEATURES}
                # carry ALL 15 behavioural (so a frozen 45/58-feature model can also be sliced),
                # even though FEATURES_AF uses only the 13 (C3 drops failed_logins/ratio)
                row.update({bfeat: float(behav[bfeat]) for bfeat in fb.BEHAV_FEATURES})
                row.update({pf: float(pc[pf]) for pf in fpc.PC_FEATURES})
                row.update({xf: float(cross[xf]) for xf in fcs.CROSS_FEATURES})
                # forensic features may legitimately be NaN (undefined) -- never zero-filled
                row.update({af: float(forensic[af]) for af in faf.FORENSIC_FEATURES})
                row["Label"] = label; row["capture"] = pcap.name
                row["flow_uid"] = f"{source}:{pcap.name}#{i}"; row["source"] = source
                rows.append(row)


def extract_real_af(sources) -> r2.RealFlows:
    live_capture._ensure_live_on_path()
    base = repo_root() / "validation"; dmap = _dmap()
    rows, invalid = [], []
    for src in sources:
        _extract_dir(base / dmap[src], src, rows, invalid)
    return r2.RealFlows(df=pd.DataFrame(rows), invalid=invalid)


def load_cic_af():
    cic_X, cic_y = rt.load_cic(); aug = cic_X.copy()
    for f in APP_FEATURES:
        aug[f] = np.nan
    return aug[list(ml.FEATURES) + APP_FEATURES], cic_y


def load_cic_test_af():
    X, y = rt.load_cic_test(); aug = X.copy()
    for f in APP_FEATURES:
        aug[f] = np.nan
    return aug[list(ml.FEATURES) + APP_FEATURES], y


def stratified_cic_subsample_af(cic_Xaug, cic_y, n):
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
    (out / f"{name}.metadata.json").write_text(json.dumps({**metadata, "scaler": None, "features": FEATURES_AF}, indent=2, default=str))
    return {"model": str(out / f"{name}.pkl")}
