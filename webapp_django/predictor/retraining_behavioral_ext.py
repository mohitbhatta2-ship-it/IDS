"""
Extended-behavioural retraining helpers (candidate only; production frozen).

Feature set = the 30 ``ml.FEATURES`` + the existing 15 ``ftp_behavioral.BEHAV_FEATURES``
+ the NEW ``ftp_behavioral_ext.EXT_FEATURES`` (57 total). The 30 packet features and 15
behavioural features are PRESERVED unchanged; EXT only adds dynamics/timing/variation.
CIC flows have no PCAP, so their 15+12 application-layer features are NaN (genuinely
missing -- HGB handles NaN; never zero-filled).

Reuses ``retraining_behavioral`` (rbh) for CIC loading / weighting / LOCO plumbing.
Candidate artifacts go under ``validation/models/`` -- never webapp_data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import ml, retraining as rt, retraining_v2 as r2, pcap_validation as pv, \
    live_capture, ftp_behavioral as fb, ftp_behavioral_ext as fbx, retraining_behavioral as rbh

SEED = 42
FTP, BENIGN = "FTP-BruteForce", "Benign"
APP_FEATURES = list(fb.BEHAV_FEATURES) + list(fbx.EXT_FEATURES)          # 15 + 12
FEATURES_AUG_EXT = list(ml.FEATURES) + APP_FEATURES                      # 30 + 27 = 57


def repo_root() -> Path:
    return rbh.repo_root()


def candidate_dir() -> Path:
    return repo_root() / "validation" / "models" / "ftp-behavioral-ext"


def _extract_dir(root: Path, source: str, rows: list, invalid: list):
    for folder, label in (("benign", BENIGN), ("ftp_bruteforce", FTP)):
        d = root / folder
        if not d.is_dir():
            continue
        for pcap in sorted(d.glob("*.pcap")):
            behav = fb.behavioural_features_for_pcap(pcap)
            ext = fbx.ext_features_for_pcap(pcap)
            for i, fl in enumerate(pv.replay_pcap(pcap)):
                problem = pv._feature_problem(fl["features"])
                if problem:
                    invalid.append({"source": source, "capture": pcap.name, "reason": problem})
                    continue
                row = {f: float(fl["features"][f]) for f in ml.FEATURES}
                row.update({bf: float(behav[bf]) for bf in fb.BEHAV_FEATURES})
                row.update({xf: float(ext[xf]) for xf in fbx.EXT_FEATURES})   # may be NaN (kept)
                row["Label"] = label
                row["capture"] = pcap.name
                row["flow_uid"] = f"{source}:{pcap.name}#{i}"
                row["source"] = source
                rows.append(row)


def extract_real_ext(sources) -> r2.RealFlows:
    live_capture._ensure_live_on_path()
    base = repo_root() / "validation"
    dmap = {"v1": "realistic_pcaps", "v2": "realistic_pcaps_v2", "targeted": "targeted_benign_pcaps",
            "independent": "independent_real_pcaps", "robustness": "robustness_pcaps",
            "robust_train": "robust_train_pcaps",
            "benign_failed_login": "benign_failed_login_pcaps",
            "independent_ftp_val": "independent_ftp_validation_pcaps"}
    rows, invalid = [], []
    for src in sources:
        _extract_dir(base / dmap[src], src, rows, invalid)
    return r2.RealFlows(df=pd.DataFrame(rows), invalid=invalid)


def load_cic_ext():
    cic_X, cic_y = rt.load_cic()
    aug = cic_X.copy()
    for f in APP_FEATURES:
        aug[f] = np.nan
    return aug[FEATURES_AUG_EXT], cic_y


def load_cic_test_ext():
    X, y = rt.load_cic_test()
    aug = X.copy()
    for f in APP_FEATURES:
        aug[f] = np.nan
    return aug[FEATURES_AUG_EXT], y


def stratified_cic_subsample_ext(cic_Xaug, cic_y, n):
    sub_X30, sub_y = r2.stratified_cic_subsample(cic_Xaug[list(ml.FEATURES)], cic_y, n)
    aug = sub_X30.copy()
    for f in APP_FEATURES:
        aug[f] = np.nan
    return aug[FEATURES_AUG_EXT], sub_y


def assemble(cic_Xaug, cic_y, real_df, exclude_captures=(), ftp_weight=1.0, benign_weight=1.0):
    enc = rt._name_to_encoded()
    real = real_df[~real_df["capture"].isin(exclude_captures)].copy()
    X = pd.concat([cic_Xaug[FEATURES_AUG_EXT], real[FEATURES_AUG_EXT]], ignore_index=True)
    y = pd.concat([cic_y.reset_index(drop=True), real["Label"].map(enc)], ignore_index=True).astype(int)
    w = np.ones(len(X), dtype=float)
    is_ftp = np.zeros(len(X), dtype=bool); is_ftp[len(cic_Xaug):] = (real["Label"] == FTP).to_numpy()
    is_ben = np.zeros(len(X), dtype=bool); is_ben[len(cic_Xaug):] = (real["Label"] == BENIGN).to_numpy()
    w[is_ftp] = ftp_weight; w[is_ben] = benign_weight
    manifest = {"cic_rows": int(len(cic_Xaug)), "real_rows": int(len(real)),
                "n_features": len(FEATURES_AUG_EXT), "sources": sorted(real["source"].unique().tolist())}
    return X, y, w, manifest


def train(X, y, sample_weight):
    from sklearn.ensemble import HistGradientBoostingClassifier
    model = HistGradientBoostingClassifier(**rt.hgb_params(), random_state=SEED)
    model.fit(X, y, sample_weight=sample_weight)
    return model


def leave_one_capture_out(cic_Xaug, cic_y, real: r2.RealFlows, held_captures, ftp_weight, benign_weight) -> dict:
    folds, pooled_truth, pooled_pred = [], [], []
    dec = rt._encoded_to_name()
    for cap in held_captures:
        held = real.df[real.df["capture"] == cap]
        X, y, w, _m = assemble(cic_Xaug, cic_y, real.df, exclude_captures=(cap,),
                               ftp_weight=ftp_weight, benign_weight=benign_weight)
        train_uids = set(real.df[~real.df["capture"].isin((cap,))]["flow_uid"])
        assert set(held["flow_uid"]).isdisjoint(train_uids), f"leak {cap}"
        model = train(X, y, w)
        pred = np.array([dec.get(int(c), str(c)) for c in model.predict(held[FEATURES_AUG_EXT])])
        truth = held["Label"].to_numpy()
        pooled_truth += list(truth); pooled_pred += list(pred)
        folds.append({"held_out_capture": cap, "true_label": held["Label"].iloc[0],
                      "n_flows": int(len(held)), "recall": float((pred == truth).mean())})
    return {"ftp_weight": ftp_weight, "benign_weight": benign_weight,
            "folds": folds, "pooled": r2._pooled_metrics(pooled_truth, pooled_pred)}


def save_candidate(model, metadata, name) -> dict:
    import json
    import joblib
    out = candidate_dir(); out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / f"{name}.pkl")
    (out / f"{name}.metadata.json").write_text(json.dumps({**metadata, "scaler": None,
                                               "features": FEATURES_AUG_EXT}, indent=2, default=str))
    return {"model": str(out / f"{name}.pkl")}
