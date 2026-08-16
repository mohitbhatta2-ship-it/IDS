"""
Targeted-benign retraining EXPERIMENT (candidate only; production + Candidate 2 frozen).

Question: does adding the 41 TARGETED benign captures
(``validation/targeted_benign_pcaps/``) to Candidate 2's training recipe reduce
its benign false-positive rate on the independent test set, while retaining its
real FTP-detection improvement and not regressing CIC?

Recipe (only the added data / its weight changes vs Candidate 2):
  * Candidate 2  = CIC(full) + v2 real (all 83 captures), unweighted.
  * New candidate = CIC(full) + v2 real (weight 1) + targeted benign (weight w_b),
    for w_b in {1 (none), 5 (moderate), 15 (strong)} -- justified because the
    targeted benign is exactly the FP-prone region; upweighting it teaches the
    model to keep those flows Benign. Over-weighting risks FTP recall, so all three
    are compared (not blindly optimised).

Strict data separation:
  * Training may use CIC + v2 real + targeted benign ONLY.
  * The independent 36-PCAP test set is FROZEN; it never enters training, weighting,
    hyperparameter/threshold/model selection. It is used once, for final eval.
  * v2 leave-one-capture-out gives the real-PCAP validation estimate (holding out
    each v2 capture; targeted benign is always training augmentation).

Reuses the frozen 30-feature ``pcap_validation`` extraction and the same tuned HGB
family/seed as every prior experiment. Writes candidate artifacts to
``validation/models/targeted-benign-candidate/`` and nothing to production.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import ml, retraining as rt, retraining_v2 as r2, pcap_validation as pv, live_capture

SEED = 42
FTP = "FTP-BruteForce"
BENIGN = "Benign"

# targeted-benign sample weights (applied to the 41 targeted benign flows only).
# moderate x5: targeted benign mass (~215*5=1075) ~ v2 benign mass (183) + more;
# strong x15: dominates the benign real region without touching FTP rows.
BENIGN_WEIGHTS = {"none": 1.0, "moderate": 5.0, "strong": 15.0}

CIC_LOCO_SAMPLE = r2.CIC_LOCO_SAMPLE      # 30k, same as v2 for comparability


def repo_root() -> Path:
    return r2.repo_root()


def targeted_dir() -> Path:
    return repo_root() / "validation" / "targeted_benign_pcaps"


def candidate_dir() -> Path:
    return repo_root() / "validation" / "models" / "targeted-benign-candidate"


# ---------------------------------------------------------------------------
# Targeted benign flows -- existing extraction, capture-tagged, no zero-fill
# ---------------------------------------------------------------------------


def extract_targeted_benign_flows() -> r2.RealFlows:
    live_capture._ensure_live_on_path()
    rows, invalid = [], []
    for pcap in sorted((targeted_dir() / "benign").glob("*.pcap")):
        for i, fl in enumerate(pv.replay_pcap(pcap)):
            problem = pv._feature_problem(fl["features"])
            if problem:
                invalid.append({"capture": pcap.name, "flow_index": i, "reason": problem})
                continue
            row = {f: float(fl["features"][f]) for f in ml.FEATURES}
            row["Label"] = BENIGN
            row["capture"] = pcap.name
            row["flow_uid"] = f"{pcap.name}#{i}"
            rows.append(row)
    return r2.RealFlows(df=pd.DataFrame(rows), invalid=invalid)


# ---------------------------------------------------------------------------
# Assemble: CIC + v2 real (weight 1) + targeted benign (weight w_b)
# ---------------------------------------------------------------------------


def assemble(cic_X, cic_y, v2_df, tgt_df, exclude_v2_captures=(), benign_weight=1.0):
    enc = rt._name_to_encoded()
    v2 = v2_df[~v2_df["capture"].isin(exclude_v2_captures)].copy()
    X = pd.concat([cic_X, v2[ml.FEATURES], tgt_df[ml.FEATURES]], ignore_index=True)
    y = pd.concat([cic_y.reset_index(drop=True),
                   v2["Label"].map(enc), tgt_df["Label"].map(enc)], ignore_index=True).astype(int)
    w = np.ones(len(X), dtype=float)
    n_tgt = len(tgt_df)
    w[len(cic_X) + len(v2):] = benign_weight       # weight ONLY the targeted benign tail
    manifest = {
        "cic_rows": int(len(cic_X)), "v2_real_rows": int(len(v2)),
        "targeted_benign_rows": int(n_tgt),
        "v2_ftp_rows": int((v2["Label"] == FTP).sum()),
        "v2_benign_rows": int((v2["Label"] == BENIGN).sum()),
        "excluded_v2_captures": list(exclude_v2_captures),
        "benign_weight": benign_weight,
        "targeted_total_weight": float(w[len(cic_X) + len(v2):].sum()),
    }
    return X, y, w, manifest


train = rt.train
evaluate = rt.evaluate


# ---------------------------------------------------------------------------
# v2 leave-one-capture-out (targeted benign always in training)
# ---------------------------------------------------------------------------


def leave_one_capture_out(cic_X, cic_y, v2_real: r2.RealFlows, tgt_df, benign_weight=1.0) -> dict:
    folds, pooled_truth, pooled_pred = [], [], []
    dec = rt._encoded_to_name()
    for cap in v2_real.captures:
        held = v2_real.df[v2_real.df["capture"] == cap]
        X, y, w, manifest = assemble(cic_X, cic_y, v2_real.df, tgt_df,
                                     exclude_v2_captures=(cap,), benign_weight=benign_weight)
        assert cap not in manifest["excluded_v2_captures"] or cap in (cap,)
        # leakage: held-out capture flows must not be in the training frame
        train_uids = set(v2_real.df[~v2_real.df["capture"].isin((cap,))]["flow_uid"]) | set(tgt_df["flow_uid"])
        assert set(held["flow_uid"]).isdisjoint(train_uids), f"leak {cap}"
        model = train(X, y, w)
        pred = np.array([dec.get(int(c), str(c)) for c in model.predict(held[ml.FEATURES])])
        truth = held["Label"].to_numpy()
        pooled_truth += list(truth); pooled_pred += list(pred)
        folds.append({
            "held_out_capture": cap, "true_label": held["Label"].iloc[0],
            "n_flows": int(len(held)), "correct": int((pred == truth).sum()),
            "recall": float((pred == truth).mean()),
            "predictions": {k: int(v) for k, v in pd.Series(pred).value_counts().items()},
        })
    pooled = r2._pooled_metrics(pooled_truth, pooled_pred)
    return {"benign_weight": benign_weight, "folds": folds, "pooled": pooled}


def save_candidate(model, metadata, name) -> dict:
    import joblib
    out = candidate_dir()
    out.mkdir(parents=True, exist_ok=True)
    mp = out / f"{name}.pkl"
    joblib.dump(model, mp)
    (out / f"{name}.metadata.json").write_text(json.dumps({**metadata, "scaler": None}, indent=2, default=str))
    return {"model": str(mp), "metadata": str(out / f"{name}.metadata.json")}
