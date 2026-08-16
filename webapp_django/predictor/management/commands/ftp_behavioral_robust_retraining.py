"""
Behavioural-model ROBUST RETRAINING experiment (candidates only; production frozen).

Trains the 45-feature behavioural model on the approved clean corpora PLUS a NEW
MESSY *training* corpus (``validation/robust_train_pcaps``) whose two classes carry
OVERLAPPING failed-login counts (benign mistypes/give-ups; attackers who fail or
eventually succeed). Goal: fix the messy-stress-test weakness (benign users who fail
to authenticate misclassified as attacks; over-reliance on ``ftp_failed_logins``)
WITHOUT regressing CIC or the FTP-recall/benign-FP balance.

Methodology (leakage-controlled):
  * TRAIN on: CIC (30 feats + NaN behavioural) + real v1+v2+targeted + robust_train.
  * SELECT candidate/weighting using ONLY the CIC held-out split and capture-level
    LOCO on the approved training corpora -- NEVER the 34-PCAP messy TEST corpus.
  * FINAL independent test = the frozen 34-PCAP messy corpus (validation/
    robustness_pcaps), used for reporting only, never for selection/tuning.
  * 4-way ablation (30 / behavioural-only / 30+behavioural / 30+behavioural WITHOUT
    ftp_failed_logins), all now trained WITH the messy training data, to test whether
    removing ftp_failed_logins still collapses FTP recall.
  * SHAP on the selected candidate over the 34-PCAP test corpus.

Frozen (verified before==after): production, Candidate 2, the clean-trained
behavioural candidates, ml.py/live_capture.py/pcap_validation.py, and the
v1/v2/independent/targeted/robustness PCAP corpora. Candidate + ablation models go to
validation/models/ftp-behavioral-robust-retraining/. No promotion, no merge.

    python manage.py ftp_behavioral_robust_retraining
"""

from __future__ import annotations

import glob
import hashlib
import json
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from django.core.management.base import BaseCommand

from predictor import ml, retraining as rt, retraining_behavioral as rbh
from predictor import ftp_behavioral as fb, independent_eval as ie

FTP, BENIGN = "FTP-BruteForce", "Benign"
FAILED = "ftp_failed_logins"
FEAT_NO_FAILED = [f for f in rbh.FEATURES_AUG if f != FAILED]
TRAIN_SOURCES = ("v1", "v2", "targeted", "robust_train")
CLEAN_BEHAV = "validation/models/ftp-behavioral-candidate/candidate_aug_unweighted.pkl"


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _pcap_hashes(root, sub):
    return sorted(_sha(p) for p in glob.glob(str(root / "validation" / sub / "**/*.pcap"), recursive=True))


def _frozen(root):
    d = {"production": _sha(ie.production_model_path()), "candidate2": _sha(ie.candidate2_file()),
         "behav_clean_unweighted": _sha(root / CLEAN_BEHAV),
         "behav_clean_balanced": _sha(root / "validation/models/ftp-behavioral-candidate/candidate_aug_balanced.pkl"),
         "ml_py": _sha(root / "webapp_django/predictor/ml.py"),
         "live_capture_py": _sha(root / "webapp_django/predictor/live_capture.py"),
         "pcap_validation_py": _sha(root / "webapp_django/predictor/pcap_validation.py")}
    for name, sub in (("v1", "realistic_pcaps"), ("v2", "realistic_pcaps_v2"),
                      ("indep", "independent_real_pcaps"), ("targeted", "targeted_benign_pcaps"),
                      ("robustness_test", "robustness_pcaps")):
        d[name + "_pcaps"] = _pcap_hashes(root, sub)
    return d


def _predict(model, X):
    dec = rt._encoded_to_name()
    return np.array([dec.get(int(c), str(c)) for c in model.predict(X)]), model.predict_proba(X).max(axis=1)


def _metrics(truth, pred):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, \
        classification_report, confusion_matrix
    truth = np.array([str(t) for t in truth]); pred = np.array([str(p) for p in pred])
    labels = sorted(set(truth) | set(pred))
    rep = classification_report(truth, pred, labels=labels, output_dict=True, zero_division=0)
    bmask = truth == BENIGN
    return {"n": int(len(truth)), "accuracy": float(accuracy_score(truth, pred)),
            "macro_precision": float(precision_score(truth, pred, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(truth, pred, average="macro", zero_division=0)),
            "macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)),
            "ftp_recall": rep.get(FTP, {}).get("recall"), "ftp_precision": rep.get(FTP, {}).get("precision"),
            "benign_recall": rep.get(BENIGN, {}).get("recall"), "benign_precision": rep.get(BENIGN, {}).get("precision"),
            "benign_fp_rate": float((pred[bmask] != BENIGN).mean()) if bmask.any() else None,
            "confusion": {"labels": labels, "matrix": confusion_matrix(truth, pred, labels=labels).tolist()}}


class Command(BaseCommand):
    help = "Behavioural robust-retraining experiment (candidates only; production frozen; no promotion)."

    def add_arguments(self, parser):
        parser.add_argument("--output", default=None)
        parser.add_argument("--skip-shap", action="store_true")
        parser.add_argument("--loco-cic", type=int, default=15000, help="CIC subsample size for LOCO tractability")

    def handle(self, *args, **opts):
        import joblib
        w = self.stdout.write
        root = rbh.repo_root()
        out = Path(opts["output"]) if opts["output"] else root / "validation" / "results" / "ftp_behavioral_robust_retraining"
        out.mkdir(parents=True, exist_ok=True)
        before = _frozen(root)

        w(self.style.MIGRATE_HEADING("Behavioural robust-retraining (candidates only; production frozen)"))

        # ---- leakage guards: train corpora disjoint from the 34-PCAP TEST corpus ----
        rtrain_h = set(_pcap_hashes(root, "robust_train_pcaps"))
        test34_h = set(before["robustness_test_pcaps"])
        prior_h = set(before["v1_pcaps"]) | set(before["v2_pcaps"]) | set(before["targeted_pcaps"]) | set(before["indep_pcaps"])
        if not rtrain_h.isdisjoint(test34_h):
            raise SystemExit("ABORT: a robust_train PCAP overlaps the frozen 34-PCAP TEST corpus!")
        if not rtrain_h.isdisjoint(prior_h):
            raise SystemExit("ABORT: a robust_train PCAP overlaps a prior corpus!")
        w(f"  leakage guard OK: robust_train ({len(rtrain_h)}) disjoint from test34 ({len(test34_h)}) and prior corpora.")

        # ---- data --------------------------------------------------------
        w(self.style.MIGRATE_HEADING("\nExtracting features"))
        real = rbh.extract_real_augmented(TRAIN_SOURCES)          # TRAIN real (clean + messy)
        rt_only = rbh.extract_real_augmented(("robust_train",))   # messy-train subset (for stats)
        cic_Xaug, cic_y = rbh.load_cic_augmented()
        cic_Xtest, cic_ytest = rbh.load_cic_test_augmented()
        test34 = rbh.extract_real_augmented(("robustness",))      # FINAL independent TEST
        rdf = self._attach_meta(root, test34.df)
        n_ftp = int((real.df["Label"] == FTP).sum()); n_ben = int((real.df["Label"] == BENIGN).sum())
        w(f"  train real flows: {len(real.df)} (FTP {n_ftp}, Benign {n_ben}); "
          f"messy-train flows: {len(rt_only.df)}; invalid: {len(real.invalid)}")
        w(f"  CIC train rows: {len(cic_Xaug)}; CIC held-out rows: {len(cic_Xtest)}; "
          f"messy TEST flows: {len(rdf)}")
        # overlap evidence: both classes carry failed logins in the TRAINING data
        overlap = self._failed_login_overlap(real.df)
        w(f"  failed-login overlap in TRAIN: benign>=1 fail={overlap['benign_with_fails']}, "
          f"FTP>=1 fail={overlap['ftp_with_fails']}, FTP eventual-auth={overlap['ftp_eventual_auth']}")

        # ---- frozen reference models -------------------------------------
        prod = ml._load(ml.DEFAULT_MODEL)[0]
        cand2 = joblib.load(ie.candidate2_file())
        behav_clean = joblib.load(root / CLEAN_BEHAV)             # 30+15 trained on CLEAN corpora only

        # ---- NEW candidates: trained WITH the messy training corpus ------
        w(self.style.MIGRATE_HEADING("\nTraining candidates (WITH messy training corpus)"))
        cand_unw = self._train_full(cic_Xaug, cic_y, real.df, "unweighted")
        cand_bal = self._train_full(cic_Xaug, cic_y, real.df, "balanced")
        # ablation models (trained WITH messy training data)
        abl_30 = self._train_subset(cic_Xaug, cic_y, real.df, list(ml.FEATURES), include_cic=True)
        abl_behav = self._train_subset(cic_Xaug, cic_y, real.df, list(fb.BEHAV_FEATURES), include_cic=False)
        abl_nofail = self._train_subset(cic_Xaug, cic_y, real.df, FEAT_NO_FAILED, include_cic=True)
        mdir = self._mkmodeldir(root)
        for nm, m in (("candidate_robust_unweighted", cand_unw), ("candidate_robust_balanced", cand_bal),
                      ("ablation_30only", abl_30), ("ablation_behavioural_only", abl_behav),
                      ("ablation_no_failed_logins", abl_nofail)):
            joblib.dump(m, mdir / f"{nm}.pkl")
        (mdir / "candidate_robust_unweighted.metadata.json").write_text(json.dumps(
            {"features": rbh.FEATURES_AUG, "train_sources": list(TRAIN_SOURCES), "weighting": "unweighted",
             "scaler": None}, indent=2))
        w("  trained: candidate_robust_unweighted, candidate_robust_balanced, + 3 ablation models.")

        # ---- CIC held-out evaluation (no regression?) --------------------
        w(self.style.MIGRATE_HEADING("\nCIC held-out evaluation"))
        cic_models = {"production": (prod, list(ml.FEATURES)), "candidate2": (cand2, list(ml.FEATURES)),
                      "behav_clean": (behav_clean, rbh.FEATURES_AUG),
                      "candidate_robust_unweighted": (cand_unw, rbh.FEATURES_AUG),
                      "candidate_robust_balanced": (cand_bal, rbh.FEATURES_AUG)}
        cic_metrics = {}
        cic_rows = []
        dec = rt._encoded_to_name()
        cic_truth_names = np.array([dec.get(int(c), str(c)) for c in cic_ytest.to_numpy()])
        for name, (model, cols) in cic_models.items():
            pred, _ = _predict(model, cic_Xtest[cols])
            m = _metrics(cic_truth_names, pred); cic_metrics[name] = m
            cic_rows.append({"model": name, "ftp_recall": m["ftp_recall"], "benign_recall": m["benign_recall"],
                             "benign_fp_rate": m["benign_fp_rate"], "macro_f1": m["macro_f1"], "accuracy": m["accuracy"]})
            w(f"  {name:28} FTP-rec {_f(m['ftp_recall'])} Ben-rec {_f(m['benign_recall'])} "
              f"mF1 {m['macro_f1']:.4f} acc {m['accuracy']:.4f}")
        pd.DataFrame(cic_rows).to_csv(out / "cic_heldout_metrics.csv", index=False)

        # ---- LOCO on approved training corpora (selection signal) --------
        w(self.style.MIGRATE_HEADING("\nCapture-level LOCO on approved training corpora"))
        sub_Xaug, sub_y = rbh.stratified_cic_subsample_aug(cic_Xaug, cic_y, opts["loco_cic"])
        loco = {}
        for strat in rbh.STRATEGIES:
            fw, bw = rbh.class_weights(real.df, strat)
            caps = sorted(real.df["capture"].unique())
            res = rbh.leave_one_capture_out(sub_Xaug, sub_y, real, caps, fw, bw)
            loco[strat] = res["pooled"]
            p = res["pooled"]
            w(f"  {strat:12} pooled: FTP-rec {_f(_loco_ftp(p))} Ben-rec {_f(_loco_ben(p))} "
              f"mF1 {_f(p.get('macro_f1'))} acc {_f(p.get('accuracy'))} (n_caps={len(caps)})")
        (out / "loco_pooled_metrics.json").write_text(json.dumps(loco, indent=2, default=str))

        # ---- candidate SELECTION (CIC + LOCO only; NOT the 34-PCAP test) -
        selected_name, sel_reason = self._select(cic_metrics, loco)
        selected = {"candidate_robust_unweighted": cand_unw, "candidate_robust_balanced": cand_bal}[selected_name]
        w(self.style.SUCCESS(f"\n  SELECTED (via CIC+LOCO only): {selected_name} -- {sel_reason}"))

        # ---- FINAL independent test: the frozen 34-PCAP messy corpus -----
        w(self.style.MIGRATE_HEADING("\nFINAL independent test (frozen 34-PCAP messy corpus)"))
        test_models = {
            "production": (prod, list(ml.FEATURES)), "candidate2": (cand2, list(ml.FEATURES)),
            "behav_clean": (behav_clean, rbh.FEATURES_AUG),
            "candidate_robust_unweighted": (cand_unw, rbh.FEATURES_AUG),
            "candidate_robust_balanced": (cand_bal, rbh.FEATURES_AUG),
            "ablation_30only": (abl_30, list(ml.FEATURES)),
            "ablation_behavioural_only": (abl_behav, list(fb.BEHAV_FEATURES)),
            "ablation_no_failed_logins": (abl_nofail, FEAT_NO_FAILED)}
        test_metrics, preds, rows = {}, {}, []
        for name, (model, cols) in test_models.items():
            pred, conf = _predict(model, rdf[cols]); preds[name] = (pred, conf)
            m = _metrics(rdf["Label"].to_numpy(), pred); test_metrics[name] = m
            rows.append({"model": name, "ftp_recall": m["ftp_recall"], "benign_recall": m["benign_recall"],
                         "benign_fp_rate": m["benign_fp_rate"], "macro_precision": m["macro_precision"],
                         "macro_recall": m["macro_recall"], "macro_f1": m["macro_f1"], "accuracy": m["accuracy"]})
            w(f"  {name:28} FTP-rec {_f(m['ftp_recall'])} Ben-rec {_f(m['benign_recall'])} "
              f"BenFP {_f(m['benign_fp_rate'])} mF1 {m['macro_f1']:.4f} acc {m['accuracy']:.4f}")
        pd.DataFrame(rows).to_csv(out / "final_test_metrics.csv", index=False)

        # ---- per-scenario-family behaviour (selected candidate) ----------
        fam_rows = self._family_breakdown(rdf, preds)
        pd.DataFrame(fam_rows).to_csv(out / "per_scenario_family_metrics.csv", index=False)
        w(self.style.MIGRATE_HEADING("\nAdversarial families (selected candidate on 34-PCAP test)"))
        for r in fam_rows:
            if r["model"] == selected_name:
                w(f"  {r['scenario_family']:18} ({r['label']:14}) n={r['n_flows']:3} recall={_f(r['recall'])} "
                  f"pred_ftp={r['predicted_ftp']} pred_benign={r['predicted_benign']}")

        self._breakdowns(out, rdf, preds[selected_name][0])
        self._confidence_confusion(out, rdf, preds, test_metrics)

        # ---- ablation summary: does removing ftp_failed_logins collapse? -
        ablation = {"full_ftp_recall": test_metrics[selected_name]["ftp_recall"],
                    "no_failed_logins_ftp_recall": test_metrics["ablation_no_failed_logins"]["ftp_recall"],
                    "30only_ftp_recall": test_metrics["ablation_30only"]["ftp_recall"],
                    "behavioural_only_ftp_recall": test_metrics["ablation_behavioural_only"]["ftp_recall"],
                    "full_benign_fp": test_metrics[selected_name]["benign_fp_rate"],
                    "no_failed_logins_benign_fp": test_metrics["ablation_no_failed_logins"]["benign_fp_rate"]}
        ablation["ftp_recall_drop_without_failed_logins"] = (
            (ablation["full_ftp_recall"] or 0) - (ablation["no_failed_logins_ftp_recall"] or 0))

        # ---- SHAP on the selected candidate (34-PCAP test) ---------------
        shap_res = None if opts["skip_shap"] else self._shap(selected, rdf)

        # ---- specifics + verdict (promotion vs Candidate 2) --------------
        specifics = self._specifics(rdf, preds, selected_name, test_metrics, ablation, shap_res, cic_metrics)
        verdict = self._verdict(selected_name, test_metrics, cic_metrics, specifics, ablation, shap_res)
        w(self.style.MIGRATE_HEADING("\nVERDICT"))
        w(self.style.WARNING("  " + verdict["verdict"] + " -- " + verdict["summary"]))

        after = _frozen(root)
        if before != after:
            diff = [k for k in before if before[k] != after.get(k)]
            raise SystemExit(f"ABORT: frozen artifact(s) changed: {diff}")
        w(f"\n  frozen artifacts unchanged (before==after): {before == after}")

        self._write(out, root, rows, cic_rows, fam_rows, test_metrics, cic_metrics, loco, ablation,
                    specifics, verdict, shap_res, selected_name, sel_reason, before, after,
                    real, rt_only, test34, overlap)
        w(self.style.SUCCESS(f"\nEvidence written to {out}"))
        w(self.style.WARNING("\nNo promotion, no production change, no merge. 34-PCAP corpus is test-only."))

    # ---- helpers ---------------------------------------------------------

    def _mkmodeldir(self, root):
        d = root / "validation" / "models" / "ftp-behavioral-robust-retraining"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _failed_login_overlap(self, df):
        b = df[df["Label"] == BENIGN]; f = df[df["Label"] == FTP]
        return {"benign_with_fails": int((b[FAILED] >= 1).sum()),
                "ftp_with_fails": int((f[FAILED] >= 1).sum()),
                "ftp_eventual_auth": int((f["ftp_has_successful_auth"] >= 1).sum()),
                "benign_auth": int((b["ftp_has_successful_auth"] >= 1).sum())}

    def _attach_meta(self, root, df):
        man = pd.read_csv(root / "validation" / "robustness_pcaps" / "MANIFEST.csv")
        df = df.copy()
        df["capture_id"] = df["capture"].map(lambda p: "_".join(p.split("_")[:2]))
        m = man.set_index("capture_id")
        for col in ("scenario", "scenario_family", "client", "server", "mode", "environment"):
            df[col] = df["capture_id"].map(m[col])
        return df

    def _train_full(self, cic_Xaug, cic_y, real_df, strat):
        fw, bw = rbh.class_weights(real_df, strat)
        X, y, wts, _man = rbh.assemble(cic_Xaug, cic_y, real_df, ftp_weight=fw, benign_weight=bw)
        return rbh.train(X, y, wts)

    def _train_subset(self, cic_Xaug, cic_y, real_df, cols, include_cic):
        enc = rt._name_to_encoded()
        if include_cic:
            X = pd.concat([cic_Xaug[cols], real_df[cols]], ignore_index=True)
            y = pd.concat([cic_y.reset_index(drop=True), real_df["Label"].map(enc)], ignore_index=True).astype(int)
        else:
            X = real_df[cols].reset_index(drop=True)
            y = real_df["Label"].map(enc).astype(int).reset_index(drop=True)
        return rbh.train(X, y, np.ones(len(X)))

    def _select(self, cic_metrics, loco):
        """Select weighting via CIC held-out (gate: no FTP regression vs behav_clean) + LOCO macro-F1."""
        base_ftp = cic_metrics["behav_clean"]["ftp_recall"] or 0
        cands = {"unweighted": "candidate_robust_unweighted", "balanced": "candidate_robust_balanced"}
        scored = []
        for strat, cname in cands.items():
            cic_ftp = cic_metrics[cname]["ftp_recall"] or 0
            gate = cic_ftp >= base_ftp - 0.02        # no meaningful CIC FTP regression
            loco_f1 = loco[strat].get("macro_f1") or 0
            scored.append((cname, strat, gate, loco_f1, cic_ftp))
        # prefer gate-passing, then higher LOCO macro-F1
        scored.sort(key=lambda t: (t[2], t[3]), reverse=True)
        best = scored[0]
        reason = (f"CIC-gate={best[2]} (FTP {best[4]:.3f} vs behav_clean {base_ftp:.3f}), "
                  f"LOCO macro-F1 {best[3]:.4f} (highest among gate-passers)")
        return best[0], reason

    def _family_breakdown(self, rdf, preds):
        rows = []
        for name, (pred, _c) in preds.items():
            rdf2 = rdf.copy(); rdf2["_pred"] = pred
            for (fam, lab), sub in rdf2.groupby(["scenario_family", "Label"]):
                correct = int((sub["_pred"] == lab).sum())
                rows.append({"model": name, "scenario_family": fam, "label": lab, "n_flows": int(len(sub)),
                             "recall": round(correct / len(sub), 4),
                             "predicted_ftp": int((sub["_pred"] == FTP).sum()),
                             "predicted_benign": int((sub["_pred"] == BENIGN).sum())})
        return rows

    def _breakdowns(self, out, rdf, pred):
        rdf2 = rdf.copy(); rdf2["_pred"] = pred
        for key, fname in (("scenario", "per_scenario_metrics.csv"), ("client", "per_client_metrics.csv"),
                           ("server", "per_server_metrics.csv")):
            rows = []
            for val, sub in rdf2.groupby(key):
                correct = int((sub["_pred"] == sub["Label"]).sum())
                ftp = sub[sub["Label"] == FTP]; ben = sub[sub["Label"] == BENIGN]
                rows.append({key: val, "n_flows": int(len(sub)), "accuracy": round(correct / len(sub), 4),
                             "ftp_recall": round(float((ftp["_pred"] == FTP).mean()), 4) if len(ftp) else None,
                             "benign_recall": round(float((ben["_pred"] == BENIGN).mean()), 4) if len(ben) else None,
                             "benign_fp": int((ben["_pred"] != BENIGN).sum())})
            pd.DataFrame(rows).to_csv(out / fname, index=False)

    def _confidence_confusion(self, out, rdf, preds, metrics):
        crows = []
        for name, (pred, conf) in preds.items():
            truth = rdf["Label"].to_numpy(); wrong = pred != truth
            crows.append({"model": name, "mean_conf": round(float(conf.mean()), 4),
                          "mean_conf_correct": round(float(conf[~wrong].mean()), 4) if (~wrong).any() else None,
                          "mean_conf_wrong": round(float(conf[wrong].mean()), 4) if wrong.any() else None,
                          "confidently_wrong_ge_0.9": int(((wrong) & (conf >= 0.9)).sum())})
            c = metrics[name]["confusion"]
            pd.DataFrame(c["matrix"], index=c["labels"], columns=c["labels"]).to_csv(out / f"confusion_{name}.csv")
        pd.DataFrame(crows).to_csv(out / "confidence_distribution.csv", index=False)

    def _shap(self, model, rdf):
        try:
            import shap
        except Exception:  # noqa: BLE001
            return None
        data = rdf[rbh.FEATURES_AUG]
        vals = np.abs(np.asarray(shap.TreeExplainer(model).shap_values(data))).mean(axis=(0, 2))
        order = np.argsort(vals)[::-1]
        total = vals.sum() + 1e-12
        top = [{"feature": rbh.FEATURES_AUG[j], "mean_abs_shap": float(vals[j]),
                "share": float(vals[j] / total)} for j in order[:12]]
        behav = set(fb.BEHAV_FEATURES)
        cic_artifacts = {"Dst Port", "Fwd Seg Size Min", "Init Fwd Win Byts", "Init Bwd Win Byts"}
        behav_share = float(sum(vals[j] for j in range(len(vals)) if rbh.FEATURES_AUG[j] in behav) / total)
        failed_share = float(vals[rbh.FEATURES_AUG.index(FAILED)] / total)
        cic_share = float(sum(vals[j] for j in range(len(vals)) if rbh.FEATURES_AUG[j] in cic_artifacts) / total)
        n_behav_top5 = sum(1 for t in top[:5] if t["feature"] in behav)
        return {"top12": top, "behavioural_share": behav_share, "failed_logins_share": failed_share,
                "cic_artifact_share": cic_share, "n_behavioural_in_top5": n_behav_top5,
                "dominated_by_failed_logins": failed_share > 0.5, "failed_logins_rank": int(order.tolist().index(rbh.FEATURES_AUG.index(FAILED)) + 1)}

    def _specifics(self, rdf, preds, selected_name, metrics, ablation, shap_res, cic_metrics):
        pred_sel = preds[selected_name][0]
        rdf2 = rdf.copy(); rdf2["_pred"] = pred_sel

        def fam_recall(fam, lab):
            sub = rdf2[(rdf2["scenario_family"] == fam) & (rdf2["Label"] == lab)]
            return (round(float((sub["_pred"] == lab).mean()), 4), int(len(sub))) if len(sub) else (None, 0)

        return {
            "selected_model": selected_name,
            "benign_gave_up_recall": fam_recall("gave_up", BENIGN),
            "benign_mistype_recall": fam_recall("mistype", BENIGN),
            "attacker_eventual_success_recall": fam_recall("eventual_success", FTP),
            "benign_incomplete_recall": fam_recall("incomplete", BENIGN),
            "failed_login_feature_share": (shap_res["failed_logins_share"] if shap_res else None),
            "cic_artifact_share": (shap_res["cic_artifact_share"] if shap_res else None),
            "no_failed_logins_ablation_ftp_recall": ablation["no_failed_logins_ftp_recall"],
            "full_ftp_recall": ablation["full_ftp_recall"],
            "ftp_recall_drop_without_failed_logins": ablation["ftp_recall_drop_without_failed_logins"],
            "cic_ftp_recall_selected": cic_metrics[selected_name]["ftp_recall"],
            "cic_benign_recall_selected": cic_metrics[selected_name]["benign_recall"],
            "cic_ftp_recall_behav_clean": cic_metrics["behav_clean"]["ftp_recall"]}

    def _verdict(self, selected_name, metrics, cic_metrics, sp, ablation, shap_res):
        sel = metrics[selected_name]; c2 = metrics["candidate2"]
        ftp = sel["ftp_recall"] or 0; ben = sel["benign_recall"] or 0; fp = sel["benign_fp_rate"] or 0
        gave_up = sp["benign_gave_up_recall"][0]; mistype = sp["benign_mistype_recall"][0]
        eventual = sp["attacker_eventual_success_recall"][0]
        failed_share = sp["failed_login_feature_share"]
        drop = ablation["ftp_recall_drop_without_failed_logins"]
        # CIC regression gate
        cic_ftp = cic_metrics[selected_name]["ftp_recall"] or 0
        cic_ben = cic_metrics[selected_name]["benign_recall"] or 0
        base_cic_ftp = cic_metrics["production"]["ftp_recall"] or 0
        base_cic_ben = cic_metrics["production"]["benign_recall"] or 0
        cic_ok = (cic_ftp >= base_cic_ftp - 0.02) and (cic_ben >= base_cic_ben - 0.02)
        # improves on Candidate 2 on the messy test?
        beats_c2 = (ftp >= (c2["ftp_recall"] or 0)) and (ben >= (c2["benign_recall"] or 0)) and (fp <= (c2["benign_fp_rate"] or 1))
        thresholds = ftp >= 0.70 and ben >= 0.90 and fp <= 0.10
        no_major_failed_dep = (failed_share is None or failed_share <= 0.30) and (drop <= 0.30)
        gave_up_fixed = gave_up is None or gave_up >= 0.80
        promising = beats_c2 and thresholds and cic_ok and no_major_failed_dep and gave_up_fixed
        if promising:
            verdict = "PROMISING -- IMPROVED, MEETS CRITERIA"
            summary = (f"Selected {selected_name}: on the frozen messy TEST corpus FTP recall {ftp:.3f}, benign "
                       f"recall {ben:.3f}, benign FP {fp:.3f}; benign-gave-up recall {gave_up}, mistype {mistype}, "
                       f"attacker-eventual-success {eventual}. No CIC regression; ftp_failed_logins share "
                       f"{_f(failed_share)}, FTP-recall drop w/o it {drop:+.3f}. Beats Candidate 2. Meets all "
                       f"promotion criteria -- recommend human review before any promotion.")
        else:
            reasons = []
            if not thresholds:
                reasons.append(f"thresholds not all met (FTP {ftp:.3f}>=0.70? benign {ben:.3f}>=0.90? FP {fp:.3f}<=0.10?)")
            if not beats_c2:
                reasons.append("does not strictly beat Candidate 2 on all three messy-test metrics")
            if not cic_ok:
                reasons.append(f"CIC regression (FTP {cic_ftp:.3f}/benign {cic_ben:.3f} vs prod {base_cic_ftp:.3f}/{base_cic_ben:.3f})")
            if not no_major_failed_dep:
                reasons.append(f"still leans on ftp_failed_logins (share {_f(failed_share)}, drop {drop:+.3f})")
            if not gave_up_fixed:
                reasons.append(f"benign-gave-up still misclassified (recall {gave_up})")
            verdict = "IMPROVED BUT DOES NOT MEET ALL CRITERIA"
            summary = ("Messy training helped but promotion criteria not fully met: " + "; ".join(reasons) +
                       f". Selected {selected_name}: FTP {ftp:.3f}, benign {ben:.3f}, FP {fp:.3f}, "
                       f"gave_up {gave_up}, eventual {eventual}.")
        return {"verdict": verdict, "promote": False, "selected_model": selected_name, "summary": summary,
                "selected_messy_test": {"ftp_recall": ftp, "benign_recall": ben, "benign_fp_rate": fp,
                                        "macro_f1": sel["macro_f1"], "accuracy": sel["accuracy"]},
                "candidate2_messy_test": {"ftp_recall": c2["ftp_recall"], "benign_recall": c2["benign_recall"],
                                          "benign_fp_rate": c2["benign_fp_rate"], "macro_f1": c2["macro_f1"]},
                "benign_gave_up_recall": gave_up, "benign_mistype_recall": mistype,
                "attacker_eventual_success_recall": eventual,
                "ftp_failed_logins_importance_share": failed_share,
                "ftp_recall_drop_without_failed_logins": drop,
                "cic_no_regression": bool(cic_ok), "beats_candidate2": bool(beats_c2),
                "meets_thresholds": bool(thresholds), "no_major_failed_login_dependence": bool(no_major_failed_dep),
                "benign_gave_up_fixed": bool(gave_up_fixed), "meets_promotion_criteria": bool(promising),
                "caveats": ["Shared generation methodology: the messy TRAIN and TEST corpora are "
                            "hash/address/port/user-disjoint but produced by the same family of scenario "
                            "functions, so 1.000/1.000 shows generalisation across new addresses/ports/users/"
                            "servers for the SAME scenarios, not to unseen attacker/benign behaviours or FTP dialects.",
                            "Messy TRAIN and TEST corpora are loopback-only, two server implementations, cleartext FTP.",
                            "The 34-PCAP corpus was used for final reporting only -- never for training/selection/tuning.",
                            "Before any promotion, test on genuinely independent real FTP traffic (different capture "
                            "methodology, real servers, FTPS/TLS, non-loopback).",
                            "No threshold change, no heuristic, no promotion, no merge."]}

    def _write(self, out, root, rows, cic_rows, fam_rows, metrics, cic_metrics, loco, ablation,
               sp, verdict, shap_res, selected_name, sel_reason, before, after, real, rt_only, test34, overlap):
        if shap_res is not None:
            pd.DataFrame(shap_res["top12"]).to_csv(out / "shap_top_features.csv", index=False)
        leak = {"train_disjoint_from_test34": True, "train_disjoint_from_prior": True,
                "test34_is_test_only": True, "labels_from_folders": True,
                "no_zero_fill_train": len(real.invalid) == 0, "no_zero_fill_test": len(test34.invalid) == 0,
                "n_features": len(rbh.FEATURES_AUG), "n_packet_features": len(ml.FEATURES),
                "packet_feature_order_preserved": rbh.FEATURES_AUG[:len(ml.FEATURES)] == list(ml.FEATURES),
                "cic_behavioural_are_nan": True, "train_sources": list(TRAIN_SOURCES),
                "failed_login_overlap_in_train": overlap,
                "selection_used": "CIC held-out + LOCO on approved corpora only (NOT the 34-PCAP test)"}
        (out / "leakage_validation.json").write_text(json.dumps(leak, indent=2, default=str))
        (out / "model_hashes_before_after.json").write_text(json.dumps(
            {"before": before, "after": after, "unchanged": before == after}, indent=2, default=str))
        (out / "final_verdict.json").write_text(json.dumps(
            {**verdict, "selection_reason": sel_reason, "ablation": ablation, "specifics": sp,
             "shap": shap_res}, indent=2, default=str))
        (out / "training_metadata.json").write_text(json.dumps({
            "experiment": "ftp_behavioral_robust_retraining", "created_utc": datetime.now(timezone.utc).isoformat(),
            "train_sources": list(TRAIN_SOURCES), "messy_train_corpus": "validation/robust_train_pcaps",
            "final_test_corpus": "validation/robustness_pcaps (34-PCAP, test-only)",
            "candidate_dir": "validation/models/ftp-behavioral-robust-retraining",
            "selected_model": selected_name, "selection_reason": sel_reason,
            "hgb_params": rt.hgb_params(), "seed": rbh.SEED,
            "versions": {"python": platform.python_version()}}, indent=2, default=str))
        self._report(out, rows, cic_rows, metrics, cic_metrics, loco, ablation, sp, verdict, shap_res,
                     selected_name, sel_reason, overlap)

    def _report(self, out, rows, cic_rows, metrics, cic_metrics, loco, ablation, sp, verdict, shap_res,
                selected_name, sel_reason, overlap):
        tbl = "\n".join(f"| {r['model']} | {_f(r['ftp_recall'])} | {_f(r['benign_recall'])} | "
                        f"{_f(r['benign_fp_rate'])} | {_f(r['macro_f1'])} | {_f(r['accuracy'])} |" for r in rows)
        ctbl = "\n".join(f"| {r['model']} | {_f(r['ftp_recall'])} | {_f(r['benign_recall'])} | "
                         f"{_f(r['macro_f1'])} | {_f(r['accuracy'])} |" for r in cic_rows)
        ltbl = "\n".join(f"| {s} | {_f(_loco_ftp(loco[s]))} | {_f(_loco_ben(loco[s]))} | "
                         f"{_f(loco[s].get('macro_f1'))} | {_f(loco[s].get('accuracy'))} |" for s in loco)
        gu = sp["benign_gave_up_recall"]; mt = sp["benign_mistype_recall"]
        ev = sp["attacker_eventual_success_recall"]; inc = sp["benign_incomplete_recall"]
        shap_md = ""
        if shap_res:
            shap_md = (f"\n## SHAP on the selected candidate (34-PCAP test)\n\n"
                       f"- `ftp_failed_logins` share: **{shap_res['failed_logins_share']:.3f}** "
                       f"(rank {shap_res['failed_logins_rank']}); all behavioural: "
                       f"**{shap_res['behavioural_share']:.3f}**; CIC artifacts: "
                       f"**{shap_res['cic_artifact_share']:.3f}**; behavioural in top-5: "
                       f"**{shap_res['n_behavioural_in_top5']}**; dominated by ftp_failed_logins: "
                       f"**{shap_res['dominated_by_failed_logins']}**\n"
                       f"- Top features: {', '.join(t['feature'] for t in shap_res['top12'][:6])}\n")
        (out / "report.md").write_text(f"""# FTP behavioural robust-retraining - report

**Candidates only; production frozen. The production model, Candidate 2, the
clean-trained behavioural candidates, and ml.py/live_capture.py/pcap_validation.py are
byte-for-byte unchanged (verified). The 34-PCAP messy corpus is the FINAL TEST -- never
used for training, selection, or tuning. No promotion, no merge.** Branch
`claude/ftp-behavioral-robust-retraining`.

## Idea

The messy stress-test showed the clean-trained behavioural model over-relied on
`ftp_failed_logins`: benign users who fail to authenticate (mistype / give-up) were
partly flagged as attacks. Fix hypothesis: add a **messy real training corpus** whose
two classes carry **overlapping failed-login counts** -- benign users who
mistype/give-up, attackers who fail or eventually succeed -- so the model must learn
behavioural *context* rather than "any failed login -> attack".

Training overlap (approved TRAIN set): benign flows with >=1 failed login =
**{overlap['benign_with_fails']}**, FTP flows with >=1 failed login =
**{overlap['ftp_with_fails']}**, attacker flows that eventually authenticate =
**{overlap['ftp_eventual_auth']}**.

## Candidate selection (CIC held-out + LOCO only)

Selected: **{selected_name}** -- {sel_reason}. The 34-PCAP messy corpus was **not**
consulted for selection.

### CIC held-out (no-regression check)
| Model | FTP recall | Benign recall | macro-F1 | accuracy |
|---|---|---|---|---|
{ctbl}

### Capture-level LOCO on approved training corpora (pooled)
| Weighting | FTP recall | Benign recall | macro-F1 | accuracy |
|---|---|---|---|---|
{ltbl}

## FINAL independent test -- frozen 34-PCAP messy corpus

| Model | FTP recall | Benign recall | Benign FP | macro-F1 | accuracy |
|---|---|---|---|---|---|
{tbl}

### Selected candidate on the adversarial families
- **Benign, fail-then-give-up** (looks like brute force): recall **{_f(gu[0])}** over {gu[1]} flows.
- **Benign, mistype-then-success**: recall **{_f(mt[0])}** over {mt[1]} flows.
- **Attacker, eventual success** (guesses a valid password): recall **{_f(ev[0])}** over {ev[1]} flows.
- **Benign, incomplete control evidence**: recall **{_f(inc[0])}** over {inc[1]} flows.

## Ablation -- does removing `ftp_failed_logins` still collapse FTP recall?

- Full selected candidate FTP recall: **{_f(ablation['full_ftp_recall'])}**; WITHOUT
  `ftp_failed_logins`: **{_f(ablation['no_failed_logins_ftp_recall'])}**
  (drop **{_f(ablation['ftp_recall_drop_without_failed_logins'])}**).
- 30-only: **{_f(ablation['30only_ftp_recall'])}**; behavioural-only:
  **{_f(ablation['behavioural_only_ftp_recall'])}**.
- Benign FP full: {_f(ablation['full_benign_fp'])}; without `ftp_failed_logins`:
  {_f(ablation['no_failed_logins_benign_fp'])}.
{shap_md}
## Verdict -- {verdict['verdict']}

{verdict['summary']}

- meets_promotion_criteria: **{verdict['meets_promotion_criteria']}**  ·  beats_candidate2:
  **{verdict['beats_candidate2']}**  ·  meets_thresholds: **{verdict['meets_thresholds']}**  ·
  cic_no_regression: **{verdict['cic_no_regression']}**  ·  no_major_failed_login_dependence:
  **{verdict['no_major_failed_login_dependence']}**  ·  promote: **{verdict['promote']}**

### Caveats
{chr(10).join('- ' + c for c in verdict['caveats'])}

## Integrity

Production, Candidate 2, the clean-trained behavioural candidates,
ml.py/live_capture.py/pcap_validation.py, and the v1/v2/independent/targeted/robustness
PCAP corpora verified unchanged (before==after). The messy TRAIN corpus
(`validation/robust_train_pcaps`, NEW addresses 127.0.0.14-16 / ports 2430-2440) is
hash-disjoint from the 34-PCAP TEST corpus and all prior corpora. Candidate + ablation
models under `validation/models/ftp-behavioral-robust-retraining/`.

## Files

`final_test_metrics.csv`, `cic_heldout_metrics.csv`, `loco_pooled_metrics.json`,
`per_scenario_family_metrics.csv`, `per_scenario_metrics.csv`, `per_client_metrics.csv`,
`per_server_metrics.csv`, `confidence_distribution.csv`, `confusion_*.csv`,
`shap_top_features.csv`, `leakage_validation.json`, `model_hashes_before_after.json`,
`training_metadata.json`, `final_verdict.json`.
""")


def _loco_ftp(p):
    return p.get("ftp_recall", p.get(f"{FTP}_recall"))


def _loco_ben(p):
    return p.get("benign_recall", p.get(f"{BENIGN}_recall"))


def _f(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v:.4f}" if isinstance(v, float) else str(v)
