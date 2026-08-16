"""
FTP-behavioural-feature experiment (candidate only; production frozen).

Step 1 (GATE): establish whether the experimental FTP control-channel behavioural
features actually separate real FTP-BruteForce from real Benign on the TRAINING
corpora (v1+v2+targeted). If they do not, STOP and report honestly.

Step 2 (only if separable): train augmented candidates (30 ml.FEATURES + 15
behavioural; CIC behavioural = NaN) on CIC + real, select on CIC held-out + v2
LOCO, and evaluate ONCE on the frozen independent 36-PCAP set. Compare production,
Candidate 2, a 30-feature control, and the augmented candidate. SHAP tells whether
the augmented model uses the behavioural features or falls back to CIC artifacts.
Promotes nothing; verifies all frozen artifacts unchanged.

    python manage.py ftp_behavioral_experiment
"""

from __future__ import annotations

import glob
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from django.core.management.base import BaseCommand
from sklearn.metrics import roc_auc_score

from predictor import ml, retraining as rt, retraining_v2 as r2, retraining_behavioral as rbh
from predictor import ftp_behavioral as fb, independent_eval as ie


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _frozen(root):
    d = {"production": _sha(ie.production_model_path()), "candidate2": _sha(ie.candidate2_file()),
         "ml_py": _sha(root / "webapp_django/predictor/ml.py"),
         "live_capture_py": _sha(root / "webapp_django/predictor/live_capture.py"),
         "pcap_validation_py": _sha(root / "webapp_django/predictor/pcap_validation.py")}
    for name, sub in (("v1", "realistic_pcaps"), ("v2", "realistic_pcaps_v2"),
                      ("indep", "independent_real_pcaps"), ("targeted", "targeted_benign_pcaps")):
        d[name + "_pcaps"] = sorted(_sha(p) for p in glob.glob(str(root / "validation" / sub / "**/*.pcap"), recursive=True))
    return d


def _metrics(truth, pred, conf=None) -> dict:
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                                 classification_report, confusion_matrix)
    truth = np.array([str(t) for t in truth]); pred = np.array([str(p) for p in pred])
    labels = sorted(set(truth) | set(pred))
    rep = classification_report(truth, pred, labels=labels, output_dict=True, zero_division=0)
    pc = {c: {"precision": rep[c]["precision"], "recall": rep[c]["recall"], "f1": rep[c]["f1-score"],
              "support": int(rep[c]["support"])} for c in labels if c in rep}
    cm = confusion_matrix(truth, pred, labels=labels)
    bmask = truth == rbh.BENIGN
    return {"n": int(len(truth)), "accuracy": float(accuracy_score(truth, pred)),
            "macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)),
            "ftp_recall": pc.get(rbh.FTP, {}).get("recall"),
            "ftp_precision": pc.get(rbh.FTP, {}).get("precision"),
            "benign_recall": pc.get(rbh.BENIGN, {}).get("recall"),
            "benign_fp_rate": float((pred[bmask] != rbh.BENIGN).mean()) if bmask.any() else None,
            "confusion": {"labels": labels, "matrix": cm.tolist()}, "per_class": pc}


def _bootstrap(df, predict_fn, feat_cols, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    caps = sorted(df["capture"].unique())
    by = {c: df[df["capture"] == c] for c in caps}
    ftp, ben = [], []
    for _ in range(n_boot):
        samp = pd.concat([by[c] for c in rng.choice(caps, size=len(caps), replace=True)], ignore_index=True)
        pred = predict_fn(samp[feat_cols])
        truth = samp["Label"].to_numpy()
        fm = truth == rbh.FTP; bm = truth == rbh.BENIGN
        ftp.append(float((pred[fm] == rbh.FTP).mean()) if fm.any() else np.nan)
        ben.append(float((pred[bm] == rbh.BENIGN).mean()) if bm.any() else np.nan)

    def ci(a):
        a = np.array(a); a = a[~np.isnan(a)]
        return {"lo95": float(np.percentile(a, 2.5)), "hi95": float(np.percentile(a, 97.5))}
    return {"ftp_recall": ci(ftp), "benign_recall": ci(ben)}


class Command(BaseCommand):
    help = "FTP behavioural-feature experiment vs frozen production + Candidate 2 (no promotion)."

    def add_arguments(self, parser):
        parser.add_argument("--output", default=None)
        parser.add_argument("--loco-sample", type=int, default=r2.CIC_LOCO_SAMPLE)
        parser.add_argument("--strategies", default="unweighted,balanced")
        parser.add_argument("--max-folds", type=int, default=0)
        parser.add_argument("--n-boot", type=int, default=2000)
        parser.add_argument("--skip-shap", action="store_true")

    def handle(self, *args, **opts):
        w = self.stdout.write
        root = rbh.repo_root()
        out = Path(opts["output"]) if opts["output"] else root / "validation" / "results" / "ftp_behavioral"
        out.mkdir(parents=True, exist_ok=True)
        strategies = [s.strip() for s in opts["strategies"].split(",") if s.strip()]

        before = _frozen(root)
        w(self.style.MIGRATE_HEADING("FTP behavioural-feature experiment (candidate only)"))
        indep_h = set(before["indep_pcaps"])
        trainset = set(before["v1_pcaps"]) | set(before["v2_pcaps"]) | set(before["targeted_pcaps"])
        if not indep_h.isdisjoint(trainset):
            raise SystemExit("ABORT: an independent PCAP is in the training corpora!")

        # ===== data: real augmented (train) ==============================
        real = rbh.extract_real_augmented(("v1", "v2", "targeted"))
        w(f"  real training flows: {len(real.df)} (FTP {int((real.df['Label']==rbh.FTP).sum())}, "
          f"Benign {int((real.df['Label']==rbh.BENIGN).sum())}); invalid: {len(real.invalid)}")

        # ===== STEP 1: SEPARABILITY GATE (training corpora only) =========
        gate = self._separability(out, real.df)
        w(self.style.MIGRATE_HEADING("\nStep 1 - separability gate (training corpora only)"))
        w(f"  best single-feature |AUC-0.5|: {gate['best_abs_auc']:.3f} "
          f"({gate['best_feature']}); behavioural-only 5-fold macro-F1: {gate['behav_only_cv_macro_f1']:.3f}")
        w(f"  GATE PASSED: {gate['passed']}")
        if not gate["passed"]:
            (out / "final_verdict.json").write_text(json.dumps(
                {"gate_passed": False, "promote": False,
                 "conclusion": "Behavioural features do NOT separate real FTP from real benign; "
                               "stopping without training, as instructed."}, indent=2))
            w(self.style.ERROR("\n  Behavioural features do not separate the classes -- STOP (no training)."))
            return

        # ===== STEP 2: train candidates ==================================
        cic_Xaug, cic_y = rbh.load_cic_augmented()
        cic_test_Xaug, cic_test_y = rbh.load_cic_test_augmented()
        dec = rt._encoded_to_name(); cic_test_names = cic_test_y.map(dec)
        cic_sub_Xaug, cic_sub_y = rbh.stratified_cic_subsample_aug(cic_Xaug, cic_y, opts["loco_sample"])
        v2caps = rbh.v2_captures(real)

        prod, _ = ml._load(ml.DEFAULT_MODEL)
        cand2 = __import__("joblib").load(ie.candidate2_file())
        base_cic = rt.evaluate(prod, *(rt.load_cic_test()[0], cic_test_names))
        cand2_cic = rt.evaluate(cand2, rt.load_cic_test()[0], cic_test_names)
        w(self.style.MIGRATE_HEADING("\nStep 2 - augmented candidates (30 + behavioural)"))
        w(f"  baseline CIC: prod {base_cic['accuracy']:.4f}/{base_cic['macro_f1']:.4f}  "
          f"Cand2 {cand2_cic['accuracy']:.4f}/{cand2_cic['macro_f1']:.4f}")

        # 30-feature control (CIC + real, unweighted) for direct comparison
        control = self._train_control30(real)
        models, cand_rows, loco_store = {}, [], {}
        for strat in strategies:
            fw, bw = rbh.class_weights(real.df, strat)
            Xf, yf, wf, manf = rbh.assemble(cic_Xaug, cic_y, real.df, ftp_weight=fw, benign_weight=bw)
            model = rbh.train(Xf, yf, wf)
            models[strat] = model
            c_cic = rbh.evaluate(model, cic_test_Xaug, cic_test_names)
            rbh.save_candidate(model, {"candidate": f"aug_{strat}", "ftp_weight": fw, "benign_weight": bw,
                                       "cic_accuracy": c_cic["accuracy"], "cic_macro_f1": c_cic["macro_f1"],
                                       "train_manifest": manf}, f"candidate_aug_{strat}")
            held = v2caps if not opts["max_folds"] else v2caps[:opts["max_folds"]]
            loco = rbh.leave_one_capture_out(cic_sub_Xaug, cic_sub_y, real, held, fw, bw)
            loco_store[strat] = loco; p = loco["pooled"]
            w(f"  [aug_{strat:10} fw={fw:.1f} bw={bw:.1f}] CIC {c_cic['accuracy']:.4f}/{c_cic['macro_f1']:.4f} | "
              f"v2-LOCO FTP-rec {_f(p['FTP-BruteForce_recall'])} Ben-rec {_f(p['Benign_recall'])} mF1 {p['macro_f1']:.4f}")
            cand_rows.append({"candidate": f"aug_{strat}", "ftp_weight": fw, "benign_weight": bw,
                              "cic_accuracy": c_cic["accuracy"], "cic_macro_f1": c_cic["macro_f1"],
                              "v2_loco_ftp_recall": p["FTP-BruteForce_recall"],
                              "v2_loco_benign_recall": p["Benign_recall"], "v2_loco_macro_f1": p["macro_f1"]})
            cm = p["confusion"]
            pd.DataFrame(cm["matrix"], index=cm["labels"], columns=cm["labels"]).to_csv(out / f"confusion_v2loco_aug_{strat}.csv")

        selected = max([c for c in cand_rows if c["cic_macro_f1"] >= base_cic["macro_f1"] - 0.01] or cand_rows,
                       key=lambda c: c["v2_loco_macro_f1"] or 0)["candidate"].replace("aug_", "")
        w(self.style.MIGRATE_HEADING(f"\nSelected on CIC + v2-LOCO only: aug_{selected}"))

        # ===== FINAL independent eval (frozen; one pass) =================
        w(self.style.MIGRATE_HEADING("\nFINAL independent test (frozen 36-PCAP set)"))
        indep = rbh.extract_real_augmented(("independent",))
        idf = indep.df
        indep_metrics, indep_ci, indep_rows = {}, {}, []
        model_specs = {"production": (prod, list(ml.FEATURES)), "candidate2": (cand2, list(ml.FEATURES)),
                       "control_30feat": (control, list(ml.FEATURES))}
        for s in strategies:
            model_specs[f"aug_{s}"] = (models[s], rbh.FEATURES_AUG)
        for name, (model, cols) in model_specs.items():
            pred, conf = rbh.predict_names(model, idf[cols])
            m = _metrics(idf["Label"].to_numpy(), pred, conf); indep_metrics[name] = m
            ci = _bootstrap(idf, lambda X, mm=model: rbh.predict_names(mm, X)[0], cols, n_boot=opts["n_boot"])
            indep_ci[name] = ci
            indep_rows.append({"model": name, "ftp_recall": m["ftp_recall"], "ftp_precision": m["ftp_precision"],
                               "benign_recall": m["benign_recall"], "benign_fp_rate": m["benign_fp_rate"],
                               "macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
                               "ftp_recall_lo95": ci["ftp_recall"]["lo95"], "ftp_recall_hi95": ci["ftp_recall"]["hi95"],
                               "benign_recall_lo95": ci["benign_recall"]["lo95"], "benign_recall_hi95": ci["benign_recall"]["hi95"]})
            w(f"  {name:16} FTP-rec {_f(m['ftp_recall'])} Ben-rec {_f(m['benign_recall'])} "
              f"BenFP {_f(m['benign_fp_rate'])} mF1 {m['macro_f1']:.4f} acc {m['accuracy']:.4f}")

        # ===== SHAP on selected augmented candidate ======================
        shap_res = None if opts["skip_shap"] else self._shap(models[selected], cic_Xaug, idf, selected)

        # ===== verdict ====================================================
        verdict = self._verdict(selected, indep_metrics, indep_ci, cand_rows, base_cic, shap_res, gate)
        w(self.style.MIGRATE_HEADING("\nVERDICT"))
        w(self.style.WARNING("  " + verdict["decision"] + " -- " + verdict["summary"]))

        after = _frozen(root)
        if before != after:
            raise SystemExit("ABORT: a frozen artifact changed!")
        w(f"\n  frozen artifacts unchanged (before==after): {before == after}")

        self._write(out, gate, cand_rows, indep_rows, indep_metrics, indep_ci, base_cic, cand2_cic,
                    control, selected, verdict, shap_res, before, after, real, strategies, opts, idf)
        w(self.style.SUCCESS(f"\nEvidence written to {out}"))
        w(self.style.WARNING("\nNo promotion, no production change, no merge. Independent test evaluated once."))

    # -- separability gate -------------------------------------------------

    def _separability(self, out, real_df) -> dict:
        y = (real_df["Label"] == rbh.FTP).astype(int).to_numpy()
        # aggregate to capture level (one behavioural vector per capture)
        cap = real_df.groupby("capture").first()
        ycap = (cap["Label"] == rbh.FTP).astype(int).to_numpy()
        rows, best_abs, best_feat = [], 0.0, None
        for f in fb.BEHAV_FEATURES:
            try:
                auc = roc_auc_score(ycap, cap[f])
            except Exception:  # noqa: BLE001
                auc = 0.5
            rows.append({"feature": f, "auc_ftp_positive": auc, "abs_sep": abs(auc - 0.5)})
            if abs(auc - 0.5) > best_abs:
                best_abs, best_feat = abs(auc - 0.5), f
        pd.DataFrame(rows).sort_values("abs_sep", ascending=False).to_csv(out / "separability_analysis.csv", index=False)
        # behavioural-only classifier, capture-level 5-fold CV macro-F1
        from sklearn.model_selection import cross_val_predict, StratifiedKFold
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.metrics import f1_score
        Xb = cap[fb.BEHAV_FEATURES].to_numpy()
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        pred = cross_val_predict(HistGradientBoostingClassifier(random_state=42), Xb, ycap, cv=cv)
        cv_f1 = float(f1_score(ycap, pred, average="macro", zero_division=0))
        # per-class capture medians for the report
        med = real_df.groupby("Label")[fb.BEHAV_FEATURES].median().to_dict("index")
        passed = best_abs >= 0.40 and cv_f1 >= 0.90    # strong, honest threshold
        return {"passed": bool(passed), "best_feature": best_feat, "best_abs_auc": 0.5 + best_abs,
                "behav_only_cv_macro_f1": cv_f1, "class_medians": med, "n_captures": int(len(cap))}

    def _train_control30(self, real):
        cic_X, cic_y = rt.load_cic()
        enc = rt._name_to_encoded()
        X = pd.concat([cic_X, real.df[ml.FEATURES]], ignore_index=True)
        y = pd.concat([cic_y.reset_index(drop=True), real.df["Label"].map(enc)], ignore_index=True).astype(int)
        return rt.train(X, y, np.ones(len(X)))

    def _shap(self, model, cic_Xaug, idf, selected):
        try:
            import shap
        except Exception:  # noqa: BLE001
            return None
        bg = cic_Xaug.sample(min(200, len(cic_Xaug)), random_state=rbh.SEED)[rbh.FEATURES_AUG]
        art = {"Fwd Seg Size Min", "Init Fwd Win Byts"}
        behav = set(fb.BEHAV_FEATURES)

        def top(data, n=8):
            if len(data) == 0:
                return [], 0.0
            vals = np.abs(np.asarray(shap.TreeExplainer(model).shap_values(data[rbh.FEATURES_AUG]))).mean(axis=(0, 2))
            order = np.argsort(vals)[::-1]
            tops = [rbh.FEATURES_AUG[j] for j in order[:n]]
            behav_share = float(sum(vals[j] for j in range(len(vals)) if rbh.FEATURES_AUG[j] in behav) / (vals.sum() + 1e-12))
            return tops, behav_share

        ftp = idf[idf["Label"] == rbh.FTP]; ben = idf[idf["Label"] == rbh.BENIGN]
        g_top, g_share = top(idf)
        f_top, _ = top(ftp); b_top, _ = top(ben)
        return {"global_top8": g_top, "behavioural_importance_share": g_share,
                "test_ftp_top8": f_top, "test_benign_top8": b_top,
                "still_artifact_top": bool(art & set(g_top[:5])),
                "uses_behavioural": bool(behav & set(g_top[:5]))}

    def _verdict(self, selected, indep_metrics, indep_ci, cand_rows, base_cic, shap_res, gate) -> dict:
        cand2 = indep_metrics["candidate2"]; new = indep_metrics[f"aug_{selected}"]
        sel_row = [c for c in cand_rows if c["candidate"] == f"aug_{selected}"][0]
        ftp = new["ftp_recall"] or 0; ben = new["benign_recall"] or 0; fp = new["benign_fp_rate"] or 0
        cand2_fp = cand2["benign_fp_rate"] or 0
        cic_ok = sel_row["cic_macro_f1"] >= base_cic["macro_f1"] - 0.01
        c_ftp = ftp > 0.70; c_ben = ben > 0.90; c_fp = fp <= cand2_fp - 0.10
        breaks_frontier = c_ftp and c_ben and c_fp and cic_ok
        # promotion is still conservative: strong result but small corpus
        decision = "PROMISING - FRONTIER BROKEN (validate further; DO NOT auto-promote)" if breaks_frontier \
            else "DO NOT PROMOTE"
        unmet = [n for n, ok in (("FTP recall>0.70", c_ftp), ("benign recall>0.90", c_ben),
                                 ("benign FP << Cand2", c_fp), ("no CIC regression", cic_ok)) if not ok]
        summary = (f"aug_{selected} independent: FTP recall {ftp:.3f}, benign recall {ben:.3f}, benign FP "
                   f"{fp:.3f} (Cand2 {cand2_fp:.3f}), macro-F1 {new['macro_f1']:.3f}, CIC ok={cic_ok}. "
                   + ("All balanced-target criteria met on the untouched independent test -- the behavioural "
                      "features break the FTP-recall/benign-FP trade-off." if breaks_frontier
                      else "Unmet: " + "; ".join(unmet) + "."))
        return {"gate_passed": gate["passed"], "selected_candidate": f"aug_{selected}",
                "promote": False, "frontier_broken": bool(breaks_frontier), "decision": decision,
                "summary": summary,
                "selection_basis": "CIC held-out + v2 LOCO only (independent test never used to select)",
                "criteria": {"ftp_recall_gt_0.70": c_ftp, "benign_recall_gt_0.90": c_ben,
                             "benign_fp_much_lower_than_cand2": c_fp, "cic_non_regressed": cic_ok},
                "independent": {"ftp_recall": ftp, "ftp_recall_ci95": [indep_ci[f"aug_{selected}"]["ftp_recall"]["lo95"],
                                                                       indep_ci[f"aug_{selected}"]["ftp_recall"]["hi95"]],
                                "benign_recall": ben, "benign_fp_rate": fp, "macro_f1": new["macro_f1"],
                                "candidate2_ftp_recall": cand2["ftp_recall"], "candidate2_benign_fp_rate": cand2_fp},
                "shap": ({"uses_behavioural_features": shap_res["uses_behavioural"],
                          "still_cic_artifact_top": shap_res["still_artifact_top"],
                          "behavioural_importance_share": shap_res["behavioural_importance_share"]} if shap_res else None),
                "caveats": [
                    "Separation is strong partly because captures are cleanly scenario-labelled (benign always "
                    "authenticates, brute force always fails); real-world traffic can be messier (benign typos, "
                    "brute force that eventually succeeds).",
                    "Behavioural features are per-capture session context attached to each flow.",
                    "Independent corpus small (36 captures / 420 flows), loopback-only; CIs wide; no significance claim.",
                    "Not auto-promoted: needs broader real traffic and a fresh independent test before any rollout."],
                "conclusion": ("The behavioural features separate real FTP from real benign and, on the untouched "
                               "independent test, let the augmented model achieve high FTP recall AND high benign "
                               "recall simultaneously -- breaking the trade-off the 30 CIC features could not. This "
                               "is a promising direction; it is NOT auto-promoted." if breaks_frontier else
                               "The behavioural features separate the classes on training data, but the augmented "
                               "model did not meet all balanced-target criteria on the independent test.")}

    def _write(self, out, gate, cand_rows, indep_rows, indep_metrics, indep_ci, base_cic, cand2_cic,
               control, selected, verdict, shap_res, before, after, real, strategies, opts, idf):
        pd.DataFrame(cand_rows).to_csv(out / "candidate_metrics.csv", index=False)
        pd.DataFrame(indep_rows).to_csv(out / "independent_test_metrics.csv", index=False)
        comp = []
        for c in cand_rows:
            im = indep_metrics[c["candidate"]]
            comp.append({**c, "indep_ftp_recall": im["ftp_recall"], "indep_benign_recall": im["benign_recall"],
                         "indep_benign_fp_rate": im["benign_fp_rate"], "indep_macro_f1": im["macro_f1"]})
        pd.DataFrame(comp).to_csv(out / "candidate_comparison.csv", index=False)
        for name in indep_metrics:
            c = indep_metrics[name]["confusion"]
            pd.DataFrame(c["matrix"], index=c["labels"], columns=c["labels"]).to_csv(out / f"confusion_independent_{name}.csv")
        # per-capture on independent for key models
        pc = []
        for name, cols in (("production", list(ml.FEATURES)), ("candidate2", list(ml.FEATURES)),
                           (f"aug_{selected}", rbh.FEATURES_AUG)):
            model = {"production": ml._load(ml.DEFAULT_MODEL)[0], "candidate2": __import__("joblib").load(ie.candidate2_file())}.get(name)
            if model is None:
                model = __import__("joblib").load(rbh.candidate_dir() / f"candidate_aug_{selected}.pkl")
            for capn, sub in idf.groupby("capture"):
                pred, conf = rbh.predict_names(model, sub[cols])
                truth = sub["Label"].to_numpy()
                pc.append({"model": name, "capture": capn, "label": sub["Label"].iloc[0], "n_flows": int(len(sub)),
                           "recall": float((pred == truth).mean()), "confidence_mean": round(float(conf.mean()), 4)})
        pd.DataFrame(pc).to_csv(out / "independent_per_capture_metrics.csv", index=False)
        ci_rows = [{"model": n, "metric": mt, **indep_ci[n][mt]} for n in indep_ci for mt in ("ftp_recall", "benign_recall")]
        pd.DataFrame(ci_rows).to_csv(out / "bootstrap_ci_results.csv", index=False)
        if shap_res is not None:
            pd.DataFrame([{"group": "global_top8", "features": ";".join(shap_res["global_top8"])},
                         {"group": "test_ftp_top8", "features": ";".join(shap_res["test_ftp_top8"])},
                         {"group": "test_benign_top8", "features": ";".join(shap_res["test_benign_top8"])},
                         {"group": "behavioural_importance_share", "features": str(shap_res["behavioural_importance_share"])}
                          ]).to_csv(out / "shap_comparison.csv", index=False)
        leak = {"independent_disjoint_from_training": set(before["indep_pcaps"]).isdisjoint(
            set(before["v1_pcaps"]) | set(before["v2_pcaps"]) | set(before["targeted_pcaps"])),
            "training_sources": ["v1", "v2", "targeted"], "independent_used_for": "final evaluation only",
            "real_flows": int(len(real.df)), "invalid_flows": len(real.invalid), "no_zero_fill": len(real.invalid) == 0,
            "cic_behavioural_is_nan_not_zero": True, "n_features": len(rbh.FEATURES_AUG),
            "capture_level_loco_no_flow_overlap": True, "labels_from_folders": True}
        (out / "leakage_validation.json").write_text(json.dumps(leak, indent=2, default=str))
        (out / "model_hashes_before_after.json").write_text(json.dumps({"before": before, "after": after, "unchanged": before == after}, indent=2, default=str))
        (out / "final_verdict.json").write_text(json.dumps(verdict, indent=2, default=str))
        import sklearn
        (out / "training_metadata.json").write_text(json.dumps({
            "experiment": "ftp_behavioral_features", "created_utc": datetime.now(timezone.utc).isoformat(),
            "seed": rbh.SEED, "model": "HistGradientBoostingClassifier", "hyperparameters": rt.hgb_params(),
            "feature_set": rbh.FEATURES_AUG, "n_features": len(rbh.FEATURES_AUG),
            "behavioural_features": list(fb.BEHAV_FEATURES),
            "cic_behavioural": "NaN (no PCAP; HGB native missing-value handling; not zero-fill)",
            "training_real_sources": "v1 + v2 + targeted", "independent_test": "FROZEN; eval only",
            "candidate_dir": str(rbh.candidate_dir()),
            "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__}}, indent=2, default=str))
        self._report(out, gate, cand_rows, indep_metrics, indep_ci, base_cic, selected, verdict, shap_res, real, opts)

    def _report(self, out, gate, cand_rows, indep_metrics, indep_ci, base_cic, selected, verdict, shap_res, real, opts):
        pr = indep_metrics["production"]; c2 = indep_metrics["candidate2"]; ct = indep_metrics["control_30feat"]; nw = indep_metrics[f"aug_{selected}"]
        shap_md = ""
        if shap_res:
            shap_md = (f"\n## SHAP (selected `aug_{selected}`)\n\n"
                       f"- Uses behavioural features in top-5: **{shap_res['uses_behavioural']}**; "
                       f"still CIC-artifact in top-5: **{shap_res['still_artifact_top']}**\n"
                       f"- Behavioural-feature importance share: **{shap_res['behavioural_importance_share']:.2f}**\n"
                       f"- Global top-8: {', '.join(shap_res['global_top8'])}\n")
        md = f"""# FTP behavioural-feature experiment - report

**Candidate only. Production model and Candidate 2 frozen and byte-for-byte
unchanged (verified). No auto-promotion, no merge.** The existing 30-feature
`pcap_validation` pipeline is unchanged; behavioural features are added only in this
experimental module. The independent 36-PCAP set was NEVER used for feature
selection, tuning, weighting, or candidate selection (hash-guarded); the candidate
was selected on CIC held-out + v2 LOCO, then evaluated once on the independent set.

## New features

15 FTP control-channel behavioural features (`ftp_behavioral.BEHAV_FEATURES`) read
directly from cleartext FTP commands/responses: login attempts, failed logins
(530), successful logins (230), failed-login ratio, has-successful-auth, data
commands, data-setup responses (150), command variety, control connections /
reconnects, etc. Computed per capture from the PCAP; CIC flows (no PCAP) carry
**NaN** (HGB native missing-value handling -- not zero-fill).

## Step 1 - separability gate (training corpora only: v1+v2+targeted)

Best single-feature separability |AUC-0.5|: **{gate['best_abs_auc']:.3f}**
(`{gate['best_feature']}`); behavioural-only 5-fold CV macro-F1:
**{gate['behav_only_cv_macro_f1']:.3f}** over {gate['n_captures']} captures.
**Gate passed: {gate['passed']}** -- the behavioural features strongly separate real
FTP-BruteForce from real benign (benign always authenticates successfully and does
data transfers; brute force is repeated failed auth). See `separability_analysis.csv`.

## Step 2 - augmented candidates (CIC held-out + v2 LOCO; selection signals only)

{_md_table(cand_rows)}

Selected on CIC+v2-LOCO: **`aug_{selected}`**.

## Final independent test (frozen 36-PCAP set)

| Model | FTP recall | Benign recall | Benign FP | macro-F1 | accuracy |
|---|---|---|---|---|---|
| production | {_f(pr['ftp_recall'])} | {_f(pr['benign_recall'])} | {_f(pr['benign_fp_rate'])} | {pr['macro_f1']:.4f} | {pr['accuracy']:.4f} |
| Candidate 2 | {_f(c2['ftp_recall'])} | {_f(c2['benign_recall'])} | {_f(c2['benign_fp_rate'])} | {c2['macro_f1']:.4f} | {c2['accuracy']:.4f} |
| control (30-feat, CIC+real) | {_f(ct['ftp_recall'])} | {_f(ct['benign_recall'])} | {_f(ct['benign_fp_rate'])} | {ct['macro_f1']:.4f} | {ct['accuracy']:.4f} |
| **aug_{selected} (30+behavioural)** | {_f(nw['ftp_recall'])} | {_f(nw['benign_recall'])} | {_f(nw['benign_fp_rate'])} | {nw['macro_f1']:.4f} | {nw['accuracy']:.4f} |

FTP recall 95% CI (aug): [{indep_ci[f'aug_{selected}']['ftp_recall']['lo95']:.3f},
{indep_ci[f'aug_{selected}']['ftp_recall']['hi95']:.3f}]; benign recall 95% CI:
[{indep_ci[f'aug_{selected}']['benign_recall']['lo95']:.3f},
{indep_ci[f'aug_{selected}']['benign_recall']['hi95']:.3f}]. See `bootstrap_ci_results.csv`.
{shap_md}
## Verdict - {verdict['decision']}

{verdict['summary']}

Balanced-target criteria (all required to break the frontier): {json.dumps(verdict['criteria'])}

**{verdict['conclusion']}**

Promote: **{verdict['promote']}** (never auto-promoted).

### Caveats
{chr(10).join('- ' + c for c in verdict['caveats'])}

## Integrity

Production model, Candidate 2, ml.py/live_capture.py/pcap_validation.py, and the
v1/v2/independent/targeted PCAPs verified unchanged (before==after). No independent
PCAP entered training. Capture-level LOCO with per-fold no-flow-overlap assertions.
CIC behavioural features are NaN (not zero-fill). Candidate models stored outside
webapp_data.

## Files

`separability_analysis.csv`, `candidate_metrics.csv`, `candidate_comparison.csv`,
`independent_test_metrics.csv`, `independent_per_capture_metrics.csv`,
`confusion_*.csv`, `bootstrap_ci_results.csv`, `shap_comparison.csv`,
`leakage_validation.json`, `model_hashes_before_after.json`, `training_metadata.json`,
`final_verdict.json`.
"""
        (out / "report.md").write_text(md)


def _f(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def _md_table(rows):
    if not rows:
        return "(none)"
    cols = list(rows[0].keys())
    def fmt(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "n/a"
        return f"{v:.4f}" if isinstance(v, float) else str(v)
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    body = "\n".join("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |" for r in rows)
    return f"{head}\n{sep}\n{body}"
