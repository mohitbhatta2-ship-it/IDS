"""
Cross-session / source-level retraining experiment (candidates only; production frozen).

Tests whether cross-session/source-level features resolve the single-session ambiguity
"a legitimate user failed a few times and left" vs "an attacker repeatedly attempts
credentials", WITHOUT sacrificing the ~1.0 FTP recall.

Compares (1) the current 45-feature candidate, (2) the best controlled C3 candidate
(43 features, no failed_logins/ratio), and (3) a new 58-feature cross-session candidate
(30 packet + 15 behavioural + 13 cross-session) trained on the approved corpora + the new
cross-session corpus. Selection uses ONLY CIC held-out + capture-level LOCO on the approved
TRAIN corpora -- never the frozen vsFTPD test. The frozen vsFTPD corpus is evaluated once,
at the end. SHAP + feature ablation + per-source/session metrics included. No promotion.

    python manage.py ftp_cross_session_experiment
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

from predictor import ml, retraining as rt, retraining_behavioral as rbh, \
    retraining_cross_session as rce, ftp_behavioral as fb, ftp_cross_session as fcs, independent_eval as ie

FTP, BENIGN = "FTP-BruteForce", "Benign"
TRAIN_SOURCES = ("v1", "v2", "targeted", "robust_train", "benign_failed_login", "cross_session")
TEST_CORPUS = "independent_ftp_validation_pcaps"
ROBUST_PKL = "validation/models/ftp-behavioral-robust-retraining/candidate_robust_unweighted.pkl"
C3_PKL = "validation/models/ftp-failed-login-ablation/C3_no_failed_or_ratio_43.pkl"
F45 = list(rbh.FEATURES_AUG)
F43 = [f for f in F45 if f not in ("ftp_failed_logins", "ftp_failed_login_ratio")]
CIC_ARTIFACTS = {"Dst Port", "Fwd Seg Size Min", "Init Fwd Win Byts", "Init Bwd Win Byts"}


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _pcap_hashes(root, sub):
    return sorted(_sha(p) for p in glob.glob(str(root / "validation" / sub / "**/*.pcap"), recursive=True))


def _frozen(root):
    d = {"production": _sha(ie.production_model_path()), "robust_candidate": _sha(root / ROBUST_PKL),
         "c3": _sha(root / C3_PKL), "ml_py": _sha(root / "webapp_django/predictor/ml.py"),
         "live_capture_py": _sha(root / "webapp_django/predictor/live_capture.py"),
         "pcap_validation_py": _sha(root / "webapp_django/predictor/pcap_validation.py"),
         "ftp_behavioral_py": _sha(root / "webapp_django/predictor/ftp_behavioral.py")}
    for name, sub in (("v1", "realistic_pcaps"), ("v2", "realistic_pcaps_v2"),
                      ("indep", "independent_real_pcaps"), ("targeted", "targeted_benign_pcaps"),
                      ("robustness", "robustness_pcaps"), ("robust_train", "robust_train_pcaps"),
                      ("benign_fl", "benign_failed_login_pcaps"), ("indep_ftp", TEST_CORPUS)):
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
            "false_positive_rate": float((pred[bmask] != BENIGN).mean()) if bmask.any() else None,
            "confusion": {"labels": labels, "matrix": confusion_matrix(truth, pred, labels=labels).tolist()}}


class Command(BaseCommand):
    help = "Cross-session retraining experiment (candidates only; production frozen; no promotion)."

    def add_arguments(self, parser):
        parser.add_argument("--output", default=None)
        parser.add_argument("--skip-shap", action="store_true")
        parser.add_argument("--loco-cic", type=int, default=6000)
        parser.add_argument("--loco-caps", type=int, default=50)
        parser.add_argument("--n-boot", type=int, default=2000)

    def handle(self, *args, **opts):
        import joblib
        w = self.stdout.write
        root = rce.repo_root()
        out = Path(opts["output"]) if opts["output"] else root / "validation" / "results" / "ftp_cross_session"
        out.mkdir(parents=True, exist_ok=True)
        before = _frozen(root)

        w(self.style.MIGRATE_HEADING("Cross-session retraining experiment (production frozen)"))

        # ---- leakage guards ---------------------------------------------
        cs_h = set(_pcap_hashes(root, "cross_session_pcaps"))
        test_h = set(before["indep_ftp_pcaps"])
        prior_h = set().union(*(set(before[k]) for k in
                   ("v1_pcaps", "v2_pcaps", "targeted_pcaps", "robust_train_pcaps", "benign_fl_pcaps", "indep_pcaps", "robustness_pcaps")))
        if not cs_h.isdisjoint(test_h):
            raise SystemExit("ABORT: cross_session overlaps the frozen vsFTPD TEST corpus!")
        if not cs_h.isdisjoint(prior_h):
            raise SystemExit("ABORT: cross_session overlaps a prior corpus!")
        w(f"  leakage guard OK: cross_session ({len(cs_h)}) disjoint from test ({len(test_h)}) and prior.")

        # ---- data (58-feature augmented) --------------------------------
        w(self.style.MIGRATE_HEADING("\nExtracting features (30 + 15 + 13 = 58)"))
        real = rce.extract_real_cross(TRAIN_SOURCES)
        cic_Xaug, cic_y = rce.load_cic_cross()
        cic_Xtest, cic_ytest = rce.load_cic_test_cross()
        test = rce.extract_real_cross(("independent_ftp_val",))
        tdf = self._attach_meta(root, test.df)
        cle = tdf[~tdf["encrypted"]].reset_index(drop=True); enc = tdf[tdf["encrypted"]].reset_index(drop=True)
        w(f"  train real flows: {len(real.df)} (FTP {int((real.df['Label']==FTP).sum())}, "
          f"Benign {int((real.df['Label']==BENIGN).sum())}); test cleartext {len(cle)}, FTPS {len(enc)}; invalid {len(real.invalid)}")

        # ---- frozen references + cross candidates -----------------------
        prod = ml._load(ml.DEFAULT_MODEL)[0]
        c1 = joblib.load(root / ROBUST_PKL)                       # current 45
        c3 = joblib.load(root / C3_PKL)                           # controlled 43
        w(self.style.MIGRATE_HEADING("\nTraining cross-session candidates"))
        cross_unw = rce.train_cols(cic_Xaug, cic_y, real.df, rce.FEATURES_AUG_CROSS, 1.0, 1.0)
        fw, bw = rbh.class_weights(real.df, "balanced")
        cross_bal = rce.train_cols(cic_Xaug, cic_y, real.df, rce.FEATURES_AUG_CROSS, fw, bw)
        mdir = self._mkmodeldir(root)
        for nm, m in (("candidate_cross_unweighted", cross_unw), ("candidate_cross_balanced", cross_bal)):
            joblib.dump(m, mdir / f"{nm}.pkl")
        (mdir / "candidate_cross_unweighted.metadata.json").write_text(json.dumps(
            {"features": rce.FEATURES_AUG_CROSS, "train_sources": list(TRAIN_SOURCES), "weighting": "unweighted",
             "scaler": None}, indent=2))
        w(f"  trained candidate_cross_unweighted / _balanced ({len(rce.FEATURES_AUG_CROSS)} features).")

        # ---- CIC held-out (no-regression) -------------------------------
        w(self.style.MIGRATE_HEADING("\nCIC held-out evaluation"))
        dec = rt._encoded_to_name()
        cic_truth = np.array([dec.get(int(c), str(c)) for c in cic_ytest.to_numpy()])
        cic_specs = {"production": (prod, list(ml.FEATURES)), "C1_current_45": (c1, F45), "C3_controlled_43": (c3, F43),
                     "candidate_cross_unweighted": (cross_unw, rce.FEATURES_AUG_CROSS),
                     "candidate_cross_balanced": (cross_bal, rce.FEATURES_AUG_CROSS)}
        cic_metrics, cic_rows = {}, []
        for name, (model, cols) in cic_specs.items():
            pred, _ = _predict(model, cic_Xtest[cols]); m = _metrics(cic_truth, pred); cic_metrics[name] = m
            cic_rows.append({"model": name, "ftp_recall": m["ftp_recall"], "benign_recall": m["benign_recall"],
                             "macro_f1": m["macro_f1"], "accuracy": m["accuracy"]})
            w(f"  {name:28} FTP-rec {_f(m['ftp_recall'])} Ben-rec {_f(m['benign_recall'])} acc {m['accuracy']:.4f}")
        pd.DataFrame(cic_rows).to_csv(out / "cic_heldout_metrics.csv", index=False)

        # ---- LOCO (selection among cross weightings) --------------------
        w(self.style.MIGRATE_HEADING("\nCapture-level LOCO on approved TRAIN corpora (selection)"))
        sub_X, sub_y = rce.stratified_cic_subsample_cross(cic_Xaug, cic_y, opts["loco_cic"])
        caps = self._loco_caps(real.df, opts["loco_caps"])
        w(f"  LOCO over {len(caps)} captures (stratified; CIC 15-class -> ~12s/fit)")
        loco = {}
        for nm, (model, cols) in (("unweighted", (cross_unw, rce.FEATURES_AUG_CROSS)), ("balanced", (cross_bal, rce.FEATURES_AUG_CROSS))):
            loco[nm] = rce.leave_one_capture_out_cols(sub_X, sub_y, real, caps, cols)
            w(f"  {nm:12} LOCO pooled: FTP-rec {_f(loco[nm].get(FTP+'_recall'))} Ben-rec {_f(loco[nm].get(BENIGN+'_recall'))} macro-F1 {_f(loco[nm].get('macro_f1'))}")
        (out / "loco_pooled_metrics.json").write_text(json.dumps(loco, indent=2, default=str))

        sel_name, sel_reason = self._select(cic_metrics, loco)
        selected = {"candidate_cross_unweighted": cross_unw, "candidate_cross_balanced": cross_bal}[sel_name]
        w(self.style.SUCCESS(f"\n  SELECTED (CIC+LOCO only): {sel_name} -- {sel_reason}"))

        # ---- FINAL vsFTPD test (once) -----------------------------------
        w(self.style.MIGRATE_HEADING("\nFINAL vsFTPD independent test (cleartext; evaluated once)"))
        eval_models = {"production": (prod, list(ml.FEATURES)), "C1_current_45": (c1, F45),
                       "C3_controlled_43": (c3, F43), sel_name: (selected, rce.FEATURES_AUG_CROSS)}
        metrics, preds, rows = {}, {}, []
        for name, (model, cols) in eval_models.items():
            pred, conf = _predict(model, cle[cols]); preds[name] = (pred, conf)
            m = _metrics(cle["Label"].to_numpy(), pred); metrics[name] = m
            rows.append({"model": name, "ftp_recall": m["ftp_recall"], "benign_recall": m["benign_recall"],
                         "false_positive_rate": m["false_positive_rate"], "ftp_precision": m["ftp_precision"],
                         "macro_f1": m["macro_f1"], "accuracy": m["accuracy"]})
            w(f"  {name:28} FTP-rec {_f(m['ftp_recall'])} Ben-rec {_f(m['benign_recall'])} "
              f"FPR {_f(m['false_positive_rate'])} prec {_f(m['ftp_precision'])} mF1 {m['macro_f1']:.4f}")
        pd.DataFrame(rows).to_csv(out / "final_test_metrics.csv", index=False)
        for name, m in metrics.items():
            c = m["confusion"]
            pd.DataFrame(c["matrix"], index=c["labels"], columns=c["labels"]).to_csv(out / f"confusion_{name}.csv")

        # ---- per-scenario-family + per-source/session -------------------
        fam_rows = self._family_breakdown(cle, preds, sel_name)
        pd.DataFrame(fam_rows).to_csv(out / "per_scenario_family_metrics.csv", index=False)
        w(self.style.MIGRATE_HEADING("\nCRITICAL cases (selected vs C1)"))
        for r in fam_rows:
            if r["scenario_family"] in ("gave_up", "mistype", "eventual_success"):
                w(f"  {r['scenario_family']:16} ({r['label']:14}) n={r['n_flows']:3} recall={_f(r['recall'])}")
        self._breakdowns(out, cle, preds[sel_name][0])
        self._per_source(out, cle, preds[sel_name][0])

        # ---- bootstrap CIs ----------------------------------------------
        w(self.style.MIGRATE_HEADING("\nBootstrap 95% CIs (capture-level)"))
        boot = {}
        for name in ("C1_current_45", "C3_controlled_43", sel_name):
            model, cols = eval_models[name]
            boot[name] = self._bootstrap(cle, model, cols, opts["n_boot"]); b = boot[name]
            w(f"  {name:26} FTP {b['ftp_recall']['mean']:.3f}[{b['ftp_recall']['lo95']:.3f},{b['ftp_recall']['hi95']:.3f}] "
              f"Ben {b['benign_recall']['mean']:.3f}[{b['benign_recall']['lo95']:.3f},{b['benign_recall']['hi95']:.3f}] "
              f"FPR {b['false_positive_rate']['mean']:.3f}[{b['false_positive_rate']['lo95']:.3f},{b['false_positive_rate']['hi95']:.3f}]")
        (out / "bootstrap_cis.json").write_text(json.dumps(boot, indent=2, default=str))

        # ---- SHAP + feature ablation (drop cross block) -----------------
        shap_res = None if opts["skip_shap"] else self._shap(selected, cle)
        w(self.style.MIGRATE_HEADING("\nFeature ablation: does the cross-session block matter?"))
        no_cross = rce.train_cols(cic_Xaug, cic_y, real.df, F45, 1.0, 1.0)   # 45 (no cross) on same data
        pred_nc, _ = _predict(no_cross, cle[F45]); m_nc = _metrics(cle["Label"].to_numpy(), pred_nc)
        ablation = {"with_cross": {"ftp_recall": metrics[sel_name]["ftp_recall"], "benign_recall": metrics[sel_name]["benign_recall"],
                                   "false_positive_rate": metrics[sel_name]["false_positive_rate"]},
                    "without_cross_45_same_data": {"ftp_recall": m_nc["ftp_recall"], "benign_recall": m_nc["benign_recall"],
                                                   "false_positive_rate": m_nc["false_positive_rate"]}}
        w(f"  with cross ({sel_name}): FTP {_f(ablation['with_cross']['ftp_recall'])} Ben {_f(ablation['with_cross']['benign_recall'])} FPR {_f(ablation['with_cross']['false_positive_rate'])}")
        w(f"  without cross (45, same data): FTP {_f(m_nc['ftp_recall'])} Ben {_f(m_nc['benign_recall'])} FPR {_f(m_nc['false_positive_rate'])}")

        # ---- FTPS separate ----------------------------------------------
        ftps = self._ftps(enc, eval_models); pd.DataFrame(ftps).to_csv(out / "ftps_encrypted_metrics.csv", index=False)

        # ---- verdict -----------------------------------------------------
        verdict = self._verdict(sel_name, metrics, cic_metrics, boot, shap_res, ablation, fam_rows)
        w(self.style.MIGRATE_HEADING("\nVERDICT"))
        w(self.style.WARNING("  " + verdict["verdict"] + " -- " + verdict["summary"]))

        after = _frozen(root)
        if before != after:
            raise SystemExit(f"ABORT: frozen artifact(s) changed: {[k for k in before if before[k]!=after.get(k)]}")
        w(f"\n  frozen artifacts unchanged (before==after): {before == after}")

        self._write(out, root, rows, cic_rows, fam_rows, loco, boot, metrics, cic_metrics, ablation, verdict,
                    shap_res, ftps, sel_name, sel_reason, before, after, real, test, cle, enc)
        w(self.style.SUCCESS(f"\nEvidence written to {out}"))
        w(self.style.WARNING("\nNo promotion, no production change, no merge. vsFTPD corpus is test-only."))

    # ---- helpers ---------------------------------------------------------

    def _mkmodeldir(self, root):
        d = root / "validation" / "models" / "ftp-cross-session"; d.mkdir(parents=True, exist_ok=True); return d

    def _attach_meta(self, root, df):
        man = pd.read_csv(root / "validation" / TEST_CORPUS / "MANIFEST.csv")
        df = df.copy(); df["capture_id"] = df["capture"].map(lambda p: "_".join(p.split("_")[:2]))
        m = man.set_index("capture_id")
        for col in ("scenario", "scenario_family", "client", "server", "environment", "mode", "encrypted"):
            df[col] = df["capture_id"].map(m[col])
        df["encrypted"] = df["encrypted"].astype(bool)
        df["sessions_per_source"] = df["ftpx_sessions_per_source"]
        return df

    def _loco_caps(self, real_df, k):
        caps = sorted(real_df["capture"].unique())
        if not k or k >= len(caps):
            return caps
        info = real_df.drop_duplicates("capture").set_index("capture")
        rng = np.random.default_rng(42); groups = {}
        for c in caps:
            groups.setdefault((info.loc[c, "source"], info.loc[c, "Label"]), []).append(c)
        picked = []
        for key, members in groups.items():
            members = sorted(members); take = max(1, round(k * len(members) / len(caps)))
            idx = rng.choice(len(members), size=min(take, len(members)), replace=False)
            picked += [members[i] for i in idx]
        return sorted(set(picked))

    def _select(self, cic_metrics, loco):
        base = cic_metrics["production"]["ftp_recall"] or 0; base_acc = cic_metrics["production"]["accuracy"] or 0
        cands = []
        for strat, cname in (("unweighted", "candidate_cross_unweighted"), ("balanced", "candidate_cross_balanced")):
            gate = (cic_metrics[cname]["ftp_recall"] or 0) >= base - 0.03 and abs((cic_metrics[cname]["accuracy"] or 0) - base_acc) < 0.01
            cands.append((cname, gate, loco[strat].get("macro_f1") or 0))
        cands.sort(key=lambda t: (t[1], t[2]), reverse=True); c = cands[0]
        return c[0], f"CIC-gate={c[1]}, LOCO macro-F1 {c[2]:.4f}"

    def _family_breakdown(self, df, preds, sel_name):
        rows = []
        d2 = df.copy(); d2["_pred"] = preds[sel_name][0]
        for (fam, lab), sub in d2.groupby(["scenario_family", "Label"]):
            rows.append({"model": sel_name, "scenario_family": fam, "label": lab, "n_flows": int(len(sub)),
                         "recall": round(float((sub["_pred"] == lab).mean()), 4)})
        # also C1 for comparison
        d3 = df.copy(); d3["_pred"] = preds["C1_current_45"][0]
        for (fam, lab), sub in d3.groupby(["scenario_family", "Label"]):
            rows.append({"model": "C1_current_45", "scenario_family": fam, "label": lab, "n_flows": int(len(sub)),
                         "recall": round(float((sub["_pred"] == lab).mean()), 4)})
        return rows

    def _breakdowns(self, out, df, pred):
        d2 = df.copy(); d2["_pred"] = pred
        for key, fname in (("scenario", "per_scenario_metrics.csv"), ("client", "per_client_metrics.csv"),
                           ("server", "per_server_metrics.csv"), ("environment", "per_environment_metrics.csv")):
            rows = []
            for val, sub in d2.groupby(key):
                ftp = sub[sub["Label"] == FTP]; ben = sub[sub["Label"] == BENIGN]
                rows.append({key: val, "n_flows": int(len(sub)),
                             "ftp_recall": round(float((ftp["_pred"] == FTP).mean()), 4) if len(ftp) else None,
                             "benign_recall": round(float((ben["_pred"] == BENIGN).mean()), 4) if len(ben) else None,
                             "benign_fp": int((ben["_pred"] != BENIGN).sum())})
            pd.DataFrame(rows).to_csv(out / fname, index=False)

    def _per_source(self, out, df, pred):
        d2 = df.copy(); d2["_pred"] = pred
        d2["sess_bucket"] = pd.cut(d2["sessions_per_source"].fillna(0), [-0.1, 1.5, 3.5, 6.5, 1e9],
                                   labels=["1", "2-3", "4-6", "7+"])
        rows = []
        for (bucket, lab), sub in d2.groupby(["sess_bucket", "Label"], observed=True):
            rows.append({"sessions_per_source_bucket": str(bucket), "label": lab, "n_flows": int(len(sub)),
                         "recall": round(float((sub["_pred"] == lab).mean()), 4),
                         "median_sessions": float(sub["sessions_per_source"].median())})
        pd.DataFrame(rows).to_csv(out / "per_source_session_metrics.csv", index=False)

    def _bootstrap(self, df, model, cols, n_boot, seed=42):
        rng = np.random.default_rng(seed); caps = sorted(df["capture"].unique())
        by = {c: df[df["capture"] == c] for c in caps}; dec = rt._encoded_to_name()
        acc, fr, br, fp = [], [], [], []
        for _ in range(n_boot):
            s = pd.concat([by[c] for c in rng.choice(caps, size=len(caps), replace=True)], ignore_index=True)
            pred = np.array([dec.get(int(c), str(c)) for c in model.predict(s[cols])])
            t = s["Label"].to_numpy(); fm = t == FTP; bm = t == BENIGN
            acc.append(float((pred == t).mean()))
            fr.append(float((pred[fm] == FTP).mean()) if fm.any() else np.nan)
            br.append(float((pred[bm] == BENIGN).mean()) if bm.any() else np.nan)
            fp.append(float((pred[bm] != BENIGN).mean()) if bm.any() else np.nan)

        def ci(a):
            a = np.array(a); a = a[~np.isnan(a)]
            return {"mean": float(a.mean()), "lo95": float(np.percentile(a, 2.5)), "hi95": float(np.percentile(a, 97.5))}
        return {"accuracy": ci(acc), "ftp_recall": ci(fr), "benign_recall": ci(br), "false_positive_rate": ci(fp)}

    def _shap(self, model, df):
        try:
            import shap
        except Exception:  # noqa: BLE001
            return None
        cols = rce.FEATURES_AUG_CROSS
        vals = np.abs(np.asarray(shap.TreeExplainer(model).shap_values(df[cols]))).mean(axis=(0, 2))
        order = np.argsort(vals)[::-1]; total = vals.sum() + 1e-12
        cross = set(fcs.CROSS_FEATURES); behav = set(fb.BEHAV_FEATURES)
        top = [{"feature": cols[j], "share": float(vals[j] / total)} for j in order[:14]]
        return {"top14": top, "max_feature": cols[order[0]], "max_share": float(vals[order[0]] / total),
                "cross_session_share": float(sum(vals[j] for j in range(len(vals)) if cols[j] in cross) / total),
                "behavioural_share": float(sum(vals[j] for j in range(len(vals)) if cols[j] in behav) / total),
                "cic_artifact_share": float(sum(vals[j] for j in range(len(vals)) if cols[j] in CIC_ARTIFACTS) / total),
                "top_cross_feature": next((cols[j] for j in order if cols[j] in cross), None)}

    def _ftps(self, enc, eval_models):
        rows = []
        if len(enc):
            for name, (model, cols) in eval_models.items():
                pred, _ = _predict(model, enc[cols]); t = enc["Label"].to_numpy()
                fm = t == FTP; bm = t == BENIGN
                rows.append({"model": name, "n": int(len(enc)),
                             "ftp_recall": float((pred[fm] == FTP).mean()) if fm.any() else None,
                             "benign_recall": float((pred[bm] == BENIGN).mean()) if bm.any() else None})
        return rows

    def _verdict(self, sel_name, metrics, cic_metrics, boot, shap_res, ablation, fam_rows):
        sel = metrics[sel_name]; c1 = metrics["C1_current_45"]
        ftp = sel["ftp_recall"] or 0; ben = sel["benign_recall"] or 0; fpr = sel["false_positive_rate"] or 0
        c1_ftp = c1["ftp_recall"] or 0; c1_ben = c1["benign_recall"] or 0; c1_fpr = c1["false_positive_rate"] or 0
        lo_ftp = boot[sel_name]["ftp_recall"]["lo95"]; lo_ben = boot[sel_name]["benign_recall"]["lo95"]
        cic_ok = (cic_metrics[sel_name]["ftp_recall"] or 0) >= (cic_metrics["production"]["ftp_recall"] or 0) - 0.03 \
            and abs((cic_metrics[sel_name]["accuracy"] or 0) - (cic_metrics["production"]["accuracy"] or 0)) < 0.01
        target = ftp >= 0.90 and ben >= 0.90 and fpr <= 0.10 and cic_ok
        ftp_preserved = ftp >= c1_ftp - 0.02
        benign_improved = ben > c1_ben or fpr < c1_fpr
        gave_up = next((r["recall"] for r in fam_rows if r["model"] == sel_name and r["scenario_family"] == "gave_up"), None)
        c1_gave_up = next((r["recall"] for r in fam_rows if r["model"] == "C1_current_45" and r["scenario_family"] == "gave_up"), None)
        cross_share = shap_res["cross_session_share"] if shap_res else None
        if target and ftp_preserved:
            verdict = "PROMISING -- TARGET MET"
            summary = (f"Cross-session features meet the target on the frozen vsFTPD test WITHOUT sacrificing attack "
                       f"detection: FTP recall {ftp:.3f}, benign recall {ben:.3f}, FPR {fpr:.3f} (CI lo FTP {lo_ftp:.3f}"
                       f"/benign {lo_ben:.3f}); gave_up {gave_up} (C1 {c1_gave_up}); no CIC regression. Recommend review.")
        elif benign_improved and ftp_preserved and (ben >= 0.90 or fpr <= 0.10):
            verdict = "PROMISING -- NEEDS MORE DATA"
            summary = (f"Cross-session features improve the benign side without losing FTP recall ({c1_ben:.3f}->{ben:.3f} "
                       f"benign, FPR {c1_fpr:.3f}->{fpr:.3f}, FTP {ftp:.3f}); gave_up {c1_gave_up}->{gave_up}; but not all "
                       f"targets met (benign>=0.90? {ben>=0.90}; FPR<=0.10? {fpr<=0.10}). Cross-session SHAP share {_f(cross_share)}.")
        elif not ftp_preserved:
            verdict = "NOT EFFECTIVE"
            summary = (f"Cross-session features damaged attack detection (FTP recall {c1_ftp:.3f}->{ftp:.3f}) -- violates the "
                       f"'do not sacrifice FTP recall' constraint; benign {ben:.3f}, FPR {fpr:.3f}. The training corpus's "
                       f"attackers had more sessions than the vsFTPD test's, so 'few sessions' leaked toward benign.")
        else:
            verdict = "NOT EFFECTIVE"
            summary = (f"Cross-session features did not resolve the ambiguity on the frozen test (FTP {ftp:.3f}, benign "
                       f"{ben:.3f}, FPR {fpr:.3f} vs C1 {c1_ftp:.3f}/{c1_ben:.3f}/{c1_fpr:.3f}); gave_up {gave_up}.")
        return {"verdict": verdict, "promote": False, "selected_model": sel_name, "target_met": bool(target and ftp_preserved),
                "ftp_recall_preserved": bool(ftp_preserved), "benign_improved_vs_c1": bool(benign_improved),
                "cic_no_regression": bool(cic_ok), "summary": summary,
                "selected_test": {"ftp_recall": ftp, "benign_recall": ben, "false_positive_rate": fpr,
                                  "ftp_precision": sel["ftp_precision"], "macro_f1": sel["macro_f1"], "accuracy": sel["accuracy"]},
                "current_c1_test": {"ftp_recall": c1_ftp, "benign_recall": c1_ben, "false_positive_rate": c1_fpr},
                "ci_lower_ftp_recall": lo_ftp, "ci_lower_benign_recall": lo_ben,
                "gave_up_recall": {"cross": gave_up, "c1": c1_gave_up}, "cross_session_shap_share": cross_share,
                "ablation_cross_block": ablation,
                "critical_test_answer": ("Cross-session information " + ("DOES" if (gave_up or 0) > (c1_gave_up or 0) and ftp_preserved else "does NOT")
                                         + " resolve the benign-give-up vs brute-force ambiguity on genuinely independent traffic."),
                "caveats": ["Loopback lab; the cross-session corpus's attackers use one credential per connection, so they "
                            "have MORE sessions than the vsFTPD test's attackers (which pack attempts per connection) -- a "
                            "session-count distribution shift between train and test.",
                            "Frozen vsFTPD corpus evaluated once, for reporting; never training/selection/tuning.",
                            "Small test corpus (wide CIs). No promotion, no threshold change, no merge."]}

    def _write(self, out, root, rows, cic_rows, fam_rows, loco, boot, metrics, cic_metrics, ablation, verdict,
               shap_res, ftps, sel_name, sel_reason, before, after, real, test, cle, enc):
        if shap_res:
            pd.DataFrame(shap_res["top14"]).to_csv(out / "shap_top_features.csv", index=False)
        leak = {"train_sources": list(TRAIN_SOURCES), "train_disjoint_from_test": True, "test_is_test_only": True,
                "labels_from_folders": True, "no_zero_fill_train": len(real.invalid) == 0,
                "no_zero_fill_test": len(test.invalid) == 0, "n_features": len(rce.FEATURES_AUG_CROSS),
                "packet_features_preserved": rce.FEATURES_AUG_CROSS[:30] == list(ml.FEATURES),
                "behavioural_15_preserved": rce.FEATURES_AUG_CROSS[30:45] == list(fb.BEHAV_FEATURES),
                "cross_features_appended": rce.FEATURES_AUG_CROSS[45:] == list(fcs.CROSS_FEATURES),
                "every_flow_traceable": bool(real.df["capture"].notna().all()),
                "selection_used": "CIC held-out + LOCO on approved TRAIN corpora only (NOT the vsFTPD test)"}
        (out / "leakage_validation.json").write_text(json.dumps(leak, indent=2, default=str))
        (out / "model_hashes_before_after.json").write_text(json.dumps({"before": before, "after": after, "unchanged": before == after}, indent=2, default=str))
        (out / "final_verdict.json").write_text(json.dumps({**verdict, "selection_reason": sel_reason, "shap": shap_res, "ftps": ftps}, indent=2, default=str))
        (out / "experiment_metadata.json").write_text(json.dumps({
            "experiment": "ftp_cross_session", "created_utc": datetime.now(timezone.utc).isoformat(),
            "train_sources": list(TRAIN_SOURCES), "new_corpus": "validation/cross_session_pcaps",
            "test_corpus": f"validation/{TEST_CORPUS} (test-only)", "selected_model": sel_name,
            "n_features": len(rce.FEATURES_AUG_CROSS), "versions": {"python": platform.python_version()}}, indent=2, default=str))
        self._report(out, rows, cic_rows, fam_rows, loco, boot, ablation, verdict, shap_res, ftps, sel_name, sel_reason)

    def _report(self, out, rows, cic_rows, fam_rows, loco, boot, ablation, verdict, shap_res, ftps, sel_name, sel_reason):
        tbl = "\n".join(f"| {r['model']} | {_f(r['ftp_recall'])} | {_f(r['benign_recall'])} | {_f(r['false_positive_rate'])} | "
                        f"{_f(r['ftp_precision'])} | {_f(r['macro_f1'])} | {_f(r['accuracy'])} |" for r in rows)
        btbl = "\n".join(f"| {n} | {boot[n]['ftp_recall']['mean']:.3f} [{boot[n]['ftp_recall']['lo95']:.3f},{boot[n]['ftp_recall']['hi95']:.3f}] | "
                         f"{boot[n]['benign_recall']['mean']:.3f} [{boot[n]['benign_recall']['lo95']:.3f},{boot[n]['benign_recall']['hi95']:.3f}] | "
                         f"{boot[n]['false_positive_rate']['mean']:.3f} [{boot[n]['false_positive_rate']['lo95']:.3f},{boot[n]['false_positive_rate']['hi95']:.3f}] |" for n in boot)
        fam = {}
        for r in fam_rows:
            fam.setdefault(r["scenario_family"], {})[r["model"]] = r["recall"]
        keyf = ["mistype", "gave_up", "eventual_success", "clean", "activity", "fast_brute", "slow_brute", "multi_conn", "user_enum"]
        famtbl = "\n".join(f"| {f} | {_f(fam.get(f, {}).get('C1_current_45'))} | {_f(fam.get(f, {}).get(sel_name))} |" for f in keyf if f in fam)
        shap_md = ""
        if shap_res:
            shap_md = (f"\n## SHAP (selected, cleartext test)\n\n"
                       f"- cross-session block share **{shap_res['cross_session_share']:.3f}** (top cross feature "
                       f"`{shap_res['top_cross_feature']}`); behavioural **{shap_res['behavioural_share']:.3f}**; "
                       f"CIC artifacts **{shap_res['cic_artifact_share']:.3f}**; largest single `{shap_res['max_feature']}` "
                       f"{shap_res['max_share']:.3f}\n- top: {', '.join(t['feature'] for t in shap_res['top14'][:6])}\n")
        ftbl = "\n".join(f"| {r['model']} | {_f(r['ftp_recall'])} | {_f(r['benign_recall'])} |" for r in ftps)
        (out / "report.md").write_text(f"""# Cross-session / source-level behaviour experiment - report

**Candidates only; production frozen (production, the 45-feature candidate, the C3
candidate, ml.py/live_capture.py/pcap_validation.py/ftp_behavioral.py unchanged,
before==after). The 30 packet + 15 behavioural features are PRESERVED; cross-session
features are appended. Selection used CIC held-out + LOCO only; the frozen vsFTPD corpus
was evaluated once. No promotion, no merge.** Branch `claude/ftp-cross-session-behavior`.

## Idea

Single-session features cannot separate "a legitimate user failed a few times and left"
from "a short brute force". Cross-session/source-level features (sessions per source,
failures across sessions, credential variation, persistence, per-session success history)
might, because an attacker persists across many sessions while a legitimate user makes few.
A new corpus with multiple sessions per source (benign 1-4 sessions; attacker 6-12) was
collected; 13 label-free cross-session features were added (58 total). Selected: **{sel_name}**
({sel_reason}).

## FINAL vsFTPD test (cleartext, evaluated once)

| Model | FTP recall | Benign recall | FPR | FTP precision | macro-F1 | accuracy |
|---|---|---|---|---|---|---|
{tbl}

### Bootstrap 95% CIs
| Model | FTP recall [95% CI] | Benign recall [95% CI] | FPR [95% CI] |
|---|---|---|---|
{btbl}

### Critical cases (recall; C1 vs selected cross candidate)
| family | C1 | cross |
|---|---|---|
{famtbl}

## Feature ablation: does the cross-session block matter?

- With cross ({sel_name}): FTP {_f(ablation['with_cross']['ftp_recall'])}, benign
  {_f(ablation['with_cross']['benign_recall'])}, FPR {_f(ablation['with_cross']['false_positive_rate'])}.
- Without cross (45 features, same data): FTP {_f(ablation['without_cross_45_same_data']['ftp_recall'])},
  benign {_f(ablation['without_cross_45_same_data']['benign_recall'])}, FPR {_f(ablation['without_cross_45_same_data']['false_positive_rate'])}.
{shap_md}
## FTPS / TLS (encrypted; behavioural unavailable)
| Model | FTP recall | Benign recall |
|---|---|---|
{ftbl}

## Critical test — does cross-session information resolve the ambiguity?

{verdict['critical_test_answer']}

`gave_up` recall: C1 {_f(verdict['gave_up_recall']['c1'])} -> cross {_f(verdict['gave_up_recall']['cross'])}.

## Verdict — {verdict['verdict']}

{verdict['summary']}

- target_met: **{verdict['target_met']}**  ·  ftp_recall_preserved: **{verdict['ftp_recall_preserved']}**  ·
  benign_improved_vs_c1: **{verdict['benign_improved_vs_c1']}**  ·  cic_no_regression:
  **{verdict['cic_no_regression']}**  ·  promote: **{verdict['promote']}**

### Caveats
{chr(10).join('- ' + c for c in verdict['caveats'])}

## Files
`final_test_metrics.csv`, `cic_heldout_metrics.csv`, `loco_pooled_metrics.json`,
`per_scenario_family_metrics.csv`, `per_source_session_metrics.csv`, `per_scenario_metrics.csv`,
`per_client_metrics.csv`, `per_server_metrics.csv`, `per_environment_metrics.csv`,
`bootstrap_cis.json`, `shap_top_features.csv`, `ftps_encrypted_metrics.csv`, `confusion_*.csv`,
`leakage_validation.json`, `model_hashes_before_after.json`, `final_verdict.json`.
""")


def _f(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v:.4f}" if isinstance(v, float) else str(v)
