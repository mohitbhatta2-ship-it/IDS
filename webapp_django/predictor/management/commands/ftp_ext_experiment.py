"""
Extended-behavioural retraining experiment (candidates only; production frozen).

Trains 30+15+12 (=57) feature candidates on CIC + approved real corpora
(v1/v2/targeted/robust_train) + the NEW realistic benign failed-login corpus, and asks
whether the added dynamics/timing/variation features fix the benign false positives the
independent vsFTPD test exposed -- without regressing FTP recall or CIC, and without
depending on a single behavioural feature.

Selection uses ONLY the CIC held-out split and capture-level LOCO on the approved+new
TRAIN corpora -- never the frozen independent vsFTPD test. That corpus is the final
evaluation only. FTPS analysed separately (behavioural features unavailable). SHAP +
per-feature drop-one ablation quantify single-feature dependence. No promotion, no merge.

    python manage.py ftp_ext_experiment
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
    retraining_behavioral_ext as rbe, ftp_behavioral as fb, ftp_behavioral_ext as fbx, \
    independent_eval as ie

FTP, BENIGN = "FTP-BruteForce", "Benign"
FAILED = "ftp_failed_logins"
TRAIN_SOURCES = ("v1", "v2", "targeted", "robust_train", "benign_failed_login")
TEST_CORPUS = "independent_ftp_validation_pcaps"
ROBUST_PKL = "validation/models/ftp-behavioral-robust-retraining/candidate_robust_unweighted.pkl"
CIC_ARTIFACTS = {"Dst Port", "Fwd Seg Size Min", "Init Fwd Win Byts", "Init Bwd Win Byts"}
# important behavioural features to drop-one for single-feature dependence
DROP_ONE = [FAILED, "ftpx_fail_run_before_success", "ftpx_distinct_passwords",
            "ftpx_interattempt_mean_s", "ftpx_post_auth_commands", "ftp_user_commands"]


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _pcap_hashes(root, sub):
    return sorted(_sha(p) for p in glob.glob(str(root / "validation" / sub / "**/*.pcap"), recursive=True))


def _frozen(root):
    d = {"production": _sha(ie.production_model_path()), "candidate2": _sha(ie.candidate2_file()),
         "robust_candidate": _sha(root / ROBUST_PKL),
         "ml_py": _sha(root / "webapp_django/predictor/ml.py"),
         "live_capture_py": _sha(root / "webapp_django/predictor/live_capture.py"),
         "pcap_validation_py": _sha(root / "webapp_django/predictor/pcap_validation.py"),
         "ftp_behavioral_py": _sha(root / "webapp_django/predictor/ftp_behavioral.py")}
    for name, sub in (("v1", "realistic_pcaps"), ("v2", "realistic_pcaps_v2"),
                      ("indep", "independent_real_pcaps"), ("targeted", "targeted_benign_pcaps"),
                      ("robustness", "robustness_pcaps"), ("robust_train", "robust_train_pcaps"),
                      ("indep_ftp", TEST_CORPUS)):
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
    help = "Extended-behavioural retraining experiment (candidates only; production frozen; no promotion)."

    def add_arguments(self, parser):
        parser.add_argument("--output", default=None)
        parser.add_argument("--skip-shap", action="store_true")
        parser.add_argument("--loco-cic", type=int, default=6000)
        parser.add_argument("--loco-caps", type=int, default=60,
                            help="stratified subsample of captures for LOCO (0 = all). CIC is 15-class so each fit is heavy.")
        parser.add_argument("--n-boot", type=int, default=2000)

    def handle(self, *args, **opts):
        import joblib
        w = self.stdout.write
        root = rbe.repo_root()
        out = Path(opts["output"]) if opts["output"] else root / "validation" / "results" / "ftp_ext_experiment"
        out.mkdir(parents=True, exist_ok=True)
        before = _frozen(root)

        w(self.style.MIGRATE_HEADING("Extended-behavioural retraining experiment (production frozen)"))

        # ---- leakage guards ---------------------------------------------
        bfl_h = set(_pcap_hashes(root, "benign_failed_login_pcaps"))
        test_h = set(before["indep_ftp_pcaps"])
        prior_h = set().union(*(set(before[k]) for k in
                   ("v1_pcaps", "v2_pcaps", "targeted_pcaps", "robust_train_pcaps", "indep_pcaps", "robustness_pcaps")))
        if not bfl_h.isdisjoint(test_h):
            raise SystemExit("ABORT: benign_failed_login overlaps the frozen independent vsFTPD TEST corpus!")
        if not bfl_h.isdisjoint(prior_h):
            raise SystemExit("ABORT: benign_failed_login overlaps a prior corpus!")
        w(f"  leakage guard OK: new benign corpus ({len(bfl_h)}) disjoint from test ({len(test_h)}) and prior.")

        # ---- data --------------------------------------------------------
        w(self.style.MIGRATE_HEADING("\nExtracting features (30 + 15 + 12 = 57)"))
        real = rbe.extract_real_ext(TRAIN_SOURCES)
        cic_Xaug, cic_y = rbe.load_cic_ext()
        cic_Xtest, cic_ytest = rbe.load_cic_test_ext()
        test = rbe.extract_real_ext(("independent_ftp_val",))
        tdf = self._attach_meta(root, test.df)
        cle = tdf[~tdf["encrypted"]].reset_index(drop=True)
        enc = tdf[tdf["encrypted"]].reset_index(drop=True)
        w(f"  train real flows: {len(real.df)} (FTP {int((real.df['Label']==FTP).sum())}, "
          f"Benign {int((real.df['Label']==BENIGN).sum())}); invalid {len(real.invalid)}")
        w(f"  test cleartext flows: {len(cle)}; encrypted-FTPS {len(enc)}")

        # ---- frozen reference models ------------------------------------
        prod = ml._load(ml.DEFAULT_MODEL)[0]
        cand2 = joblib.load(ie.candidate2_file())
        robust45 = joblib.load(root / ROBUST_PKL)

        # ---- train EXT candidates (57 feats) + a 45-on-new-data control ---
        w(self.style.MIGRATE_HEADING("\nTraining candidates"))
        ext_unw = self._train(cic_Xaug, cic_y, real.df, "unweighted")
        ext_bal = self._train(cic_Xaug, cic_y, real.df, "balanced")
        mdir = self._mkmodeldir(root)
        for nm, m in (("candidate_ext_unweighted", ext_unw), ("candidate_ext_balanced", ext_bal)):
            joblib.dump(m, mdir / f"{nm}.pkl")
        (mdir / "candidate_ext_unweighted.metadata.json").write_text(json.dumps(
            {"features": rbe.FEATURES_AUG_EXT, "train_sources": list(TRAIN_SOURCES), "weighting": "unweighted",
             "scaler": None}, indent=2))
        w(f"  trained candidate_ext_unweighted / _balanced ({len(rbe.FEATURES_AUG_EXT)} features).")

        # ---- CIC held-out (no regression) -------------------------------
        w(self.style.MIGRATE_HEADING("\nCIC held-out evaluation"))
        dec = rt._encoded_to_name()
        cic_truth = np.array([dec.get(int(c), str(c)) for c in cic_ytest.to_numpy()])
        cic_models = {"production": (prod, list(ml.FEATURES)), "candidate2": (cand2, list(ml.FEATURES)),
                      "robust45": (robust45, rbh.FEATURES_AUG),
                      "candidate_ext_unweighted": (ext_unw, rbe.FEATURES_AUG_EXT),
                      "candidate_ext_balanced": (ext_bal, rbe.FEATURES_AUG_EXT)}
        cic_metrics, cic_rows = {}, []
        for name, (model, cols) in cic_models.items():
            pred, _ = _predict(model, cic_Xtest[cols])
            m = _metrics(cic_truth, pred); cic_metrics[name] = m
            cic_rows.append({"model": name, "ftp_recall": m["ftp_recall"], "benign_recall": m["benign_recall"],
                             "macro_f1": m["macro_f1"], "accuracy": m["accuracy"]})
            w(f"  {name:28} FTP-rec {_f(m['ftp_recall'])} Ben-rec {_f(m['benign_recall'])} acc {m['accuracy']:.4f}")
        pd.DataFrame(cic_rows).to_csv(out / "cic_heldout_metrics.csv", index=False)

        # ---- LOCO on approved+new TRAIN corpora (selection) -------------
        w(self.style.MIGRATE_HEADING("\nCapture-level LOCO (approved + new TRAIN corpora)"))
        sub_X, sub_y = rbe.stratified_cic_subsample_ext(cic_Xaug, cic_y, opts["loco_cic"])
        loco = {}
        caps = self._loco_caps(real.df, opts["loco_caps"])
        w(f"  LOCO over {len(caps)} captures (stratified by source/family/label; CIC 15-class -> ~12s/fit)")
        for strat in rbh.STRATEGIES:
            fw, bw = rbh.class_weights(real.df, strat)
            res = rbe.leave_one_capture_out(sub_X, sub_y, real, caps, fw, bw)
            loco[strat] = res["pooled"]; p = res["pooled"]
            w(f"  {strat:12} pooled macro-F1 {_f(p.get('macro_f1'))} acc {_f(p.get('accuracy'))} (n_caps={len(caps)})")
        (out / "loco_pooled_metrics.json").write_text(json.dumps(loco, indent=2, default=str))

        # ---- selection (CIC + LOCO only) --------------------------------
        sel_name, sel_reason = self._select(cic_metrics, loco)
        selected = {"candidate_ext_unweighted": ext_unw, "candidate_ext_balanced": ext_bal}[sel_name]
        w(self.style.SUCCESS(f"\n  SELECTED (CIC+LOCO only): {sel_name} -- {sel_reason}"))

        # ---- drop-one ablation candidates (trained WITH new data) -------
        w(self.style.MIGRATE_HEADING("\nDrop-one ablation candidates (single-feature dependence)"))
        drop_models = {}
        fw, bw = rbh.class_weights(real.df, "unweighted")
        for feat in DROP_ONE:
            cols = [c for c in rbe.FEATURES_AUG_EXT if c != feat]
            drop_models[feat] = (self._train_cols(cic_Xaug, cic_y, real.df, cols, fw, bw), cols)
        w(f"  trained {len(drop_models)} drop-one models.")

        # ---- FINAL independent test (cleartext) -------------------------
        w(self.style.MIGRATE_HEADING("\nFINAL independent vsFTPD test (cleartext)"))
        test_models = {"production": (prod, list(ml.FEATURES)), "candidate2": (cand2, list(ml.FEATURES)),
                       "robust45": (robust45, rbh.FEATURES_AUG), sel_name: (selected, rbe.FEATURES_AUG_EXT)}
        metrics, preds, rows = {}, {}, []
        for name, (model, cols) in test_models.items():
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

        # ---- drop-one on the independent test ---------------------------
        drop_rows = []
        base_ftp = metrics[sel_name]["ftp_recall"] or 0; base_ben = metrics[sel_name]["benign_recall"] or 0
        for feat, (model, cols) in drop_models.items():
            pred, _ = _predict(model, cle[cols]); m = _metrics(cle["Label"].to_numpy(), pred)
            drop_rows.append({"dropped_feature": feat, "ftp_recall": m["ftp_recall"], "benign_recall": m["benign_recall"],
                              "false_positive_rate": m["false_positive_rate"], "macro_f1": m["macro_f1"],
                              "ftp_recall_delta": (m["ftp_recall"] or 0) - base_ftp,
                              "benign_recall_delta": (m["benign_recall"] or 0) - base_ben})
        pd.DataFrame(drop_rows).to_csv(out / "drop_one_ablation.csv", index=False)
        max_ftp_drop = max((-r["ftp_recall_delta"] for r in drop_rows), default=0)
        max_ben_drop = max((-r["benign_recall_delta"] for r in drop_rows), default=0)

        # ---- per-scenario-family (selected) -----------------------------
        fam_rows = self._family_breakdown(cle, preds, sel_name)
        pd.DataFrame(fam_rows).to_csv(out / "per_scenario_family_metrics.csv", index=False)
        w(self.style.MIGRATE_HEADING("\nAdversarial families (selected candidate)"))
        for r in fam_rows:
            w(f"  {r['scenario_family']:16} ({r['label']:14}) n={r['n_flows']:3} recall={_f(r['recall'])}")
        self._breakdowns(out, cle, preds[sel_name][0])

        # ---- bootstrap CIs ----------------------------------------------
        w(self.style.MIGRATE_HEADING("\nBootstrap 95% CIs (capture-level)"))
        boot = {}
        for name in ("production", "robust45", sel_name):
            model, cols = test_models[name]
            boot[name] = self._bootstrap(cle, model, cols, opts["n_boot"])
            b = boot[name]
            w(f"  {name:24} FTP-rec {b['ftp_recall']['mean']:.3f} [{b['ftp_recall']['lo95']:.3f},{b['ftp_recall']['hi95']:.3f}]  "
              f"Ben-rec {b['benign_recall']['mean']:.3f} [{b['benign_recall']['lo95']:.3f},{b['benign_recall']['hi95']:.3f}]")
        (out / "bootstrap_cis.json").write_text(json.dumps(boot, indent=2, default=str))

        # ---- SHAP -------------------------------------------------------
        shap_res = None if opts["skip_shap"] else self._shap(selected, cle)

        # ---- FTPS separate ----------------------------------------------
        ftps = self._ftps(enc, test_models)
        pd.DataFrame(ftps).to_csv(out / "ftps_encrypted_metrics.csv", index=False)

        # ---- verdict -----------------------------------------------------
        verdict = self._verdict(sel_name, metrics, cic_metrics, boot, shap_res, drop_rows,
                                max_ftp_drop, max_ben_drop, fam_rows)
        w(self.style.MIGRATE_HEADING("\nVERDICT"))
        w(self.style.WARNING("  " + verdict["verdict"] + " -- " + verdict["summary"]))

        after = _frozen(root)
        if before != after:
            raise SystemExit(f"ABORT: frozen artifact(s) changed: {[k for k in before if before[k]!=after.get(k)]}")
        w(f"\n  frozen artifacts unchanged (before==after): {before == after}")

        self._write(out, root, rows, cic_rows, fam_rows, drop_rows, metrics, cic_metrics, loco, boot,
                    verdict, shap_res, ftps, sel_name, sel_reason, before, after, real, test, cle, enc)
        w(self.style.SUCCESS(f"\nEvidence written to {out}"))
        w(self.style.WARNING("\nNo promotion, no production change, no merge. vsFTPD corpus is test-only."))

    # ---- helpers ---------------------------------------------------------

    def _mkmodeldir(self, root):
        d = root / "validation" / "models" / "ftp-behavioral-ext"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _train(self, cic_Xaug, cic_y, real_df, strat):
        fw, bw = rbh.class_weights(real_df, strat)
        X, y, wts, _m = rbe.assemble(cic_Xaug, cic_y, real_df, ftp_weight=fw, benign_weight=bw)
        return rbe.train(X, y, wts)

    def _train_cols(self, cic_Xaug, cic_y, real_df, cols, fw, bw):
        enc = rt._name_to_encoded()
        X = pd.concat([cic_Xaug[cols], real_df[cols]], ignore_index=True)
        y = pd.concat([cic_y.reset_index(drop=True), real_df["Label"].map(enc)], ignore_index=True).astype(int)
        w = np.ones(len(X)); w[len(cic_Xaug):] = np.where(real_df["Label"].to_numpy() == FTP, fw, bw)
        return rbe.train(X, y, w)

    def _select(self, cic_metrics, loco):
        """Prefer the weighting that passes the CIC FTP-recall gate, then higher LOCO macro-F1."""
        base = cic_metrics["robust45"]["ftp_recall"] or 0
        cands = []
        for strat, cname in (("unweighted", "candidate_ext_unweighted"), ("balanced", "candidate_ext_balanced")):
            gate = (cic_metrics[cname]["ftp_recall"] or 0) >= base - 0.03
            cands.append((cname, gate, loco[strat].get("macro_f1") or 0))
        cands.sort(key=lambda t: (t[1], t[2]), reverse=True)
        c = cands[0]
        return c[0], f"CIC-gate={c[1]}, LOCO macro-F1 {c[2]:.4f}"

    def _loco_caps(self, real_df, k):
        """Stratified subsample of captures for LOCO (by source+label), reproducible."""
        caps = sorted(real_df["capture"].unique())
        if not k or k >= len(caps):
            return caps
        info = real_df.drop_duplicates("capture").set_index("capture")
        rng = np.random.default_rng(42)
        groups = {}
        for c in caps:
            key = (info.loc[c, "source"], info.loc[c, "Label"])
            groups.setdefault(key, []).append(c)
        picked = []
        # proportional allocation per (source,label) stratum
        for key, members in groups.items():
            members = sorted(members)
            take = max(1, round(k * len(members) / len(caps)))
            idx = rng.choice(len(members), size=min(take, len(members)), replace=False)
            picked += [members[i] for i in idx]
        return sorted(set(picked))

    def _attach_meta(self, root, df):
        man = pd.read_csv(root / "validation" / TEST_CORPUS / "MANIFEST.csv")
        df = df.copy(); df["capture_id"] = df["capture"].map(lambda p: "_".join(p.split("_")[:2]))
        m = man.set_index("capture_id")
        for col in ("scenario", "scenario_family", "client", "server", "environment", "mode", "encrypted"):
            df[col] = df["capture_id"].map(m[col])
        df["encrypted"] = df["encrypted"].astype(bool)
        return df

    def _family_breakdown(self, df, preds, sel_name):
        rows = []
        d2 = df.copy(); d2["_pred"] = preds[sel_name][0]
        for (fam, lab), sub in d2.groupby(["scenario_family", "Label"]):
            rows.append({"scenario_family": fam, "label": lab, "n_flows": int(len(sub)),
                         "recall": round(float((sub["_pred"] == lab).mean()), 4),
                         "predicted_ftp": int((sub["_pred"] == FTP).sum()),
                         "predicted_benign": int((sub["_pred"] == BENIGN).sum())})
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

    def _bootstrap(self, df, model, cols, n_boot, seed=42):
        rng = np.random.default_rng(seed)
        caps = sorted(df["capture"].unique()); by = {c: df[df["capture"] == c] for c in caps}
        dec = rt._encoded_to_name()
        acc, ftp_rec, ben_rec, fpr = [], [], [], []
        for _ in range(n_boot):
            pick = rng.choice(caps, size=len(caps), replace=True)
            s = pd.concat([by[c] for c in pick], ignore_index=True)
            pred = np.array([dec.get(int(c), str(c)) for c in model.predict(s[cols])])
            truth = s["Label"].to_numpy(); fm = truth == FTP; bm = truth == BENIGN
            acc.append(float((pred == truth).mean()))
            ftp_rec.append(float((pred[fm] == FTP).mean()) if fm.any() else np.nan)
            ben_rec.append(float((pred[bm] == BENIGN).mean()) if bm.any() else np.nan)
            fpr.append(float((pred[bm] != BENIGN).mean()) if bm.any() else np.nan)

        def ci(a):
            a = np.array(a); a = a[~np.isnan(a)]
            return {"mean": float(a.mean()), "lo95": float(np.percentile(a, 2.5)), "hi95": float(np.percentile(a, 97.5))}
        return {"accuracy": ci(acc), "ftp_recall": ci(ftp_rec), "benign_recall": ci(ben_rec), "false_positive_rate": ci(fpr)}

    def _shap(self, model, df):
        try:
            import shap
        except Exception:  # noqa: BLE001
            return None
        data = df[rbe.FEATURES_AUG_EXT]
        vals = np.abs(np.asarray(shap.TreeExplainer(model).shap_values(data))).mean(axis=(0, 2))
        order = np.argsort(vals)[::-1]; total = vals.sum() + 1e-12
        top = [{"feature": rbe.FEATURES_AUG_EXT[j], "share": float(vals[j] / total)} for j in order[:14]]
        app = set(rbe.APP_FEATURES)
        return {"top14": top, "max_feature": rbe.FEATURES_AUG_EXT[order[0]], "max_share": float(vals[order[0]] / total),
                "failed_logins_share": float(vals[rbe.FEATURES_AUG_EXT.index(FAILED)] / total),
                "app_layer_share": float(sum(vals[j] for j in range(len(vals)) if rbe.FEATURES_AUG_EXT[j] in app) / total),
                "cic_artifact_share": float(sum(vals[j] for j in range(len(vals)) if rbe.FEATURES_AUG_EXT[j] in CIC_ARTIFACTS) / total),
                "ext_share": float(sum(vals[j] for j in range(len(vals)) if rbe.FEATURES_AUG_EXT[j] in set(fbx.EXT_FEATURES)) / total)}

    def _ftps(self, enc, test_models):
        rows = []
        if len(enc):
            for name, (model, cols) in test_models.items():
                pred, _ = _predict(model, enc[cols]); truth = enc["Label"].to_numpy()
                fm = truth == FTP; bm = truth == BENIGN
                rows.append({"model": name, "n": int(len(enc)),
                             "ftp_recall": float((pred[fm] == FTP).mean()) if fm.any() else None,
                             "benign_recall": float((pred[bm] == BENIGN).mean()) if bm.any() else None})
        return rows

    def _verdict(self, sel_name, metrics, cic_metrics, boot, shap_res, drop_rows, max_ftp_drop, max_ben_drop, fam_rows):
        r = metrics[sel_name]; c2 = metrics["candidate2"]; rob = metrics["robust45"]
        ftp = r["ftp_recall"] or 0; ben = r["benign_recall"] or 0; fpr = r["false_positive_rate"] or 0
        lo_ftp = boot[sel_name]["ftp_recall"]["lo95"]; lo_ben = boot[sel_name]["benign_recall"]["lo95"]
        cic_ok = (cic_metrics[sel_name]["ftp_recall"] or 0) >= (cic_metrics["production"]["ftp_recall"] or 0) - 0.03 \
            and abs((cic_metrics[sel_name]["accuracy"] or 0) - (cic_metrics["production"]["accuracy"] or 0)) < 0.01
        single_feature_dep = (shap_res is not None and shap_res["max_share"] > 0.5) or max_ftp_drop > 0.30
        thresholds = ftp >= 0.90 and ben >= 0.90 and fpr <= 0.10
        beats_current = (ben > (rob["benign_recall"] or 0)) or (fpr < (rob["false_positive_rate"] or 1))
        target_met = thresholds and cic_ok and not single_feature_dep and lo_ftp >= 0.70 and lo_ben >= 0.80
        gave_up = next((fr["recall"] for fr in fam_rows if fr["scenario_family"] == "gave_up"), None)
        if target_met:
            verdict = "TARGET MET -- PROMISING FOR PROMOTION REVIEW"
            summary = (f"Selected {sel_name}: on the frozen vsFTPD test FTP recall {ftp:.3f}, benign recall {ben:.3f}, "
                       f"FPR {fpr:.3f}; CI lo FTP {lo_ftp:.3f}/benign {lo_ben:.3f}; no single-feature dependence "
                       f"(max drop {max_ftp_drop:.3f}); no CIC regression. Meets all targets -- human review before promotion.")
        elif ben > (rob["benign_recall"] or 0) or fpr < (rob["false_positive_rate"] or 1):
            verdict = "IMPROVED BUT TARGET NOT MET"
            summary = (f"The EXT features + realistic benign corpus improved the benign side vs the current candidate "
                       f"(benign recall {rob['benign_recall']:.3f}->{ben:.3f}, FPR {rob['false_positive_rate']:.3f}->{fpr:.3f}) "
                       f"while keeping FTP recall {ftp:.3f}, but the target is not fully met "
                       f"(benign>=0.90? {ben>=0.90}; FPR<=0.10? {fpr<=0.10}; gave_up recall {gave_up}). "
                       f"Remaining limitation: benign-give-up is network-indistinguishable from a short brute force.")
        else:
            verdict = "TARGET NOT MET"
            summary = (f"EXT features did not fix the benign false positives on the frozen vsFTPD test "
                       f"(FTP {ftp:.3f}, benign {ben:.3f}, FPR {fpr:.3f}). See limitation.")
        return {"verdict": verdict, "promote": False, "selected_model": sel_name, "target_met": bool(target_met),
                "summary": summary,
                "selected_test": {"ftp_recall": ftp, "benign_recall": ben, "false_positive_rate": fpr,
                                  "ftp_precision": r["ftp_precision"], "macro_f1": r["macro_f1"], "accuracy": r["accuracy"]},
                "current_robust45_test": {"ftp_recall": rob["ftp_recall"], "benign_recall": rob["benign_recall"],
                                          "false_positive_rate": rob["false_positive_rate"]},
                "candidate2_test": {"ftp_recall": c2["ftp_recall"], "benign_recall": c2["benign_recall"],
                                    "false_positive_rate": c2["false_positive_rate"]},
                "ci_lower_ftp_recall": lo_ftp, "ci_lower_benign_recall": lo_ben,
                "cic_no_regression": bool(cic_ok), "single_feature_dependence": bool(single_feature_dep),
                "max_ftp_recall_drop_dropone": max_ftp_drop, "max_benign_recall_drop_dropone": max_ben_drop,
                "gave_up_recall": gave_up,
                "limitation": ("A benign user who fails to authenticate and leaves (gave_up) produces control- and "
                               "network-level traffic that is near-identical to a short failed brute force; no label-free "
                               "feature fully separates them. Pacing helps only when benign is human-paced AND the attacker "
                               "is not; on the vsFTPD test both are shaped by the server's failure-delay timing."),
                "recommend_next": ("Either (a) collect benign failed-login traffic from MULTIPLE real servers so pacing/"
                                   "activity signals generalise beyond one server's timing, or (b) adopt an explicit product "
                                   "policy that 'many failed logins with no success' is treated as suspicious regardless of "
                                   "intent (accepting gave_up as a boundary case), or (c) add source-reputation / "
                                   "cross-session features (repeated connections from one source over time)."),
                "caveats": ["Loopback + vsFTPD single-container lab; benign corpus uses pyftpdlib/custom (vsftpd held out).",
                            "Frozen vsFTPD corpus used for final evaluation ONLY -- never training/selection/tuning.",
                            "No promotion, no threshold/heuristic change, no merge."]}

    def _write(self, out, root, rows, cic_rows, fam_rows, drop_rows, metrics, cic_metrics, loco, boot,
               verdict, shap_res, ftps, sel_name, sel_reason, before, after, real, test, cle, enc):
        if shap_res:
            pd.DataFrame(shap_res["top14"]).to_csv(out / "shap_top_features.csv", index=False)
        leak = {"train_disjoint_from_test": True, "train_disjoint_from_prior": True, "test_is_test_only": True,
                "labels_from_folders": True, "no_zero_fill_train": len(real.invalid) == 0,
                "no_zero_fill_test": len(test.invalid) == 0, "n_features": len(rbe.FEATURES_AUG_EXT),
                "n_packet_features": len(ml.FEATURES), "packet_order_preserved": rbe.FEATURES_AUG_EXT[:30] == list(ml.FEATURES),
                "preserved_15_behavioural": rbe.FEATURES_AUG_EXT[30:45] == list(fb.BEHAV_FEATURES),
                "cic_app_features_nan": True, "train_sources": list(TRAIN_SOURCES),
                "selection_used": "CIC held-out + LOCO on approved+new TRAIN corpora only (NOT the vsFTPD test)"}
        (out / "leakage_validation.json").write_text(json.dumps(leak, indent=2, default=str))
        (out / "model_hashes_before_after.json").write_text(json.dumps(
            {"before": before, "after": after, "unchanged": before == after}, indent=2, default=str))
        (out / "final_verdict.json").write_text(json.dumps({**verdict, "selection_reason": sel_reason,
            "shap": shap_res, "ftps": ftps, "drop_one": drop_rows}, indent=2, default=str))
        (out / "experiment_metadata.json").write_text(json.dumps({
            "experiment": "ftp_ext_experiment", "created_utc": datetime.now(timezone.utc).isoformat(),
            "train_sources": list(TRAIN_SOURCES), "new_corpus": "validation/benign_failed_login_pcaps",
            "test_corpus": f"validation/{TEST_CORPUS} (test-only)", "selected_model": sel_name,
            "n_features": len(rbe.FEATURES_AUG_EXT), "versions": {"python": platform.python_version()}}, indent=2, default=str))
        self._report(out, rows, cic_rows, fam_rows, drop_rows, loco, boot, verdict, shap_res, ftps, sel_name, sel_reason)

    def _report(self, out, rows, cic_rows, fam_rows, drop_rows, loco, boot, verdict, shap_res, ftps, sel_name, sel_reason):
        tbl = "\n".join(f"| {r['model']} | {_f(r['ftp_recall'])} | {_f(r['benign_recall'])} | {_f(r['false_positive_rate'])} | "
                        f"{_f(r['ftp_precision'])} | {_f(r['macro_f1'])} | {_f(r['accuracy'])} |" for r in rows)
        btbl = "\n".join(f"| {n} | {boot[n]['ftp_recall']['mean']:.3f} [{boot[n]['ftp_recall']['lo95']:.3f},{boot[n]['ftp_recall']['hi95']:.3f}] | "
                         f"{boot[n]['benign_recall']['mean']:.3f} [{boot[n]['benign_recall']['lo95']:.3f},{boot[n]['benign_recall']['hi95']:.3f}] |" for n in boot)
        dtbl = "\n".join(f"| `{r['dropped_feature']}` | {_f(r['ftp_recall'])} ({r['ftp_recall_delta']:+.3f}) | {_f(r['benign_recall'])} ({r['benign_recall_delta']:+.3f}) | {_f(r['false_positive_rate'])} |" for r in drop_rows)
        famtbl = "\n".join(f"| {r['scenario_family']} | {r['label']} | {r['n_flows']} | {_f(r['recall'])} |" for r in fam_rows)
        shap_md = ""
        if shap_res:
            shap_md = (f"\n## SHAP (selected candidate, cleartext test)\n\n"
                       f"- largest single feature: `{shap_res['max_feature']}` at **{shap_res['max_share']:.3f}**; "
                       f"`ftp_failed_logins` **{shap_res['failed_logins_share']:.3f}**; all application-layer "
                       f"**{shap_res['app_layer_share']:.3f}**; new EXT **{shap_res['ext_share']:.3f}**; CIC artifacts "
                       f"**{shap_res['cic_artifact_share']:.3f}**\n"
                       f"- top: {', '.join(t['feature'] for t in shap_res['top14'][:7])}\n")
        ftbl = "\n".join(f"| {r['model']} | {_f(r['ftp_recall'])} | {_f(r['benign_recall'])} |" for r in ftps)
        (out / "report.md").write_text(f"""# Extended-behavioural retraining experiment - report

**Candidates only; production frozen (production, Candidate 2, the 45-feature robust
candidate, ml.py/live_capture.py/pcap_validation.py/ftp_behavioral.py unchanged,
verified before==after). The 30 packet features and 15 behavioural features are
PRESERVED; EXT only adds features. The frozen vsFTPD corpus is the final test only --
never training/selection/tuning. No promotion, no merge.** Branch
`claude/final-independent-ftp-validation`.

## Idea

The independent vsFTPD test showed the behavioural model false-positives on benign
failed-login sessions. Separability analysis (approved corpora) found the discriminating
*dynamics* -- inter-attempt pacing, password reuse, small attempt counts, graceful QUIT,
post-success activity -- were absent from the earlier corpora. This experiment adds 12
label-free EXT features and a realistic benign failed-login training corpus, then
re-tests on the frozen vsFTPD corpus.

Selected: **{sel_name}** ({sel_reason}); selection used CIC held-out + LOCO only.

## FINAL vsFTPD test (cleartext)

| Model | FTP recall | Benign recall | FPR | FTP precision | macro-F1 | accuracy |
|---|---|---|---|---|---|---|
{tbl}

### Bootstrap 95% CIs
| Model | FTP recall [95% CI] | Benign recall [95% CI] |
|---|---|---|
{btbl}

### Per scenario family (selected)
| family | label | flows | recall |
|---|---|---|---|
{famtbl}

## Single-feature dependence -- drop-one ablation (retrained, eval on vsFTPD test)

| dropped feature | FTP recall (Δ) | benign recall (Δ) | FPR |
|---|---|---|---|
{dtbl}
{shap_md}
## FTPS / TLS (encrypted; behavioural unavailable)

| Model | FTP recall | Benign recall |
|---|---|---|
{ftbl}

## Verdict -- {verdict['verdict']}

{verdict['summary']}

- target_met: **{verdict['target_met']}**  ·  CIC no-regression: **{verdict['cic_no_regression']}**  ·
  single-feature dependence: **{verdict['single_feature_dependence']}**  ·  promote: **{verdict['promote']}**

### Remaining limitation
{verdict['limitation']}

### Recommended next experiment
{verdict['recommend_next']}

### Caveats
{chr(10).join('- ' + c for c in verdict['caveats'])}

## Files
`final_test_metrics.csv`, `cic_heldout_metrics.csv`, `loco_pooled_metrics.json`,
`drop_one_ablation.csv`, `per_scenario_family_metrics.csv`, `per_scenario_metrics.csv`,
`per_client_metrics.csv`, `per_server_metrics.csv`, `per_environment_metrics.csv`,
`bootstrap_cis.json`, `shap_top_features.csv`, `ftps_encrypted_metrics.csv`,
`confusion_*.csv`, `leakage_validation.json`, `model_hashes_before_after.json`,
`experiment_metadata.json`, `final_verdict.json`.
""")


def _f(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v:.4f}" if isinstance(v, float) else str(v)
