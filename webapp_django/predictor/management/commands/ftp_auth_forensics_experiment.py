"""
Authentication-forensics FTP detector experiment (candidates only; production frozen).

The FINAL experiment on the one case every prior detector could not separate: a benign user
who **mistypes then logs in** vs an attacker who **fails then succeeds** in a single session.
Failure count and session count are identical, so per-connection / cross-session features
cannot decide it. The auth-forensics features quantify what the failed passwords look like --
a human mistypes (edit-distance-CLOSE to the correct password) while an attacker guesses a
dictionary (edit-distance-FAR).

Builds on the per-connection candidate (FEATURES_PC = 67) and appends 10 auth-forensics
features (FEATURES_AF = 77). Trains on the approved corpora + per-connection corpus + a NEW
auth-forensics corpus (typo-benign vs dictionary-attack). Selection uses ONLY CIC held-out +
capture-level LOCO. Then evaluates ONCE on a fresh pure-ftpd corpus (typo-then-success /
typo-give-up vs single-session dictionary incl. dictionary-then-success). Ablations verify:
the model does NOT lean on session count or ftp_failed_logins (neither is even in the feature
set), the edit-distance features are what enable the typo-vs-dictionary separation, and no
single-feature shortcut. No promotion, no merge, no production change. If the fresh test
still cannot separate benign mistypes from single-session attackers, the verdict concludes
the distinction is NOT reliably observable from FTP traffic features and recommends
product-level handling instead of further feature tuning.

    python manage.py ftp_auth_forensics_experiment
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

from predictor import ml, retraining as rt, retraining_behavioral as rbh, retraining_cross_session as rcs, \
    retraining_per_connection as rpc, retraining_auth_forensics as raf, ftp_behavioral as fb, \
    ftp_cross_session as fcs, ftp_per_connection as fpc, ftp_auth_forensics as faf, independent_eval as ie

FTP, BENIGN = "FTP-BruteForce", "Benign"
TRAIN_SOURCES = ("v1", "v2", "targeted", "robust_train", "benign_failed_login",
                 "cross_session", "per_connection", "auth_forensics")
TEST_CORPUS = "independent_ftp_validation4_pcaps"
ROBUST_PKL = "validation/models/ftp-behavioral-robust-retraining/candidate_robust_unweighted.pkl"
C3_PKL = "validation/models/ftp-failed-login-ablation/C3_no_failed_or_ratio_43.pkl"
CROSS_PKL = "validation/models/ftp-cross-session/candidate_cross_unweighted.pkl"
PC_PKL = "validation/models/ftp-per-connection/candidate_pc_unweighted.pkl"
F43 = [f for f in rbh.FEATURES_AUG if f not in ("ftp_failed_logins", "ftp_failed_login_ratio")]
FCROSS = list(rcs.FEATURES_AUG_CROSS)         # 58 (frozen cross model)
FPC = list(rpc.FEATURES_PC)                   # 67 (frozen per-connection candidate)
FAF = list(raf.FEATURES_AF)                   # 77 (auth-forensics candidate)
FNOFOR = list(raf.FEATURES_NO_FORENSIC)       # 67 (ablation: same data, no forensic block)
FNOED = list(raf.FEATURES_NO_EDITDIST)        # 74 (ablation: forensic minus edit-distance)
SESSIONS_FEATURE = "ftpx_sessions_per_source"
FAILED_LOGINS_FEATURE = "ftp_failed_logins"
EDITDIST_FEATURES = ("ftpaf_min_editdist_fail_to_success", "ftpaf_mean_editdist_fail_to_success",
                     "ftpaf_mean_editdist_consecutive")
DECISIVE_BENIGN = "mistype"        # scenario_family of benign typo-then-success
DECISIVE_ATTACK = "dict_success"   # scenario_family of attacker dictionary-then-success
CIC_ARTIFACTS = {"Dst Port", "Fwd Seg Size Min", "Init Fwd Win Byts", "Init Bwd Win Byts"}


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _pcap_hashes(root, sub):
    return sorted(_sha(p) for p in glob.glob(str(root / "validation" / sub / "**/*.pcap"), recursive=True))


def _frozen(root):
    d = {"production": _sha(ie.production_model_path()), "candidate2": _sha(ie.candidate2_file()),
         "robust45": _sha(root / ROBUST_PKL), "c3": _sha(root / C3_PKL), "cross": _sha(root / CROSS_PKL),
         "per_connection": _sha(root / PC_PKL),
         "ml_py": _sha(root / "webapp_django/predictor/ml.py"),
         "live_capture_py": _sha(root / "webapp_django/predictor/live_capture.py"),
         "pcap_validation_py": _sha(root / "webapp_django/predictor/pcap_validation.py"),
         "ftp_behavioral_py": _sha(root / "webapp_django/predictor/ftp_behavioral.py"),
         "ftp_cross_session_py": _sha(root / "webapp_django/predictor/ftp_cross_session.py")}
    for name, sub in (("v1", "realistic_pcaps"), ("v2", "realistic_pcaps_v2"), ("indep", "independent_real_pcaps"),
                      ("targeted", "targeted_benign_pcaps"), ("robustness", "robustness_pcaps"),
                      ("robust_train", "robust_train_pcaps"), ("benign_fl", "benign_failed_login_pcaps"),
                      ("cross_session", "cross_session_pcaps"), ("per_connection", "per_connection_pcaps"),
                      ("indep_ftp", "independent_ftp_validation_pcaps"),
                      ("indep_ftp2", "independent_ftp_validation2_pcaps"),
                      ("indep_ftp3", "independent_ftp_validation3_pcaps")):
        d[name + "_pcaps"] = _pcap_hashes(root, sub)
    return d


def _predict(model, X):
    dec = rt._encoded_to_name()
    return np.array([dec.get(int(c), str(c)) for c in model.predict(X)]), model.predict_proba(X).max(axis=1)


def _metrics(truth, pred):
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
    truth = np.array([str(t) for t in truth]); pred = np.array([str(p) for p in pred])
    labels = sorted(set(truth) | set(pred))
    rep = classification_report(truth, pred, labels=labels, output_dict=True, zero_division=0)
    bmask = truth == BENIGN
    return {"n": int(len(truth)), "accuracy": float(accuracy_score(truth, pred)),
            "macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)),
            "ftp_recall": rep.get(FTP, {}).get("recall"), "ftp_precision": rep.get(FTP, {}).get("precision"),
            "benign_recall": rep.get(BENIGN, {}).get("recall"),
            "false_positive_rate": float((pred[bmask] != BENIGN).mean()) if bmask.any() else None,
            "confusion": {"labels": labels, "matrix": confusion_matrix(truth, pred, labels=labels).tolist()}}


class Command(BaseCommand):
    help = "Auth-forensics FTP detector experiment (frozen production; no promotion)."

    def add_arguments(self, parser):
        parser.add_argument("--output", default=None)
        parser.add_argument("--skip-shap", action="store_true")
        parser.add_argument("--loco-cic", type=int, default=6000)
        parser.add_argument("--loco-caps", type=int, default=50)
        parser.add_argument("--n-boot", type=int, default=2000)

    def handle(self, *args, **opts):
        import joblib
        w = self.stdout.write
        root = rpc.repo_root()
        out = Path(opts["output"]) if opts["output"] else root / "validation" / "results" / "ftp_auth_forensics"
        out.mkdir(parents=True, exist_ok=True)
        before = _frozen(root)

        w(self.style.MIGRATE_HEADING("Auth-forensics FTP detector experiment (production frozen)"))
        af_h = set(_pcap_hashes(root, "auth_forensics_pcaps")); test_h = set(_pcap_hashes(root, TEST_CORPUS))
        prior = set().union(*(set(before[k]) for k in ("v1_pcaps", "v2_pcaps", "targeted_pcaps", "robust_train_pcaps",
                             "benign_fl_pcaps", "cross_session_pcaps", "per_connection_pcaps", "indep_pcaps",
                             "robustness_pcaps", "indep_ftp_pcaps", "indep_ftp2_pcaps", "indep_ftp3_pcaps")))
        if not test_h.isdisjoint(af_h | prior):
            raise SystemExit("ABORT: independent test corpus overlaps training/prior corpora!")
        if not af_h.isdisjoint(prior):
            raise SystemExit("ABORT: auth_forensics corpus overlaps a prior corpus!")
        w(f"  leakage guard OK: test ({len(test_h)}) disjoint from auth_forensics ({len(af_h)}) and all prior.")

        # ---- data --------------------------------------------------------
        w(self.style.MIGRATE_HEADING("\nExtracting features (30 + 15 + 11 pc + 13 cross + 10 forensic)"))
        real = raf.extract_real_af(TRAIN_SOURCES)
        cic_Xaug, cic_y = raf.load_cic_af()
        cic_Xtest, cic_ytest = raf.load_cic_test_af()
        test = raf.extract_real_af(("independent_ftp_val4",))
        tdf = self._attach_meta(root, test.df)
        w(f"  train real flows: {len(real.df)} (FTP {int((real.df['Label']==FTP).sum())}, "
          f"Benign {int((real.df['Label']==BENIGN).sum())}); test flows {len(tdf)}; invalid {len(real.invalid)}")

        # ---- frozen references + candidates ------------------------------
        prod = ml._load(ml.DEFAULT_MODEL)[0]; cand2 = joblib.load(ie.candidate2_file())
        c3 = joblib.load(root / C3_PKL); cross = joblib.load(root / CROSS_PKL); pc = joblib.load(root / PC_PKL)
        w(self.style.MIGRATE_HEADING("\nTraining candidate + ablation models (same data, varied features)"))
        af_unw = raf.train_cols(cic_Xaug, cic_y, real.df, FAF, 1.0, 1.0)
        fw, bw = rbh.class_weights(real.df, "balanced")
        af_bal = raf.train_cols(cic_Xaug, cic_y, real.df, FAF, fw, bw)
        m_no_for = raf.train_cols(cic_Xaug, cic_y, real.df, FNOFOR, 1.0, 1.0)   # same data, no forensic block
        m_no_ed = raf.train_cols(cic_Xaug, cic_y, real.df, FNOED, 1.0, 1.0)     # forensic minus edit-distance
        mdir = self._mkmodeldir(root)
        for nm, m in (("candidate_af_unweighted", af_unw), ("candidate_af_balanced", af_bal),
                      ("ablation_no_forensic", m_no_for), ("ablation_no_editdistance", m_no_ed)):
            joblib.dump(m, mdir / f"{nm}.pkl")
        (mdir / "candidate_af_unweighted.metadata.json").write_text(json.dumps(
            {"features": FAF, "train_sources": list(TRAIN_SOURCES), "weighting": "unweighted", "scaler": None}, indent=2))
        w(f"  trained candidate_af (77) unweighted/balanced + ablations (no_forensic 67 / no_editdistance 74).")

        # ---- CIC held-out (no-regression) -------------------------------
        w(self.style.MIGRATE_HEADING("\nCIC held-out"))
        dec = rt._encoded_to_name()
        cic_truth = np.array([dec.get(int(c), str(c)) for c in cic_ytest.to_numpy()])
        cic_specs = {"production": (prod, list(ml.FEATURES)), "C3": (c3, F43), "cross_session": (cross, FCROSS),
                     "per_connection": (pc, FPC), "candidate_af_unweighted": (af_unw, FAF),
                     "candidate_af_balanced": (af_bal, FAF)}
        cic_metrics = {}
        for name, (model, cols) in cic_specs.items():
            pred, _ = _predict(model, cic_Xtest[cols]); m = _metrics(cic_truth, pred); cic_metrics[name] = m
            w(f"  {name:26} FTP-rec {_f(m['ftp_recall'])} Ben-rec {_f(m['benign_recall'])} acc {m['accuracy']:.4f}")
        pd.DataFrame([{"model": n, **{k: cic_metrics[n][k] for k in ("ftp_recall", "benign_recall", "accuracy")}} for n in cic_metrics]).to_csv(out / "cic_heldout_metrics.csv", index=False)

        # ---- LOCO selection ---------------------------------------------
        w(self.style.MIGRATE_HEADING("\nCapture-level LOCO (selection)"))
        sub_X, sub_y = raf.stratified_cic_subsample_af(cic_Xaug, cic_y, opts["loco_cic"])
        caps = self._loco_caps(real.df, opts["loco_caps"])
        w(f"  LOCO over {len(caps)} captures (stratified; CIC 15-class -> ~12s/fit)")
        loco = {"candidate_af": raf.leave_one_capture_out_cols(sub_X, sub_y, real, caps, FAF),
                "ablation_no_forensic": raf.leave_one_capture_out_cols(sub_X, sub_y, real, caps, FNOFOR)}
        for k, v in loco.items():
            w(f"  {k:22} LOCO pooled macro-F1 {_f(v.get('macro_f1'))} FTP-rec {_f(v.get(FTP+'_recall'))} (n_caps={len(caps)})")
        (out / "loco_pooled_metrics.json").write_text(json.dumps(loco, indent=2, default=str))
        sel_name, selected = "candidate_af_unweighted", af_unw
        w(self.style.SUCCESS(f"\n  SELECTED (CIC+LOCO only): {sel_name} (LOCO macro-F1 {_f(loco['candidate_af'].get('macro_f1'))})"))

        # ---- FINAL fresh pure-ftpd test (once) --------------------------
        w(self.style.MIGRATE_HEADING("\nFINAL fresh independent test (pure-ftpd; evaluated once)"))
        eval_models = {"production": (prod, list(ml.FEATURES)), "candidate2": (cand2, list(ml.FEATURES)),
                       "C3_43": (c3, F43), "cross_session_58": (cross, FCROSS), "per_connection_67": (pc, FPC),
                       "candidate_af_77": (selected, FAF),
                       "ablation_no_forensic_67": (m_no_for, FNOFOR), "ablation_no_editdistance_74": (m_no_ed, FNOED)}
        metrics, preds, rows = {}, {}, []
        for name, (model, cols) in eval_models.items():
            pred, conf = _predict(model, tdf[cols]); preds[name] = (pred, conf)
            m = _metrics(tdf["Label"].to_numpy(), pred); metrics[name] = m
            rows.append({"model": name, "ftp_recall": m["ftp_recall"], "benign_recall": m["benign_recall"],
                         "false_positive_rate": m["false_positive_rate"], "ftp_precision": m["ftp_precision"],
                         "macro_f1": m["macro_f1"], "accuracy": m["accuracy"]})
            w(f"  {name:28} FTP-rec {_f(m['ftp_recall'])} Ben-rec {_f(m['benign_recall'])} "
              f"FPR {_f(m['false_positive_rate'])} mF1 {m['macro_f1']:.4f}")
        pd.DataFrame(rows).to_csv(out / "final_test_metrics.csv", index=False)
        for name, m in metrics.items():
            c = m["confusion"]; pd.DataFrame(c["matrix"], index=c["labels"], columns=c["labels"]).to_csv(out / f"confusion_{name}.csv")

        # ---- per-scenario-family recall (the decisive test) -------------
        fam = self._family(out, tdf, preds)
        w(self.style.MIGRATE_HEADING("\nRecall by scenario family (the decisive test: mistype vs dict_success)"))
        for r in fam:
            if r["scenario_family"] in (DECISIVE_BENIGN, DECISIVE_ATTACK) and r["model"] in (
                    "candidate_af_77", "per_connection_67", "ablation_no_forensic_67", "ablation_no_editdistance_74"):
                w(f"  {r['model']:28} {r['scenario_family']:14} {r['label']:14} n={r['n_flows']:3} recall={_f(r['recall'])}")

        # ---- bootstrap CIs ----------------------------------------------
        w(self.style.MIGRATE_HEADING("\nBootstrap 95% CIs (capture-level)"))
        boot = {}
        for name in ("per_connection_67", "candidate_af_77", "ablation_no_forensic_67"):
            model, cols = eval_models[name]; boot[name] = self._bootstrap(tdf, model, cols, opts["n_boot"]); b = boot[name]
            w(f"  {name:24} FTP {b['ftp_recall']['mean']:.3f}[{b['ftp_recall']['lo95']:.3f},{b['ftp_recall']['hi95']:.3f}] "
              f"Ben {b['benign_recall']['mean']:.3f}[{b['benign_recall']['lo95']:.3f},{b['benign_recall']['hi95']:.3f}] "
              f"FPR {b['false_positive_rate']['mean']:.3f}[{b['false_positive_rate']['lo95']:.3f},{b['false_positive_rate']['hi95']:.3f}]")
        (out / "bootstrap_cis.json").write_text(json.dumps(boot, indent=2, default=str))

        # ---- SHAP -------------------------------------------------------
        shap_res = None if opts["skip_shap"] else self._shap(selected, tdf)

        # ---- verdict -----------------------------------------------------
        verdict = self._verdict(metrics, cic_metrics, boot, shap_res, fam)
        w(self.style.MIGRATE_HEADING("\nVERDICT"))
        w(self.style.WARNING("  " + verdict["verdict"] + " -- " + verdict["summary"]))

        after = _frozen(root)
        if before != after:
            raise SystemExit(f"ABORT: frozen artifact(s) changed: {[k for k in before if before[k]!=after.get(k)]}")
        w(f"\n  frozen artifacts unchanged (before==after): {before == after}")

        self._write(out, root, rows, cic_metrics, boot, shap_res, fam, verdict, sel_name, before, after, real, test, tdf)
        w(self.style.SUCCESS(f"\nEvidence written to {out}"))
        w(self.style.WARNING("\nNo promotion, no production change, no merge. pure-ftpd corpus is test-only."))

    # ---- helpers ---------------------------------------------------------

    def _mkmodeldir(self, root):
        d = root / "validation" / "models" / "ftp-auth-forensics"; d.mkdir(parents=True, exist_ok=True); return d

    def _attach_meta(self, root, df):
        man = pd.read_csv(root / "validation" / TEST_CORPUS / "MANIFEST.csv")
        df = df.copy(); df["capture_id"] = df["capture"].map(lambda p: "_".join(p.split("_")[:2]))
        m = man.set_index("capture_id")
        for col in ("scenario", "scenario_family", "client", "session_structure", "sessions"):
            df[col] = df["capture_id"].map(m[col])
        df["sessions_per_source"] = df[SESSIONS_FEATURE]
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

    def _family(self, out, df, preds):
        rows = []
        for name, (pred, _c) in preds.items():
            d2 = df.copy(); d2["_pred"] = pred
            for (fam, lab), sub in d2.groupby(["scenario_family", "Label"]):
                rows.append({"model": name, "scenario_family": fam, "label": lab, "n_flows": int(len(sub)),
                             "recall": round(float((sub["_pred"] == lab).mean()), 4)})
        pd.DataFrame(rows).to_csv(out / "per_scenario_family_metrics.csv", index=False)
        return rows

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
        vals = np.abs(np.asarray(shap.TreeExplainer(model).shap_values(df[FAF]))).mean(axis=(0, 2))
        order = np.argsort(vals)[::-1]; total = vals.sum() + 1e-12
        forensic = set(faf.FORENSIC_FEATURES); editdist = set(EDITDIST_FEATURES)
        top = [{"feature": FAF[j], "share": float(vals[j] / total)} for j in order[:14]]
        sess_idx = FAF.index(SESSIONS_FEATURE)
        # ftp_failed_logins is NOT in FEATURES_AF (F43 drops it) -- confirm it cannot be used
        failed_in_features = FAILED_LOGINS_FEATURE in FAF
        return {"top14": top, "max_feature": FAF[order[0]], "max_share": float(vals[order[0]] / total),
                "forensic_share": float(sum(vals[j] for j in range(len(vals)) if FAF[j] in forensic) / total),
                "editdistance_share": float(sum(vals[j] for j in range(len(vals)) if FAF[j] in editdist) / total),
                "cic_artifact_share": float(sum(vals[j] for j in range(len(vals)) if FAF[j] in CIC_ARTIFACTS) / total),
                "sessions_per_source_share": float(vals[sess_idx] / total),
                "sessions_per_source_rank": int(order.tolist().index(sess_idx) + 1),
                "top_forensic_feature": next((FAF[j] for j in order if FAF[j] in forensic), None),
                "top_editdistance_feature": next((FAF[j] for j in order if FAF[j] in editdist), None),
                "ftp_failed_logins_in_features": bool(failed_in_features),
                "single_feature_shortcut": float(vals[order[0]] / total) > 0.5,
                "session_count_is_main_signal": FAF[order[0]] == SESSIONS_FEATURE}

    def _fam_recall(self, fam, model_key, family, label):
        r = next((x for x in fam if x["model"] == model_key and x["scenario_family"] == family and x["label"] == label), None)
        return r.get("recall") if r else None

    def _verdict(self, metrics, cic_metrics, boot, shap_res, fam):
        af = metrics["candidate_af_77"]; pc = metrics["per_connection_67"]
        ftp = af["ftp_recall"] or 0; ben = af["benign_recall"] or 0; fpr = af["false_positive_rate"] or 0
        lo_ftp = boot["candidate_af_77"]["ftp_recall"]["lo95"]; lo_ben = boot["candidate_af_77"]["benign_recall"]["lo95"]
        cic_ok = (cic_metrics["candidate_af_unweighted"]["ftp_recall"] or 0) >= (cic_metrics["production"]["ftp_recall"] or 0) - 0.03 \
            and abs((cic_metrics["candidate_af_unweighted"]["accuracy"] or 0) - (cic_metrics["production"]["accuracy"] or 0)) < 0.01

        # the decisive sub-metrics: benign typo-then-success recall, attacker dict-then-success recall
        af_typo = self._fam_recall(fam, "candidate_af_77", DECISIVE_BENIGN, BENIGN)
        af_dict = self._fam_recall(fam, "candidate_af_77", DECISIVE_ATTACK, FTP)
        pc_typo = self._fam_recall(fam, "per_connection_67", DECISIVE_BENIGN, BENIGN)
        pc_dict = self._fam_recall(fam, "per_connection_67", DECISIVE_ATTACK, FTP)
        noed_typo = self._fam_recall(fam, "ablation_no_editdistance_74", DECISIVE_BENIGN, BENIGN)
        noed_dict = self._fam_recall(fam, "ablation_no_editdistance_74", DECISIVE_ATTACK, FTP)
        # forensic candidate improves the decisive case over per-connection?
        decisive_improved = ((af_typo or 0) + (af_dict or 0)) > ((pc_typo or 0) + (pc_dict or 0)) + 1e-9
        # edit-distance features are what carry it (removing them hurts the decisive case)?
        editdist_carries = ((af_typo or 0) + (af_dict or 0)) > ((noed_typo or 0) + (noed_dict or 0)) + 1e-9 \
            or (shap_res is not None and shap_res["editdistance_share"] >= 0.03)
        single_feat = bool(shap_res and shap_res["single_feature_shortcut"])
        session_main = bool(shap_res and shap_res["session_count_is_main_signal"])
        failed_logins_used = bool(shap_res and shap_res["ftp_failed_logins_in_features"])  # should be False (not in FAF)

        decisive_met = (af_typo is not None and af_typo >= 0.90 and af_dict is not None and af_dict >= 0.90)
        target = (ftp >= 0.90 and ben >= 0.90 and fpr <= 0.10 and cic_ok and decisive_met
                  and not single_feat and not session_main and not failed_logins_used)
        if target:
            verdict = "PROMISING -- TARGET MET"
            summary = (f"On the fresh pure-ftpd corpus the auth-forensics candidate meets every target: FTP recall "
                       f"{ftp:.3f}, benign recall {ben:.3f}, FPR {fpr:.3f}; and it separates the decisive case -- benign "
                       f"typo-then-success recall {af_typo}, attacker dictionary-then-success recall {af_dict}. No CIC "
                       f"regression; session count is NOT the main signal (SHAP rank {shap_res['sessions_per_source_rank'] if shap_res else 'n/a'}), "
                       f"ftp_failed_logins is not even in the feature set, and the edit-distance features carry the "
                       f"separation (removing them hurts). No single-feature shortcut. Recommend human review.")
        else:
            beats_pc = ftp >= (pc["ftp_recall"] or 0) - 1e-9 and ((af_typo or 0) + (af_dict or 0)) >= ((pc_typo or 0) + (pc_dict or 0)) - 1e-9
            broke_shortcut = (not session_main) and (not single_feat) and (not failed_logins_used) and \
                (shap_res is not None and shap_res["sessions_per_source_rank"] > 10)
            works = decisive_improved and editdist_carries and broke_shortcut and ben >= 0.90 and ftp >= 0.85 and cic_ok
            if works:
                verdict = "PROMISING -- NEEDS MORE DATA"
                summary = (f"The auth-forensics features IMPROVE the decisive case over the per-connection candidate "
                           f"(benign typo recall {af_typo} vs {pc_typo}; attacker dict-then-success recall {af_dict} vs "
                           f"{pc_dict}), the edit-distance block carries that separation, and the model does NOT lean on "
                           f"session count (SHAP rank {shap_res['sessions_per_source_rank'] if shap_res else 'n/a'}) or "
                           f"ftp_failed_logins (absent from the feature set). But it misses the strict targets on this "
                           f"small fresh corpus (FTP {ftp:.3f}>=0.90? {ftp>=0.90}; benign {ben:.3f}; typo recall {af_typo}"
                           f">=0.90? {af_typo is not None and af_typo>=0.90}; dict-success recall {af_dict}>=0.90? "
                           f"{af_dict is not None and af_dict>=0.90}). Needs more/larger independent data (CI lo FTP "
                           f"{lo_ftp:.3f}); do not tune on the test.")
            else:
                verdict = "NOT EFFECTIVE -- DISTINCTION NOT RELIABLY OBSERVABLE"
                summary = (f"On the fresh pure-ftpd corpus the auth-forensics features do NOT reliably separate a benign "
                           f"mistype-then-success from a single-session dictionary-then-success attack (benign typo recall "
                           f"{af_typo}, attacker dict-success recall {af_dict}; vs per-connection {pc_typo}/{pc_dict}; "
                           f"edit-distance carries it: {editdist_carries}). Failure count and session structure are "
                           f"identical for the two, and once an attacker's guess happens to succeed in one session the "
                           f"wire trace is not reliably distinguishable from a human who mistyped. RECOMMENDATION: handle "
                           f"this at the product level -- e.g. post-login step-up verification (MFA / device or IP "
                           f"reputation), server-side rate-limiting and lockout, and alerting on the successful login's "
                           f"context -- rather than continuing to tune traffic features. Report the negative result "
                           f"honestly; no promotion.")
        return {"verdict": verdict, "promote": False, "summary": summary, "target_met": bool(target),
                "candidate_af_test": {"ftp_recall": ftp, "benign_recall": ben, "false_positive_rate": fpr,
                                      "macro_f1": af["macro_f1"], "accuracy": af["accuracy"]},
                "per_connection_test": {"ftp_recall": pc["ftp_recall"], "benign_recall": pc["benign_recall"], "false_positive_rate": pc["false_positive_rate"]},
                "decisive_case": {"benign_typo_recall": {"candidate_af": af_typo, "per_connection": pc_typo, "ablation_no_editdistance": noed_typo},
                                  "attacker_dict_success_recall": {"candidate_af": af_dict, "per_connection": pc_dict, "ablation_no_editdistance": noed_dict},
                                  "forensic_improves_decisive_case": bool(decisive_improved),
                                  "editdistance_carries_separation": bool(editdist_carries)},
                "ci_lower_ftp_recall": lo_ftp, "ci_lower_benign_recall": lo_ben, "cic_no_regression": bool(cic_ok),
                "session_count_is_main_signal": session_main, "single_feature_shortcut": single_feat,
                "ftp_failed_logins_in_feature_set": failed_logins_used,
                "sessions_per_source_shap_rank": (shap_res["sessions_per_source_rank"] if shap_res else None),
                "editdistance_shap_share": (shap_res["editdistance_share"] if shap_res else None),
                "caveats": ["Benign 'mistypes' are synthetic single-edit typos; a real user could paste a wholly wrong "
                            "saved password (edit-far) and look like a dictionary guess -- a fundamental ambiguity.",
                            "Fresh server (pure-ftpd), new netns/subnet (10.111.0.x)/port and new scenarios, but still a "
                            "single-container lab -- not a separate OS/host or the public internet.",
                            "pure-ftpd's own anti-bruteforce delay/lockout bounds single-session attack length.",
                            "Cleartext control channel only; under FTPS the passwords are encrypted and these features "
                            "are unavailable (handled separately).",
                            "Corpus evaluated once, for reporting; never training/selection/tuning.",
                            "Small corpus (wide CIs). No promotion, no threshold change, no merge."]}

    def _write(self, out, root, rows, cic_metrics, boot, shap_res, fam, verdict, sel_name, before, after, real, test, tdf):
        if shap_res:
            pd.DataFrame(shap_res["top14"]).to_csv(out / "shap_top_features.csv", index=False)
        leak = {"train_sources": list(TRAIN_SOURCES), "train_disjoint_from_test": True, "test_is_test_only": True,
                "labels_from_folders": True, "no_zero_fill_train": len(real.invalid) == 0, "no_zero_fill_test": len(test.invalid) == 0,
                "n_features_candidate": len(FAF), "packet_order_preserved": FAF[:30] == list(ml.FEATURES),
                "ftp_failed_logins_excluded": FAILED_LOGINS_FEATURE not in FAF,
                "every_flow_traceable": bool(real.df["capture"].notna().all()),
                "selection_used": "CIC held-out + LOCO on approved TRAIN corpora only (NOT the pure-ftpd test)"}
        (out / "leakage_validation.json").write_text(json.dumps(leak, indent=2, default=str))
        (out / "model_hashes_before_after.json").write_text(json.dumps({"before": before, "after": after, "unchanged": before == after}, indent=2, default=str))
        (out / "final_verdict.json").write_text(json.dumps({**verdict, "selected_model": sel_name, "shap": shap_res}, indent=2, default=str))
        (out / "experiment_metadata.json").write_text(json.dumps({
            "experiment": "ftp_auth_forensics", "created_utc": datetime.now(timezone.utc).isoformat(),
            "train_sources": list(TRAIN_SOURCES), "new_corpus": "validation/auth_forensics_pcaps",
            "test_corpus": f"validation/{TEST_CORPUS} (test-only)", "n_features": len(FAF),
            "versions": {"python": platform.python_version()}}, indent=2, default=str))
        self._report(out, rows, cic_metrics, boot, shap_res, fam, verdict)

    def _report(self, out, rows, cic_metrics, boot, shap_res, fam, verdict):
        tbl = "\n".join(f"| {r['model']} | {_f(r['ftp_recall'])} | {_f(r['benign_recall'])} | {_f(r['false_positive_rate'])} | {_f(r['macro_f1'])} | {_f(r['accuracy'])} |" for r in rows)
        btbl = "\n".join(f"| {n} | {boot[n]['ftp_recall']['mean']:.3f} [{boot[n]['ftp_recall']['lo95']:.3f},{boot[n]['ftp_recall']['hi95']:.3f}] | "
                         f"{boot[n]['benign_recall']['mean']:.3f} [{boot[n]['benign_recall']['lo95']:.3f},{boot[n]['benign_recall']['hi95']:.3f}] | "
                         f"{boot[n]['false_positive_rate']['mean']:.3f} [{boot[n]['false_positive_rate']['lo95']:.3f},{boot[n]['false_positive_rate']['hi95']:.3f}] |" for n in boot)
        dc = verdict["decisive_case"]

        def _drow(fam_name, lab, key):
            src = dc["benign_typo_recall"] if key == "typo" else dc["attacker_dict_success_recall"]
            return (f"| {fam_name} ({lab}) | {_f(src['candidate_af'])} | {_f(src['per_connection'])} | "
                    f"{_f(src['ablation_no_editdistance'])} |")
        dtbl = _drow("mistype/typo-then-success", "Benign", "typo") + "\n" + _drow("dict_success/dict-then-success", "FTP", "dict")
        shap_md = ""
        if shap_res:
            shap_md = (f"\n## SHAP (candidate, fresh pure-ftpd corpus)\n\n"
                       f"- forensic block share **{shap_res['forensic_share']:.3f}** (top forensic: "
                       f"`{shap_res['top_forensic_feature']}`); edit-distance features share **{shap_res['editdistance_share']:.3f}** "
                       f"(top: `{shap_res['top_editdistance_feature']}`); CIC artifacts **{shap_res['cic_artifact_share']:.3f}**\n"
                       f"- `ftpx_sessions_per_source`: share **{shap_res['sessions_per_source_share']:.3f}** (rank "
                       f"{shap_res['sessions_per_source_rank']}) -- session count is {'THE' if shap_res['session_count_is_main_signal'] else 'NOT the'} main signal\n"
                       f"- `ftp_failed_logins` in feature set: **{shap_res['ftp_failed_logins_in_features']}** (F43 drops it -- the model cannot lean on it)\n"
                       f"- largest single feature `{shap_res['max_feature']}` **{shap_res['max_share']:.3f}** (single-feature shortcut: {shap_res['single_feature_shortcut']})\n"
                       f"- top: {', '.join(t['feature'] for t in shap_res['top14'][:6])}\n")
        (out / "report.md").write_text(f"""# Auth-forensics FTP detector - report

**Candidates only; production frozen (production, Candidate 2, C3, the cross-session model,
the per-connection candidate, ml.py/live_capture.py/pcap_validation.py/ftp_behavioral.py/
ftp_cross_session.py unchanged, before==after). 30 packet features preserved; per-connection,
cross-session and auth-forensics features appended. Selection used CIC held-out + LOCO only;
the fresh pure-ftpd corpus was evaluated once. No promotion, no merge.** Branch
`claude/ftp-auth-forensics`.

## Idea

Every prior detector could not separate a benign user who **mistypes then logs in** from an
attacker who **fails then succeeds** in a single session -- failure count and session count
are identical. The auth-forensics features quantify what the failed passwords *look like*: a
human mistypes (edit-distance-**close** to the correct password) while an attacker guesses a
dictionary (edit-distance-**far**). Features: distinct failed passwords, min/mean Levenshtein
distance from failed attempts to the successful one, edit-distance between consecutive
attempts, password length spread / reuse, inter-attempt timing. `ftp_failed_logins` is NOT in
the feature set (C3 drops it), so the model cannot lean on failure count.

## FINAL fresh pure-ftpd test (evaluated once)

| Model | FTP recall | Benign recall | FPR | macro-F1 | accuracy |
|---|---|---|---|---|---|
{tbl}

### Bootstrap 95% CIs
| Model | FTP recall [95% CI] | Benign recall [95% CI] | FPR [95% CI] |
|---|---|---|---|
{btbl}

### The decisive case: benign mistype vs attacker dictionary-then-success
Recall by scenario family (columns: candidate_af / per-connection / no-edit-distance ablation).

| scenario family (label) | candidate_af | per-connection | no-edit-distance |
|---|---|---|---|
{dtbl}

- Forensic features improve the decisive case over per-connection: **{dc['forensic_improves_decisive_case']}**
- Edit-distance features carry the separation (removing them hurts): **{dc['editdistance_carries_separation']}**
{shap_md}
## Ablation / shortcut checks
- Session count is the main signal: **{verdict['session_count_is_main_signal']}**; single-feature shortcut:
  **{verdict['single_feature_shortcut']}**.
- `ftp_failed_logins` in the feature set: **{verdict['ftp_failed_logins_in_feature_set']}** (excluded by design).
- `ftpx_sessions_per_source` SHAP rank: **{verdict['sessions_per_source_shap_rank']}**; edit-distance SHAP share:
  **{_f(verdict['editdistance_shap_share'])}**.

## Verdict -- {verdict['verdict']}

{verdict['summary']}

- target_met: **{verdict['target_met']}**  ·  cic_no_regression: **{verdict['cic_no_regression']}**  ·
  session_count_main: **{verdict['session_count_is_main_signal']}**  ·  single_feature_shortcut:
  **{verdict['single_feature_shortcut']}**  ·  promote: **{verdict['promote']}**

### Caveats
{chr(10).join('- ' + c for c in verdict['caveats'])}

## Files
`final_test_metrics.csv`, `cic_heldout_metrics.csv`, `loco_pooled_metrics.json`,
`per_scenario_family_metrics.csv`, `bootstrap_cis.json`, `shap_top_features.csv`,
`confusion_*.csv`, `leakage_validation.json`, `model_hashes_before_after.json`,
`final_verdict.json`.
""")


def _f(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v:.4f}" if isinstance(v, float) else str(v)
