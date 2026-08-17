"""
FINAL independent validation of the cross-session FTP detector (all models frozen).

Evaluates -- ONCE, for reporting only -- the production model, Candidate 2, the current
45-feature behavioural candidate, the C3 controlled candidate, and the cross-session
candidate on a SECOND genuinely-independent corpus (pure-ftpd server, ncftp client, a new
netns/subnet, fresh scenario code with single- vs multi-session structure, plus FTPS).
Nothing is trained, tuned, selected, or thresholded on this corpus. SHAP + a frozen
with/without-cross-block ablation check whether the cross-session block genuinely
contributes and whether the model leans on a single feature.

Verdict: PROMISING -- TARGET MET / PROMISING -- NEEDS MORE DATA / NOT EFFECTIVE /
INVALID -- LEAKAGE/INTEGRITY FAILURE. No promotion, no merge, no production change.

    python manage.py ftp_cross_session_final_validation
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
CORPUS = "independent_ftp_validation2_pcaps"
ROBUST_PKL = "validation/models/ftp-behavioral-robust-retraining/candidate_robust_unweighted.pkl"
C3_PKL = "validation/models/ftp-failed-login-ablation/C3_no_failed_or_ratio_43.pkl"
CROSS_PKL = "validation/models/ftp-cross-session/candidate_cross_unweighted.pkl"
F45 = list(rbh.FEATURES_AUG)
F43 = [f for f in F45 if f not in ("ftp_failed_logins", "ftp_failed_login_ratio")]
FCROSS = list(rce.FEATURES_AUG_CROSS)
CIC_ARTIFACTS = {"Dst Port", "Fwd Seg Size Min", "Init Fwd Win Byts", "Init Bwd Win Byts"}


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _pcap_hashes(root, sub):
    return sorted(_sha(p) for p in glob.glob(str(root / "validation" / sub / "**/*.pcap"), recursive=True))


def _frozen(root):
    d = {"production": _sha(ie.production_model_path()), "candidate2": _sha(ie.candidate2_file()),
         "robust45": _sha(root / ROBUST_PKL), "c3": _sha(root / C3_PKL), "cross": _sha(root / CROSS_PKL),
         "ml_py": _sha(root / "webapp_django/predictor/ml.py"),
         "live_capture_py": _sha(root / "webapp_django/predictor/live_capture.py"),
         "pcap_validation_py": _sha(root / "webapp_django/predictor/pcap_validation.py"),
         "ftp_behavioral_py": _sha(root / "webapp_django/predictor/ftp_behavioral.py"),
         "ftp_cross_session_py": _sha(root / "webapp_django/predictor/ftp_cross_session.py")}
    for name, sub in (("v1", "realistic_pcaps"), ("v2", "realistic_pcaps_v2"), ("indep", "independent_real_pcaps"),
                      ("targeted", "targeted_benign_pcaps"), ("robustness", "robustness_pcaps"),
                      ("robust_train", "robust_train_pcaps"), ("benign_fl", "benign_failed_login_pcaps"),
                      ("cross_session", "cross_session_pcaps"), ("indep_ftp", "independent_ftp_validation_pcaps")):
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
    help = "Final independent validation of the cross-session detector on a 2nd corpus (frozen models)."

    def add_arguments(self, parser):
        parser.add_argument("--output", default=None)
        parser.add_argument("--skip-shap", action="store_true")
        parser.add_argument("--n-boot", type=int, default=2000)

    def handle(self, *args, **opts):
        import joblib
        w = self.stdout.write
        root = rce.repo_root()
        out = Path(opts["output"]) if opts["output"] else root / "validation" / "results" / "ftp_cross_session_independent_validation"
        out.mkdir(parents=True, exist_ok=True)
        before = _frozen(root)

        w(self.style.MIGRATE_HEADING("FINAL cross-session independent validation (frozen models; test-only)"))

        # ---- integrity / leakage guard ----------------------------------
        integ = self._integrity(root, before)
        if not integ["all_pass"]:
            (out / "final_verdict.json").write_text(json.dumps(
                {"verdict": "INVALID -- LEAKAGE/INTEGRITY FAILURE", "promote": False, "integrity": integ}, indent=2))
            (out / "leakage_validation.json").write_text(json.dumps(integ, indent=2, default=str))
            w(self.style.ERROR("  INTEGRITY/LEAKAGE FAILURE -> verdict INVALID")); return
        w(f"  integrity OK: new corpus disjoint from all {integ['n_prior']} prior corpora; {integ['n_new']} new pcaps.")

        # ---- extract (58-feature) + CIC held-out ------------------------
        real = rce.extract_real_cross(("independent_ftp_val2",))
        df = self._attach_meta(root, real.df)
        cle = df[~df["encrypted"]].reset_index(drop=True); enc = df[df["encrypted"]].reset_index(drop=True)
        w(f"  flows: {len(df)} (cleartext {len(cle)}: FTP {int((cle['Label']==FTP).sum())}, Benign "
          f"{int((cle['Label']==BENIGN).sum())}; FTPS {len(enc)}); invalid {len(real.invalid)}")

        cic_Xtest, cic_ytest = rce.load_cic_test_cross()
        dec = rt._encoded_to_name()
        cic_truth = np.array([dec.get(int(c), str(c)) for c in cic_ytest.to_numpy()])

        # ---- models (all frozen) ----------------------------------------
        prod = ml._load(ml.DEFAULT_MODEL)[0]; cand2 = joblib.load(ie.candidate2_file())
        robust45 = joblib.load(root / ROBUST_PKL); c3 = joblib.load(root / C3_PKL); cross = joblib.load(root / CROSS_PKL)
        model_specs = {"production": (prod, list(ml.FEATURES)), "candidate2": (cand2, list(ml.FEATURES)),
                       "C1_current_45": (robust45, F45), "C3_controlled_43": (c3, F43),
                       "cross_session_58": (cross, FCROSS)}

        # ---- CIC held-out (no-regression, all models) -------------------
        w(self.style.MIGRATE_HEADING("\nCIC held-out (no-regression check)"))
        cic_metrics = {}
        for name, (model, cols) in model_specs.items():
            pred, _ = _predict(model, cic_Xtest[cols]); m = _metrics(cic_truth, pred); cic_metrics[name] = m
            w(f"  {name:20} FTP-rec {_f(m['ftp_recall'])} Ben-rec {_f(m['benign_recall'])} acc {m['accuracy']:.4f}")
        pd.DataFrame([{"model": n, **{k: cic_metrics[n][k] for k in ("ftp_recall", "benign_recall", "accuracy")}} for n in cic_metrics]).to_csv(out / "cic_heldout_metrics.csv", index=False)

        # ---- FINAL cleartext evaluation (once) --------------------------
        w(self.style.MIGRATE_HEADING("\nFINAL cleartext evaluation (2nd independent corpus)"))
        metrics, preds, rows = {}, {}, []
        for name, (model, cols) in model_specs.items():
            pred, conf = _predict(model, cle[cols]); preds[name] = (pred, conf)
            m = _metrics(cle["Label"].to_numpy(), pred); metrics[name] = m
            rows.append({"model": name, "ftp_recall": m["ftp_recall"], "benign_recall": m["benign_recall"],
                         "false_positive_rate": m["false_positive_rate"], "ftp_precision": m["ftp_precision"],
                         "macro_f1": m["macro_f1"], "accuracy": m["accuracy"]})
            w(f"  {name:20} FTP-rec {_f(m['ftp_recall'])} Ben-rec {_f(m['benign_recall'])} "
              f"FPR {_f(m['false_positive_rate'])} prec {_f(m['ftp_precision'])} mF1 {m['macro_f1']:.4f}")
        pd.DataFrame(rows).to_csv(out / "final_test_metrics.csv", index=False)
        for name, m in metrics.items():
            c = m["confusion"]; pd.DataFrame(c["matrix"], index=c["labels"], columns=c["labels"]).to_csv(out / f"confusion_{name}.csv")

        # ---- per-scenario / structure / sessions-per-source -------------
        self._family(out, cle, preds)
        self._structure_and_buckets(w, out, cle, preds)
        self._breakdowns(out, cle, preds["cross_session_58"][0])

        # ---- bootstrap CIs ----------------------------------------------
        w(self.style.MIGRATE_HEADING("\nBootstrap 95% CIs (capture-level)"))
        boot = {}
        for name in ("C1_current_45", "C3_controlled_43", "cross_session_58"):
            model, cols = model_specs[name]; boot[name] = self._bootstrap(cle, model, cols, opts["n_boot"]); b = boot[name]
            w(f"  {name:20} FTP {b['ftp_recall']['mean']:.3f}[{b['ftp_recall']['lo95']:.3f},{b['ftp_recall']['hi95']:.3f}] "
              f"Ben {b['benign_recall']['mean']:.3f}[{b['benign_recall']['lo95']:.3f},{b['benign_recall']['hi95']:.3f}] "
              f"FPR {b['false_positive_rate']['mean']:.3f}[{b['false_positive_rate']['lo95']:.3f},{b['false_positive_rate']['hi95']:.3f}]")
        (out / "bootstrap_cis.json").write_text(json.dumps(boot, indent=2, default=str))

        # ---- SHAP + frozen with/without-cross ablation ------------------
        shap_res = None if opts["skip_shap"] else self._shap(cross, cle)
        ablation = {"with_cross_block_cross_session_58": {k: metrics["cross_session_58"][k] for k in ("ftp_recall", "benign_recall", "false_positive_rate")},
                    "without_cross_block_C1_45": {k: metrics["C1_current_45"][k] for k in ("ftp_recall", "benign_recall", "false_positive_rate")}}

        # ---- FTPS separate ----------------------------------------------
        ftps = self._ftps(enc, model_specs); pd.DataFrame(ftps).to_csv(out / "ftps_encrypted_metrics.csv", index=False)

        # ---- verdict -----------------------------------------------------
        verdict = self._verdict(metrics, cic_metrics, boot, shap_res, ablation)
        # attach the concrete failure/robustness mode: attack recall by session structure
        d2 = cle.copy(); d2["_pred"] = preds["cross_session_58"][0]
        d3 = cle.copy(); d3["_pred"] = preds["C3_controlled_43"][0]

        def _atk_recall(dd, mask):
            s = dd[mask & (dd["Label"] == FTP)]
            return round(float((s["_pred"] == FTP).mean()), 4) if len(s) else None
        ss = d2["session_structure"] == "single_session"
        low = d2["sessions_per_source"].fillna(0) <= 3
        verdict["failure_mode"] = {
            "single_session_attack_recall_cross": _atk_recall(d2, ss.to_numpy()),
            "single_session_attack_recall_C3": _atk_recall(d3, (d3["session_structure"] == "single_session").to_numpy()),
            "low_session_1to3_attack_recall_cross": _atk_recall(d2, low.to_numpy()),
            "high_session_4plus_attack_recall_cross": _atk_recall(d2, (~low).to_numpy()),
            "note": ("Cross-session model recall on attacks by session structure -- if single/low-session attack recall "
                     "is much lower than high-session, the model over-relies on session count.")}
        w(self.style.MIGRATE_HEADING("\nVERDICT"))
        w(self.style.WARNING("  " + verdict["verdict"] + " -- " + verdict["summary"]))

        after = _frozen(root)
        if before != after:
            raise SystemExit(f"ABORT: frozen artifact(s) changed: {[k for k in before if before[k]!=after.get(k)]}")
        w(f"\n  frozen artifacts unchanged (before==after): {before == after}")

        self._write(out, root, rows, metrics, cic_metrics, boot, shap_res, ablation, ftps, verdict, integ, before, after, real, cle, enc)
        w(self.style.SUCCESS(f"\nEvidence written to {out}"))
        w(self.style.WARNING("\nNo promotion, no production change, no merge. 2nd corpus is test-only."))

    # ---- helpers ---------------------------------------------------------

    def _integrity(self, root, before):
        new_paths = sorted(glob.glob(str(root / "validation" / CORPUS / "**/*.pcap"), recursive=True))
        new_h = [_sha(p) for p in new_paths]; new_set = set(new_h)
        prior = {k: before[k + "_pcaps"] for k in ("v1", "v2", "indep", "targeted", "robustness", "robust_train", "benign_fl", "cross_session", "indep_ftp")}
        checks = {f"disjoint_from_{k}": new_set.isdisjoint(set(v)) for k, v in prior.items()}
        checks["no_duplicate_pcaps"] = len(new_h) == len(new_set)
        checks["feature_count_58"] = len(FCROSS) == 58
        checks["packet_order_preserved"] = FCROSS[:30] == list(ml.FEATURES)
        checks["behavioural_15_preserved"] = FCROSS[30:45] == list(fb.BEHAV_FEATURES)
        checks["cross_appended"] = FCROSS[45:] == list(fcs.CROSS_FEATURES)
        return {"all_pass": all(checks.values()), "checks": checks, "n_new": len(new_h), "n_prior": len(prior)}

    def _attach_meta(self, root, df):
        man = pd.read_csv(root / "validation" / CORPUS / "MANIFEST.csv")
        df = df.copy(); df["capture_id"] = df["capture"].map(lambda p: "_".join(p.split("_")[:2]))
        m = man.set_index("capture_id")
        for col in ("scenario", "scenario_family", "client", "server", "session_structure", "encrypted", "sessions"):
            df[col] = df["capture_id"].map(m[col])
        df["encrypted"] = df["encrypted"].astype(bool)
        df["sessions_per_source"] = df["ftpx_sessions_per_source"]
        return df

    def _family(self, out, df, preds):
        rows = []
        for name, (pred, _c) in preds.items():
            d2 = df.copy(); d2["_pred"] = pred
            for (fam, lab), sub in d2.groupby(["scenario_family", "Label"]):
                rows.append({"model": name, "scenario_family": fam, "label": lab, "n_flows": int(len(sub)),
                             "recall": round(float((sub["_pred"] == lab).mean()), 4)})
        pd.DataFrame(rows).to_csv(out / "per_scenario_family_metrics.csv", index=False)

    def _structure_and_buckets(self, w, out, df, preds):
        # per session_structure (single vs multi) and per sessions-per-source bucket, for the cross model
        d2 = df.copy(); d2["_pred"] = preds["cross_session_58"][0]
        rows = []
        for key in ("session_structure",):
            for val, sub in d2.groupby(key):
                for lab in (FTP, BENIGN):
                    s = sub[sub["Label"] == lab]
                    if len(s):
                        rows.append({"dimension": key, "value": str(val), "label": lab, "n_flows": int(len(s)),
                                     "recall": round(float((s["_pred"] == lab).mean()), 4)})
        d2["bucket"] = pd.cut(d2["sessions_per_source"].fillna(0), [-0.1, 1.5, 3.5, 6.5, 1e9], labels=["1", "2-3", "4-6", "7+"])
        for val, sub in d2.groupby("bucket", observed=True):
            for lab in (FTP, BENIGN):
                s = sub[sub["Label"] == lab]
                if len(s):
                    rows.append({"dimension": "sessions_per_source", "value": str(val), "label": lab,
                                 "n_flows": int(len(s)), "recall": round(float((s["_pred"] == lab).mean()), 4)})
        pd.DataFrame(rows).to_csv(out / "per_session_structure_metrics.csv", index=False)
        w(self.style.MIGRATE_HEADING("\nSession-structure robustness (cross candidate)"))
        for r in rows:
            if r["dimension"] == "session_structure" or (r["dimension"] == "sessions_per_source"):
                w(f"  {r['dimension']:20} {r['value']:8} {r['label']:14} n={r['n_flows']:3} recall={_f(r['recall'])}")

    def _breakdowns(self, out, df, pred):
        d2 = df.copy(); d2["_pred"] = pred
        for key, fname in (("scenario", "per_scenario_metrics.csv"), ("client", "per_client_metrics.csv")):
            rows = []
            for val, sub in d2.groupby(key):
                ftp = sub[sub["Label"] == FTP]; ben = sub[sub["Label"] == BENIGN]
                rows.append({key: val, "n_flows": int(len(sub)),
                             "ftp_recall": round(float((ftp["_pred"] == FTP).mean()), 4) if len(ftp) else None,
                             "benign_recall": round(float((ben["_pred"] == BENIGN).mean()), 4) if len(ben) else None})
            pd.DataFrame(rows).to_csv(out / fname, index=False)

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
        vals = np.abs(np.asarray(shap.TreeExplainer(model).shap_values(df[FCROSS]))).mean(axis=(0, 2))
        order = np.argsort(vals)[::-1]; total = vals.sum() + 1e-12
        cross = set(fcs.CROSS_FEATURES); behav = set(fb.BEHAV_FEATURES)
        top = [{"feature": FCROSS[j], "share": float(vals[j] / total)} for j in order[:14]]
        return {"top14": top, "max_feature": FCROSS[order[0]], "max_share": float(vals[order[0]] / total),
                "cross_session_share": float(sum(vals[j] for j in range(len(vals)) if FCROSS[j] in cross) / total),
                "behavioural_share": float(sum(vals[j] for j in range(len(vals)) if FCROSS[j] in behav) / total),
                "cic_artifact_share": float(sum(vals[j] for j in range(len(vals)) if FCROSS[j] in CIC_ARTIFACTS) / total),
                "top_cross_feature": next((FCROSS[j] for j in order if FCROSS[j] in cross), None),
                "single_feature_shortcut": float(vals[order[0]] / total) > 0.5}

    def _ftps(self, enc, model_specs):
        rows = []
        if len(enc):
            for name, (model, cols) in model_specs.items():
                pred, _ = _predict(model, enc[cols]); t = enc["Label"].to_numpy(); fm = t == FTP; bm = t == BENIGN
                rows.append({"model": name, "n": int(len(enc)),
                             "ftp_recall": float((pred[fm] == FTP).mean()) if fm.any() else None,
                             "benign_recall": float((pred[bm] == BENIGN).mean()) if bm.any() else None})
        return rows

    def _verdict(self, metrics, cic_metrics, boot, shap_res, ablation):
        cr = metrics["cross_session_58"]; c1 = metrics["C1_current_45"]
        ftp = cr["ftp_recall"] or 0; ben = cr["benign_recall"] or 0; fpr = cr["false_positive_rate"] or 0
        lo_ftp = boot["cross_session_58"]["ftp_recall"]["lo95"]; lo_ben = boot["cross_session_58"]["benign_recall"]["lo95"]
        cic_ok = (cic_metrics["cross_session_58"]["ftp_recall"] or 0) >= (cic_metrics["production"]["ftp_recall"] or 0) - 0.03 \
            and abs((cic_metrics["cross_session_58"]["accuracy"] or 0) - (cic_metrics["production"]["accuracy"] or 0)) < 0.01
        single_feat = bool(shap_res and shap_res["single_feature_shortcut"])
        cross_contributes = (ftp > (c1["ftp_recall"] or 0)) or (ben > (c1["benign_recall"] or 0)) or (fpr < (c1["false_positive_rate"] or 1))
        target = ftp >= 0.90 and ben >= 0.90 and fpr <= 0.10 and cic_ok and not single_feat
        if target:
            verdict = "PROMISING -- TARGET MET"
            summary = (f"On the 2nd genuinely-independent corpus (pure-ftpd/ncftp/10.88.0.x) the cross-session candidate "
                       f"meets the target: FTP recall {ftp:.3f}, benign recall {ben:.3f}, FPR {fpr:.3f} (CI lo FTP {lo_ftp:.3f}"
                       f"/benign {lo_ben:.3f}); no CIC regression; no single-feature shortcut (max SHAP "
                       f"{_f(shap_res['max_share'] if shap_res else None)}). Holds across a different server/client/network.")
        elif ftp >= 0.90 and (ben >= 0.90 or fpr <= 0.10) and cic_ok and not single_feat:
            verdict = "PROMISING -- NEEDS MORE DATA"
            summary = (f"The cross-session candidate largely holds on the 2nd independent corpus (FTP {ftp:.3f}, benign "
                       f"{ben:.3f}, FPR {fpr:.3f}) but not every target threshold is met (benign>=0.90? {ben>=0.90}; "
                       f"FPR<=0.10? {fpr<=0.10}; CI lo benign {lo_ben:.3f}). Needs more independent data.")
        else:
            verdict = "NOT EFFECTIVE"
            summary = (f"On the 2nd independent corpus the cross-session candidate did not meet the target "
                       f"(FTP {ftp:.3f}>=0.90? benign {ben:.3f}>=0.90? FPR {fpr:.3f}<=0.10?; CIC no-regression {cic_ok}; "
                       f"single-feature shortcut {single_feat}). See failure mode.")
        return {"verdict": verdict, "promote": False, "summary": summary,
                "cross_test": {"ftp_recall": ftp, "benign_recall": ben, "false_positive_rate": fpr,
                               "ftp_precision": cr["ftp_precision"], "macro_f1": cr["macro_f1"], "accuracy": cr["accuracy"]},
                "c1_test": {"ftp_recall": c1["ftp_recall"], "benign_recall": c1["benign_recall"], "false_positive_rate": c1["false_positive_rate"]},
                "ci_lower_ftp_recall": lo_ftp, "ci_lower_benign_recall": lo_ben,
                "cic_no_regression": bool(cic_ok), "single_feature_shortcut": single_feat,
                "cross_block_contributes_vs_c1": bool(cross_contributes), "ablation": ablation,
                "cross_session_shap_share": (shap_res["cross_session_share"] if shap_res else None),
                "caveats": ["Different server (pure-ftpd), client (ncftp), netns/subnet (10.88.0.x) and fresh scenario "
                            "code, but still a single-container lab -- not a separate OS/host or the public internet.",
                            "Pure-FTPd's escalating failure delay bounds attempt counts, so single-session brute force is "
                            "shorter here than a real attacker could sustain.",
                            "The corpus was evaluated once, for reporting; never training/selection/tuning.",
                            "Small corpus (wide CIs). No promotion, no threshold change, no merge."]}

    def _write(self, out, root, rows, metrics, cic_metrics, boot, shap_res, ablation, ftps, verdict, integ, before, after, real, cle, enc):
        if shap_res:
            pd.DataFrame(shap_res["top14"]).to_csv(out / "shap_top_features.csv", index=False)
        leak = {**integ, "labels_from_folders": True, "no_zero_fill": len(real.invalid) == 0,
                "cleartext_flows": int(len(cle)), "encrypted_flows": int(len(enc)),
                "every_flow_traceable": bool(real.df["capture"].notna().all()),
                "used_for": "evaluation only (never training/selection/threshold/tuning)"}
        (out / "leakage_validation.json").write_text(json.dumps(leak, indent=2, default=str))
        (out / "model_hashes_before_after.json").write_text(json.dumps({"before": before, "after": after, "unchanged": before == after}, indent=2, default=str))
        (out / "final_verdict.json").write_text(json.dumps({**verdict, "shap": shap_res, "ftps": ftps}, indent=2, default=str))
        (out / "evaluation_metadata.json").write_text(json.dumps({
            "experiment": "ftp_cross_session_independent_validation", "created_utc": datetime.now(timezone.utc).isoformat(),
            "corpus": f"validation/{CORPUS}", "server": "pure-ftpd", "client_new": "ncftp",
            "network": "veth + ip netns 10.88.0.0/24", "evaluated_models": list(metrics.keys()), "test_only": True,
            "versions": {"python": platform.python_version()}}, indent=2, default=str))
        self._report(out, rows, cic_metrics, boot, shap_res, ablation, ftps, verdict)

    def _report(self, out, rows, cic_metrics, boot, shap_res, ablation, ftps, verdict):
        tbl = "\n".join(f"| {r['model']} | {_f(r['ftp_recall'])} | {_f(r['benign_recall'])} | {_f(r['false_positive_rate'])} | "
                        f"{_f(r['ftp_precision'])} | {_f(r['macro_f1'])} | {_f(r['accuracy'])} |" for r in rows)
        btbl = "\n".join(f"| {n} | {boot[n]['ftp_recall']['mean']:.3f} [{boot[n]['ftp_recall']['lo95']:.3f},{boot[n]['ftp_recall']['hi95']:.3f}] | "
                         f"{boot[n]['benign_recall']['mean']:.3f} [{boot[n]['benign_recall']['lo95']:.3f},{boot[n]['benign_recall']['hi95']:.3f}] | "
                         f"{boot[n]['false_positive_rate']['mean']:.3f} [{boot[n]['false_positive_rate']['lo95']:.3f},{boot[n]['false_positive_rate']['hi95']:.3f}] |" for n in boot)
        ctbl = "\n".join(f"| {n} | {_f(cic_metrics[n]['ftp_recall'])} | {_f(cic_metrics[n]['benign_recall'])} | {_f(cic_metrics[n]['accuracy'])} |" for n in cic_metrics)
        ftbl = "\n".join(f"| {r['model']} | {_f(r['ftp_recall'])} | {_f(r['benign_recall'])} |" for r in ftps)
        shap_md = ""
        if shap_res:
            shap_md = (f"\n## SHAP (cross candidate, 2nd corpus)\n\n"
                       f"- cross-session block share **{shap_res['cross_session_share']:.3f}** (top cross feature "
                       f"`{shap_res['top_cross_feature']}`); behavioural **{shap_res['behavioural_share']:.3f}**; CIC "
                       f"artifacts **{shap_res['cic_artifact_share']:.3f}**; largest single feature `{shap_res['max_feature']}` "
                       f"**{shap_res['max_share']:.3f}** (single-feature shortcut: {shap_res['single_feature_shortcut']})\n"
                       f"- top: {', '.join(t['feature'] for t in shap_res['top14'][:6])}\n")
        (out / "report.md").write_text(f"""# Final independent validation of the cross-session FTP detector - report

**All models frozen (production, Candidate 2, the 45-feature candidate, C3, the
cross-session candidate, and ml.py/live_capture.py/pcap_validation.py/ftp_behavioral.py/
ftp_cross_session.py verified before==after). This 2nd independent corpus is TEST-ONLY --
never training/selection/tuning. No promotion, no merge.** Branch
`claude/ftp-cross-session-independent-validation`.

## What this test is

A SECOND genuinely-independent corpus, different from every prior one and from the first
vsFTPD test: **pure-ftpd** server (new), **ncftp** client (new), a new **netns/subnet**
(10.88.0.0/24) and ports (2222/2323), **fresh scenario code** with deliberate
**single- vs multi-session** structure, plus **FTPS**. It tests whether the cross-session
detector's benefit holds on a different server/client/network -- and whether it still
catches **single-session** attacks (which could expose over-reliance on session count).

## CIC held-out (no-regression check)
| Model | FTP recall | Benign recall | accuracy |
|---|---|---|---|
{ctbl}

## FINAL cleartext evaluation (2nd independent corpus, evaluated once)
| Model | FTP recall | Benign recall | FPR | FTP precision | macro-F1 | accuracy |
|---|---|---|---|---|---|---|
{tbl}

### Bootstrap 95% CIs
| Model | FTP recall [95% CI] | Benign recall [95% CI] | FPR [95% CI] |
|---|---|---|---|
{btbl}

## With / without the cross-session block (frozen models)

- with cross (cross_session_58): FTP {_f(ablation['with_cross_block_cross_session_58']['ftp_recall'])},
  benign {_f(ablation['with_cross_block_cross_session_58']['benign_recall'])},
  FPR {_f(ablation['with_cross_block_cross_session_58']['false_positive_rate'])}.
- without cross (C1 45): FTP {_f(ablation['without_cross_block_C1_45']['ftp_recall'])},
  benign {_f(ablation['without_cross_block_C1_45']['benign_recall'])},
  FPR {_f(ablation['without_cross_block_C1_45']['false_positive_rate'])}.
{shap_md}
## FTPS / TLS (encrypted; behavioural + cross-session unavailable)
| Model | FTP recall | Benign recall |
|---|---|---|
{ftbl}

## Verdict — {verdict['verdict']}

{verdict['summary']}

- cic_no_regression: **{verdict['cic_no_regression']}**  ·  single_feature_shortcut:
  **{verdict['single_feature_shortcut']}**  ·  cross_block_contributes_vs_c1:
  **{verdict['cross_block_contributes_vs_c1']}**  ·  promote: **{verdict['promote']}**

### What it proves / does not prove
Proves the cross-session detector's behaviour on a **different server (pure-ftpd), client
(ncftp), and network** with fresh scenarios and single/multi-session structure. Does NOT
prove generalisation to a different OS/host, the public internet, or attackers who sustain
long single-session brute force (bounded here by pure-ftpd's failure delay).

### Caveats
{chr(10).join('- ' + c for c in verdict['caveats'])}

## Files
`final_test_metrics.csv`, `cic_heldout_metrics.csv`, `per_scenario_family_metrics.csv`,
`per_session_structure_metrics.csv`, `per_scenario_metrics.csv`, `per_client_metrics.csv`,
`bootstrap_cis.json`, `shap_top_features.csv`, `ftps_encrypted_metrics.csv`,
`confusion_*.csv`, `leakage_validation.json`, `model_hashes_before_after.json`,
`final_verdict.json`.
""")


def _f(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v:.4f}" if isinstance(v, float) else str(v)
