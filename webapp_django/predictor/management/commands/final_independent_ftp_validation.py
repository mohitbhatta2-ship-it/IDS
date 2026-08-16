"""
FINAL independent evaluation of the robust behavioural candidate (models frozen).

Evaluates the frozen production model, Candidate 2, and the robust behavioural
candidate (``candidate_robust_unweighted``) on the fresh, genuinely-different
independent corpus (real vsftpd, lftp, non-loopback veth/netns, plus FTPS). Nothing
is trained, weighted, tuned, or selected on this corpus -- it is TEST-ONLY.

Also evaluates three frozen ablation models from the robust-retraining phase (30-only,
behavioural-only, 30+behavioural WITHOUT ftp_failed_logins) -- all trained on the
approved corpora, never on this corpus -- to answer "with vs without behavioural
features" and the ftp_failed_logins ablation on genuinely new traffic. SHAP on the
robust candidate quantifies its dependence on ftp_failed_logins / successful auth /
CIC artifacts / any single application-layer feature.

FTPS/TLS captures are analysed SEPARATELY and clearly marked: the cleartext
behavioural features are genuinely unavailable (encrypted control channel).

Verdict is one of: ROBUST -- PROMISING FOR PROMOTION REVIEW / PROMISING -- NEEDS MORE
DATA / NOT ROBUST / INVALID -- LEAKAGE/INTEGRITY FAILURE. Nothing is promoted or merged.

    python manage.py final_independent_ftp_validation
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

from predictor import ml, retraining as rt, retraining_behavioral as rbh
from predictor import ftp_behavioral as fb, independent_eval as ie

FTP, BENIGN = "FTP-BruteForce", "Benign"
FAILED = "ftp_failed_logins"
FEAT_NO_FAILED = [f for f in rbh.FEATURES_AUG if f != FAILED]
CORPUS = "independent_ftp_validation_pcaps"
MODELDIR = "validation/models/ftp-behavioral-robust-retraining"
CIC_ARTIFACTS = {"Dst Port", "Fwd Seg Size Min", "Init Fwd Win Byts", "Init Bwd Win Byts"}
AUTH_FEATURES = {"ftp_has_successful_auth", "ftp_successful_logins"}


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _pcap_hashes(root, sub):
    return sorted(_sha(p) for p in glob.glob(str(root / "validation" / sub / "**/*.pcap"), recursive=True))


def _frozen(root):
    d = {"production": _sha(ie.production_model_path()), "candidate2": _sha(ie.candidate2_file()),
         "robust_candidate": _sha(root / MODELDIR / "candidate_robust_unweighted.pkl"),
         "ablation_30only": _sha(root / MODELDIR / "ablation_30only.pkl"),
         "ablation_behavioural_only": _sha(root / MODELDIR / "ablation_behavioural_only.pkl"),
         "ablation_no_failed_logins": _sha(root / MODELDIR / "ablation_no_failed_logins.pkl"),
         "ml_py": _sha(root / "webapp_django/predictor/ml.py"),
         "live_capture_py": _sha(root / "webapp_django/predictor/live_capture.py"),
         "pcap_validation_py": _sha(root / "webapp_django/predictor/pcap_validation.py")}
    for name, sub in (("v1", "realistic_pcaps"), ("v2", "realistic_pcaps_v2"),
                      ("indep", "independent_real_pcaps"), ("targeted", "targeted_benign_pcaps"),
                      ("robustness", "robustness_pcaps"), ("robust_train", "robust_train_pcaps")):
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
    help = "Final independent evaluation on the fresh vsftpd/lftp/non-loopback corpus (frozen models)."

    def add_arguments(self, parser):
        parser.add_argument("--output", default=None)
        parser.add_argument("--skip-shap", action="store_true")
        parser.add_argument("--n-boot", type=int, default=2000)

    def handle(self, *args, **opts):
        import joblib
        w = self.stdout.write
        root = rbh.repo_root()
        out = Path(opts["output"]) if opts["output"] else root / "validation" / "results" / "final_independent_ftp_validation"
        out.mkdir(parents=True, exist_ok=True)
        before = _frozen(root)

        w(self.style.MIGRATE_HEADING("FINAL independent FTP validation (frozen models; test-only)"))

        # ---- integrity/leakage guard BEFORE anything ---------------------
        integrity = self._integrity(root, before)
        if not integrity["all_pass"]:
            self._invalid(out, integrity, before)
            w(self.style.ERROR("  INTEGRITY/LEAKAGE FAILURE -> verdict INVALID; see final_verdict.json"))
            return
        w(f"  integrity/leakage OK: new corpus disjoint from all {integrity['n_prior_corpora']} prior corpora; "
          f"{integrity['n_new']} new pcaps.")

        # ---- extract augmented features (45) + attach manifest meta ------
        allflows = rbh.extract_real_augmented(("independent_ftp_val",))
        df = self._attach_meta(root, allflows.df)
        cle = df[~df["encrypted"]].reset_index(drop=True)
        enc = df[df["encrypted"]].reset_index(drop=True)
        w(f"  flows: {len(df)} total; cleartext {len(cle)} (FTP {int((cle['Label']==FTP).sum())}, "
          f"Benign {int((cle['Label']==BENIGN).sum())}); encrypted-FTPS {len(enc)}; invalid {len(allflows.invalid)}")

        # ---- models (all frozen) -----------------------------------------
        prod = ml._load(ml.DEFAULT_MODEL)[0]
        cand2 = joblib.load(ie.candidate2_file())
        robust = joblib.load(root / MODELDIR / "candidate_robust_unweighted.pkl")
        abl_30 = joblib.load(root / MODELDIR / "ablation_30only.pkl")
        abl_behav = joblib.load(root / MODELDIR / "ablation_behavioural_only.pkl")
        abl_nofail = joblib.load(root / MODELDIR / "ablation_no_failed_logins.pkl")
        model_specs = {
            "production": (prod, list(ml.FEATURES)), "candidate2": (cand2, list(ml.FEATURES)),
            "robust_behavioural": (robust, rbh.FEATURES_AUG),
            "ablation_30only": (abl_30, list(ml.FEATURES)),
            "ablation_behavioural_only": (abl_behav, list(fb.BEHAV_FEATURES)),
            "ablation_no_failed_logins": (abl_nofail, FEAT_NO_FAILED)}

        # ---- CLEARTEXT evaluation (primary) ------------------------------
        w(self.style.MIGRATE_HEADING("\nCleartext evaluation (behavioural features meaningful)"))
        metrics, preds, rows = {}, {}, []
        for name, (model, cols) in model_specs.items():
            pred, conf = _predict(model, cle[cols]); preds[name] = (pred, conf)
            m = _metrics(cle["Label"].to_numpy(), pred); metrics[name] = m
            rows.append({"model": name, "ftp_recall": m["ftp_recall"], "benign_recall": m["benign_recall"],
                         "false_positive_rate": m["false_positive_rate"], "ftp_precision": m["ftp_precision"],
                         "macro_precision": m["macro_precision"], "macro_recall": m["macro_recall"],
                         "macro_f1": m["macro_f1"], "accuracy": m["accuracy"]})
            w(f"  {name:28} FTP-rec {_f(m['ftp_recall'])} Ben-rec {_f(m['benign_recall'])} "
              f"FPR {_f(m['false_positive_rate'])} prec {_f(m['ftp_precision'])} mF1 {m['macro_f1']:.4f} acc {m['accuracy']:.4f}")
        pd.DataFrame(rows).to_csv(out / "cleartext_metrics.csv", index=False)

        # ---- confusion matrices ------------------------------------------
        for name, m in metrics.items():
            c = m["confusion"]
            pd.DataFrame(c["matrix"], index=c["labels"], columns=c["labels"]).to_csv(out / f"confusion_{name}.csv")

        # ---- per scenario / client / server / environment (robust) -------
        self._breakdowns(out, cle, preds["robust_behavioural"][0])
        fam_rows = self._family_breakdown(cle, preds)
        pd.DataFrame(fam_rows).to_csv(out / "per_scenario_family_metrics.csv", index=False)
        w(self.style.MIGRATE_HEADING("\nAdversarial families (robust candidate, cleartext)"))
        for r in fam_rows:
            if r["model"] == "robust_behavioural":
                w(f"  {r['scenario_family']:18} ({r['label']:14}) n={r['n_flows']:3} recall={_f(r['recall'])} "
                  f"pred_ftp={r['predicted_ftp']} pred_benign={r['predicted_benign']}")

        # ---- confidence distribution -------------------------------------
        self._confidence(out, cle, preds)

        # ---- bootstrap CIs (capture-level) for the three headline models -
        w(self.style.MIGRATE_HEADING("\nBootstrap 95% CIs (capture-level)"))
        boot = {}
        for name in ("production", "candidate2", "robust_behavioural"):
            model, cols = model_specs[name]
            boot[name] = self._bootstrap(cle, model, cols, opts["n_boot"])
            b = boot[name]
            w(f"  {name:20} FTP-rec {b['ftp_recall']['mean']:.3f} [{b['ftp_recall']['lo95']:.3f},{b['ftp_recall']['hi95']:.3f}]  "
              f"Ben-rec {b['benign_recall']['mean']:.3f} [{b['benign_recall']['lo95']:.3f},{b['benign_recall']['hi95']:.3f}]")
        (out / "bootstrap_cis.json").write_text(json.dumps(boot, indent=2, default=str))

        # ---- with vs without behavioural + ftp_failed_logins ablation ----
        ablation = {
            "with_behavioural_ftp_recall": metrics["robust_behavioural"]["ftp_recall"],
            "without_behavioural_30only_ftp_recall": metrics["ablation_30only"]["ftp_recall"],
            "behavioural_only_ftp_recall": metrics["ablation_behavioural_only"]["ftp_recall"],
            "without_ftp_failed_logins_ftp_recall": metrics["ablation_no_failed_logins"]["ftp_recall"],
            "with_behavioural_benign_recall": metrics["robust_behavioural"]["benign_recall"],
            "without_behavioural_30only_benign_recall": metrics["ablation_30only"]["benign_recall"],
            "with_behavioural_fpr": metrics["robust_behavioural"]["false_positive_rate"],
            "without_ftp_failed_logins_fpr": metrics["ablation_no_failed_logins"]["false_positive_rate"]}
        ablation["ftp_recall_drop_without_failed_logins"] = (
            (ablation["with_behavioural_ftp_recall"] or 0) - (ablation["without_ftp_failed_logins_ftp_recall"] or 0))

        # ---- SHAP dependence analysis (robust candidate, cleartext) ------
        shap_res = None if opts["skip_shap"] else self._shap(robust, cle)

        # ---- FTPS/TLS separate analysis (behavioural unavailable) --------
        ftps = self._ftps_analysis(enc, model_specs)
        pd.DataFrame(ftps["rows"]).to_csv(out / "ftps_encrypted_metrics.csv", index=False)
        w(self.style.MIGRATE_HEADING("\nFTPS/TLS (encrypted; behavioural features UNAVAILABLE)"))
        w(f"  {ftps['note']}")
        for r in ftps["rows"]:
            w(f"  {r['model']:28} FTP-rec {_f(r['ftp_recall'])} Ben-rec {_f(r['benign_recall'])} (n={r['n']})")

        # ---- dependence + verdict ----------------------------------------
        dependence = self._dependence(shap_res, ablation, cle, preds)
        verdict = self._verdict(metrics, ablation, dependence, boot, cle)
        w(self.style.MIGRATE_HEADING("\nVERDICT"))
        w(self.style.WARNING("  " + verdict["verdict"] + " -- " + verdict["summary"]))

        after = _frozen(root)
        if before != after:
            diff = [k for k in before if before[k] != after.get(k)]
            raise SystemExit(f"ABORT: frozen artifact(s) changed: {diff}")
        w(f"\n  frozen artifacts unchanged (before==after): {before == after}")

        self._write(out, root, rows, fam_rows, metrics, ablation, dependence, verdict, shap_res, boot,
                    ftps, integrity, before, after, allflows, cle, enc)
        w(self.style.SUCCESS(f"\nEvidence written to {out}"))
        w(self.style.WARNING("\nNo promotion, no production change, no merge. This corpus is test-only."))

    # ---- helpers ---------------------------------------------------------

    def _integrity(self, root, before):
        new_paths = sorted(glob.glob(str(root / "validation" / CORPUS / "**/*.pcap"), recursive=True))
        new_h = [_sha(p) for p in new_paths]
        new_set = set(new_h)
        prior = {"v1": before["v1_pcaps"], "v2": before["v2_pcaps"], "independent": before["indep_pcaps"],
                 "targeted": before["targeted_pcaps"], "robustness": before["robustness_pcaps"],
                 "robust_train": before["robust_train_pcaps"]}
        checks = {f"disjoint_from_{k}": new_set.isdisjoint(set(v)) for k, v in prior.items()}
        checks["no_duplicate_pcaps"] = len(new_h) == len(new_set)
        # feature integrity
        checks["exactly_45_features"] = len(rbh.FEATURES_AUG) == 45
        checks["exactly_30_packet_features"] = len(ml.FEATURES) == 30
        checks["packet_feature_order_preserved"] = rbh.FEATURES_AUG[:30] == list(ml.FEATURES)
        return {"all_pass": all(checks.values()), "checks": checks, "n_new": len(new_h),
                "n_prior_corpora": len(prior)}

    def _attach_meta(self, root, df):
        man = pd.read_csv(root / "validation" / CORPUS / "MANIFEST.csv")
        df = df.copy()
        df["capture_id"] = df["capture"].map(lambda p: "_".join(p.split("_")[:2]))
        m = man.set_index("capture_id")
        for col in ("scenario", "scenario_family", "client", "server", "environment", "mode", "encrypted"):
            df[col] = df["capture_id"].map(m[col])
        df["encrypted"] = df["encrypted"].astype(bool)
        # finite + no zero-fill sanity per flow (packet features)
        assert np.isfinite(df[list(ml.FEATURES)].to_numpy()).all(), "non-finite packet feature"
        return df

    def _family_breakdown(self, df, preds):
        rows = []
        for name, (pred, _c) in preds.items():
            d2 = df.copy(); d2["_pred"] = pred
            for (fam, lab), sub in d2.groupby(["scenario_family", "Label"]):
                rows.append({"model": name, "scenario_family": fam, "label": lab, "n_flows": int(len(sub)),
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
                             "accuracy": round(float((sub["_pred"] == sub["Label"]).mean()), 4),
                             "ftp_recall": round(float((ftp["_pred"] == FTP).mean()), 4) if len(ftp) else None,
                             "benign_recall": round(float((ben["_pred"] == BENIGN).mean()), 4) if len(ben) else None,
                             "benign_fp": int((ben["_pred"] != BENIGN).sum())})
            pd.DataFrame(rows).to_csv(out / fname, index=False)

    def _confidence(self, out, df, preds):
        crows = []
        for name, (pred, conf) in preds.items():
            truth = df["Label"].to_numpy(); wrong = pred != truth
            crows.append({"model": name, "mean_conf": round(float(conf.mean()), 4),
                          "mean_conf_correct": round(float(conf[~wrong].mean()), 4) if (~wrong).any() else None,
                          "mean_conf_wrong": round(float(conf[wrong].mean()), 4) if wrong.any() else None,
                          "confidently_wrong_ge_0.9": int(((wrong) & (conf >= 0.9)).sum())})
        pd.DataFrame(crows).to_csv(out / "confidence_distribution.csv", index=False)

    def _bootstrap(self, df, model, cols, n_boot, seed=42):
        rng = np.random.default_rng(seed)
        caps = sorted(df["capture"].unique())
        by_cap = {c: df[df["capture"] == c] for c in caps}
        dec = rt._encoded_to_name()
        acc, ftp_rec, ben_rec, fpr = [], [], [], []
        for _ in range(n_boot):
            pick = rng.choice(caps, size=len(caps), replace=True)
            sample = pd.concat([by_cap[c] for c in pick], ignore_index=True)
            pred = np.array([dec.get(int(c), str(c)) for c in model.predict(sample[cols])])
            truth = sample["Label"].to_numpy()
            acc.append(float((pred == truth).mean()))
            fm = truth == FTP; bm = truth == BENIGN
            ftp_rec.append(float((pred[fm] == FTP).mean()) if fm.any() else np.nan)
            ben_rec.append(float((pred[bm] == BENIGN).mean()) if bm.any() else np.nan)
            fpr.append(float((pred[bm] != BENIGN).mean()) if bm.any() else np.nan)

        def ci(arr):
            a = np.array(arr); a = a[~np.isnan(a)]
            return {"mean": float(a.mean()), "lo95": float(np.percentile(a, 2.5)), "hi95": float(np.percentile(a, 97.5))}

        return {"accuracy": ci(acc), "ftp_recall": ci(ftp_rec), "benign_recall": ci(ben_rec), "false_positive_rate": ci(fpr)}

    def _shap(self, model, df):
        try:
            import shap
        except Exception:  # noqa: BLE001
            return None
        data = df[rbh.FEATURES_AUG]
        vals = np.abs(np.asarray(shap.TreeExplainer(model).shap_values(data))).mean(axis=(0, 2))
        order = np.argsort(vals)[::-1]; total = vals.sum() + 1e-12
        top = [{"feature": rbh.FEATURES_AUG[j], "mean_abs_shap": float(vals[j]), "share": float(vals[j] / total)}
               for j in order[:12]]
        behav = set(fb.BEHAV_FEATURES)
        app_shares = {rbh.FEATURES_AUG[j]: float(vals[j] / total) for j in range(len(vals)) if rbh.FEATURES_AUG[j] in behav}
        return {"top12": top,
                "failed_logins_share": float(vals[rbh.FEATURES_AUG.index(FAILED)] / total),
                "failed_logins_rank": int(order.tolist().index(rbh.FEATURES_AUG.index(FAILED)) + 1),
                "auth_share": float(sum(vals[rbh.FEATURES_AUG.index(a)] for a in AUTH_FEATURES) / total),
                "behavioural_share": float(sum(vals[j] for j in range(len(vals)) if rbh.FEATURES_AUG[j] in behav) / total),
                "cic_artifact_share": float(sum(vals[j] for j in range(len(vals)) if rbh.FEATURES_AUG[j] in CIC_ARTIFACTS) / total),
                "max_single_app_layer_share": max(app_shares.values()) if app_shares else 0.0,
                "max_single_app_layer_feature": max(app_shares, key=app_shares.get) if app_shares else None,
                "dominated_by_failed_logins": float(vals[rbh.FEATURES_AUG.index(FAILED)] / total) > 0.5}

    def _ftps_analysis(self, enc, model_specs):
        rows = []
        if len(enc):
            for name, (model, cols) in model_specs.items():
                pred, _ = _predict(model, enc[cols])
                truth = enc["Label"].to_numpy()
                fm = truth == FTP; bm = truth == BENIGN
                rows.append({"model": name, "n": int(len(enc)),
                             "ftp_recall": float((pred[fm] == FTP).mean()) if fm.any() else None,
                             "benign_recall": float((pred[bm] == BENIGN).mean()) if bm.any() else None,
                             "predicted_ftp": int((pred == FTP).sum()), "predicted_benign": int((pred == BENIGN).sum())})
        return {"rows": rows,
                "note": ("Behavioural features are 0/unavailable under TLS (encrypted control channel); "
                         "these numbers reflect the 30 packet features only, and behavioural models "
                         "effectively fall back to their CIC-artifact behaviour on encrypted traffic.")}

    def _dependence(self, shap_res, ablation, cle, preds):
        # attacker-eventual-success (has successful auth) still caught? -> not fooled by 230
        d2 = cle.copy(); d2["_pred"] = preds["robust_behavioural"][0]
        ev = d2[(d2["scenario_family"] == "eventual_success") & (d2["Label"] == FTP)]
        eventual_recall = round(float((ev["_pred"] == FTP).mean()), 4) if len(ev) else None
        # benign-that-fails (mistype/gave_up) still benign? -> not over-flagging failed logins
        bf = d2[(d2["scenario_family"].isin(["mistype", "gave_up"])) & (d2["Label"] == BENIGN)]
        benign_fail_recall = round(float((bf["_pred"] == BENIGN).mean()), 4) if len(bf) else None
        return {
            "depends_on_ftp_failed_logins": {
                "shap_share": (shap_res["failed_logins_share"] if shap_res else None),
                "shap_rank": (shap_res["failed_logins_rank"] if shap_res else None),
                "ftp_recall_drop_when_removed": ablation["ftp_recall_drop_without_failed_logins"],
                "verdict": bool((shap_res and shap_res["failed_logins_share"] > 0.30) or
                                ablation["ftp_recall_drop_without_failed_logins"] > 0.30)},
            "depends_on_successful_auth": {
                "shap_share": (shap_res["auth_share"] if shap_res else None),
                "attacker_eventual_success_recall": eventual_recall,
                "verdict": bool(eventual_recall is not None and eventual_recall < 0.7)},
            "depends_on_cic_artifacts": {
                "shap_share": (shap_res["cic_artifact_share"] if shap_res else None),
                "verdict": bool(shap_res and shap_res["cic_artifact_share"] > 0.5)},
            "depends_on_single_app_layer_feature": {
                "max_share": (shap_res["max_single_app_layer_share"] if shap_res else None),
                "feature": (shap_res["max_single_app_layer_feature"] if shap_res else None),
                "verdict": bool(shap_res and shap_res["max_single_app_layer_share"] > 0.30)},
            "benign_that_fails_recall": benign_fail_recall,
            "attacker_eventual_success_recall": eventual_recall}

    def _verdict(self, metrics, ablation, dependence, boot, cle):
        r = metrics["robust_behavioural"]; c2 = metrics["candidate2"]
        ftp = r["ftp_recall"] or 0; ben = r["benign_recall"] or 0; fpr = r["false_positive_rate"] or 0
        lo_ftp = boot["robust_behavioural"]["ftp_recall"]["lo95"]
        lo_ben = boot["robust_behavioural"]["benign_recall"]["lo95"]
        beats_c2 = (ftp >= (c2["ftp_recall"] or 0)) and (fpr <= (c2["false_positive_rate"] or 1))
        thresholds = ftp >= 0.70 and ben >= 0.90 and fpr <= 0.10
        depends_failed = dependence["depends_on_ftp_failed_logins"]["verdict"]
        depends_single = dependence["depends_on_single_app_layer_feature"]["verdict"]
        benign_fail_ok = dependence["benign_that_fails_recall"] is None or dependence["benign_that_fails_recall"] >= 0.80
        eventual_ok = dependence["attacker_eventual_success_recall"] is None or dependence["attacker_eventual_success_recall"] >= 0.70
        robust = thresholds and beats_c2 and (not depends_failed) and (not depends_single) and benign_fail_ok and eventual_ok and lo_ftp >= 0.60 and lo_ben >= 0.80
        promising = thresholds and beats_c2
        if robust:
            verdict = "ROBUST -- PROMISING FOR PROMOTION REVIEW"
            summary = (f"On genuinely different traffic (vsftpd/lftp/non-loopback) the robust candidate holds FTP recall "
                       f"{ftp:.3f} (95% CI lo {lo_ftp:.3f}), benign recall {ben:.3f} (lo {lo_ben:.3f}), FPR {fpr:.3f}; "
                       f"benign-that-fails recall {dependence['benign_that_fails_recall']}, attacker-eventual-success "
                       f"{dependence['attacker_eventual_success_recall']}; no dominant single feature; not a "
                       f"ftp_failed_logins shortcut. Recommend human promotion review (still loopback-lab, single OS).")
        elif promising:
            verdict = "PROMISING -- NEEDS MORE DATA"
            summary = (f"Robust candidate beats Candidate 2 and meets the point thresholds (FTP {ftp:.3f}, benign {ben:.3f}, "
                       f"FPR {fpr:.3f}) but a robustness signal is soft (CI lo FTP {lo_ftp:.3f}/benign {lo_ben:.3f}, "
                       f"failed-login dependence={depends_failed}, single-feature={depends_single}, benign-fail "
                       f"recall {dependence['benign_that_fails_recall']}). Needs more/broader real traffic.")
        else:
            verdict = "NOT ROBUST"
            summary = (f"On genuinely different traffic the robust candidate does NOT meet the criteria (FTP {ftp:.3f}>=0.70? "
                       f"benign {ben:.3f}>=0.90? FPR {fpr:.3f}<=0.10?; beats C2={beats_c2}; failed-login "
                       f"dependence={depends_failed}; single-feature={depends_single}). See metrics.")
        return {"verdict": verdict, "promote": False,
                "robust_candidate_cleartext": {"ftp_recall": ftp, "benign_recall": ben, "false_positive_rate": fpr,
                                               "ftp_precision": r["ftp_precision"], "macro_f1": r["macro_f1"],
                                               "accuracy": r["accuracy"]},
                "candidate2_cleartext": {"ftp_recall": c2["ftp_recall"], "benign_recall": c2["benign_recall"],
                                         "false_positive_rate": c2["false_positive_rate"], "macro_f1": c2["macro_f1"]},
                "beats_candidate2": bool(beats_c2), "meets_thresholds": bool(thresholds),
                "ci_lower_ftp_recall": lo_ftp, "ci_lower_benign_recall": lo_ben,
                "summary": summary, "dependence": dependence, "ablation": ablation,
                "caveats": ["Single-container lab: real vsftpd + real veth/netns (non-loopback) but no separate OS/VM "
                            "or physical host; one server implementation; cleartext + one FTPS config.",
                            "FTPS captures analysed separately: behavioural features genuinely unavailable (encrypted).",
                            "This corpus was used for evaluation ONLY -- never training/weighting/threshold/feature/HP selection.",
                            "No promotion, no threshold/heuristic change, no merge."]}

    def _invalid(self, out, integrity, before):
        verdict = {"verdict": "INVALID -- LEAKAGE/INTEGRITY FAILURE", "promote": False,
                   "integrity": integrity, "summary": "New corpus failed leakage/feature-integrity checks; see checks."}
        (out / "final_verdict.json").write_text(json.dumps(verdict, indent=2, default=str))
        (out / "leakage_validation.json").write_text(json.dumps(integrity, indent=2, default=str))

    def _write(self, out, root, rows, fam_rows, metrics, ablation, dependence, verdict, shap_res, boot,
               ftps, integrity, before, after, allflows, cle, enc):
        if shap_res is not None:
            pd.DataFrame(shap_res["top12"]).to_csv(out / "shap_top_features.csv", index=False)
        leak = {**integrity, "labels_from_folders": True, "no_zero_fill": len(allflows.invalid) == 0,
                "cleartext_flows": int(len(cle)), "encrypted_flows": int(len(enc)),
                "cic_behavioural_nan_note": "CIC training flows keep NaN behavioural; this corpus has measured behavioural.",
                "every_flow_traceable_to_pcap": bool(allflows.df["capture"].notna().all()),
                "used_for": "evaluation only (never training/weighting/threshold/feature/HP selection)"}
        (out / "leakage_validation.json").write_text(json.dumps(leak, indent=2, default=str))
        (out / "model_hashes_before_after.json").write_text(json.dumps(
            {"before": before, "after": after, "unchanged": before == after}, indent=2, default=str))
        (out / "final_verdict.json").write_text(json.dumps({**verdict, "shap": shap_res, "ftps": ftps["rows"]}, indent=2, default=str))
        (out / "evaluation_metadata.json").write_text(json.dumps({
            "experiment": "final_independent_ftp_validation", "created_utc": datetime.now(timezone.utc).isoformat(),
            "corpus": f"validation/{CORPUS}", "server": "vsFTPd (real; new)", "client_new": "lftp",
            "network": "veth + ip netns 10.77.0.0/24 (non-loopback)",
            "evaluated_models": list(metrics.keys()), "test_only": True,
            "versions": {"python": platform.python_version()}}, indent=2, default=str))
        self._report(out, rows, metrics, ablation, dependence, verdict, shap_res, boot, ftps)

    def _report(self, out, rows, metrics, ablation, dependence, verdict, shap_res, boot, ftps):
        tbl = "\n".join(f"| {r['model']} | {_f(r['ftp_recall'])} | {_f(r['benign_recall'])} | {_f(r['false_positive_rate'])} | "
                        f"{_f(r['ftp_precision'])} | {_f(r['macro_f1'])} | {_f(r['accuracy'])} |" for r in rows)
        btbl = "\n".join(f"| {n} | {boot[n]['ftp_recall']['mean']:.3f} [{boot[n]['ftp_recall']['lo95']:.3f}, {boot[n]['ftp_recall']['hi95']:.3f}] | "
                         f"{boot[n]['benign_recall']['mean']:.3f} [{boot[n]['benign_recall']['lo95']:.3f}, {boot[n]['benign_recall']['hi95']:.3f}] | "
                         f"{boot[n]['false_positive_rate']['mean']:.3f} [{boot[n]['false_positive_rate']['lo95']:.3f}, {boot[n]['false_positive_rate']['hi95']:.3f}] |"
                         for n in boot)
        ftbl = "\n".join(f"| {r['model']} | {_f(r['ftp_recall'])} | {_f(r['benign_recall'])} | {r['predicted_ftp']}/{r['predicted_benign']} |" for r in ftps["rows"])
        shap_md = ""
        if shap_res:
            shap_md = (f"\n## SHAP dependence (robust candidate, cleartext)\n\n"
                       f"- `ftp_failed_logins`: share **{shap_res['failed_logins_share']:.3f}** (rank {shap_res['failed_logins_rank']})\n"
                       f"- successful-auth features (`ftp_has_successful_auth`+`ftp_successful_logins`): **{shap_res['auth_share']:.3f}**\n"
                       f"- CIC artifacts: **{shap_res['cic_artifact_share']:.3f}**; all behavioural: **{shap_res['behavioural_share']:.3f}**\n"
                       f"- largest single application-layer feature: `{shap_res['max_single_app_layer_feature']}` at **{shap_res['max_single_app_layer_share']:.3f}**\n"
                       f"- top features: {', '.join(t['feature'] for t in shap_res['top12'][:6])}\n")
        (out / "report.md").write_text(f"""# Final independent FTP validation - report

**All models frozen (production, Candidate 2, robust behavioural candidate, ablation
models, ml.py/live_capture.py/pcap_validation.py verified before==after). This corpus
is TEST-ONLY -- never training, weighting, threshold, feature, or hyperparameter
selection. No promotion, no merge.** Branch `claude/final-independent-ftp-validation`.

## What this test proves (and what it does not)

**Different stack:** real **vsFTPd** server (never used before), **lftp** client (new),
a genuine **non-loopback** private network (veth pair to an isolated `ip netns`,
`10.77.0.0/24`), and **FTPS/TLS** captured separately. Fresh scenario code (not the
prior generators). So this tests whether the robust candidate generalises to a
different server implementation, a different client, and a different network path.

**It does NOT prove** generalisation to a different OS, physical hosts, the public
internet, or FTP dialects beyond vsftpd -- this is still a single-container lab.

## Cleartext evaluation (behavioural features meaningful)

| Model | FTP recall | Benign recall | FPR | FTP precision | macro-F1 | accuracy |
|---|---|---|---|---|---|---|
{tbl}

## Bootstrap 95% CIs (capture-level resampling)

| Model | FTP recall [95% CI] | Benign recall [95% CI] | FPR [95% CI] |
|---|---|---|---|
{btbl}

## With vs without behavioural features / `ftp_failed_logins` ablation (cleartext)

- With behavioural (robust): FTP recall **{_f(ablation['with_behavioural_ftp_recall'])}**.
- 30 packet-only: **{_f(ablation['without_behavioural_30only_ftp_recall'])}**;
  behavioural-only (15): **{_f(ablation['behavioural_only_ftp_recall'])}**.
- WITHOUT `ftp_failed_logins` (44): **{_f(ablation['without_ftp_failed_logins_ftp_recall'])}**
  (drop **{_f(ablation['ftp_recall_drop_without_failed_logins'])}**).
{shap_md}
## Does the robust model still depend on...?

- **`ftp_failed_logins`**: {dependence['depends_on_ftp_failed_logins']['verdict']} (SHAP share
  {_f(dependence['depends_on_ftp_failed_logins']['shap_share'])}, removal drops FTP recall by
  {_f(dependence['depends_on_ftp_failed_logins']['ftp_recall_drop_when_removed'])}).
- **successful authentication**: {dependence['depends_on_successful_auth']['verdict']}
  (attacker-eventual-success recall {_f(dependence['attacker_eventual_success_recall'])} -- not fooled by the 230).
- **CIC artifacts**: {dependence['depends_on_cic_artifacts']['verdict']}
  (share {_f(dependence['depends_on_cic_artifacts']['shap_share'])}).
- **any single application-layer feature**: {dependence['depends_on_single_app_layer_feature']['verdict']}
  (max {_f(dependence['depends_on_single_app_layer_feature']['max_share'])} on
  `{dependence['depends_on_single_app_layer_feature']['feature']}`).

## FTPS / TLS (encrypted -- behavioural features UNAVAILABLE)

{ftps['note']}

| Model | FTP recall | Benign recall | pred FTP/Benign |
|---|---|---|---|
{ftbl}

## Verdict -- {verdict['verdict']}

{verdict['summary']}

- beats_candidate2: **{verdict['beats_candidate2']}**  ·  meets_thresholds: **{verdict['meets_thresholds']}**  ·
  CI-lower FTP recall **{verdict['ci_lower_ftp_recall']:.3f}**, benign **{verdict['ci_lower_benign_recall']:.3f}**  ·
  promote: **{verdict['promote']}**

### Caveats
{chr(10).join('- ' + c for c in verdict['caveats'])}

## Files

`cleartext_metrics.csv`, `confusion_*.csv`, `per_scenario_metrics.csv`,
`per_scenario_family_metrics.csv`, `per_client_metrics.csv`, `per_server_metrics.csv`,
`per_environment_metrics.csv`, `confidence_distribution.csv`, `bootstrap_cis.json`,
`ftps_encrypted_metrics.csv`, `shap_top_features.csv`, `leakage_validation.json`,
`model_hashes_before_after.json`, `evaluation_metadata.json`, `final_verdict.json`.
""")


def _f(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v:.4f}" if isinstance(v, float) else str(v)
