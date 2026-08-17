"""
Controlled ablation: is ``ftp_failed_logins`` causing the benign false positives?

Holds the training data constant (the same approved corpora the current 45-feature
behavioural candidate used: CIC + v1 + v2 + targeted + robust_train) and varies ONLY the
feature set, so any difference is attributable to the feature change -- not a data change.

Candidates:
  * C1  full 45  (frozen ``candidate_robust_unweighted`` -- the current model, evaluated as-is)
  * C2  44       (45 minus ``ftp_failed_logins``)
  * C3  43       (45 minus ``ftp_failed_logins`` AND ``ftp_failed_login_ratio`` -- remove the raw
                  failed-login count and the failure ratio, retain every other behavioural feature)
  * C4  40       (alternative small set: 30 packet + positive-evidence behavioural, dropping every
                  direct failure-volume encoder -- failed_logins, failed_login_ratio,
                  error_responses_5xx -- and the raw volume proxies login_attempts / total_commands)

Selection uses ONLY CIC held-out + capture-level LOCO on the approved training corpora --
never the frozen vsFTPD test. The frozen vsFTPD corpus is evaluated once, at the end, for
reporting. Candidate models go under ``validation/models/ftp-failed-login-ablation`` --
never webapp_data. Nothing here changes ml.py / live_capture.py / pcap_validation.py /
ftp_behavioral.py or any frozen model. No promotion, no merge.

    python manage.py ftp_failed_login_ablation
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
    ftp_behavioral as fb, independent_eval as ie

FTP, BENIGN = "FTP-BruteForce", "Benign"
FAILED = "ftp_failed_logins"
RATIO = "ftp_failed_login_ratio"
ERR5XX = "ftp_error_responses_5xx"
TRAIN_SOURCES = ("v1", "v2", "targeted", "robust_train")     # same as the current 45-feature candidate
TEST_CORPUS = "independent_ftp_validation_pcaps"
ROBUST_PKL = "validation/models/ftp-behavioral-robust-retraining/candidate_robust_unweighted.pkl"
CIC_ARTIFACTS = {"Dst Port", "Fwd Seg Size Min", "Init Fwd Win Byts", "Init Bwd Win Byts"}

F45 = list(rbh.FEATURES_AUG)                                   # 30 + 15
F44 = [f for f in F45 if f != FAILED]                          # C2
F43 = [f for f in F45 if f not in (FAILED, RATIO)]             # C3
# C4: positive-evidence behavioural only (no direct failure-volume encoders / volume proxies)
_C4_BEHAV = ["ftp_successful_logins", "ftp_has_successful_auth", "ftp_user_commands",
             "ftp_distinct_commands", "ftp_data_setup_responses", "ftp_data_commands",
             "ftp_ok_responses_2xx", "ftp_control_connections", "ftp_reconnects", "ftp_cmds_per_connection"]
F40 = list(ml.FEATURES) + _C4_BEHAV                            # C4 (30 + 10)

CANDIDATES = {"C1_full_45": F45, "C2_no_failed_logins_44": F44,
              "C3_no_failed_or_ratio_43": F43, "C4_alt_no_failure_volume_40": F40}


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
    help = "Controlled ftp_failed_logins ablation (candidates only; production frozen; no promotion)."

    def add_arguments(self, parser):
        parser.add_argument("--output", default=None)
        parser.add_argument("--skip-shap", action="store_true")
        parser.add_argument("--loco-cic", type=int, default=6000)
        parser.add_argument("--loco-caps", type=int, default=50)
        parser.add_argument("--n-boot", type=int, default=2000)

    def handle(self, *args, **opts):
        import joblib
        w = self.stdout.write
        root = rbh.repo_root()
        out = Path(opts["output"]) if opts["output"] else root / "validation" / "results" / "ftp_failed_login_ablation"
        out.mkdir(parents=True, exist_ok=True)
        before = _frozen(root)

        w(self.style.MIGRATE_HEADING("Controlled ftp_failed_logins ablation (production frozen)"))

        # ---- leakage guard: training corpora disjoint from the frozen test ----
        train_h = set().union(*(set(before[k]) for k in ("v1_pcaps", "v2_pcaps", "targeted_pcaps", "robust_train_pcaps")))
        if not train_h.isdisjoint(set(before["indep_ftp_pcaps"])):
            raise SystemExit("ABORT: a training PCAP overlaps the frozen vsFTPD test!")
        w(f"  leakage guard OK: training corpora disjoint from the frozen vsFTPD test.")
        w(f"  feature-set sizes: C1={len(F45)} C2={len(F44)} C3={len(F43)} C4={len(F40)} (data held constant)")

        # ---- data (45-feature augmented; subset columns per candidate) ---
        real = rbh.extract_real_augmented(TRAIN_SOURCES)
        cic_Xaug, cic_y = rbh.load_cic_augmented()
        cic_Xtest, cic_ytest = rbh.load_cic_test_augmented()
        test = self._extract_test(root)                          # the frozen vsFTPD independent test corpus
        tdf = self._attach_meta(root, test.df)
        cle = tdf[~tdf["encrypted"]].reset_index(drop=True); enc = tdf[tdf["encrypted"]].reset_index(drop=True)
        w(f"  train real flows: {len(real.df)} (FTP {int((real.df['Label']==FTP).sum())}, "
          f"Benign {int((real.df['Label']==BENIGN).sum())}); test cleartext {len(cle)}, FTPS {len(enc)}")

        # ---- frozen references + retrained ablation candidates -----------
        prod = ml._load(ml.DEFAULT_MODEL)[0]
        cand2 = joblib.load(ie.candidate2_file())
        c1 = joblib.load(root / ROBUST_PKL)                       # frozen full-45 candidate (C1)
        w(self.style.MIGRATE_HEADING("\nTraining ablation candidates (same data, varied features)"))
        models = {"C1_full_45": (c1, F45)}
        mdir = self._mkmodeldir(root)
        for name in ("C2_no_failed_logins_44", "C3_no_failed_or_ratio_43", "C4_alt_no_failure_volume_40"):
            cols = CANDIDATES[name]
            m = self._train_cols(cic_Xaug, cic_y, real.df, cols)
            models[name] = (m, cols)
            joblib.dump(m, mdir / f"{name}.pkl")
        (mdir / "README.json").write_text(json.dumps(
            {n: {"n_features": len(c), "features": c} for n, c in CANDIDATES.items()}, indent=2))
        w(f"  trained C2/C3/C4 (C1 is the frozen current candidate).")

        # ---- CIC held-out (no-regression gate) --------------------------
        w(self.style.MIGRATE_HEADING("\nCIC held-out evaluation"))
        dec = rt._encoded_to_name()
        cic_truth = np.array([dec.get(int(c), str(c)) for c in cic_ytest.to_numpy()])
        cic_metrics, cic_rows = {}, []
        ref = {"production": (prod, list(ml.FEATURES)), "candidate2": (cand2, list(ml.FEATURES))}
        for name, (model, cols) in {**ref, **models}.items():
            pred, _ = _predict(model, cic_Xtest[cols]); m = _metrics(cic_truth, pred); cic_metrics[name] = m
            cic_rows.append({"model": name, "ftp_recall": m["ftp_recall"], "benign_recall": m["benign_recall"],
                             "macro_f1": m["macro_f1"], "accuracy": m["accuracy"]})
            w(f"  {name:30} FTP-rec {_f(m['ftp_recall'])} Ben-rec {_f(m['benign_recall'])} acc {m['accuracy']:.4f}")
        pd.DataFrame(cic_rows).to_csv(out / "cic_heldout_metrics.csv", index=False)

        # ---- LOCO on approved corpora (selection among C2/C3/C4) --------
        w(self.style.MIGRATE_HEADING("\nCapture-level LOCO on approved corpora (selection)"))
        sub_X, sub_y = rbh.stratified_cic_subsample_aug(cic_Xaug, cic_y, opts["loco_cic"])
        caps = self._loco_caps(real.df, opts["loco_caps"])
        w(f"  LOCO over {len(caps)} captures (stratified; CIC 15-class -> ~12s/fit)")
        loco = {}
        for name in ("C2_no_failed_logins_44", "C3_no_failed_or_ratio_43", "C4_alt_no_failure_volume_40"):
            loco[name] = self._loco(sub_X, sub_y, real, caps, CANDIDATES[name])
            w(f"  {name:30} LOCO pooled: FTP-rec {_f(loco[name].get(FTP+'_recall'))} "
              f"Ben-rec {_f(loco[name].get(BENIGN+'_recall'))} macro-F1 {_f(loco[name].get('macro_f1'))}")
        (out / "loco_pooled_metrics.json").write_text(json.dumps(loco, indent=2, default=str))

        # ---- selection (CIC gate + LOCO; NOT the vsFTPD test) -----------
        sel_name, sel_reason = self._select(cic_metrics, loco)
        w(self.style.SUCCESS(f"\n  SELECTED (CIC+LOCO only): {sel_name} -- {sel_reason}"))

        # ---- FINAL vsFTPD test (once) -----------------------------------
        w(self.style.MIGRATE_HEADING("\nFINAL vsFTPD independent test (cleartext; evaluated once)"))
        eval_models = {"production": (prod, list(ml.FEATURES)), "candidate2": (cand2, list(ml.FEATURES)), **models}
        metrics, preds, rows = {}, {}, []
        for name, (model, cols) in eval_models.items():
            pred, conf = _predict(model, cle[cols]); preds[name] = (pred, conf)
            m = _metrics(cle["Label"].to_numpy(), pred); metrics[name] = m
            rows.append({"model": name, "ftp_recall": m["ftp_recall"], "benign_recall": m["benign_recall"],
                         "false_positive_rate": m["false_positive_rate"], "ftp_precision": m["ftp_precision"],
                         "macro_precision": m["macro_precision"], "macro_f1": m["macro_f1"], "accuracy": m["accuracy"]})
            w(f"  {name:30} FTP-rec {_f(m['ftp_recall'])} Ben-rec {_f(m['benign_recall'])} "
              f"FPR {_f(m['false_positive_rate'])} prec {_f(m['ftp_precision'])} mF1 {m['macro_f1']:.4f}")
        pd.DataFrame(rows).to_csv(out / "final_test_metrics.csv", index=False)
        for name, m in metrics.items():
            c = m["confusion"]
            pd.DataFrame(c["matrix"], index=c["labels"], columns=c["labels"]).to_csv(out / f"confusion_{name}.csv")

        # ---- per-scenario-family robustness (all candidates) ------------
        fam_rows = self._family_breakdown(cle, preds)
        pd.DataFrame(fam_rows).to_csv(out / "per_scenario_family_metrics.csv", index=False)
        w(self.style.MIGRATE_HEADING("\nRobustness across scenarios (selected + C1)"))
        for r in fam_rows:
            if r["model"] in (sel_name, "C1_full_45"):
                w(f"  {r['model']:26} {r['scenario_family']:16} ({r['label']:14}) n={r['n_flows']:3} recall={_f(r['recall'])}")
        self._breakdowns(out, cle, preds[sel_name][0])

        # ---- bootstrap CIs ----------------------------------------------
        w(self.style.MIGRATE_HEADING("\nBootstrap 95% CIs (capture-level)"))
        boot = {}
        for name in ("C1_full_45", sel_name) if sel_name != "C1_full_45" else ("C1_full_45",):
            model, cols = eval_models[name]
            boot[name] = self._bootstrap(cle, model, cols, opts["n_boot"])
            b = boot[name]
            w(f"  {name:26} FTP {b['ftp_recall']['mean']:.3f}[{b['ftp_recall']['lo95']:.3f},{b['ftp_recall']['hi95']:.3f}] "
              f"Ben {b['benign_recall']['mean']:.3f}[{b['benign_recall']['lo95']:.3f},{b['benign_recall']['hi95']:.3f}] "
              f"FPR {b['false_positive_rate']['mean']:.3f}[{b['false_positive_rate']['lo95']:.3f},{b['false_positive_rate']['hi95']:.3f}]")
        (out / "bootstrap_cis.json").write_text(json.dumps(boot, indent=2, default=str))

        # ---- SHAP on selected -------------------------------------------
        shap_res = None if opts["skip_shap"] else self._shap(*eval_models[sel_name], cle)

        # ---- FTPS separate ----------------------------------------------
        ftps = self._ftps(enc, eval_models)
        pd.DataFrame(ftps).to_csv(out / "ftps_encrypted_metrics.csv", index=False)

        # ---- verdict -----------------------------------------------------
        verdict = self._verdict(sel_name, metrics, cic_metrics, boot, shap_res, fam_rows)
        w(self.style.MIGRATE_HEADING("\nVERDICT"))
        w(self.style.WARNING("  " + verdict["verdict"] + " -- " + verdict["summary"]))

        after = _frozen(root)
        if before != after:
            raise SystemExit(f"ABORT: frozen artifact(s) changed: {[k for k in before if before[k]!=after.get(k)]}")
        w(f"\n  frozen artifacts unchanged (before==after): {before == after}")

        self._write(out, root, rows, cic_rows, fam_rows, loco, boot, metrics, cic_metrics, verdict,
                    shap_res, ftps, sel_name, sel_reason, before, after, real, test, cle, enc)
        w(self.style.SUCCESS(f"\nEvidence written to {out}"))
        w(self.style.WARNING("\nNo promotion, no production change, no merge. vsFTPD corpus is test-only."))

    # ---- helpers ---------------------------------------------------------

    def _mkmodeldir(self, root):
        d = root / "validation" / "models" / "ftp-failed-login-ablation"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _extract_test(self, root):
        # extract the frozen vsFTPD independent test corpus with the 45-feature schema
        from predictor import retraining_v2 as r2, pcap_validation as pv, live_capture
        live_capture._ensure_live_on_path()
        base = root / "validation" / TEST_CORPUS
        rows, invalid = [], []
        for folder, label in (("benign", BENIGN), ("ftp_bruteforce", FTP)):
            d = base / folder
            if not d.is_dir():
                continue
            for pcap in sorted(d.glob("*.pcap")):
                behav = fb.behavioural_features_for_pcap(pcap)
                for i, fl in enumerate(pv.replay_pcap(pcap)):
                    if pv._feature_problem(fl["features"]):
                        invalid.append({"capture": pcap.name}); continue
                    row = {f: float(fl["features"][f]) for f in ml.FEATURES}
                    row.update({bfeat: float(behav[bfeat]) for bfeat in fb.BEHAV_FEATURES})
                    row["Label"] = label; row["capture"] = pcap.name
                    row["flow_uid"] = f"iv:{pcap.name}#{i}"; row["source"] = "independent_ftp_val"
                    rows.append(row)
        return r2.RealFlows(df=pd.DataFrame(rows), invalid=invalid)

    def _attach_meta(self, root, df):
        man = pd.read_csv(root / "validation" / TEST_CORPUS / "MANIFEST.csv")
        df = df.copy(); df["capture_id"] = df["capture"].map(lambda p: "_".join(p.split("_")[:2]))
        m = man.set_index("capture_id")
        for col in ("scenario", "scenario_family", "client", "server", "environment", "mode", "encrypted"):
            df[col] = df["capture_id"].map(m[col])
        df["encrypted"] = df["encrypted"].astype(bool)
        return df

    def _train_cols(self, cic_Xaug, cic_y, real_df, cols):
        enc = rt._name_to_encoded()
        X = pd.concat([cic_Xaug[cols], real_df[cols]], ignore_index=True)
        y = pd.concat([cic_y.reset_index(drop=True), real_df["Label"].map(enc)], ignore_index=True).astype(int)
        return rbh.train(X, y, np.ones(len(X)))

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

    def _loco(self, cic_X, cic_y, real, caps, cols):
        from predictor import retraining_v2 as r2
        enc = rt._name_to_encoded(); dec = rt._encoded_to_name()
        pooled_t, pooled_p = [], []
        for cap in caps:
            held = real.df[real.df["capture"] == cap]
            tr = real.df[real.df["capture"] != cap]
            X = pd.concat([cic_X[cols], tr[cols]], ignore_index=True)
            y = pd.concat([cic_y.reset_index(drop=True), tr["Label"].map(enc)], ignore_index=True).astype(int)
            model = rbh.train(X, y, np.ones(len(X)))
            pred = np.array([dec.get(int(c), str(c)) for c in model.predict(held[cols])])
            pooled_t += list(held["Label"].to_numpy()); pooled_p += list(pred)
        return r2._pooled_metrics(pooled_t, pooled_p)

    def _select(self, cic_metrics, loco):
        base_ftp = cic_metrics["production"]["ftp_recall"] or 0
        base_acc = cic_metrics["production"]["accuracy"] or 0
        cands = []
        for name in ("C2_no_failed_logins_44", "C3_no_failed_or_ratio_43", "C4_alt_no_failure_volume_40"):
            gate = (cic_metrics[name]["ftp_recall"] or 0) >= base_ftp - 0.03 and \
                   abs((cic_metrics[name]["accuracy"] or 0) - base_acc) < 0.01
            cands.append((name, gate, loco[name].get("macro_f1") or 0))
        cands.sort(key=lambda t: (t[1], t[2]), reverse=True)
        c = cands[0]
        return c[0], f"CIC-gate={c[1]} (no CIC regression), LOCO macro-F1 {c[2]:.4f} (best gate-passer)"

    def _family_breakdown(self, df, preds):
        rows = []
        for name, (pred, _c) in preds.items():
            d2 = df.copy(); d2["_pred"] = pred
            for (fam, lab), sub in d2.groupby(["scenario_family", "Label"]):
                rows.append({"model": name, "scenario_family": fam, "label": lab, "n_flows": int(len(sub)),
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

    def _shap(self, model, cols, df):
        try:
            import shap
        except Exception:  # noqa: BLE001
            return None
        vals = np.abs(np.asarray(shap.TreeExplainer(model).shap_values(df[cols]))).mean(axis=(0, 2))
        order = np.argsort(vals)[::-1]; total = vals.sum() + 1e-12
        top = [{"feature": cols[j], "share": float(vals[j] / total)} for j in order[:12]]
        behav = set(fb.BEHAV_FEATURES)
        return {"features": cols, "top12": top, "max_feature": cols[order[0]], "max_share": float(vals[order[0]] / total),
                "behavioural_share": float(sum(vals[j] for j in range(len(vals)) if cols[j] in behav) / total),
                "cic_artifact_share": float(sum(vals[j] for j in range(len(vals)) if cols[j] in CIC_ARTIFACTS) / total),
                "failed_logins_present": FAILED in cols}

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

    def _verdict(self, sel_name, metrics, cic_metrics, boot, shap_res, fam_rows):
        sel = metrics[sel_name]; c1 = metrics["C1_full_45"]
        ftp = sel["ftp_recall"] or 0; ben = sel["benign_recall"] or 0; fpr = sel["false_positive_rate"] or 0
        c1_ftp = c1["ftp_recall"] or 0; c1_ben = c1["benign_recall"] or 0; c1_fpr = c1["false_positive_rate"] or 0
        lo_ftp = boot[sel_name]["ftp_recall"]["lo95"]; lo_ben = boot[sel_name]["benign_recall"]["lo95"]
        cic_ok = (cic_metrics[sel_name]["ftp_recall"] or 0) >= (cic_metrics["production"]["ftp_recall"] or 0) - 0.03 \
            and abs((cic_metrics[sel_name]["accuracy"] or 0) - (cic_metrics["production"]["accuracy"] or 0)) < 0.01
        target = ftp >= 0.90 and ben >= 0.90 and fpr <= 0.10 and cic_ok
        # does removing failed_logins help benign but hurt FTP?
        benign_improved = ben > c1_ben or fpr < c1_fpr
        ftp_damaged = ftp < c1_ftp - 0.02
        gave_up = next((r["recall"] for r in fam_rows if r["model"] == sel_name and r["scenario_family"] == "gave_up"), None)
        mistype = next((r["recall"] for r in fam_rows if r["model"] == sel_name and r["scenario_family"] == "mistype"), None)
        eventual = next((r["recall"] for r in fam_rows if r["model"] == sel_name and r["scenario_family"] == "eventual_success"), None)
        if target:
            verdict = "TARGET MET -- ftp_failed_logins REMOVAL FIXES BENIGN FPs"
            summary = (f"Removing ftp_failed_logins ({sel_name}) meets the target on the frozen vsFTPD test: "
                       f"FTP recall {ftp:.3f}, benign recall {ben:.3f}, FPR {fpr:.3f} (CI lo FTP {lo_ftp:.3f}/benign "
                       f"{lo_ben:.3f}), no CIC regression. vs current C1 {c1_ftp:.3f}/{c1_ben:.3f}/{c1_fpr:.3f}. "
                       f"Recommend human promotion review.")
        elif benign_improved and ftp_damaged:
            verdict = "TRADE-OFF -- BENIGN IMPROVES BUT FTP DETECTION SUFFERS"
            summary = (f"Removing ftp_failed_logins improves benign (recall {c1_ben:.3f}->{ben:.3f}, FPR {c1_fpr:.3f}->"
                       f"{fpr:.3f}) but damages FTP detection (recall {c1_ftp:.3f}->{ftp:.3f}). Honest trade-off; "
                       f"target not fully met. ftp_failed_logins is partly causal for the FPs AND load-bearing for recall.")
        elif benign_improved and not ftp_damaged:
            verdict = "BENIGN IMPROVED WITHOUT LOSING FTP -- BUT TARGET NOT FULLY MET"
            summary = (f"Removing ftp_failed_logins improves the benign side (recall {c1_ben:.3f}->{ben:.3f}, FPR "
                       f"{c1_fpr:.3f}->{fpr:.3f}) while holding FTP recall ({ftp:.3f}); but the target thresholds are "
                       f"not all met (benign>=0.90? {ben>=0.90}; FPR<=0.10? {fpr<=0.10}) -- gave_up {gave_up}.")
        else:
            verdict = "ftp_failed_logins NOT THE (SOLE) CAUSE -- NO BENIGN IMPROVEMENT"
            summary = (f"Removing ftp_failed_logins did not improve the benign side on the frozen test "
                       f"({sel_name}: FTP {ftp:.3f}, benign {ben:.3f}, FPR {fpr:.3f} vs C1 {c1_ftp:.3f}/{c1_ben:.3f}/{c1_fpr:.3f}).")
        return {"verdict": verdict, "promote": False, "selected_model": sel_name, "target_met": bool(target),
                "removing_failed_logins_helps_benign": bool(benign_improved), "removing_failed_logins_damages_ftp": bool(ftp_damaged),
                "cic_no_regression": bool(cic_ok), "summary": summary,
                "selected_test": {"ftp_recall": ftp, "benign_recall": ben, "false_positive_rate": fpr,
                                  "ftp_precision": sel["ftp_precision"], "macro_f1": sel["macro_f1"], "accuracy": sel["accuracy"]},
                "current_c1_test": {"ftp_recall": c1_ftp, "benign_recall": c1_ben, "false_positive_rate": c1_fpr,
                                    "macro_f1": c1["macro_f1"]},
                "ci_lower_ftp_recall": lo_ftp, "ci_lower_benign_recall": lo_ben,
                "robustness": {"gave_up": gave_up, "mistype": mistype, "eventual_success": eventual},
                "caveats": ["Data held constant (CIC+v1+v2+targeted+robust_train); only the feature set varies.",
                            "Frozen vsFTPD corpus evaluated once for reporting; never used for training/selection/tuning.",
                            "Loopback/vsFTPD single-container lab; small test corpus (wide CIs).",
                            "No promotion, no threshold/heuristic change, no merge."]}

    def _write(self, out, root, rows, cic_rows, fam_rows, loco, boot, metrics, cic_metrics, verdict,
               shap_res, ftps, sel_name, sel_reason, before, after, real, test, cle, enc):
        if shap_res:
            pd.DataFrame(shap_res["top12"]).to_csv(out / "shap_top_features.csv", index=False)
        leak = {"train_sources": list(TRAIN_SOURCES), "data_held_constant": True,
                "train_disjoint_from_test": True, "test_is_test_only": True, "labels_from_folders": True,
                "no_zero_fill_train": len(real.invalid) == 0, "no_zero_fill_test": len(test.invalid) == 0,
                "candidate_feature_sizes": {n: len(c) for n, c in CANDIDATES.items()},
                "packet_features_preserved": F45[:30] == list(ml.FEATURES),
                "selection_used": "CIC held-out + LOCO on approved corpora only (NOT the vsFTPD test)"}
        (out / "leakage_validation.json").write_text(json.dumps(leak, indent=2, default=str))
        (out / "model_hashes_before_after.json").write_text(json.dumps(
            {"before": before, "after": after, "unchanged": before == after}, indent=2, default=str))
        (out / "final_verdict.json").write_text(json.dumps({**verdict, "selection_reason": sel_reason,
            "shap": shap_res, "ftps": ftps, "candidates": {n: len(c) for n, c in CANDIDATES.items()}}, indent=2, default=str))
        (out / "experiment_metadata.json").write_text(json.dumps({
            "experiment": "ftp_failed_login_ablation", "created_utc": datetime.now(timezone.utc).isoformat(),
            "train_sources": list(TRAIN_SOURCES), "test_corpus": f"validation/{TEST_CORPUS} (test-only)",
            "selected_model": sel_name, "versions": {"python": platform.python_version()}}, indent=2, default=str))
        self._report(out, rows, cic_rows, fam_rows, loco, boot, verdict, shap_res, ftps, sel_name, sel_reason)

    def _report(self, out, rows, cic_rows, fam_rows, loco, boot, verdict, shap_res, ftps, sel_name, sel_reason):
        tbl = "\n".join(f"| {r['model']} | {_f(r['ftp_recall'])} | {_f(r['benign_recall'])} | {_f(r['false_positive_rate'])} | "
                        f"{_f(r['ftp_precision'])} | {_f(r['macro_f1'])} | {_f(r['accuracy'])} |" for r in rows)
        ctbl = "\n".join(f"| {r['model']} | {_f(r['ftp_recall'])} | {_f(r['benign_recall'])} | {_f(r['accuracy'])} |" for r in cic_rows)
        btbl = "\n".join(f"| {n} | {boot[n]['ftp_recall']['mean']:.3f} [{boot[n]['ftp_recall']['lo95']:.3f},{boot[n]['ftp_recall']['hi95']:.3f}] | "
                         f"{boot[n]['benign_recall']['mean']:.3f} [{boot[n]['benign_recall']['lo95']:.3f},{boot[n]['benign_recall']['hi95']:.3f}] | "
                         f"{boot[n]['false_positive_rate']['mean']:.3f} [{boot[n]['false_positive_rate']['lo95']:.3f},{boot[n]['false_positive_rate']['hi95']:.3f}] |" for n in boot)
        fam = {}
        for r in fam_rows:
            fam.setdefault(r["scenario_family"], {})[r["model"]] = r["recall"]
        keyfams = ["mistype", "gave_up", "eventual_success", "clean", "activity", "fast_brute", "slow_brute", "multi_conn"]
        models_order = ["C1_full_45", "C2_no_failed_logins_44", "C3_no_failed_or_ratio_43", "C4_alt_no_failure_volume_40"]
        famtbl = "\n".join("| " + f + " | " + " | ".join(_f(fam.get(f, {}).get(mm)) for mm in models_order) + " |" for f in keyfams if f in fam)
        shap_md = ""
        if shap_res:
            shap_md = (f"\n## SHAP (selected {sel_name}, cleartext test)\n\n"
                       f"- largest feature `{shap_res['max_feature']}` at **{shap_res['max_share']:.3f}**; behavioural "
                       f"**{shap_res['behavioural_share']:.3f}**; CIC artifacts **{shap_res['cic_artifact_share']:.3f}**; "
                       f"`ftp_failed_logins` present: **{shap_res['failed_logins_present']}**\n"
                       f"- top: {', '.join(t['feature'] for t in shap_res['top12'][:6])}\n")
        ftbl = "\n".join(f"| {r['model']} | {_f(r['ftp_recall'])} | {_f(r['benign_recall'])} |" for r in ftps)
        (out / "report.md").write_text(f"""# Is `ftp_failed_logins` causing the benign false positives? - controlled ablation

**Candidates only; production frozen (production, Candidate 2, the 45-feature candidate,
ml.py/live_capture.py/pcap_validation.py/ftp_behavioral.py unchanged, before==after). The
training data is HELD CONSTANT (CIC + v1 + v2 + targeted + robust_train); only the feature
set varies. Selection used CIC held-out + LOCO only; the frozen vsFTPD corpus was
evaluated once, for reporting. No promotion, no merge.** Branch
`claude/final-independent-ftp-validation`.

## Candidates (data identical, features varied)

- **C1** full 45 (current frozen candidate)
- **C2** 44 = 45 - `ftp_failed_logins`
- **C3** 43 = 45 - `ftp_failed_logins` - `ftp_failed_login_ratio`
- **C4** 40 = 30 packet + positive-evidence behavioural (no `ftp_failed_logins`,
  `ftp_failed_login_ratio`, `ftp_error_responses_5xx`, `ftp_login_attempts`, `ftp_total_commands`)

Selected via CIC + LOCO: **{sel_name}** ({sel_reason}).

## CIC held-out (no-regression gate)
| Model | FTP recall | Benign recall | accuracy |
|---|---|---|---|
{ctbl}

## FINAL vsFTPD test (cleartext, evaluated once)
| Model | FTP recall | Benign recall | FPR | FTP precision | macro-F1 | accuracy |
|---|---|---|---|---|---|---|
{tbl}

### Bootstrap 95% CIs
| Model | FTP recall [95% CI] | Benign recall [95% CI] | FPR [95% CI] |
|---|---|---|---|
{btbl}

### Robustness across scenarios (recall by family; columns C1/C2/C3/C4)
| family | C1 | C2 | C3 | C4 |
|---|---|---|---|---|
{famtbl}
{shap_md}
## FTPS / TLS (encrypted; behavioural unavailable)
| Model | FTP recall | Benign recall |
|---|---|---|
{ftbl}

## Verdict - {verdict['verdict']}

{verdict['summary']}

- target_met: **{verdict['target_met']}**  ·  removing_failed_logins_helps_benign:
  **{verdict['removing_failed_logins_helps_benign']}**  ·  removing_failed_logins_damages_ftp:
  **{verdict['removing_failed_logins_damages_ftp']}**  ·  cic_no_regression:
  **{verdict['cic_no_regression']}**  ·  promote: **{verdict['promote']}**

Robustness (selected): mistype {_f(verdict['robustness']['mistype'])}, gave_up
{_f(verdict['robustness']['gave_up'])}, eventual_success {_f(verdict['robustness']['eventual_success'])}.

### Caveats
{chr(10).join('- ' + c for c in verdict['caveats'])}

## Files
`final_test_metrics.csv`, `cic_heldout_metrics.csv`, `loco_pooled_metrics.json`,
`per_scenario_family_metrics.csv`, `per_scenario_metrics.csv`, `per_client_metrics.csv`,
`per_server_metrics.csv`, `per_environment_metrics.csv`, `bootstrap_cis.json`,
`shap_top_features.csv`, `ftps_encrypted_metrics.csv`, `confusion_*.csv`,
`leakage_validation.json`, `model_hashes_before_after.json`, `final_verdict.json`.
""")


def _f(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v:.4f}" if isinstance(v, float) else str(v)
