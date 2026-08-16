"""
Behavioural-model robustness stress-test + feature ablation (frozen models).

Evaluates the frozen production model, Candidate 2, and the existing 45-feature
behavioural candidate on the NEW MESSY robustness corpus, and runs a 4-way ablation
(30 packet-flow only / behavioural only / 30+behavioural / 30+behavioural WITHOUT
ftp_failed_logins) to determine whether ftp_failed_logins is acting as a label
shortcut. Ablation models are trained on the APPROVED corpora (v1+v2+targeted) only;
the messy corpus is TEST-ONLY (never training/selection/tuning). Verifies all frozen
artifacts unchanged. Promotes nothing.

    python manage.py ftp_behavioral_robustness
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


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _frozen(root):
    d = {"production": _sha(ie.production_model_path()), "candidate2": _sha(ie.candidate2_file()),
         "aug_unweighted": _sha(root / "validation/models/ftp-behavioral-candidate/candidate_aug_unweighted.pkl"),
         "ml_py": _sha(root / "webapp_django/predictor/ml.py"),
         "live_capture_py": _sha(root / "webapp_django/predictor/live_capture.py"),
         "pcap_validation_py": _sha(root / "webapp_django/predictor/pcap_validation.py")}
    for name, sub in (("v1", "realistic_pcaps"), ("v2", "realistic_pcaps_v2"),
                      ("indep", "independent_real_pcaps"), ("targeted", "targeted_benign_pcaps")):
        d[name + "_pcaps"] = sorted(_sha(p) for p in glob.glob(str(root / "validation" / sub / "**/*.pcap"), recursive=True))
    return d


def _predict(model, X):
    dec = rt._encoded_to_name()
    return np.array([dec.get(int(c), str(c)) for c in model.predict(X)]), model.predict_proba(X).max(axis=1)


def _metrics(truth, pred, conf=None):
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
    truth = np.array([str(t) for t in truth]); pred = np.array([str(p) for p in pred])
    labels = sorted(set(truth) | set(pred))
    rep = classification_report(truth, pred, labels=labels, output_dict=True, zero_division=0)
    bmask = truth == BENIGN
    return {"n": int(len(truth)), "accuracy": float(accuracy_score(truth, pred)),
            "macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)),
            "ftp_recall": rep.get(FTP, {}).get("recall"), "ftp_precision": rep.get(FTP, {}).get("precision"),
            "benign_recall": rep.get(BENIGN, {}).get("recall"),
            "benign_fp_rate": float((pred[bmask] != BENIGN).mean()) if bmask.any() else None,
            "confusion": {"labels": labels, "matrix": confusion_matrix(truth, pred, labels=labels).tolist()}}


class Command(BaseCommand):
    help = "Behavioural-model robustness stress-test + ablation on the messy corpus (frozen models)."

    def add_arguments(self, parser):
        parser.add_argument("--output", default=None)
        parser.add_argument("--skip-shap", action="store_true")

    def handle(self, *args, **opts):
        w = self.stdout.write
        root = rbh.repo_root()
        out = Path(opts["output"]) if opts["output"] else root / "validation" / "results" / "ftp_behavioral_robustness"
        out.mkdir(parents=True, exist_ok=True)
        before = _frozen(root)

        w(self.style.MIGRATE_HEADING("Behavioural robustness stress-test + ablation (frozen models)"))
        # leak guard: robustness corpus disjoint from all training corpora
        rob_h = {_sha(p) for p in glob.glob(str(root / "validation/robustness_pcaps/**/*.pcap"), recursive=True)}
        train_h = set(before["v1_pcaps"]) | set(before["v2_pcaps"]) | set(before["targeted_pcaps"]) | set(before["indep_pcaps"])
        if not rob_h.isdisjoint(train_h):
            raise SystemExit("ABORT: a robustness PCAP overlaps a prior corpus!")

        # ---- messy test corpus (45 features) ----------------------------
        rob = rbh.extract_real_augmented(("robustness",))
        rdf = self._attach_meta(root, rob.df)
        w(f"  messy corpus flows: {len(rdf)} (FTP {int((rdf['Label']==FTP).sum())}, "
          f"Benign {int((rdf['Label']==BENIGN).sum())}); invalid: {len(rob.invalid)}")

        # ---- approved training data (v1+v2+targeted) --------------------
        real = rbh.extract_real_augmented(("v1", "v2", "targeted"))
        cic_Xaug, cic_y = rbh.load_cic_augmented()

        # ---- models: frozen + ablation ----------------------------------
        import joblib
        prod = ml._load(ml.DEFAULT_MODEL)[0]
        cand2 = joblib.load(ie.candidate2_file())
        aug_full = joblib.load(root / "validation/models/ftp-behavioral-candidate/candidate_aug_unweighted.pkl")

        w(self.style.MIGRATE_HEADING("\nTraining ablation models (approved corpora only)"))
        abl_30 = self._train_subset(cic_Xaug, cic_y, real.df, list(ml.FEATURES), include_cic=True)
        abl_nofail = self._train_subset(cic_Xaug, cic_y, real.df, FEAT_NO_FAILED, include_cic=True)
        abl_behav = self._train_subset(cic_Xaug, cic_y, real.df, list(fb.BEHAV_FEATURES), include_cic=False)
        for nm, m, cols in (("ablation_30only", abl_30, list(ml.FEATURES)),
                            ("ablation_no_failed_logins", abl_nofail, FEAT_NO_FAILED),
                            ("ablation_behavioural_only", abl_behav, list(fb.BEHAV_FEATURES))):
            joblib.dump(m, self._mkmodeldir(root) / f"{nm}.pkl")
        w("  ablation models trained + saved (30only / behavioural-only / no-failed-logins).")

        # ---- evaluate everything on the messy corpus --------------------
        w(self.style.MIGRATE_HEADING("\nEvaluation on messy robustness corpus"))
        model_specs = {
            "production": (prod, list(ml.FEATURES)), "candidate2": (cand2, list(ml.FEATURES)),
            "behav_full_30+15": (aug_full, rbh.FEATURES_AUG),
            "ablation_30only": (abl_30, list(ml.FEATURES)),
            "ablation_behavioural_only": (abl_behav, list(fb.BEHAV_FEATURES)),
            "ablation_no_failed_logins": (abl_nofail, FEAT_NO_FAILED)}
        metrics, preds = {}, {}
        rows = []
        for name, (model, cols) in model_specs.items():
            pred, conf = _predict(model, rdf[cols]); preds[name] = (pred, conf)
            m = _metrics(rdf["Label"].to_numpy(), pred, conf); metrics[name] = m
            rows.append({"model": name, "ftp_recall": m["ftp_recall"], "benign_recall": m["benign_recall"],
                         "benign_fp_rate": m["benign_fp_rate"], "macro_f1": m["macro_f1"], "accuracy": m["accuracy"]})
            w(f"  {name:26} FTP-rec {_f(m['ftp_recall'])} Ben-rec {_f(m['benign_recall'])} "
              f"BenFP {_f(m['benign_fp_rate'])} mF1 {m['macro_f1']:.4f} acc {m['accuracy']:.4f}")
        pd.DataFrame(rows).to_csv(out / "ablation_and_model_metrics.csv", index=False)

        # ---- per-scenario-family behaviour (the messy cases) ------------
        fam_rows = self._family_breakdown(rdf, preds)
        pd.DataFrame(fam_rows).to_csv(out / "per_scenario_family_metrics.csv", index=False)
        w(self.style.MIGRATE_HEADING("\nBehaviour on the adversarial families (behav_full model)"))
        for r in fam_rows:
            if r["model"] == "behav_full_30+15":
                w(f"  {r['scenario_family']:18} ({r['label']:14}) n={r['n_flows']:3} recall={_f(r['recall'])} "
                  f"predicted_ftp={r['predicted_ftp']} predicted_benign={r['predicted_benign']}")

        # ---- per-scenario / per-client / per-server for behav_full ------
        self._breakdowns(out, rdf, preds["behav_full_30+15"][0])

        # ---- confidence + confusion for key models ----------------------
        self._confidence_confusion(out, rdf, preds, metrics)

        # ---- SHAP: does behav_full use ONE feature or several? ----------
        shap_res = None if opts["skip_shap"] else self._shap(aug_full, cic_Xaug, rdf)

        # ---- specific measurements --------------------------------------
        specifics = self._specifics(rdf, preds, metrics, shap_res)

        # ---- verdict -----------------------------------------------------
        verdict = self._verdict(metrics, specifics, shap_res)
        w(self.style.MIGRATE_HEADING("\nVERDICT"))
        w(self.style.WARNING("  " + verdict["verdict"] + " -- " + verdict["summary"]))

        after = _frozen(root)
        if before != after:
            raise SystemExit("ABORT: a frozen artifact changed!")
        w(f"\n  frozen artifacts unchanged (before==after): {before == after}")

        self._write(out, rows, fam_rows, metrics, specifics, verdict, shap_res, before, after, rob, real, opts)
        w(self.style.SUCCESS(f"\nEvidence written to {out}"))
        w(self.style.WARNING("\nNo promotion, no production change, no merge. Messy corpus is test-only."))

    # -- helpers -----------------------------------------------------------

    def _mkmodeldir(self, root):
        d = root / "validation" / "models" / "ftp-behavioral-robustness"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _attach_meta(self, root, df):
        man = pd.read_csv(root / "validation" / "robustness_pcaps" / "MANIFEST.csv")
        man["cap2"] = man["capture_id"]
        # map pcap filename -> capture_id (benign_01_..)
        df = df.copy()
        df["capture_id"] = df["capture"].map(lambda p: "_".join(p.split("_")[:2]))
        m = man.set_index("capture_id")
        for col in ("scenario", "scenario_family", "client", "server", "mode", "environment"):
            df[col] = df["capture_id"].map(m[col])
        return df

    def _train_subset(self, cic_Xaug, cic_y, real_df, cols, include_cic):
        enc = rt._name_to_encoded()
        if include_cic:
            X = pd.concat([cic_Xaug[cols], real_df[cols]], ignore_index=True)
            y = pd.concat([cic_y.reset_index(drop=True), real_df["Label"].map(enc)], ignore_index=True).astype(int)
        else:
            X = real_df[cols].reset_index(drop=True)
            y = real_df["Label"].map(enc).astype(int).reset_index(drop=True)
        return rbh.train(X, y, np.ones(len(X)))

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

    def _shap(self, model, cic_Xaug, rdf):
        try:
            import shap
        except Exception:  # noqa: BLE001
            return None
        data = rdf[rbh.FEATURES_AUG]
        vals = np.abs(np.asarray(shap.TreeExplainer(model).shap_values(data))).mean(axis=(0, 2))
        order = np.argsort(vals)[::-1]
        total = vals.sum() + 1e-12
        top = [{"feature": rbh.FEATURES_AUG[j], "mean_abs_shap": float(vals[j]),
                "share": float(vals[j] / total)} for j in order[:10]]
        behav = set(fb.BEHAV_FEATURES)
        behav_share = float(sum(vals[j] for j in range(len(vals)) if rbh.FEATURES_AUG[j] in behav) / total)
        failed_share = float(vals[rbh.FEATURES_AUG.index(FAILED)] / total)
        n_behav_top5 = sum(1 for t in top[:5] if t["feature"] in behav)
        return {"top10": top, "behavioural_share": behav_share, "failed_logins_share": failed_share,
                "n_behavioural_in_top5": n_behav_top5,
                "dominated_by_failed_logins": failed_share > 0.5}

    def _specifics(self, rdf, preds, metrics, shap_res):
        pred_full = preds["behav_full_30+15"][0]
        rdf2 = rdf.copy(); rdf2["_pred"] = pred_full

        def fam_recall(fam, lab):
            sub = rdf2[(rdf2["scenario_family"] == fam) & (rdf2["Label"] == lab)]
            return (round(float((sub["_pred"] == lab).mean()), 4), int(len(sub))) if len(sub) else (None, 0)

        gave_up_rec, gave_up_n = fam_recall("gave_up", BENIGN)          # benign that never authenticates
        mistype_rec, mistype_n = fam_recall("mistype", BENIGN)          # benign that fails then succeeds
        eventual_rec, eventual_n = fam_recall("eventual_success", FTP)  # attack that guesses right
        incomplete_rec, incomplete_n = fam_recall("incomplete", BENIGN)  # no auth evidence
        return {
            "benign_mistake_recall": {"gave_up": [gave_up_rec, gave_up_n], "mistype": [mistype_rec, mistype_n]},
            "attacker_eventual_success_recall": [eventual_rec, eventual_n],
            "incomplete_control_evidence_recall": [incomplete_rec, incomplete_n],
            "failed_login_feature_share": (shap_res["failed_logins_share"] if shap_res else None),
            "successful_login_feature_present": True,
            "no_failed_logins_ablation_ftp_recall": metrics["ablation_no_failed_logins"]["ftp_recall"],
            "full_ftp_recall": metrics["behav_full_30+15"]["ftp_recall"],
            "no_failed_logins_ablation_benign_fp": metrics["ablation_no_failed_logins"]["benign_fp_rate"],
            "full_benign_fp": metrics["behav_full_30+15"]["benign_fp_rate"]}

    def _verdict(self, metrics, sp, shap_res):
        full = metrics["behav_full_30+15"]
        ftp = full["ftp_recall"] or 0; ben = full["benign_recall"] or 0; fp = full["benign_fp_rate"] or 0
        gave_up = sp["benign_mistake_recall"]["gave_up"][0]
        eventual = sp["attacker_eventual_success_recall"][0]
        failed_share = sp["failed_login_feature_share"] or 0
        # shortcut signature: benign that fails (gave_up) is flagged as attack AND the model leans
        # heavily on ftp_failed_logins AND removing it drops FTP recall a lot.
        no_fail_ftp = metrics["ablation_no_failed_logins"]["ftp_recall"] or 0
        drop_without_failed = ftp - no_fail_ftp
        shortcut = ((gave_up is not None and gave_up < 0.5)
                    and (failed_share > 0.30 or drop_without_failed > 0.30))
        robust = ftp > 0.70 and ben > 0.90 and (gave_up is None or gave_up > 0.8) and (eventual is None or eventual > 0.7)
        if robust:
            verdict = "PROMISING AND ROBUST"
            summary = (f"On messy traffic the behavioural model keeps FTP recall {ftp:.3f}, benign recall {ben:.3f}, "
                       f"handles benign-that-fails (gave_up recall {gave_up}) and attacker-eventual-success "
                       f"({eventual}). Still validate on broader real traffic.")
        elif shortcut:
            verdict = "BEHAVIORAL FEATURE IS A LABEL SHORTCUT"
            summary = (f"Benign sessions that merely fail to log in (gave_up recall {gave_up}) are flagged as "
                       f"attacks, ftp_failed_logins carries {failed_share:.2f} of importance, and removing it drops "
                       f"FTP recall by {drop_without_failed:+.3f}. The 'break' was largely ftp_failed_logins acting "
                       f"as a proxy for the label; it does not survive messy authentication.")
        else:
            verdict = "PROMISING BUT NEEDS MORE DATA"
            summary = (f"Behavioural model is better than the 30-feature models but not robust on messy cases "
                       f"(FTP recall {ftp:.3f}, benign recall {ben:.3f}, benign FP {fp:.3f}; gave_up recall {gave_up}, "
                       f"attacker-eventual-success recall {eventual}). Needs broader/messier real traffic.")
        return {"verdict": verdict, "promote": False, "summary": summary,
                "full_model": {"ftp_recall": ftp, "benign_recall": ben, "benign_fp_rate": fp, "macro_f1": full["macro_f1"]},
                "benign_gave_up_recall": gave_up, "attacker_eventual_success_recall": eventual,
                "ftp_failed_logins_importance_share": failed_share,
                "ftp_recall_drop_without_failed_logins": drop_without_failed,
                "is_label_shortcut": bool(shortcut), "is_robust": bool(robust),
                "caveats": ["Messy corpus is small (loopback-only, two servers); results are diagnostic, not final.",
                            "The messy corpus was never used for training/selection/tuning.",
                            "No threshold change, no heuristic, no promotion."]}

    def _write(self, out, rows, fam_rows, metrics, sp, verdict, shap_res, before, after, rob, real, opts):
        pd.DataFrame(fam_rows).to_csv(out / "per_scenario_family_metrics.csv", index=False)
        if shap_res is not None:
            pd.DataFrame(shap_res["top10"]).to_csv(out / "shap_top_features.csv", index=False)
        leak = {"robustness_disjoint_from_all_prior": True, "test_only": True,
                "labels_from_folders": True, "no_zero_fill": len(rob.invalid) == 0,
                "n_features": len(rbh.FEATURES_AUG), "training_sources": ["v1", "v2", "targeted"],
                "messy_corpus_used_for": "evaluation/ablation only (never training/selection)"}
        (out / "leakage_validation.json").write_text(json.dumps(leak, indent=2, default=str))
        (out / "model_hashes_before_after.json").write_text(json.dumps({"before": before, "after": after, "unchanged": before == after}, indent=2, default=str))
        (out / "final_verdict.json").write_text(json.dumps({**verdict, "specifics": sp}, indent=2, default=str))
        (out / "training_metadata.json").write_text(json.dumps({
            "experiment": "ftp_behavioral_robustness", "created_utc": datetime.now(timezone.utc).isoformat(),
            "ablations": ["30 packet-flow only", "behavioural only", "30+behavioural (full)", "30+behavioural without ftp_failed_logins"],
            "messy_corpus": "validation/robustness_pcaps (test only)", "ablation_model_dir": "validation/models/ftp-behavioral-robustness",
            "versions": {"python": platform.python_version()}}, indent=2, default=str))
        self._report(out, rows, metrics, sp, verdict, shap_res)

    def _report(self, out, rows, metrics, sp, verdict, shap_res):
        shap_md = ""
        if shap_res:
            shap_md = (f"\n## SHAP on the messy corpus (does it use one feature or many?)\n\n"
                       f"- `ftp_failed_logins` importance share: **{shap_res['failed_logins_share']:.2f}**; "
                       f"all behavioural features: **{shap_res['behavioural_share']:.2f}**; "
                       f"behavioural features in top-5: **{shap_res['n_behavioural_in_top5']}**; "
                       f"dominated by ftp_failed_logins: **{shap_res['dominated_by_failed_logins']}**\n"
                       f"- Top features: {', '.join(t['feature'] for t in shap_res['top10'][:6])}\n")
        tbl = "\n".join(f"| {r['model']} | {_f(r['ftp_recall'])} | {_f(r['benign_recall'])} | {_f(r['benign_fp_rate'])} | {r['macro_f1']:.4f} | {r['accuracy']:.4f} |" for r in rows)
        gu = sp["benign_mistake_recall"]["gave_up"]; mt = sp["benign_mistake_recall"]["mistype"]
        ev = sp["attacker_eventual_success_recall"]; inc = sp["incomplete_control_evidence_recall"]
        (out / "report.md").write_text(f"""# Behavioural-model robustness stress-test + ablation - report

**Frozen models only (production, Candidate 2, and the 45-feature behavioural
candidate are byte-for-byte unchanged, verified). Ablation models are diagnostic,
trained on the approved corpora (v1+v2+targeted) only. The messy corpus is TEST-ONLY
(never training/selection/tuning). No promotion, no threshold/heuristic change, no
merge.**

## Why this test

The earlier 45-feature model scored 1.000/1.000 on a CLEAN corpus (benign always
authenticated, brute force always failed) -- so a single feature `ftp_failed_logins`
could separate the classes. This corpus deliberately contains the messy cases where
that proxy breaks: benign users who mistype then succeed, benign who fail then give
up (look like brute force), and attackers who eventually guess a correct password
(look partially benign).

## Evaluation + ablation on the messy corpus

| Model | FTP recall | Benign recall | Benign FP | macro-F1 | accuracy |
|---|---|---|---|---|---|
{tbl}

## Behaviour on the adversarial families (full behavioural model)

- **Benign, fail-then-give-up** (looks like brute force): recall **{_f(gu[0])}** over {gu[1]} flows.
- **Benign, mistype-then-success**: recall **{_f(mt[0])}** over {mt[1]} flows.
- **Attacker, eventual success** (guesses a valid password): recall **{_f(ev[0])}** over {ev[1]} flows.
- **Benign, incomplete control evidence** (no auth outcome): recall **{_f(inc[0])}** over {inc[1]} flows.

## Ablation - is `ftp_failed_logins` a shortcut?

- Full (30+behavioural) FTP recall: **{_f(sp['full_ftp_recall'])}**; WITHOUT `ftp_failed_logins`:
  **{_f(sp['no_failed_logins_ablation_ftp_recall'])}** (drop {_f(verdict['ftp_recall_drop_without_failed_logins'])}).
- Benign FP full: {_f(sp['full_benign_fp'])}; without `ftp_failed_logins`: {_f(sp['no_failed_logins_ablation_benign_fp'])}.
{shap_md}
## Verdict - {verdict['verdict']}

{verdict['summary']}

- is_label_shortcut: **{verdict['is_label_shortcut']}**  ·  is_robust: **{verdict['is_robust']}**  ·  promote: **{verdict['promote']}**

### Caveats
{chr(10).join('- ' + c for c in verdict['caveats'])}

## Integrity

Production model, Candidate 2, the 45-feature candidate, ml.py/live_capture.py/
pcap_validation.py, and the v1/v2/independent/targeted PCAPs verified unchanged
(before==after). The messy corpus is hash-disjoint from all prior corpora and was
used for evaluation only. Ablation models stored under
validation/models/ftp-behavioral-robustness/.

## Files

`ablation_and_model_metrics.csv`, `per_scenario_family_metrics.csv`,
`per_scenario_metrics.csv`, `per_client_metrics.csv`, `per_server_metrics.csv`,
`confidence_distribution.csv`, `confusion_*.csv`, `shap_top_features.csv`,
`leakage_validation.json`, `model_hashes_before_after.json`, `training_metadata.json`,
`final_verdict.json`.
""")


def _f(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v:.4f}" if isinstance(v, float) else str(v)
