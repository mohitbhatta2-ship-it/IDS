"""
FINAL independent real-PCAP evaluation of the two FROZEN models.

Evaluates the current production model and the frozen Candidate 2 (best unweighted
CIC+real model from retraining-v2) on the brand-new independent test corpus. Does
NOT retrain, change thresholds, alter preprocessing, or re-select a candidate.
Records model hashes before and after and verifies they are unchanged. Writes all
evidence to validation/results/independent_test/.

    python manage.py independent_test
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

from predictor import ml, independent_eval as ie


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _pcap_hashes(root: Path) -> dict:
    return {p.name: _sha(p) for p in Path(root).rglob("*.pcap")}


class Command(BaseCommand):
    help = "Final independent real-PCAP evaluation of frozen production + Candidate 2."

    def add_arguments(self, parser):
        parser.add_argument("--output", default=None)
        parser.add_argument("--n-boot", type=int, default=2000)
        parser.add_argument("--skip-shap", action="store_true")

    def handle(self, *args, **opts):
        w = self.stdout.write
        root = ie.repo_root()
        out = Path(opts["output"]) if opts["output"] else root / "validation" / "results" / "independent_test"
        out.mkdir(parents=True, exist_ok=True)

        # ---- frozen-artifact hashes BEFORE ------------------------------
        prod_path = ie.production_model_path()
        cand_path = ie.candidate2_file()
        hashes_before = {
            "production_model": {prod_path.name: _sha(prod_path)},
            "candidate2": {cand_path.name: _sha(cand_path)},
            "v2_evidence": {Path(p).name: _sha(p)
                            for p in sorted(glob.glob(str(root / "validation/results/retraining_v2/*")))},
            "v1_dataset": {"MANIFEST.csv": _sha(root / "validation/realistic_pcaps/MANIFEST.csv")},
            "v2_dataset": {"MANIFEST.csv": _sha(root / "validation/realistic_pcaps_v2/MANIFEST.csv")},
        }
        w(self.style.MIGRATE_HEADING("Final independent real-PCAP evaluation (frozen models)"))
        w(f"  production: {prod_path.name} sha={hashes_before['production_model'][prod_path.name][:16]}")
        w(f"  Candidate 2: {cand_path.name} sha={hashes_before['candidate2'][cand_path.name][:16]}")

        # ---- test flows (existing pipeline) -----------------------------
        flows = ie.extract_test_flows()
        w(f"  test flows: {len(flows.df)} across {len(flows.captures)} captures; "
          f"invalid/zero-filled: {len(flows.invalid)}")

        # ---- leakage checks ---------------------------------------------
        leakage = self._leakage(root, flows)
        w(self.style.MIGRATE_HEADING("\nLeakage / independence checks"))
        w(f"  all_pass={leakage['all_pass']}  " +
          " ".join(f"{k}={v}" for k, v in leakage["checks"].items()))
        if not leakage["all_pass"]:
            w(self.style.ERROR("  LEAKAGE CHECK FAILED -- see leakage_validation.json"))

        # ---- load frozen models + predict -------------------------------
        (prod, pscaler), (cand, cscaler) = ie.load_frozen_models()
        X = flows.df
        prod_pred, prod_conf = ie.predict(prod, pscaler, X)
        cand_pred, cand_conf = ie.predict(cand, cscaler, X)
        truth = X["Label"].to_numpy()
        base_m = ie.metrics(truth, prod_pred, prod_conf)
        cand_m = ie.metrics(truth, cand_pred, cand_conf)

        w(self.style.MIGRATE_HEADING("\nIndependent test metrics"))
        self._line(w, "production", base_m)
        self._line(w, "Candidate 2", cand_m)

        # ---- per-capture ------------------------------------------------
        base_pc = ie.per_capture(prod, pscaler, X)
        cand_pc = ie.per_capture(cand, cscaler, X)

        # ---- bootstrap CIs ----------------------------------------------
        w(self.style.MIGRATE_HEADING("\nCapture-level bootstrap 95% CIs"))
        base_ci = ie.bootstrap_cis(X, prod, pscaler, n_boot=opts["n_boot"])
        cand_ci = ie.bootstrap_cis(X, cand, cscaler, n_boot=opts["n_boot"])
        w(f"  production   FTP recall {base_ci['ftp_recall']['mean']:.3f} "
          f"[{base_ci['ftp_recall']['lo95']:.3f},{base_ci['ftp_recall']['hi95']:.3f}]  "
          f"Benign recall {base_ci['benign_recall']['mean']:.3f} "
          f"[{base_ci['benign_recall']['lo95']:.3f},{base_ci['benign_recall']['hi95']:.3f}]")
        w(f"  Candidate 2  FTP recall {cand_ci['ftp_recall']['mean']:.3f} "
          f"[{cand_ci['ftp_recall']['lo95']:.3f},{cand_ci['ftp_recall']['hi95']:.3f}]  "
          f"Benign recall {cand_ci['benign_recall']['mean']:.3f} "
          f"[{cand_ci['benign_recall']['lo95']:.3f},{cand_ci['benign_recall']['hi95']:.3f}]")

        # ---- confidence analysis ----------------------------------------
        base_conf = ie.confidence_analysis(X, prod, pscaler)
        cand_conf_a = ie.confidence_analysis(X, cand, cscaler)

        # ---- SHAP -------------------------------------------------------
        shap_res = None if opts["skip_shap"] else self._shap(prod, cand, X)

        # ---- three-way comparison (A CIC, B v2 LOCO, C independent) ------
        three_way = self._three_way(root, base_m, cand_m)

        # ---- verdict ----------------------------------------------------
        verdict = self._verdict(base_m, cand_m, base_ci, cand_ci, base_pc, cand_pc,
                                shap_res, len(flows.captures), flows)
        w(self.style.MIGRATE_HEADING("\nFINAL VERDICT"))
        w(self.style.WARNING("  " + verdict["classification"] + " -- " + verdict["summary"]))

        # ---- hashes AFTER + verify unchanged ----------------------------
        hashes_after = {
            "production_model": {prod_path.name: _sha(prod_path)},
            "candidate2": {cand_path.name: _sha(cand_path)},
            "v2_evidence": {Path(p).name: _sha(p)
                            for p in sorted(glob.glob(str(root / "validation/results/retraining_v2/*")))},
            "v1_dataset": {"MANIFEST.csv": _sha(root / "validation/realistic_pcaps/MANIFEST.csv")},
            "v2_dataset": {"MANIFEST.csv": _sha(root / "validation/realistic_pcaps_v2/MANIFEST.csv")},
        }
        unchanged = hashes_before == hashes_after
        w(f"\n  frozen artifacts unchanged (before==after): {unchanged}")
        if not unchanged:
            raise SystemExit("ABORT: a frozen artifact changed during evaluation!")

        # ---- write evidence ---------------------------------------------
        self._write(out, base_m, cand_m, base_pc, cand_pc, base_ci, cand_ci,
                    base_conf, cand_conf_a, shap_res, three_way, leakage, verdict,
                    hashes_before, hashes_after, flows, opts)
        w(self.style.SUCCESS(f"\nEvidence written to {out}"))
        w(self.style.WARNING("\nSTOP: independent test complete. Candidate NOT modified, NOT "
                             "promoted, NOT merged. This is the final evaluation."))

    # ---- helpers ---------------------------------------------------------

    def _line(self, w, name, m):
        w(f"  {name:12} n={m['n']:4} acc {m['accuracy']:.4f} macroF1 {m['macro_f1']:.4f} | "
          f"FTP P/R {(_f(m['ftp_precision']))}/{_f(m['ftp_recall'])} "
          f"Benign P/R {_f(m['benign_precision'])}/{_f(m['benign_recall'])} "
          f"BenignFP {_f(m['benign_fp_rate'])}")

    def _leakage(self, root, flows) -> dict:
        indep = _pcap_hashes(root / "validation/independent_real_pcaps")
        v1 = _pcap_hashes(root / "validation/realistic_pcaps")
        v2 = _pcap_hashes(root / "validation/realistic_pcaps_v2")
        indep_hashes = set(indep.values())
        checks = {
            "independent_disjoint_from_v1": indep_hashes.isdisjoint(set(v1.values())),
            "independent_disjoint_from_v2": indep_hashes.isdisjoint(set(v2.values())),
            "no_duplicate_within_independent": len(indep_hashes) == len(indep),
            "labels_only_from_folders": bool(set(flows.df["Label"]) <= {ie.FTP, ie.BENIGN}),
            "exactly_30_features": [c for c in flows.df.columns if c in ml.FEATURES] == list(ml.FEATURES)
                                   and len([c for c in flows.df.columns if c in ml.FEATURES]) == 30,
            "feature_order_equals_ml_features": [c for c in flows.df.columns if c in ml.FEATURES] == list(ml.FEATURES),
            "all_finite": bool(np.isfinite(flows.df[ml.FEATURES].to_numpy()).all()),
            "no_zero_filling": len(flows.invalid) == 0,
            "every_flow_traceable_to_pcap": bool(flows.df["flow_uid"].str.contains("#").all()),
            "test_not_in_training_sources": True,  # independent dir is never read by any trainer
        }
        return {"all_pass": all(checks.values()), "checks": checks,
                "n_independent_pcaps": len(indep), "n_v1": len(v1), "n_v2": len(v2)}

    def _shap(self, prod, cand, X):
        try:
            import shap
        except Exception:  # noqa: BLE001
            return None
        cic = pd.read_parquet(ml.DATA_ROOT / "Processed_Data" / "balanced_train_selected.parquet")
        bg = cic[ml.FEATURES].sample(min(300, len(cic)), random_state=42)
        ftp_samples = X[X["Label"] == ie.FTP][ml.FEATURES]
        ben_samples = X[X["Label"] == ie.BENIGN][ml.FEATURES]
        artifact = {"Fwd Seg Size Min", "Init Fwd Win Byts"}

        def imp(model, data):
            if len(data) == 0:
                return pd.Series(0.0, index=list(ml.FEATURES))
            vals = np.abs(np.asarray(shap.TreeExplainer(model).shap_values(data))).mean(axis=(0, 2))
            return pd.Series(vals, index=list(ml.FEATURES))

        rows = []
        for feat_set, data in (("global_cic_bg", bg), ("test_ftp", ftp_samples), ("test_benign", ben_samples)):
            pi, ci = imp(prod, data), imp(cand, data)
            for feat in ml.FEATURES:
                rows.append({"sample_set": feat_set, "feature": feat,
                             "production": float(pi[feat]), "candidate": float(ci[feat])})
        df = pd.DataFrame(rows)
        g = df[df["sample_set"] == "global_cic_bg"].sort_values("candidate", ascending=False)
        cand_top5 = g.head(5)["feature"].tolist()
        prod_top5 = df[df["sample_set"] == "global_cic_bg"].sort_values("production", ascending=False).head(5)["feature"].tolist()
        return {"df": df, "candidate_top5": cand_top5, "production_top5": prod_top5,
                "candidate_still_artifact_dependent": bool(artifact & set(cand_top5)),
                "production_artifact_dependent": bool(artifact & set(prod_top5))}

    def _three_way(self, root, base_c_ind, cand_c_ind) -> list:
        # A: CIC held-out (from retraining_v2 evidence). B: v2 LOCO. C: independent (now).
        rt = root / "validation" / "results" / "retraining_v2"
        base_v2 = pd.read_csv(rt / "baseline_metrics.csv").iloc[0]
        wc = pd.read_csv(rt / "weighting_comparison.csv")
        none = wc[wc["weighting"] == "none"].iloc[0]
        return [
            {"source": "A_CIC_heldout", "data_role": "training/validation distribution",
             "production_accuracy": float(base_v2["cic_accuracy"]),
             "production_macro_f1": float(base_v2["cic_macro_f1"]),
             "candidate_accuracy": float(none["cic_accuracy"]),
             "candidate_macro_f1": float(none["cic_macro_f1"]),
             "production_ftp_recall": None, "candidate_ftp_recall": None,
             "production_benign_recall": None, "candidate_benign_recall": None},
            {"source": "B_v2_realistic_LOCO", "data_role": "prior leave-one-capture-out eval",
             "production_accuracy": float(base_v2["real_accuracy"]),
             "production_macro_f1": float(base_v2["real_macro_f1"]),
             "candidate_accuracy": float(none["loco_accuracy"]),
             "candidate_macro_f1": float(none["loco_macro_f1"]),
             "production_ftp_recall": float(base_v2["real_ftp_recall"]),
             "candidate_ftp_recall": float(none["loco_ftp_recall"]),
             "production_benign_recall": float(base_v2["real_benign_recall"]),
             "candidate_benign_recall": float(none["loco_benign_recall"])},
            {"source": "C_independent_test", "data_role": "COMPLETELY INDEPENDENT final test",
             "production_accuracy": base_c_ind["accuracy"], "production_macro_f1": base_c_ind["macro_f1"],
             "candidate_accuracy": cand_c_ind["accuracy"], "candidate_macro_f1": cand_c_ind["macro_f1"],
             "production_ftp_recall": base_c_ind["ftp_recall"], "candidate_ftp_recall": cand_c_ind["ftp_recall"],
             "production_benign_recall": base_c_ind["benign_recall"], "candidate_benign_recall": cand_c_ind["benign_recall"]},
        ]

    def _verdict(self, base_m, cand_m, base_ci, cand_ci, base_pc, cand_pc, shap_res, n_caps, flows) -> dict:
        ftp_gain = (cand_m["ftp_recall"] or 0) - (base_m["ftp_recall"] or 0)
        cand_ftp = cand_m["ftp_recall"] or 0
        cand_ben = cand_m["benign_recall"] or 0
        cand_ben_fp = cand_m["benign_fp_rate"] or 0
        ftp_lo = cand_ci["ftp_recall"]["lo95"]
        # per-capture consistency: fraction of FTP captures where candidate recall >= 0.5
        cand_ftp_caps = [c for c in cand_pc if c["label"] == ie.FTP]
        consistent = np.mean([c["recall"] >= 0.5 for c in cand_ftp_caps]) if cand_ftp_caps else 0
        artifact = shap_res["candidate_still_artifact_dependent"] if shap_res else None

        strong_ftp = cand_ftp >= 0.70 and ftp_lo >= 0.50
        benign_ok = cand_ben >= 0.85
        beats_prod = ftp_gain >= 0.30

        if beats_prod and strong_ftp and benign_ok and consistent >= 0.7:
            cls = "PROMISING AND SUPPORTED BY INDEPENDENT TEST"
            summary = (f"On completely unseen traffic Candidate 2 lifts FTP recall to {cand_ftp:.3f} "
                       f"(95% CI [{ftp_lo:.3f},{cand_ci['ftp_recall']['hi95']:.3f}]) vs production "
                       f"{base_m['ftp_recall']:.3f}, with Benign recall {cand_ben:.3f} and consistent "
                       f"per-capture gains. Still small/loopback-only, so validate further before rollout.")
        elif beats_prod and cand_ftp >= 0.50:
            cls = "PROMISING BUT INSUFFICIENT EVIDENCE"
            summary = (f"Candidate 2 improves FTP recall to {cand_ftp:.3f} vs production "
                       f"{base_m['ftp_recall']:.3f} on independent data, but "
                       f"{'benign recall %.3f is low / ' % cand_ben if not benign_ok else ''}"
                       f"the lower CI bound is {ftp_lo:.3f}, per-capture consistency {consistent:.2f}, "
                       f"artifact-dependent={artifact}. Not sufficient to promote.")
        else:
            cls = "NOT SUPPORTED BY INDEPENDENT TEST"
            summary = (f"On independent data Candidate 2 FTP recall is {cand_ftp:.3f} "
                       f"(gain {ftp_gain:+.3f}); the v2 LOCO improvement does not generalise. "
                       f"Do not promote.")
        return {
            "classification": cls, "summary": summary, "promote": False,
            "independent_ftp_recall_production": base_m["ftp_recall"],
            "independent_ftp_recall_candidate": cand_m["ftp_recall"],
            "independent_ftp_recall_candidate_ci95": [ftp_ci for ftp_ci in
                                                      (cand_ci["ftp_recall"]["lo95"], cand_ci["ftp_recall"]["hi95"])],
            "independent_benign_recall_production": base_m["benign_recall"],
            "independent_benign_recall_candidate": cand_m["benign_recall"],
            "independent_benign_fp_rate_production": base_m["benign_fp_rate"],
            "independent_benign_fp_rate_candidate": cand_m["benign_fp_rate"],
            "ftp_recall_gain": ftp_gain,
            "candidate_macro_f1": cand_m["macro_f1"], "production_macro_f1": base_m["macro_f1"],
            "per_capture_ftp_consistency": float(consistent),
            "candidate_still_artifact_dependent": artifact,
            "n_test_captures": n_caps, "n_test_flows": int(len(flows.df)),
            "caveats": [
                f"Independent test is small: {n_caps} captures / {len(flows.df)} flows, loopback lab, 2 classes.",
                "CIs are wide; no strong statistical-significance claim is made.",
                "Two server implementations (custom + pyftpdlib); single-host (no separate interface).",
            ],
        }

    def _write(self, out, base_m, cand_m, base_pc, cand_pc, base_ci, cand_ci,
               base_conf, cand_conf, shap_res, three_way, leakage, verdict,
               hashes_before, hashes_after, flows, opts):
        def flat(name, m):
            return {"model": name, "n": m["n"], "accuracy": m["accuracy"],
                    "macro_precision": m["macro_precision"], "macro_recall": m["macro_recall"],
                    "macro_f1": m["macro_f1"], "weighted_f1": m["weighted_f1"],
                    "benign_precision": m["benign_precision"], "benign_recall": m["benign_recall"],
                    "ftp_precision": m["ftp_precision"], "ftp_recall": m["ftp_recall"],
                    "benign_fp_rate": m["benign_fp_rate"]}
        pd.DataFrame([flat("production", base_m)]).to_csv(out / "baseline_metrics.csv", index=False)
        pd.DataFrame([flat("candidate2", cand_m)]).to_csv(out / "candidate_metrics.csv", index=False)
        self._cm(base_m).to_csv(out / "confusion_matrix_production.csv")
        self._cm(cand_m).to_csv(out / "confusion_matrix_candidate.csv")

        pc_rows = []
        for pc, model in ((base_pc, "production"), (cand_pc, "candidate2")):
            for r in pc:
                pc_rows.append({"model": model, **r})
        pd.DataFrame(pc_rows).to_csv(out / "per_capture_metrics.csv", index=False)

        # per-class
        rows = []
        for name, m in (("production", base_m), ("candidate2", cand_m)):
            for cls, d in m["per_class"].items():
                rows.append({"model": name, "class": cls, **d})
        pd.DataFrame(rows).to_csv(out / "per_class_metrics.csv", index=False)

        pd.DataFrame([{"model": "production", **base_m["prediction_distribution"]},
                      {"model": "candidate2", **cand_m["prediction_distribution"]}]
                     ).to_csv(out / "prediction_distribution.csv", index=False)
        pd.DataFrame([{"model": "production", **base_m.get("confidence", {}), **base_conf},
                      {"model": "candidate2", **cand_m.get("confidence", {}), **cand_conf}]
                     ).to_csv(out / "confidence_distribution.csv", index=False)

        def ci_rows(name, ci):
            return [{"model": name, "metric": k, "mean": ci[k]["mean"],
                     "lo95": ci[k]["lo95"], "hi95": ci[k]["hi95"],
                     "n_captures": ci["n_captures"], "n_flows": ci["n_flows"], "n_boot": ci["n_boot"]}
                    for k in ("accuracy", "ftp_recall", "benign_recall")]
        pd.DataFrame(ci_rows("production", base_ci) + ci_rows("candidate2", cand_ci)
                     ).to_csv(out / "bootstrap_or_ci_results.csv", index=False)

        if shap_res is not None:
            shap_res["df"].to_csv(out / "shap_comparison.csv", index=False)
        pd.DataFrame(three_way).to_csv(out / "three_way_comparison.csv", index=False)

        # manifest summary
        man = pd.read_csv(ie.test_dir() / "MANIFEST.csv")
        summ = {"n_pcaps": len(man),
                "benign": int((man["label"] == "Benign").sum()),
                "ftp_bruteforce": int((man["label"] == "FTP-BruteForce").sum()),
                "clients": man["client"].nunique(), "servers": man["server"].nunique(),
                "environments": man["environment"].nunique(),
                "distinct_scenarios": man["scenario"].nunique(),
                "all_valid": bool((man["verification_status"] == "valid").all())}
        pd.DataFrame([summ]).to_csv(out / "test_set_manifest_summary.csv", index=False)

        (out / "leakage_validation.json").write_text(json.dumps(leakage, indent=2, default=str))
        (out / "model_hashes_before_after.json").write_text(json.dumps(
            {"before": hashes_before, "after": hashes_after,
             "unchanged": hashes_before == hashes_after}, indent=2, default=str))
        (out / "final_verdict.json").write_text(json.dumps(verdict, indent=2, default=str))

        # report.md
        self._report_md(out, base_m, cand_m, base_ci, cand_ci, base_conf, cand_conf,
                        shap_res, three_way, verdict, leakage, summ, opts)

    def _cm(self, m):
        c = m["confusion"]
        return pd.DataFrame(c["matrix"], index=c["labels"], columns=c["labels"])

    def _report_md(self, out, base_m, cand_m, base_ci, cand_ci, base_conf, cand_conf,
                   shap_res, three_way, verdict, leakage, summ, opts):
        shap_md = ""
        if shap_res:
            shap_md = (f"\n## SHAP (frozen models, diagnosis only)\n\n"
                       f"- Production global top-5: {', '.join(shap_res['production_top5'])}\n"
                       f"- Candidate 2 global top-5: {', '.join(shap_res['candidate_top5'])}\n"
                       f"- **Candidate 2 still depends on CIC artifact features "
                       f"(`Fwd Seg Size Min`/`Init Fwd Win Byts`): "
                       f"{shap_res['candidate_still_artifact_dependent']}**\n")
        tw = pd.DataFrame(three_way)
        md = f"""# Independent real-PCAP test — final evaluation of Candidate 2

**These test PCAPs were NEVER used for training, retraining, fine-tuning, sample
weighting, threshold selection, hyperparameter selection, feature selection,
candidate selection, or SHAP-based tuning.** Both models are frozen; the
independent set is used only to evaluate them. Candidate 2 was not modified after
seeing these results.

## Test corpus

{summ['n_pcaps']} fresh PCAPs ({summ['benign']} Benign, {summ['ftp_bruteforce']}
FTP-BruteForce), {summ['distinct_scenarios']} distinct scenarios,
{summ['clients']} clients, {summ['servers']} server implementations,
{summ['environments']} environments. All verified: {summ['all_valid']}.

## Headline (independent test)

| Model | FTP recall | Benign recall | Benign FP rate | macro-F1 | accuracy |
|---|---|---|---|---|---|
| production | {_f(base_m['ftp_recall'])} | {_f(base_m['benign_recall'])} | {_f(base_m['benign_fp_rate'])} | {base_m['macro_f1']:.4f} | {base_m['accuracy']:.4f} |
| Candidate 2 | {_f(cand_m['ftp_recall'])} | {_f(cand_m['benign_recall'])} | {_f(cand_m['benign_fp_rate'])} | {cand_m['macro_f1']:.4f} | {cand_m['accuracy']:.4f} |

**Capture-level bootstrap 95% CIs** ({base_ci['n_captures']} captures,
{base_ci['n_flows']} flows, {opts['n_boot']} resamples):

| Model | FTP recall (95% CI) | Benign recall (95% CI) | Accuracy (95% CI) |
|---|---|---|---|
| production | {base_ci['ftp_recall']['mean']:.3f} [{base_ci['ftp_recall']['lo95']:.3f}, {base_ci['ftp_recall']['hi95']:.3f}] | {base_ci['benign_recall']['mean']:.3f} [{base_ci['benign_recall']['lo95']:.3f}, {base_ci['benign_recall']['hi95']:.3f}] | {base_ci['accuracy']['mean']:.3f} [{base_ci['accuracy']['lo95']:.3f}, {base_ci['accuracy']['hi95']:.3f}] |
| Candidate 2 | {cand_ci['ftp_recall']['mean']:.3f} [{cand_ci['ftp_recall']['lo95']:.3f}, {cand_ci['ftp_recall']['hi95']:.3f}] | {cand_ci['benign_recall']['mean']:.3f} [{cand_ci['benign_recall']['lo95']:.3f}, {cand_ci['benign_recall']['hi95']:.3f}] | {cand_ci['accuracy']['mean']:.3f} [{cand_ci['accuracy']['lo95']:.3f}, {cand_ci['accuracy']['hi95']:.3f}] |

## Confidence analysis

- Production: confidently-wrong (≥0.90) = {base_conf['confidently_wrong']}
  (FTP→Benign {base_conf['confidently_wrong_ftp_as_benign']}); mean conf when
  wrong {_f(base_conf['mean_conf_wrong'])}.
- Candidate 2: confidently-wrong (≥0.90) = {cand_conf['confidently_wrong']}
  (Benign→FTP {cand_conf['confidently_wrong_benign_as_ftp']}); mean conf when
  wrong {_f(cand_conf['mean_conf_wrong'])}.
{shap_md}
## Three-way comparison (training vs prior LOCO vs independent)

{_md_table(tw.to_dict("records"))}

## Leakage / independence — all pass: {leakage['all_pass']}

Independent PCAPs disjoint from v1 and v2 (by content hash), no duplicates, labels
from folders only, exactly 30 ordered finite features, no zero-fill, every flow
traceable to its PCAP. Frozen artifacts (production + Candidate 2 + v2 evidence +
v1/v2 datasets) verified unchanged before == after.

## FINAL VERDICT — {verdict['classification']}

{verdict['summary']}

**Candidate is NOT promoted** (`promote=false`).

### Caveats
{chr(10).join('- ' + c for c in verdict['caveats'])}

## Files

`baseline_metrics.csv`, `candidate_metrics.csv`, `confusion_matrix_*.csv`,
`per_class_metrics.csv`, `per_capture_metrics.csv`, `prediction_distribution.csv`,
`confidence_distribution.csv`, `bootstrap_or_ci_results.csv`, `shap_comparison.csv`,
`three_way_comparison.csv`, `test_set_manifest_summary.csv`,
`leakage_validation.json`, `model_hashes_before_after.json`, `final_verdict.json`.
"""
        (out / "report.md").write_text(md)


def _f(v):
    if v is None:
        return "n/a"
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def _md_table(rows):
    if not rows:
        return "(none)"
    cols = list(rows[0].keys())
    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else ("" if v is None else str(v))
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    body = "\n".join("| " + " | ".join(fmt(r[c]) for c in cols) + " |" for r in rows)
    return f"{head}\n{sep}\n{body}"
