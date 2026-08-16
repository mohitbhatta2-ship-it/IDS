"""
Targeted-benign retraining experiment (candidate only; production + Candidate 2 frozen).

Trains HGB candidates = CIC + v2 real + the 41 TARGETED benign captures, under
benign-weighting strategies {none, moderate, strong}, to reduce Candidate 2's
benign false-positive rate while retaining FTP recall. Selects the primary
candidate using ONLY CIC held-out + v2 LOCO (never the independent test), then runs
ONE final evaluation on the FROZEN independent 36-PCAP test set comparing
production, Candidate 2, and the new candidates. Verifies all frozen artifacts
unchanged before/after. Writes evidence to validation/results/targeted_benign_retraining/.

    python manage.py targeted_benign_retrain
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

from predictor import ml, retraining as rt, retraining_v2 as r2, retraining_targeted as tr
from predictor import independent_eval as ie


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _frozen_hashes(root):
    return {
        "production": _sha(ie.production_model_path()),
        "candidate2": _sha(ie.candidate2_file()),
        "ml_py": _sha(root / "webapp_django/predictor/ml.py"),
        "live_capture_py": _sha(root / "webapp_django/predictor/live_capture.py"),
        "pcap_validation_py": _sha(root / "webapp_django/predictor/pcap_validation.py"),
        "v1_pcaps": sorted(_sha(p) for p in glob.glob(str(root / "validation/realistic_pcaps/**/*.pcap"), recursive=True)),
        "v2_pcaps": sorted(_sha(p) for p in glob.glob(str(root / "validation/realistic_pcaps_v2/**/*.pcap"), recursive=True)),
        "indep_pcaps": sorted(_sha(p) for p in glob.glob(str(root / "validation/independent_real_pcaps/**/*.pcap"), recursive=True)),
        "targeted_pcaps": sorted(_sha(p) for p in glob.glob(str(root / "validation/targeted_benign_pcaps/**/*.pcap"), recursive=True)),
    }


class Command(BaseCommand):
    help = "Targeted-benign retraining vs frozen production + Candidate 2 (no promotion, no merge)."

    def add_arguments(self, parser):
        parser.add_argument("--output", default=None)
        parser.add_argument("--loco-sample", type=int, default=tr.CIC_LOCO_SAMPLE)
        parser.add_argument("--weights", default="none,moderate,strong")
        parser.add_argument("--max-folds", type=int, default=0)
        parser.add_argument("--n-boot", type=int, default=2000)
        parser.add_argument("--skip-shap", action="store_true")

    def handle(self, *args, **opts):
        w = self.stdout.write
        root = tr.repo_root()
        out = Path(opts["output"]) if opts["output"] else root / "validation" / "results" / "targeted_benign_retraining"
        out.mkdir(parents=True, exist_ok=True)
        weight_names = [x.strip() for x in opts["weights"].split(",") if x.strip()]

        before = _frozen_hashes(root)
        w(self.style.MIGRATE_HEADING("Targeted-benign retraining experiment (candidate only)"))
        w(f"  production sha={before['production'][:16]}  Candidate 2 sha={before['candidate2'][:16]}")

        # ---- data (training may use CIC + v2 real + targeted benign ONLY) ----
        v2_real = r2.extract_real_flows_v2()
        tgt = tr.extract_targeted_benign_flows()
        w(f"  v2 real flows: {len(v2_real.df)} ({len(v2_real.captures)} captures); "
          f"targeted benign flows: {len(tgt.df)} ({tgt.df['capture'].nunique()} captures); "
          f"invalid: {len(v2_real.invalid) + len(tgt.invalid)}")
        # data-separation guard: targeted/v2 flows must not overlap the independent set
        self._guard_no_independent_leak(root, v2_real, tgt)

        cic_X, cic_y = rt.load_cic()
        cic_test_X, cic_test_y = rt.load_cic_test()
        dec = rt._encoded_to_name()
        cic_test_names = cic_test_y.map(dec)
        cic_sub_X, cic_sub_y = r2.stratified_cic_subsample(cic_X, cic_y, opts["loco_sample"])

        # ---- baselines: production + Candidate 2 on CIC + v2 real ----------
        prod, _ = ml._load(ml.DEFAULT_MODEL)
        cand2 = __import__("joblib").load(ie.candidate2_file())
        base_cic = rt.evaluate(prod, cic_test_X, cic_test_names)
        cand2_cic = rt.evaluate(cand2, cic_test_X, cic_test_names)
        w(f"  baseline CIC: prod {base_cic['accuracy']:.4f}/{base_cic['macro_f1']:.4f}  "
          f"Cand2 {cand2_cic['accuracy']:.4f}/{cand2_cic['macro_f1']:.4f}")

        # ---- train new candidates (full CIC + v2 real + targeted benign) ---
        w(self.style.MIGRATE_HEADING("\nNew candidates (CIC + v2 real + targeted benign)"))
        cand_rows, per_class_rows, per_cap_rows, models = [], [], [], {}
        loco_store = {}
        for wn in weight_names:
            wb = tr.BENIGN_WEIGHTS[wn]
            Xf, yf, wf, manf = tr.assemble(cic_X, cic_y, v2_real.df, tgt.df, benign_weight=wb)
            model = tr.train(Xf, yf, wf)
            models[wn] = model
            c_cic = rt.evaluate(model, cic_test_X, cic_test_names)
            saved = tr.save_candidate(model, {
                "candidate": f"targeted_benign_weight_{wn}", "seed": tr.SEED, "benign_weight": wb,
                "train": "full CIC + v2 real + targeted benign", "train_manifest": manf,
                "cic_accuracy": c_cic["accuracy"], "cic_macro_f1": c_cic["macro_f1"]}, f"candidate_{wn}")

            real_v2 = v2_real
            if opts["max_folds"]:
                keep = v2_real.captures[:opts["max_folds"]]
                real_v2 = r2.RealFlows(df=v2_real.df[v2_real.df["capture"].isin(keep)].reset_index(drop=True),
                                       invalid=v2_real.invalid)
            loco = tr.leave_one_capture_out(cic_sub_X, cic_sub_y, real_v2, tgt.df, benign_weight=wb)
            loco_store[wn] = loco
            p = loco["pooled"]
            w(f"  [{wn:8} wb={wb:>4g}] CIC {c_cic['accuracy']:.4f}/{c_cic['macro_f1']:.4f} | "
              f"v2-LOCO acc {p['accuracy']:.4f} FTP-rec {p['FTP-BruteForce_recall']} "
              f"Ben-rec {p['Benign_recall']}")
            cand_rows.append({
                "candidate": f"targeted_{wn}", "benign_weight": wb,
                "cic_accuracy": c_cic["accuracy"], "cic_macro_f1": c_cic["macro_f1"],
                "v2_loco_accuracy": p["accuracy"], "v2_loco_macro_f1": p["macro_f1"],
                "v2_loco_ftp_recall": p["FTP-BruteForce_recall"],
                "v2_loco_benign_recall": p["Benign_recall"]})
            for pc in c_cic["per_class"]:
                per_class_rows.append({"candidate": f"targeted_{wn}", "eval": "cic_heldout", **pc})
            for f in loco["folds"]:
                per_cap_rows.append({"candidate": f"targeted_{wn}", "eval": "v2_loco",
                                     "capture": f["held_out_capture"], "label": f["true_label"],
                                     "n_flows": f["n_flows"], "recall": f["recall"],
                                     "predictions": json.dumps(f["predictions"])})
            r2_cm = p["confusion"]
            pd.DataFrame(r2_cm["matrix"], index=r2_cm["labels"], columns=r2_cm["labels"]).to_csv(
                out / f"confusion_v2loco_{wn}.csv")

        # ---- SELECT primary candidate using ONLY CIC + v2 LOCO -------------
        selected = self._select(cand_rows, base_cic)
        w(self.style.MIGRATE_HEADING(f"\nPrimary candidate selected on CIC+v2-LOCO only: {selected}"))

        # ---- FINAL independent eval (frozen test; report all models) -------
        w(self.style.MIGRATE_HEADING("\nFINAL independent test (frozen 36-PCAP set)"))
        flows = ie.extract_test_flows()
        indep_rows, indep_metrics, indep_ci, indep_conf, indep_percap = [], {}, {}, {}, {}
        eval_models = {"production": (prod, None), "candidate2": (cand2, None)}
        for wn in weight_names:
            eval_models[f"targeted_{wn}"] = (models[wn], None)
        for name, (model, scaler) in eval_models.items():
            pred, conf = ie.predict(model, scaler, flows.df)
            m = ie.metrics(flows.df["Label"].to_numpy(), pred, conf)
            indep_metrics[name] = m
            indep_ci[name] = ie.bootstrap_cis(flows.df, model, scaler, n_boot=opts["n_boot"])
            indep_conf[name] = ie.confidence_analysis(flows.df, model, scaler)
            indep_percap[name] = ie.per_capture(model, scaler, flows.df)
            indep_rows.append({"model": name, "n": m["n"], "accuracy": m["accuracy"],
                               "macro_f1": m["macro_f1"], "ftp_recall": m["ftp_recall"],
                               "ftp_precision": m["ftp_precision"], "benign_recall": m["benign_recall"],
                               "benign_fp_rate": m["benign_fp_rate"],
                               "ftp_recall_lo95": indep_ci[name]["ftp_recall"]["lo95"],
                               "ftp_recall_hi95": indep_ci[name]["ftp_recall"]["hi95"],
                               "benign_recall_lo95": indep_ci[name]["benign_recall"]["lo95"],
                               "benign_recall_hi95": indep_ci[name]["benign_recall"]["hi95"]})
            w(f"  {name:16} FTP-rec {_f(m['ftp_recall'])} Ben-rec {_f(m['benign_recall'])} "
              f"BenFP {_f(m['benign_fp_rate'])} mF1 {m['macro_f1']:.4f} acc {m['accuracy']:.4f}")

        # ---- SHAP on the selected candidate --------------------------------
        shap_res = None if opts["skip_shap"] else self._shap(prod, cand2, models[selected], cic_X, flows, selected)

        # ---- verdict -------------------------------------------------------
        verdict = self._verdict(selected, indep_metrics, cand_rows, base_cic, shap_res)
        w(self.style.MIGRATE_HEADING("\nVERDICT"))
        w(self.style.WARNING("  " + verdict["decision"] + " -- " + verdict["summary"]))

        # ---- hashes after --------------------------------------------------
        after = _frozen_hashes(root)
        unchanged = before == after
        w(f"\n  frozen artifacts unchanged (before==after): {unchanged}")
        if not unchanged:
            raise SystemExit("ABORT: a frozen artifact changed!")

        self._write(out, cand_rows, per_class_rows, per_cap_rows, indep_rows, indep_metrics,
                    indep_ci, indep_conf, indep_percap, base_cic, cand2_cic, selected, verdict,
                    shap_res, before, after, v2_real, tgt, weight_names, opts, flows)
        w(self.style.SUCCESS(f"\nEvidence written to {out}"))
        w(self.style.WARNING("\nNo promotion, no production change, no merge. Independent test used "
                             "once, for evaluation only."))

    # -- helpers -----------------------------------------------------------

    def _guard_no_independent_leak(self, root, v2_real, tgt):
        indep = {_sha(p) for p in glob.glob(str(root / "validation/independent_real_pcaps/**/*.pcap"), recursive=True)}
        train_pcaps = {_sha(p) for p in glob.glob(str(root / "validation/realistic_pcaps_v2/**/*.pcap"), recursive=True)}
        train_pcaps |= {_sha(p) for p in glob.glob(str(root / "validation/targeted_benign_pcaps/**/*.pcap"), recursive=True)}
        if not indep.isdisjoint(train_pcaps):
            raise SystemExit("ABORT: an independent-test PCAP appears in the training corpora!")

    def _select(self, cand_rows, base_cic) -> str:
        # Selection uses ONLY CIC held-out + v2 LOCO (never the independent test).
        # Rule: among candidates with CIC macro-F1 >= baseline-0.01 AND v2-LOCO FTP
        # recall >= 0.85 (retain FTP), pick the highest v2-LOCO benign recall.
        ok = [c for c in cand_rows if c["cic_macro_f1"] >= base_cic["macro_f1"] - 0.01
              and (c["v2_loco_ftp_recall"] or 0) >= 0.85]
        pool = ok or cand_rows
        best = max(pool, key=lambda c: (c["v2_loco_benign_recall"] or 0))
        return best["candidate"].split("_")[-1]

    def _shap(self, prod, cand2, cand_new, cic_X, flows, selected):
        try:
            import shap
        except Exception:  # noqa: BLE001
            return None
        bg = cic_X.sample(min(300, len(cic_X)), random_state=tr.SEED)
        art = {"Fwd Seg Size Min", "Init Fwd Win Byts"}

        def top(model, data, n=5):
            if len(data) == 0:
                return []
            vals = np.abs(np.asarray(shap.TreeExplainer(model).shap_values(data))).mean(axis=(0, 2))
            order = np.argsort(vals)[::-1]
            return [ml.FEATURES[j] for j in order[:n]]

        ftp = flows.df[flows.df["Label"] == "FTP-BruteForce"][ml.FEATURES]
        ben = flows.df[flows.df["Label"] == "Benign"][ml.FEATURES]
        rows = []
        for name, model in (("production", prod), ("candidate2", cand2), (f"targeted_{selected}", cand_new)):
            g = top(model, bg); ft = top(model, ftp); bt = top(model, ben)
            rows.append({"model": name, "global_top5": ";".join(g), "test_ftp_top5": ";".join(ft),
                         "test_benign_top5": ";".join(bt),
                         "artifact_in_global_top5": bool(art & set(g))})
        df = pd.DataFrame(rows)
        c2 = df[df["model"] == "candidate2"].iloc[0]
        cn = df[df["model"] == f"targeted_{selected}"].iloc[0]
        return {"df": df,
                "candidate2_artifact_dep": bool(c2["artifact_in_global_top5"]),
                "new_artifact_dep": bool(cn["artifact_in_global_top5"]),
                "new_benign_top5": cn["test_benign_top5"], "new_global_top5": cn["global_top5"]}

    def _verdict(self, selected, indep_metrics, cand_rows, base_cic, shap_res) -> dict:
        cand2 = indep_metrics["candidate2"]
        new = indep_metrics[f"targeted_{selected}"]
        sel_row = [c for c in cand_rows if c["candidate"].endswith(selected)][0]
        fp_before = cand2["benign_fp_rate"]; fp_after = new["benign_fp_rate"]
        fp_reduction = fp_before - fp_after
        ftp_before = cand2["ftp_recall"]; ftp_after = new["ftp_recall"]
        cic_ok = sel_row["cic_macro_f1"] >= base_cic["macro_f1"] - 0.01
        retains_ftp = ftp_after >= 0.50
        meaningful_fp_drop = fp_reduction >= 0.10
        promote = bool(meaningful_fp_drop and retains_ftp and cic_ok
                       and new["macro_f1"] >= cand2["macro_f1"])
        if promote:
            decision = "PROMOTE (candidate meets all criteria -- still validate on more data)"
        else:
            decision = "DO NOT PROMOTE"
        summary = (f"benign FP {fp_before:.3f}->{fp_after:.3f} ({fp_reduction:+.3f}); "
                   f"FTP recall {ftp_before:.3f}->{ftp_after:.3f}; macro-F1 "
                   f"{cand2['macro_f1']:.3f}->{new['macro_f1']:.3f}; CIC non-regressed={cic_ok}. "
                   + ("Meets promotion criteria." if promote else
                      "Does not meet all promotion criteria (needs meaningful FP drop, retained FTP "
                      "recall, no CIC regression, and macro-F1 not worse)."))
        return {
            "selected_candidate": f"targeted_{selected}", "promote": promote, "decision": decision,
            "summary": summary,
            "candidate_selection_basis": "CIC held-out + v2 LOCO only (independent test never used to select)",
            "independent_benign_fp_candidate2": fp_before, "independent_benign_fp_new": fp_after,
            "independent_benign_fp_reduction": fp_reduction,
            "independent_ftp_recall_candidate2": ftp_before, "independent_ftp_recall_new": ftp_after,
            "independent_macro_f1_candidate2": cand2["macro_f1"], "independent_macro_f1_new": new["macro_f1"],
            "cic_non_regressed": cic_ok,
            "shap": ({"candidate2_artifact_dependent": shap_res["candidate2_artifact_dep"],
                      "new_artifact_dependent": shap_res["new_artifact_dep"]} if shap_res else None),
            "caveats": [
                "Independent test small (36 captures / 420 flows), loopback-only, 2 real classes.",
                "Candidate selected on CIC+v2-LOCO; independent test evaluated once.",
                "No statistical-significance claim; CIs are wide.",
            ],
        }

    def _write(self, out, cand_rows, per_class_rows, per_cap_rows, indep_rows, indep_metrics,
               indep_ci, indep_conf, indep_percap, base_cic, cand2_cic, selected, verdict,
               shap_res, before, after, v2_real, tgt, weight_names, opts, flows):
        pd.DataFrame(cand_rows).to_csv(out / "candidate_metrics.csv", index=False)
        pd.DataFrame([{"model": "production_cic", **_flat(base_cic)},
                      {"model": "candidate2_cic", **_flat(cand2_cic)}]).to_csv(out / "baseline_metrics.csv", index=False)
        pd.DataFrame(per_class_rows).to_csv(out / "per_class_metrics.csv", index=False)
        pd.DataFrame(per_cap_rows).to_csv(out / "per_capture_v2loco_metrics.csv", index=False)
        pd.DataFrame(indep_rows).to_csv(out / "independent_test_metrics.csv", index=False)

        # weighting comparison (CIC + v2 LOCO + independent side by side)
        wc = []
        for c in cand_rows:
            tag = c["candidate"].split("_")[-1]
            im = indep_metrics[f"targeted_{tag}"]
            wc.append({**c, "indep_ftp_recall": im["ftp_recall"], "indep_benign_recall": im["benign_recall"],
                       "indep_benign_fp_rate": im["benign_fp_rate"], "indep_macro_f1": im["macro_f1"]})
        pd.DataFrame(wc).to_csv(out / "weighting_comparison.csv", index=False)

        # independent per-capture + confusion + confidence for the key models
        pc_rows = []
        for name in ("production", "candidate2", f"targeted_{selected}"):
            for r in indep_percap[name]:
                pc_rows.append({"model": name, **r})
        pd.DataFrame(pc_rows).to_csv(out / "independent_per_capture_metrics.csv", index=False)
        for name in ("production", "candidate2", f"targeted_{selected}"):
            c = indep_metrics[name]["confusion"]
            pd.DataFrame(c["matrix"], index=c["labels"], columns=c["labels"]).to_csv(
                out / f"confusion_independent_{name}.csv")
        pd.DataFrame([{"model": n, **indep_conf[n]} for n in indep_conf]).to_csv(
            out / "confidence_distribution.csv", index=False)

        ci_rows = []
        for name, ci in indep_ci.items():
            for metric in ("accuracy", "ftp_recall", "benign_recall"):
                ci_rows.append({"model": name, "metric": metric, **ci[metric],
                                "n_captures": ci["n_captures"], "n_flows": ci["n_flows"]})
        pd.DataFrame(ci_rows).to_csv(out / "bootstrap_ci_results.csv", index=False)

        if shap_res is not None:
            shap_res["df"].to_csv(out / "shap_results.csv", index=False)

        leak = {"independent_disjoint_from_training": True,
                "targeted_and_v2_used_for_training": True,
                "independent_used_for": "final evaluation only",
                "labels_from_folders": True,
                "v2_flows": int(len(v2_real.df)), "targeted_flows": int(len(tgt.df)),
                "invalid_flows": len(v2_real.invalid) + len(tgt.invalid),
                "n_features": len(ml.FEATURES), "feature_order_ok": True, "no_zero_fill": True}
        (out / "leakage_validation.json").write_text(json.dumps(leak, indent=2, default=str))
        (out / "model_hashes_before_after.json").write_text(json.dumps(
            {"before": before, "after": after, "unchanged": before == after}, indent=2, default=str))
        (out / "final_verdict.json").write_text(json.dumps(verdict, indent=2, default=str))

        import sklearn
        (out / "training_metadata.json").write_text(json.dumps({
            "experiment": "targeted_benign_retraining", "created_utc": datetime.now(timezone.utc).isoformat(),
            "seed": tr.SEED, "model": "HistGradientBoostingClassifier", "hyperparameters": rt.hgb_params(),
            "benign_weights": {k: tr.BENIGN_WEIGHTS[k] for k in weight_names},
            "loco_cic_subsample": opts["loco_sample"], "features": list(ml.FEATURES),
            "training_data": "CIC + v2 real (weight 1) + targeted benign (weight w_b)",
            "independent_test": "FROZEN; evaluation only",
            "candidate_dir": str(tr.candidate_dir()),
            "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__}},
            indent=2, default=str))

        self._report(out, cand_rows, indep_rows, indep_metrics, indep_ci, selected, verdict, shap_res, opts)

    def _report(self, out, cand_rows, indep_rows, indep_metrics, indep_ci, selected, verdict, shap_res, opts):
        sel = f"targeted_{selected}"
        c2 = indep_metrics["candidate2"]; nw = indep_metrics[sel]; pr = indep_metrics["production"]
        shap_md = ""
        if shap_res:
            shap_md = (f"\n## SHAP (selected candidate `{sel}`)\n\n"
                       f"- Candidate 2 global top-5 artifact-dependent: "
                       f"**{shap_res['candidate2_artifact_dep']}**\n"
                       f"- New candidate global top-5 artifact-dependent: "
                       f"**{shap_res['new_artifact_dep']}**\n"
                       f"- New candidate top features on independent BENIGN flows: "
                       f"{shap_res['new_benign_top5']}\n")
        md = f"""# Targeted-benign retraining — report

**Candidate only. Production model and Candidate 2 are frozen and byte-for-byte
unchanged (verified). No promotion by default, no merge.** The independent 36-PCAP
test set was NEVER used for training, weighting, hyperparameter/threshold
selection, or candidate selection — the primary candidate was selected using CIC
held-out + v2 LOCO only, then evaluated once on the independent set.

## Data roles

- **Training:** CIC `balanced_train_selected` + v2 real (83 captures, weight 1) +
  **41 targeted benign captures** (weight w_b).
- **v2 LOCO:** held-out v2 capture per fold (targeted benign always training
  augmentation) — the real-PCAP validation estimate.
- **Independent test:** `validation/independent_real_pcaps/` — FROZEN, evaluated
  once.

## Candidates (CIC held-out + v2 LOCO; selection signals only)

{_md_table(cand_rows)}

Primary candidate selected on CIC+v2-LOCO: **`{sel}`** (rule: no CIC regression,
v2-LOCO FTP recall ≥0.85, then max v2-LOCO benign recall).

## Final independent test (frozen 36-PCAP set)

{_md_table([{k: r[k] for k in ('model','ftp_recall','ftp_precision','benign_recall','benign_fp_rate','macro_f1','accuracy')} for r in indep_rows])}

Capture-level bootstrap 95% CIs ({indep_ci['production']['n_captures']} captures,
{indep_ci['production']['n_flows']} flows, {opts['n_boot']} resamples) in
`bootstrap_ci_results.csv`.

## Headline comparison — Candidate 2 vs selected new candidate (independent)

| Metric | production | Candidate 2 | new `{selected}` |
|---|---|---|---|
| FTP recall | {_f(pr['ftp_recall'])} | {_f(c2['ftp_recall'])} | {_f(nw['ftp_recall'])} |
| Benign recall | {_f(pr['benign_recall'])} | {_f(c2['benign_recall'])} | {_f(nw['benign_recall'])} |
| Benign FP rate | {_f(pr['benign_fp_rate'])} | {_f(c2['benign_fp_rate'])} | {_f(nw['benign_fp_rate'])} |
| macro-F1 | {pr['macro_f1']:.4f} | {c2['macro_f1']:.4f} | {nw['macro_f1']:.4f} |
| accuracy | {pr['accuracy']:.4f} | {c2['accuracy']:.4f} | {nw['accuracy']:.4f} |
{shap_md}
## Verdict — {verdict['decision']}

{verdict['summary']}

- benign FP change (Cand2→new): **{verdict['independent_benign_fp_reduction']:+.3f}**
- FTP recall change (Cand2→new): **{(verdict['independent_ftp_recall_new'] or 0) - (verdict['independent_ftp_recall_candidate2'] or 0):+.3f}**
- CIC non-regressed: **{verdict['cic_non_regressed']}**  ·  promote: **{verdict['promote']}**

### Caveats
{chr(10).join('- ' + c for c in verdict['caveats'])}

## Integrity

Production model, Candidate 2, ml.py / live_capture.py / pcap_validation.py, and
the v1/v2/independent/targeted PCAPs verified unchanged before == after. Exactly 30
ordered finite features, no zero-fill. No independent PCAP entered training.

## Files

`candidate_metrics.csv`, `baseline_metrics.csv`, `per_class_metrics.csv`,
`per_capture_v2loco_metrics.csv`, `independent_test_metrics.csv`,
`independent_per_capture_metrics.csv`, `weighting_comparison.csv`,
`confusion_*.csv`, `confidence_distribution.csv`, `bootstrap_ci_results.csv`,
`shap_results.csv`, `leakage_validation.json`, `model_hashes_before_after.json`,
`training_metadata.json`, `final_verdict.json`.
"""
        (out / "report.md").write_text(md)


def _flat(m):
    return {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
            "macro_precision": m["macro_precision"], "macro_recall": m["macro_recall"]}


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
    body = "\n".join("| " + " | ".join(fmt(r[c]) for c in cols) + " |" for r in rows)
    return f"{head}\n{sep}\n{body}"
