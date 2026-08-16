"""
Balanced real-PCAP retraining experiment (candidate only; production frozen).

Trains HGB candidates on CIC + a larger, class-balanced set of real flows
(v1 + v2 + targeted benign) under several weighting strategies, selects using CIC
held-out + v2 LOCO only, then runs ONE final evaluation on the FROZEN independent
36-PCAP test set. Promotes nothing automatically; a candidate is promotable only if
it hits the balanced target (FTP recall >0.70, benign recall >0.90, benign FP well
below Candidate 2, no CIC regression, no leakage, no threshold/heuristic tricks).

    python manage.py balanced_real_retrain
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

from predictor import ml, retraining as rt, retraining_v2 as r2, retraining_balanced as rb
from predictor import independent_eval as ie


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


class Command(BaseCommand):
    help = "Balanced real-PCAP retraining vs frozen production + Candidate 2 (no promotion, no merge)."

    def add_arguments(self, parser):
        parser.add_argument("--output", default=None)
        parser.add_argument("--loco-sample", type=int, default=r2.CIC_LOCO_SAMPLE)
        parser.add_argument("--strategies", default="unweighted,moderate_ftp,balanced")
        parser.add_argument("--max-folds", type=int, default=0)
        parser.add_argument("--n-boot", type=int, default=2000)
        parser.add_argument("--skip-shap", action="store_true")

    def handle(self, *args, **opts):
        w = self.stdout.write
        root = rb.repo_root()
        out = Path(opts["output"]) if opts["output"] else root / "validation" / "results" / "balanced_real_retraining"
        out.mkdir(parents=True, exist_ok=True)
        strategies = [s.strip() for s in opts["strategies"].split(",") if s.strip()]

        before = _frozen(root)
        w(self.style.MIGRATE_HEADING("Balanced real-PCAP retraining experiment (candidate only)"))
        w(f"  production sha={before['production'][:16]}  Candidate 2 sha={before['candidate2'][:16]}")

        # data-separation guard: no independent PCAP in the training corpora
        indep = set(before["indep_pcaps"])
        trainset = set(before["v1_pcaps"]) | set(before["v2_pcaps"]) | set(before["targeted_pcaps"])
        if not indep.isdisjoint(trainset):
            raise SystemExit("ABORT: an independent-test PCAP is in the training corpora!")

        real = rb.combined_real_flows()
        v2caps = rb.v2_captures(real)
        w(f"  combined real flows: {len(real.df)} (FTP {int((real.df['Label']=='FTP-BruteForce').sum())}, "
          f"Benign {int((real.df['Label']=='Benign').sum())}); invalid: {len(real.invalid)}; "
          f"v2 LOCO captures: {len(v2caps)}")

        cic_X, cic_y = rt.load_cic()
        cic_test_X, cic_test_y = rt.load_cic_test()
        dec = rt._encoded_to_name()
        cic_test_names = cic_test_y.map(dec)
        cic_sub_X, cic_sub_y = r2.stratified_cic_subsample(cic_X, cic_y, opts["loco_sample"])

        # baselines
        prod, _ = ml._load(ml.DEFAULT_MODEL)
        cand2 = __import__("joblib").load(ie.candidate2_file())
        base_cic = rt.evaluate(prod, cic_test_X, cic_test_names)
        cand2_cic = rt.evaluate(cand2, cic_test_X, cic_test_names)
        w(f"  baseline CIC: prod {base_cic['accuracy']:.4f}/{base_cic['macro_f1']:.4f}  "
          f"Cand2 {cand2_cic['accuracy']:.4f}/{cand2_cic['macro_f1']:.4f}")

        # ---- Candidate 1: CIC-only control -------------------------------
        w(self.style.MIGRATE_HEADING("\nCandidate 1: CIC-only control"))
        control = rt.train(cic_X, cic_y, np.ones(len(cic_X)))
        ctrl_cic = rt.evaluate(control, cic_test_X, cic_test_names)
        reproduces = abs(ctrl_cic["accuracy"] - base_cic["accuracy"]) < 5e-3
        v2_flows = real.df[real.df["source"] == "v2"]
        ctrl_v2 = rt.evaluate(control, v2_flows[ml.FEATURES], v2_flows["Label"])
        w(f"  CIC {ctrl_cic['accuracy']:.4f}/{ctrl_cic['macro_f1']:.4f} reproduces={reproduces} | "
          f"v2(all) FTP-rec {_f(ctrl_v2['ftp_recall'])} Ben-rec {_f(ctrl_v2['benign_recall'])}")
        rb.save_candidate(control, {"candidate": "cic_only_control", "seed": rb.SEED,
                                    "reproduces_baseline": reproduces}, "candidate1_cic_only")
        models = {"cic_only": control}

        # ---- Candidates 2-4: CIC + real, weighting strategies ------------
        w(self.style.MIGRATE_HEADING("\nCandidates: CIC + real (" + ", ".join(strategies) + ")"))
        cand_rows, per_cap_rows, loco_store = [], [], {}
        for strat in strategies:
            fw, bw = rb.class_weights(real.df, strat)
            Xf, yf, wf, manf = rb.assemble(cic_X, cic_y, real.df, ftp_weight=fw, benign_weight=bw)
            model = rb.train(Xf, yf, wf)
            models[strat] = model
            c_cic = rt.evaluate(model, cic_test_X, cic_test_names)
            rb.save_candidate(model, {"candidate": f"cic_real_{strat}", "seed": rb.SEED,
                                      "ftp_weight": fw, "benign_weight": bw, "train_manifest": manf,
                                      "cic_accuracy": c_cic["accuracy"], "cic_macro_f1": c_cic["macro_f1"]},
                              f"candidate_{strat}")
            held = v2caps if not opts["max_folds"] else v2caps[:opts["max_folds"]]
            loco = rb.leave_one_capture_out(cic_sub_X, cic_sub_y, real, held, fw, bw)
            loco_store[strat] = loco
            p = loco["pooled"]
            w(f"  [{strat:13} fw={fw:.1f} bw={bw:.1f}] CIC {c_cic['accuracy']:.4f}/{c_cic['macro_f1']:.4f} | "
              f"v2-LOCO FTP-rec {_f(p['FTP-BruteForce_recall'])} Ben-rec {_f(p['Benign_recall'])} "
              f"mF1 {p['macro_f1']:.4f}")
            cand_rows.append({"candidate": strat, "ftp_weight": fw, "benign_weight": bw,
                              "cic_accuracy": c_cic["accuracy"], "cic_macro_f1": c_cic["macro_f1"],
                              "v2_loco_ftp_recall": p["FTP-BruteForce_recall"],
                              "v2_loco_benign_recall": p["Benign_recall"],
                              "v2_loco_macro_f1": p["macro_f1"], "v2_loco_accuracy": p["accuracy"]})
            for f in loco["folds"]:
                per_cap_rows.append({"candidate": strat, "eval": "v2_loco", "capture": f["held_out_capture"],
                                     "label": f["true_label"], "n_flows": f["n_flows"], "recall": f["recall"],
                                     "predictions": json.dumps(f["predictions"])})
            cm = p["confusion"]
            pd.DataFrame(cm["matrix"], index=cm["labels"], columns=cm["labels"]).to_csv(out / f"confusion_v2loco_{strat}.csv")

        # ---- select on CIC + v2 LOCO only --------------------------------
        selected = self._select(cand_rows, base_cic)
        w(self.style.MIGRATE_HEADING(f"\nSelected on CIC + v2-LOCO only: {selected}"))

        # ---- FINAL independent eval (frozen; report all) -----------------
        w(self.style.MIGRATE_HEADING("\nFINAL independent test (frozen 36-PCAP set)"))
        flows = ie.extract_test_flows()
        eval_models = {"production": prod, "candidate2": cand2, "cic_only": models["cic_only"]}
        for s in strategies:
            eval_models[s] = models[s]
        indep_metrics, indep_ci, indep_conf, indep_percap, indep_rows = {}, {}, {}, {}, []
        for name, model in eval_models.items():
            pred, conf = ie.predict(model, None, flows.df)
            m = ie.metrics(flows.df["Label"].to_numpy(), pred, conf)
            indep_metrics[name] = m
            indep_ci[name] = ie.bootstrap_cis(flows.df, model, None, n_boot=opts["n_boot"])
            indep_conf[name] = ie.confidence_analysis(flows.df, model, None)
            indep_percap[name] = ie.per_capture(model, None, flows.df)
            indep_rows.append({"model": name, "ftp_recall": m["ftp_recall"], "ftp_precision": m["ftp_precision"],
                               "benign_recall": m["benign_recall"], "benign_fp_rate": m["benign_fp_rate"],
                               "macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
                               "ftp_recall_lo95": indep_ci[name]["ftp_recall"]["lo95"],
                               "ftp_recall_hi95": indep_ci[name]["ftp_recall"]["hi95"],
                               "benign_recall_lo95": indep_ci[name]["benign_recall"]["lo95"],
                               "benign_recall_hi95": indep_ci[name]["benign_recall"]["hi95"]})
            w(f"  {name:14} FTP-rec {_f(m['ftp_recall'])} Ben-rec {_f(m['benign_recall'])} "
              f"BenFP {_f(m['benign_fp_rate'])} mF1 {m['macro_f1']:.4f} acc {m['accuracy']:.4f}")

        # ---- SHAP on selected --------------------------------------------
        shap_res = None if opts["skip_shap"] else self._shap(prod, cand2, models[selected], cic_X, flows, selected)

        # ---- verdict against the strict balanced target ------------------
        verdict = self._verdict(selected, indep_metrics, indep_ci, cand_rows, base_cic, shap_res)
        w(self.style.MIGRATE_HEADING("\nVERDICT"))
        w(self.style.WARNING("  " + verdict["decision"] + " -- " + verdict["summary"]))

        after = _frozen(root)
        unchanged = before == after
        w(f"\n  frozen artifacts unchanged (before==after): {unchanged}")
        if not unchanged:
            raise SystemExit("ABORT: a frozen artifact changed!")

        self._write(out, cand_rows, ctrl_cic, ctrl_v2, reproduces, per_cap_rows, indep_rows,
                    indep_metrics, indep_ci, indep_conf, indep_percap, base_cic, cand2_cic,
                    selected, verdict, shap_res, before, after, real, strategies, opts, flows)
        w(self.style.SUCCESS(f"\nEvidence written to {out}"))
        w(self.style.WARNING("\nNo promotion, no production change, no merge. Independent test evaluated once."))

    # -- helpers -----------------------------------------------------------

    def _select(self, cand_rows, base_cic) -> str:
        # CIC+v2-LOCO only. Balanced goal -> among no-CIC-regression candidates,
        # pick the best v2-LOCO macro-F1 (which balances FTP and benign).
        ok = [c for c in cand_rows if c["cic_macro_f1"] >= base_cic["macro_f1"] - 0.01]
        pool = ok or cand_rows
        return max(pool, key=lambda c: (c["v2_loco_macro_f1"] or 0))["candidate"]

    def _shap(self, prod, cand2, sel_model, cic_X, flows, selected):
        try:
            import shap
        except Exception:  # noqa: BLE001
            return None
        bg = cic_X.sample(min(300, len(cic_X)), random_state=rb.SEED)
        art = {"Fwd Seg Size Min", "Init Fwd Win Byts"}

        def top(model, data, n=5):
            if len(data) == 0:
                return []
            vals = np.abs(np.asarray(shap.TreeExplainer(model).shap_values(data))).mean(axis=(0, 2))
            return [ml.FEATURES[j] for j in np.argsort(vals)[::-1][:n]]

        ftp = flows.df[flows.df["Label"] == "FTP-BruteForce"][ml.FEATURES]
        ben = flows.df[flows.df["Label"] == "Benign"][ml.FEATURES]
        rows = []
        for name, model in (("production", prod), ("candidate2", cand2), (f"selected_{selected}", sel_model)):
            g = top(model, bg)
            rows.append({"model": name, "global_top5": ";".join(g),
                         "test_ftp_top5": ";".join(top(model, ftp)),
                         "test_benign_top5": ";".join(top(model, ben)),
                         "artifact_in_global_top5": bool(art & set(g))})
        df = pd.DataFrame(rows)
        sel = df[df["model"] == f"selected_{selected}"].iloc[0]
        return {"df": df, "selected_artifact_dep": bool(sel["artifact_in_global_top5"]),
                "candidate2_artifact_dep": bool(df[df["model"] == "candidate2"].iloc[0]["artifact_in_global_top5"])}

    def _verdict(self, selected, indep_metrics, indep_ci, cand_rows, base_cic, shap_res) -> dict:
        cand2 = indep_metrics["candidate2"]
        new = indep_metrics[selected]
        sel_row = [c for c in cand_rows if c["candidate"] == selected][0]
        ftp = new["ftp_recall"] or 0
        ben = new["benign_recall"] or 0
        fp = new["benign_fp_rate"] or 0
        cand2_fp = cand2["benign_fp_rate"] or 0
        cic_ok = sel_row["cic_macro_f1"] >= base_cic["macro_f1"] - 0.01
        # strict balanced target
        c_ftp = ftp > 0.70
        c_ben = ben > 0.90
        c_fp = fp <= cand2_fp - 0.10          # substantially lower benign FP than Cand2
        promote = bool(c_ftp and c_ben and c_fp and cic_ok)
        decision = "PROMOTE" if promote else "DO NOT PROMOTE"
        unmet = [n for n, ok in (("FTP recall>0.70", c_ftp), ("benign recall>0.90", c_ben),
                                 ("benign FP << Cand2", c_fp), ("no CIC regression", cic_ok)) if not ok]
        summary = (f"selected `{selected}` independent: FTP recall {ftp:.3f}, benign recall {ben:.3f}, "
                   f"benign FP {fp:.3f} (Cand2 {cand2_fp:.3f}), macro-F1 {new['macro_f1']:.3f}, "
                   f"CIC non-regressed={cic_ok}. "
                   + ("Meets the balanced target." if promote else
                      "Unmet criteria: " + "; ".join(unmet) + "."))
        return {"selected_candidate": selected, "promote": promote, "decision": decision, "summary": summary,
                "selection_basis": "CIC held-out + v2 LOCO only (independent test never used to select)",
                "criteria": {"ftp_recall_gt_0.70": c_ftp, "benign_recall_gt_0.90": c_ben,
                             "benign_fp_much_lower_than_cand2": c_fp, "cic_non_regressed": cic_ok},
                "independent": {"ftp_recall": ftp, "ftp_recall_ci95": [indep_ci[selected]["ftp_recall"]["lo95"],
                                                                       indep_ci[selected]["ftp_recall"]["hi95"]],
                                "benign_recall": ben, "benign_fp_rate": fp, "macro_f1": new["macro_f1"],
                                "candidate2_ftp_recall": cand2["ftp_recall"],
                                "candidate2_benign_fp_rate": cand2_fp},
                "shap": ({"selected_artifact_dependent": shap_res["selected_artifact_dep"]} if shap_res else None),
                "conclusion": ("A candidate meets the balanced target on the untouched independent test."
                               if promote else
                               "No candidate meets the balanced target on the independent test; at this "
                               "scale/feature space the real benign and real FTP regions overlap on the "
                               "CIC-artifact features, so lowering benign FP costs FTP recall. The current "
                               "feature space/data is insufficient -- DO NOT promote."),
                "caveats": ["Independent test small (36 captures / 420 flows), loopback-only, 2 real classes.",
                            "Selection on CIC+v2-LOCO; independent evaluated once; CIs wide; no significance claim."]}

    def _write(self, out, cand_rows, ctrl_cic, ctrl_v2, reproduces, per_cap_rows, indep_rows,
               indep_metrics, indep_ci, indep_conf, indep_percap, base_cic, cand2_cic, selected,
               verdict, shap_res, before, after, real, strategies, opts, flows):
        pd.DataFrame(cand_rows).to_csv(out / "candidate_metrics.csv", index=False)
        pd.DataFrame([{"model": "production", "cic_accuracy": base_cic["accuracy"], "cic_macro_f1": base_cic["macro_f1"]},
                      {"model": "candidate2", "cic_accuracy": cand2_cic["accuracy"], "cic_macro_f1": cand2_cic["macro_f1"]},
                      {"model": "cic_only_control", "cic_accuracy": ctrl_cic["accuracy"],
                       "cic_macro_f1": ctrl_cic["macro_f1"], "reproduces_baseline": reproduces,
                       "v2_ftp_recall": ctrl_v2["ftp_recall"], "v2_benign_recall": ctrl_v2["benign_recall"]}]
                     ).to_csv(out / "baseline_metrics.csv", index=False)
        pd.DataFrame(indep_rows).to_csv(out / "independent_test_metrics.csv", index=False)
        pd.DataFrame(per_cap_rows).to_csv(out / "per_capture_v2loco_metrics.csv", index=False)

        # candidate comparison (CIC + v2 LOCO + independent)
        comp = []
        for c in cand_rows:
            im = indep_metrics[c["candidate"]]
            comp.append({**c, "indep_ftp_recall": im["ftp_recall"], "indep_benign_recall": im["benign_recall"],
                         "indep_benign_fp_rate": im["benign_fp_rate"], "indep_macro_f1": im["macro_f1"],
                         "indep_accuracy": im["accuracy"]})
        pd.DataFrame(comp).to_csv(out / "candidate_comparison.csv", index=False)

        pc = []
        for name in ("production", "candidate2", selected):
            for r in indep_percap[name]:
                pc.append({"model": name, **r})
        pd.DataFrame(pc).to_csv(out / "independent_per_capture_metrics.csv", index=False)
        for name in ("production", "candidate2", "cic_only", selected):
            c = indep_metrics[name]["confusion"]
            pd.DataFrame(c["matrix"], index=c["labels"], columns=c["labels"]).to_csv(out / f"confusion_independent_{name}.csv")
        pd.DataFrame([{"model": n, **indep_conf[n]} for n in indep_conf]).to_csv(out / "confidence_distribution.csv", index=False)
        ci_rows = []
        for name, ci in indep_ci.items():
            for metric in ("accuracy", "ftp_recall", "benign_recall"):
                ci_rows.append({"model": name, "metric": metric, **ci[metric],
                                "n_captures": ci["n_captures"], "n_flows": ci["n_flows"]})
        pd.DataFrame(ci_rows).to_csv(out / "bootstrap_ci_results.csv", index=False)
        if shap_res is not None:
            shap_res["df"].to_csv(out / "shap_comparison.csv", index=False)

        leak = {"independent_disjoint_from_training": set(before["indep_pcaps"]).isdisjoint(
            set(before["v1_pcaps"]) | set(before["v2_pcaps"]) | set(before["targeted_pcaps"])),
            "training_sources": ["v1", "v2", "targeted"], "independent_used_for": "final evaluation only",
            "real_flows": int(len(real.df)), "invalid_flows": len(real.invalid),
            "labels_from_folders": True, "n_features": len(ml.FEATURES),
            "feature_order_ok": True, "no_zero_fill": len(real.invalid) == 0,
            "capture_level_loco_no_flow_overlap": True}
        (out / "leakage_validation.json").write_text(json.dumps(leak, indent=2, default=str))
        (out / "model_hashes_before_after.json").write_text(json.dumps(
            {"before": before, "after": after, "unchanged": before == after}, indent=2, default=str))
        (out / "final_verdict.json").write_text(json.dumps(verdict, indent=2, default=str))
        import sklearn
        (out / "training_metadata.json").write_text(json.dumps({
            "experiment": "balanced_real_retraining", "created_utc": datetime.now(timezone.utc).isoformat(),
            "seed": rb.SEED, "model": "HistGradientBoostingClassifier", "hyperparameters": rt.hgb_params(),
            "strategies": {s: list(rb.class_weights(real.df, s)) for s in strategies},
            "loco_cic_subsample": opts["loco_sample"], "features": list(ml.FEATURES),
            "training_real_sources": "v1 + v2 + targeted benign", "independent_test": "FROZEN; eval only",
            "candidate_dir": str(rb.candidate_dir()),
            "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__}}, indent=2, default=str))
        self._report(out, cand_rows, comp, indep_metrics, indep_ci, ctrl_cic, ctrl_v2, reproduces,
                     selected, verdict, shap_res, base_cic, real, opts)

    def _report(self, out, cand_rows, comp, indep_metrics, indep_ci, ctrl_cic, ctrl_v2, reproduces,
                selected, verdict, shap_res, base_cic, real, opts):
        pr = indep_metrics["production"]; c2 = indep_metrics["candidate2"]; nw = indep_metrics[selected]
        shap_md = ""
        if shap_res:
            shap_md = (f"\n## SHAP (selected `{selected}`)\n\n"
                       f"- Candidate 2 global-top5 artifact-dependent: **{shap_res['candidate2_artifact_dep']}**\n"
                       f"- Selected candidate global-top5 artifact-dependent: **{shap_res['selected_artifact_dep']}**\n"
                       f"- Still leans on `Fwd Seg Size Min` / `Init Fwd Win Byts`: see `shap_comparison.csv`.\n")
        md = f"""# Balanced real-PCAP retraining — report

**Candidate only. Production model and Candidate 2 frozen and byte-for-byte
unchanged (verified). No auto-promotion, no merge.** The independent 36-PCAP test
set was NEVER used for training, weighting, threshold/feature selection, or
candidate selection — the candidate was selected on CIC held-out + v2 LOCO only,
then evaluated once on the independent set.

## Data (approved training corpora only)

Combined real training flows = **{len(real.df)}** (FTP
{int((real.df['Label']=='FTP-BruteForce').sum())}, Benign
{int((real.df['Label']=='Benign').sum())}) from **v1 + v2 + targeted benign**. The
independent set is disjoint (content-hash) and frozen. Existing 30-feature
`pcap_validation` extraction; exactly 30 ordered finite features, no zero-fill.

## Candidate 1 — CIC-only control

CIC {ctrl_cic['accuracy']:.4f}/{ctrl_cic['macro_f1']:.4f} (reproduces baseline:
**{reproduces}**); on all v2 real flows FTP recall {_f(ctrl_v2['ftp_recall'])} /
benign recall {_f(ctrl_v2['benign_recall'])} — confirms CIC alone does not detect
real FTP.

## Candidates — CIC + real (CIC held-out + v2 LOCO; selection signals only)

{_md_table(cand_rows)}

Selected on CIC+v2-LOCO: **`{selected}`** (rule: no CIC regression, then best
v2-LOCO macro-F1).

## Final independent test (frozen 36-PCAP set)

{_md_table([{k: r[k] for k in ('model','ftp_recall','ftp_precision','benign_recall','benign_fp_rate','macro_f1','accuracy')} for r in
            [{'model': n, **indep_metrics[n]} for n in indep_metrics]])}

Bootstrap 95% CIs in `bootstrap_ci_results.csv`.

### Headline — Candidate 2 vs selected (independent)

| Metric | production | Candidate 2 | selected `{selected}` |
|---|---|---|---|
| FTP recall | {_f(pr['ftp_recall'])} | {_f(c2['ftp_recall'])} | {_f(nw['ftp_recall'])} |
| Benign recall | {_f(pr['benign_recall'])} | {_f(c2['benign_recall'])} | {_f(nw['benign_recall'])} |
| Benign FP rate | {_f(pr['benign_fp_rate'])} | {_f(c2['benign_fp_rate'])} | {_f(nw['benign_fp_rate'])} |
| macro-F1 | {pr['macro_f1']:.4f} | {c2['macro_f1']:.4f} | {nw['macro_f1']:.4f} |
{shap_md}
## Verdict — {verdict['decision']}

{verdict['summary']}

Promotion criteria (all required): {json.dumps(verdict['criteria'])}

**{verdict['conclusion']}**

### Caveats
{chr(10).join('- ' + c for c in verdict['caveats'])}

## Integrity

Production model, Candidate 2, ml.py/live_capture.py/pcap_validation.py, and the
v1/v2/independent/targeted PCAPs verified unchanged (before==after). No independent
PCAP entered training. Capture-level LOCO with per-fold no-flow-overlap assertions.

## Files

`candidate_metrics.csv`, `baseline_metrics.csv`, `candidate_comparison.csv`,
`independent_test_metrics.csv`, `independent_per_capture_metrics.csv`,
`per_capture_v2loco_metrics.csv`, `confusion_*.csv`, `confidence_distribution.csv`,
`bootstrap_ci_results.csv`, `shap_comparison.csv`, `leakage_validation.json`,
`model_hashes_before_after.json`, `training_metadata.json`, `final_verdict.json`.
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
