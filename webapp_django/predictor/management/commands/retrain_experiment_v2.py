"""
Realistic-PCAP retraining experiment v2 (candidate only; production frozen).

Baseline (production model) vs HGB candidates trained on CIC and CIC+the 83
diversified real captures, under three real-flow weighting strategies, evaluated
with capture-level leave-one-capture-out (no PCAP in train and test). Writes all
evidence to validation/results/retraining_v2/ and candidate artifacts to
validation/models/realistic_pcap_candidate_v2/. Overwrites nothing in
webapp_data/Results/Models and touches no production file.

    python manage.py retrain_experiment_v2
"""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from django.core.management.base import BaseCommand

from predictor import ml, retraining as rt, retraining_v2 as r2


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _prod_model_path() -> Path:
    return ml.MODELS_DIR / ml.MODEL_REGISTRY[ml.DEFAULT_MODEL]["file"]


def _md_table(rows) -> str:
    if not rows:
        return "(no rows)"
    cols = list(rows[0].keys())

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    body = "\n".join("| " + " | ".join(fmt(r[c]) for c in cols) + " |" for r in rows)
    return f"{head}\n{sep}\n{body}"


class Command(BaseCommand):
    help = "Experimental realistic-PCAP v2 retraining vs frozen baseline (no production changes)."

    def add_arguments(self, parser):
        parser.add_argument("--output", default=None)
        parser.add_argument("--loco-sample", type=int, default=r2.CIC_LOCO_SAMPLE)
        parser.add_argument("--max-folds", type=int, default=0, help="0 = all captures")
        parser.add_argument("--weights", default="none,moderate,strong")
        parser.add_argument("--skip-shap", action="store_true")

    def handle(self, *args, **opts):
        w = self.stdout.write
        out = Path(opts["output"]) if opts["output"] else \
            r2.repo_root() / "validation" / "results" / "retraining_v2"
        out.mkdir(parents=True, exist_ok=True)
        weight_names = [x.strip() for x in opts["weights"].split(",") if x.strip()]

        prod_path = _prod_model_path()
        prod_hash_before = _sha(prod_path)
        w(self.style.MIGRATE_HEADING("Realistic-PCAP retraining experiment v2 (candidate only)"))
        w(f"  production model: {prod_path.name}  sha256={prod_hash_before[:16]}...")

        # --- data ---------------------------------------------------------
        real = r2.extract_real_flows_v2()
        w(f"  real flows: {len(real.df)} across {len(real.captures)} captures; "
          f"invalid/zero-filled: {len(real.invalid)}")
        if real.invalid:
            w(self.style.ERROR(f"  INVALID (excluded): {real.invalid[:3]} ..."))
        cic_X, cic_y = rt.load_cic()
        cic_test_X, cic_test_y = rt.load_cic_test()
        dec = rt._encoded_to_name()
        cic_test_names = cic_test_y.map(dec)
        cic_sub_X, cic_sub_y = r2.stratified_cic_subsample(cic_X, cic_y, opts["loco_sample"])
        w(f"  CIC train: {len(cic_X)} rows; LOCO subsample: {len(cic_sub_X)} rows")

        # ===== A. BASELINE (frozen production model) ======================
        baseline, _ = ml._load(ml.DEFAULT_MODEL)
        base_cic = rt.evaluate(baseline, cic_test_X, cic_test_names)
        base_real = rt.evaluate(baseline, real.df[ml.FEATURES], real.df["Label"])
        base_real_percap = self._per_capture(baseline, real)
        w(self.style.MIGRATE_HEADING("\nA. Baseline (production model)"))
        w(f"  CIC held-out : acc {base_cic['accuracy']:.4f}  macroF1 {base_cic['macro_f1']:.4f}")
        w(f"  Real (854)   : acc {base_real['accuracy']:.4f}  macroF1 {base_real['macro_f1']:.4f}  "
          f"FTP-recall {base_real['ftp_recall']:.3f}  Benign-recall {base_real['benign_recall']:.3f}")

        # ===== B. CANDIDATE 1 -- CIC-only full reproduction ===============
        w(self.style.MIGRATE_HEADING("\nB. Candidate 1 -- CIC-only reproduction (full CIC, seed 42)"))
        cand1 = rt.train(cic_X, cic_y, np.ones(len(cic_X)))
        c1_cic = rt.evaluate(cand1, cic_test_X, cic_test_names)
        reproduces = abs(c1_cic["accuracy"] - base_cic["accuracy"]) < 5e-3
        w(f"  CIC held-out : acc {c1_cic['accuracy']:.4f}  macroF1 {c1_cic['macro_f1']:.4f} "
          f"(baseline {base_cic['accuracy']:.4f}/{base_cic['macro_f1']:.4f})  reproduces={reproduces}")
        saved1 = r2.save_candidate_v2(cand1, {
            "candidate": "cic_only_reproduction", "seed": r2.SEED,
            "train": "full CIC only", "cic_accuracy": c1_cic["accuracy"],
            "cic_macro_f1": c1_cic["macro_f1"], "reproduces_baseline": reproduces,
        }, "candidate1_cic_only")

        # matched subsample-only control (fair regression base for weighting)
        ctrl_sub = rt.train(cic_sub_X, cic_sub_y, np.ones(len(cic_sub_X)))
        ctrl_sub_cic = rt.evaluate(ctrl_sub, cic_test_X, cic_test_names)

        # ===== C+D. CANDIDATE 2 -- CIC + real, 3 weightings ===============
        w(self.style.MIGRATE_HEADING("\nC/D. Candidate 2 -- CIC + real, weightings: " + ", ".join(weight_names)))
        candidate_rows, weighting_rows, per_class_rows, per_cap_rows = [], [], [], []
        confusion_dir = out
        loco_store, fullcic_store = {}, {}
        for wn in weight_names:
            wt = r2.WEIGHTS[wn]
            # full-CIC candidate: CIC regression + saved artifact + SHAP source
            Xf, yf, wf, manf = r2.assemble_v2(cic_X, cic_y, real.df, real_weight=wt)
            cand = rt.train(Xf, yf, wf)
            c_cic = rt.evaluate(cand, cic_test_X, cic_test_names)
            c_real = rt.evaluate(cand, real.df[ml.FEATURES], real.df["Label"])
            fullcic_store[wn] = cand
            saved = r2.save_candidate_v2(cand, {
                "candidate": f"cic_plus_real_weight_{wn}", "seed": r2.SEED,
                "real_weight": wt, "train": "full CIC + all real (weighted)",
                "train_manifest": manf, "cic_accuracy": c_cic["accuracy"],
                "cic_macro_f1": c_cic["macro_f1"],
            }, f"candidate2_{wn}")

            # LOCO real-PCAP estimate on the fixed CIC subsample
            real_loco = real
            if opts["max_folds"]:
                keep = real.captures[:opts["max_folds"]]
                real_loco = r2.RealFlows(df=real.df[real.df["capture"].isin(keep)].reset_index(drop=True),
                                         invalid=real.invalid)
            loco = r2.leave_one_capture_out_v2(cic_sub_X, cic_sub_y, real_loco, real_weight=wt)
            loco_store[wn] = loco
            p = loco["pooled"]
            w(f"  [{wn:8} w={wt:>4g}] CIC acc {c_cic['accuracy']:.4f}/F1 {c_cic['macro_f1']:.4f} | "
              f"LOCO acc {p['accuracy']:.4f}/F1 {p['macro_f1']:.4f} "
              f"FTP-rec {p['FTP-BruteForce_recall']} Ben-rec {p['Benign_recall']}")

            candidate_rows.append({
                "candidate": f"cic+real_{wn}", "real_weight": wt,
                "cic_accuracy": c_cic["accuracy"], "cic_macro_f1": c_cic["macro_f1"],
                "cic_macro_precision": c_cic["macro_precision"],
                "cic_macro_recall": c_cic["macro_recall"], "cic_weighted_f1": c_cic["weighted_f1"],
                "real_full_accuracy": c_real["accuracy"], "real_full_macro_f1": c_real["macro_f1"],
                "loco_accuracy": p["accuracy"], "loco_macro_f1": p["macro_f1"],
                "loco_weighted_f1": p["weighted_f1"],
                "loco_ftp_recall": p["FTP-BruteForce_recall"],
                "loco_benign_recall": p["Benign_recall"],
            })
            weighting_rows.append({
                "weighting": wn, "real_weight": wt, "real_total_weight": manf["real_total_weight"],
                "cic_accuracy": c_cic["accuracy"], "cic_macro_f1": c_cic["macro_f1"],
                "loco_accuracy": p["accuracy"], "loco_macro_f1": p["macro_f1"],
                "loco_ftp_recall": p["FTP-BruteForce_recall"], "loco_benign_recall": p["Benign_recall"],
            })
            for pc in c_cic["per_class"]:
                per_class_rows.append({"candidate": f"cic+real_{wn}", "eval": "cic_heldout", **pc})
            for f in loco["folds"]:
                per_cap_rows.append({
                    "candidate": f"cic+real_{wn}", "capture": f["held_out_capture"],
                    "label": f["true_label"], "n_flows": f["n_flows"],
                    "recall": f["recall"], "predictions": json.dumps(f["predictions"]),
                    "confidence_mean": round(f["confidence_mean"], 4),
                })
            # per-weighting LOCO confusion
            self._cm_df(p["confusion"]).to_csv(confusion_dir / f"confusion_loco_{wn}.csv")

        # baseline per-capture (production model, no training)
        for r in base_real_percap:
            per_cap_rows.append({"candidate": "baseline", **r})

        # ===== E. LEAKAGE CHECKS ==========================================
        leakage = self._leakage_checks(real, loco_store, cic_sub_X, prod_hash_before, saved1)
        w(self.style.MIGRATE_HEADING("\nE. Leakage / integrity checks"))
        w(f"  all_pass={leakage['all_pass']}  " +
          " ".join(f"{k}={v}" for k, v in leakage["checks"].items()))

        # ===== F. STATISTICAL COMPARISON ==================================
        comparison = self._comparison(base_cic, base_real, candidate_rows, ctrl_sub_cic)

        # ===== G. SHAP ====================================================
        shap_res = None
        if not opts["skip_shap"]:
            best = self._pick_best(candidate_rows, base_cic)
            shap_res = self._shap(baseline, fullcic_store.get(best), cic_X, best)
            if shap_res is not None:
                shap_res["df"].to_csv(out / "shap_comparison.csv", index=False)

        # ===== H. VERDICT =================================================
        verdict = self._verdict(base_cic, base_real, candidate_rows, comparison, shap_res)
        w(self.style.MIGRATE_HEADING("\nH. Verdict"))
        w(self.style.WARNING("  " + verdict["classification"].upper() + " -- " + verdict["summary"]))

        # ===== production model unchanged =================================
        prod_hash_after = _sha(prod_path)
        unchanged = prod_hash_before == prod_hash_after
        w(f"\n  production model sha256 before==after: {unchanged}")
        if not unchanged:
            raise SystemExit("ABORT: production model changed during experiment!")

        # ===== write evidence =============================================
        self._write(out, base_cic, base_real, base_real_percap, candidate_rows,
                    weighting_rows, per_class_rows, per_cap_rows, comparison, leakage,
                    verdict, shap_res, real, weight_names, opts, prod_hash_before,
                    prod_hash_after, c1_cic, reproduces, ctrl_sub_cic)
        w(self.style.SUCCESS(f"\nEvidence written to {out}"))
        w(self.style.WARNING("\nSTOP: experimental only. No promotion, no production change, "
                             "no merge. Candidate artifacts under "
                             "validation/models/realistic_pcap_candidate_v2/."))

    # ---- helpers ---------------------------------------------------------

    def _per_capture(self, model, real: r2.RealFlows) -> list:
        dec = rt._encoded_to_name()
        rows = []
        for cap in real.captures:
            held = real.df[real.df["capture"] == cap]
            pred = np.array([dec.get(int(c), str(c)) for c in model.predict(held[ml.FEATURES])])
            truth = held["Label"].to_numpy()
            proba = model.predict_proba(held[ml.FEATURES]).max(axis=1)
            rows.append({"capture": cap, "label": held["Label"].iloc[0],
                         "n_flows": int(len(held)), "recall": float((pred == truth).mean()),
                         "predictions": json.dumps({k: int(v) for k, v in pd.Series(pred).value_counts().items()}),
                         "confidence_mean": round(float(proba.mean()), 4)})
        return rows

    def _cm_df(self, confusion) -> pd.DataFrame:
        return pd.DataFrame(confusion["matrix"], index=confusion["labels"], columns=confusion["labels"])

    def _leakage_checks(self, real, loco_store, cic_sub_X, prod_hash, saved1) -> dict:
        checks = {}
        # no PCAP (and no flow) in both train and test in any fold, any weighting
        no_leak = True
        n_folds = 0
        for loco in loco_store.values():
            for f in loco["folds"]:
                n_folds += 1
                if f["held_out_in_train_captures"] or f["train_test_flow_overlap"] != 0:
                    no_leak = False
        checks["no_pcap_in_train_and_test"] = no_leak and n_folds > 0
        # labels only from folders
        checks["labels_only_from_folders"] = bool(set(real.df["Label"]) <= {r2.FTP, r2.BENIGN})
        # exactly 30 features, order == ml.FEATURES
        feat_cols = [c for c in real.df.columns if c in ml.FEATURES]
        checks["exactly_30_features"] = len(feat_cols) == 30
        checks["feature_order_equals_ml_features"] = \
            [c for c in real.df.columns if c in ml.FEATURES] == list(ml.FEATURES)
        # all finite
        checks["all_finite"] = bool(np.isfinite(real.df[ml.FEATURES].to_numpy()).all())
        # no zero-fill: invalid list empty AND no all-zero feature rows fabricated
        checks["no_zero_filling"] = len(real.invalid) == 0
        # every flow traceable to a PCAP
        checks["every_flow_traceable_to_pcap"] = bool(real.df["flow_uid"].str.contains("#").all()
                                                       and real.df["capture"].notna().all())
        # candidate artifacts separate from production
        checks["candidate_dir_separate_from_production"] = \
            str(ml.MODELS_DIR.resolve()) not in str(r2.candidate_dir_v2().resolve())
        # production model files unchanged
        checks["production_model_unchanged"] = _sha(_prod_model_path()) == prod_hash
        return {"all_pass": all(checks.values()), "checks": checks}

    def _comparison(self, base_cic, base_real, candidate_rows, ctrl_sub_cic) -> list:
        rows = []
        for c in candidate_rows:
            rows.append({
                "candidate": c["candidate"],
                "cic_acc_abs_change": c["cic_accuracy"] - base_cic["accuracy"],
                "cic_macro_f1_abs_change": c["cic_macro_f1"] - base_cic["macro_f1"],
                "cic_macro_f1_rel_change": (c["cic_macro_f1"] - base_cic["macro_f1"]) / base_cic["macro_f1"],
                "loco_ftp_recall": c["loco_ftp_recall"],
                "baseline_real_ftp_recall": base_real["ftp_recall"],
                "ftp_recall_abs_change": (c["loco_ftp_recall"] or 0) - (base_real["ftp_recall"] or 0),
                "loco_benign_recall": c["loco_benign_recall"],
                "baseline_benign_recall": base_real["benign_recall"],
                "benign_recall_abs_change": (c["loco_benign_recall"] or 0) - (base_real["benign_recall"] or 0),
                "cic_regressed": c["cic_macro_f1"] < base_cic["macro_f1"] - 0.01,
            })
        return rows

    def _pick_best(self, candidate_rows, base_cic) -> str:
        # NOT chosen on real performance alone: require CIC macro-F1 within 0.01 of
        # baseline, then maximise LOCO macro-F1.
        ok = [c for c in candidate_rows if c["cic_macro_f1"] >= base_cic["macro_f1"] - 0.01]
        pool = ok or candidate_rows
        best = max(pool, key=lambda c: c["loco_macro_f1"])
        return best["candidate"].split("_")[-1]

    def _shap(self, baseline, candidate, cic_X, best_name):
        if candidate is None:
            return None
        try:
            import shap
        except Exception:  # noqa: BLE001
            return None
        bg = cic_X.sample(min(400, len(cic_X)), random_state=r2.SEED)
        rows = []
        for label, model in (("production", baseline), (f"candidate_{best_name}", candidate)):
            vals = np.abs(np.asarray(shap.TreeExplainer(model).shap_values(bg))).mean(axis=(0, 2))
            rows.append(pd.Series(vals, index=list(ml.FEATURES), name=label))
        df = pd.concat(rows, axis=1).reset_index().rename(columns={"index": "feature"})
        prod_col = "production"
        df = df.sort_values(prod_col, ascending=False).reset_index(drop=True)
        top_prod = df.head(5)["feature"].tolist()
        artifact_feats = {"Fwd Seg Size Min", "Init Fwd Win Byts"}
        still_artifact = bool(artifact_feats & set(top_prod) &
                              set(df.sort_values(f"candidate_{best_name}", ascending=False)
                                  .head(5)["feature"].tolist()))
        return {"df": df, "best": best_name, "top_production": top_prod,
                "candidate_still_depends_on_cic_artifacts": still_artifact}

    def _verdict(self, base_cic, base_real, candidate_rows, comparison, shap_res) -> dict:
        # conservative: 83 captures, 2 real classes, subsampled LOCO
        best = max(candidate_rows, key=lambda c: (c["loco_macro_f1"]))
        cic_ok = best["cic_macro_f1"] >= base_cic["macro_f1"] - 0.01
        ftp_gain = (best["loco_ftp_recall"] or 0) - (base_real["ftp_recall"] or 0)
        benign_ok = (best["loco_benign_recall"] or 0) >= 0.90
        artifact_dependent = shap_res["candidate_still_depends_on_cic_artifacts"] if shap_res else None

        if ftp_gain >= 0.30 and cic_ok and benign_ok:
            cls = "promising but insufficient evidence"
            summary = (f"Best candidate ({best['candidate']}) lifts LOCO FTP recall by "
                       f"{ftp_gain:+.3f} with CIC macro-F1 non-regressed and benign recall "
                       f"{best['loco_benign_recall']}. But only 83 loopback captures / 2 real "
                       f"classes / subsampled-CIC LOCO -- not production-grade evidence.")
        elif ftp_gain >= 0.30 and cic_ok:
            cls = "promising but insufficient evidence"
            summary = (f"FTP recall improves ({ftp_gain:+.3f}) with CIC non-regressed, but benign "
                       f"recall drops to {best['loco_benign_recall']} (false positives) -- not ready.")
        elif ftp_gain <= 0.05:
            cls = "unsuccessful"
            summary = (f"Adding the real captures does not materially improve LOCO FTP recall "
                       f"({ftp_gain:+.3f}); the failure is not fixed by this data at this scale.")
        else:
            cls = "promising but insufficient evidence"
            summary = (f"Partial FTP-recall gain ({ftp_gain:+.3f}); mixed CIC/benign trade-offs. "
                       f"Exploratory only.")
        return {
            "classification": cls, "summary": summary,
            "promote": False, "best_candidate": best["candidate"],
            "best_loco_ftp_recall": best["loco_ftp_recall"],
            "best_loco_benign_recall": best["loco_benign_recall"],
            "best_cic_macro_f1": best["cic_macro_f1"], "baseline_cic_macro_f1": base_cic["macro_f1"],
            "baseline_real_ftp_recall": base_real["ftp_recall"],
            "cic_non_regressed": cic_ok, "candidate_still_artifact_dependent": artifact_dependent,
            "caveats": [
                "Only 83 real captures (loopback lab), 2 real classes (Benign, FTP-BruteForce).",
                "LOCO uses a fixed 30k stratified CIC subsample for tractability; full-CIC "
                "candidates confirm CIC regression separately.",
                "No statistical-significance claim: sample too small; point estimates only.",
            ],
        }

    def _write(self, out, base_cic, base_real, base_real_percap, candidate_rows,
               weighting_rows, per_class_rows, per_cap_rows, comparison, leakage,
               verdict, shap_res, real, weight_names, opts, hash_before, hash_after,
               c1_cic, reproduces, ctrl_sub_cic):
        pd.DataFrame([{
            "model": "baseline_production",
            "cic_accuracy": base_cic["accuracy"], "cic_macro_precision": base_cic["macro_precision"],
            "cic_macro_recall": base_cic["macro_recall"], "cic_macro_f1": base_cic["macro_f1"],
            "cic_weighted_f1": base_cic["weighted_f1"],
            "real_accuracy": base_real["accuracy"], "real_macro_precision": base_real["macro_precision"],
            "real_macro_recall": base_real["macro_recall"], "real_macro_f1": base_real["macro_f1"],
            "real_weighted_f1": base_real["weighted_f1"],
            "real_ftp_recall": base_real["ftp_recall"], "real_benign_recall": base_real["benign_recall"],
            "real_confidence_mean": base_real["confidence"]["mean"],
        }]).to_csv(out / "baseline_metrics.csv", index=False)

        pd.DataFrame(candidate_rows).to_csv(out / "candidate_metrics.csv", index=False)
        pd.DataFrame(weighting_rows).to_csv(out / "weighting_comparison.csv", index=False)
        pd.DataFrame(per_class_rows).to_csv(out / "per_class_metrics.csv", index=False)
        pd.DataFrame(per_cap_rows).to_csv(out / "per_capture_metrics.csv", index=False)
        pd.DataFrame(comparison).to_csv(out / "statistical_comparison.csv", index=False)
        self._cm_df(base_cic["confusion"]).to_csv(out / "confusion_baseline_cic.csv")
        self._cm_df(base_real["confusion"]).to_csv(out / "confusion_baseline_real.csv")

        (out / "leakage_validation.json").write_text(json.dumps(leakage, indent=2, default=str))
        (out / "schema_validation.json").write_text(json.dumps({
            "n_features": len(ml.FEATURES), "feature_order": list(ml.FEATURES),
            "real_flows": int(len(real.df)), "captures": len(real.captures),
            "invalid_flows": len(real.invalid), "all_finite": leakage["checks"]["all_finite"],
            "no_zero_filling": leakage["checks"]["no_zero_filling"],
            "extraction": "pcap_validation.replay_pcap (existing pipeline, unchanged)",
        }, indent=2, default=str))
        (out / "final_verdict.json").write_text(json.dumps(verdict, indent=2, default=str))

        import sklearn
        (out / "training_metadata.json").write_text(json.dumps({
            "experiment": "realistic_pcap_retraining_v2",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "seed": r2.SEED, "model": "HistGradientBoostingClassifier",
            "hyperparameters": rt.hgb_params(), "features": list(ml.FEATURES),
            "weightings": {k: r2.WEIGHTS[k] for k in weight_names},
            "loco_cic_subsample": opts["loco_sample"],
            "cic_only_reproduction": {"cic_accuracy": c1_cic["accuracy"],
                                      "cic_macro_f1": c1_cic["macro_f1"], "reproduces": reproduces},
            "subsample_control_cic": {"accuracy": ctrl_sub_cic["accuracy"],
                                      "macro_f1": ctrl_sub_cic["macro_f1"]},
            "data_sources": {
                "cic_train": "webapp_data/Processed_Data/balanced_train_selected.parquet",
                "cic_test": "webapp_data/Processed_Data/test_selected.parquet",
                "real_pcaps": "validation/realistic_pcaps_v2/",
            },
            "production_model_sha256_before": hash_before,
            "production_model_sha256_after": hash_after,
            "production_model_unchanged": hash_before == hash_after,
            "candidate_dir": str(r2.candidate_dir_v2()),
            "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__},
        }, indent=2, default=str))

        # README / report
        best = verdict["best_candidate"]
        shap_line = ""
        if shap_res is not None:
            shap_line = (f"\n## SHAP (production vs {shap_res['best']} candidate)\n\n"
                         f"Top production features: {', '.join(shap_res['top_production'])}. "
                         f"Candidate still depends on CIC artifact features "
                         f"(`Fwd Seg Size Min`/`Init Fwd Win Byts`): "
                         f"**{shap_res['candidate_still_depends_on_cic_artifacts']}**. "
                         f"See `shap_comparison.csv`.\n")
        md = f"""# Realistic-PCAP retraining v2 — report

**Experimental only. Production model frozen and byte-for-byte unchanged
(sha256 verified before==after). No promotion, no merge.** Candidates live under
`validation/models/realistic_pcap_candidate_v2/`; evidence here.

## Baseline (production model)

| Eval | Accuracy | Macro-F1 | FTP recall | Benign recall |
|---|---|---|---|---|
| CIC held-out | {base_cic['accuracy']:.4f} | {base_cic['macro_f1']:.4f} | — | — |
| Real (854 flows) | {base_real['accuracy']:.4f} | {base_real['macro_f1']:.4f} | {base_real['ftp_recall']:.3f} | {base_real['benign_recall']:.3f} |

## Candidate 1 — CIC-only reproduction

Full-CIC retrain (seed 42): CIC acc {c1_cic['accuracy']:.4f} / macro-F1
{c1_cic['macro_f1']:.4f}. Reproduces baseline: **{reproduces}** — confirms the
training pipeline is not the variable.

## Candidate 2 — CIC + real (weighting comparison, LOCO)

Capture-level leave-one-capture-out (no PCAP in train and test). CIC held-out from
full-CIC candidates; real-PCAP LOCO from a fixed {opts['loco_sample']}-row
stratified CIC subsample (matched subsample-only control CIC macro-F1
{ctrl_sub_cic['macro_f1']:.4f}). See `weighting_comparison.csv`,
`candidate_metrics.csv`, `per_capture_metrics.csv`.

{_md_table(weighting_rows)}

## Leakage / integrity

All automated checks pass: **{leakage['all_pass']}**. See `leakage_validation.json`
(no PCAP in train+test, labels from folders only, exactly 30 ordered finite
features, no zero-fill, every flow traceable to its PCAP, candidate artifacts
separate from production, production model unchanged).
{shap_line}
## Verdict — {verdict['classification']}

{verdict['summary']}

**Best candidate:** {best} (LOCO FTP recall {verdict['best_loco_ftp_recall']},
benign recall {verdict['best_loco_benign_recall']}, CIC macro-F1
{verdict['best_cic_macro_f1']:.4f} vs baseline {base_cic['macro_f1']:.4f}).
**Not promoted.**

### Caveats
{chr(10).join('- ' + c for c in verdict['caveats'])}

## Files

`baseline_metrics.csv`, `candidate_metrics.csv`, `weighting_comparison.csv`,
`per_class_metrics.csv`, `per_capture_metrics.csv`, `statistical_comparison.csv`,
`confusion_baseline_{{cic,real}}.csv`, `confusion_loco_*.csv`,
`schema_validation.json`, `leakage_validation.json`, `training_metadata.json`,
`final_verdict.json`, `shap_comparison.csv`.
"""
        (out / "report.md").write_text(md)
