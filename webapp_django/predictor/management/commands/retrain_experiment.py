"""
Realistic-PCAP retraining experiment (candidate only; baseline frozen).

Trains HGB candidates on CIC + real FTP flows (same tuned hyperparameters and 30
features as the baseline, so only the data differs), evaluates them with
leave-one-capture-out on the real PCAPs (no leakage) and on CIC held-out (for
regression), runs a SHAP comparison, and writes all evidence to
validation/results/retraining/. Overwrites nothing in webapp_data/Results/Models.

    python manage.py retrain_experiment --output ../validation/results/retraining
"""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from django.core.management.base import BaseCommand

from predictor import ml, retraining as rt


class Command(BaseCommand):
    help = "Experimental realistic-PCAP retraining vs frozen baseline (no production changes)."

    def add_arguments(self, parser):
        parser.add_argument("--output", default=None)
        parser.add_argument("--ftp-weight", type=float, default=rt.FTP_UPWEIGHT)

    def handle(self, *args, **opts):
        w = self.stdout.write
        out = Path(opts["output"]) if opts["output"] else rt.repo_root() / "validation" / "results" / "retraining"
        out.mkdir(parents=True, exist_ok=True)
        W = opts["ftp_weight"]

        w(self.style.MIGRATE_HEADING("Realistic-PCAP retraining experiment (candidate only)"))

        # --- data ---------------------------------------------------------
        real = rt.extract_real_flows()
        w(f"  real flows: {len(real.df)} across {len(real.captures)} captures; "
          f"invalid/zero-filled: {len(real.invalid)}")
        if real.invalid:
            w(self.style.ERROR(f"  INVALID (excluded, not zero-filled): {real.invalid}"))
        cic_X, cic_y = rt.load_cic()
        cic_test_X, cic_test_y = rt.load_cic_test()
        dec = rt._encoded_to_name()
        cic_test_names = cic_test_y.map(dec)

        # --- BASELINE (frozen saved model) --------------------------------
        baseline, _ = ml._load("histgradientboosting")
        base_cic = rt.evaluate(baseline, cic_test_X, cic_test_names)
        base_real = rt.evaluate(baseline, real.df[ml.FEATURES], real.df["Label"])
        w(self.style.MIGRATE_HEADING("\nBaseline (frozen saved model)"))
        w(f"  CIC held-out : acc {base_cic['accuracy']:.4f}  macroF1 {base_cic['macro_f1']:.4f}")
        w(f"  Real PCAP    : acc {base_real['accuracy']:.4f}  macroF1 {base_real['macro_f1']:.4f}  "
          f"FTP recall {base_real['ftp_recall']:.3f}  Benign recall {base_real['benign_recall']:.3f}")

        # --- CONTROL: retrain on CIC only (should reproduce baseline) -----
        Xc, yc, wc, _ = rt.assemble_training(cic_X, cic_y, real.df.iloc[0:0], ftp_weight=1.0)
        control = rt.train(Xc, yc, wc)
        ctrl_cic = rt.evaluate(control, cic_test_X, cic_test_names)
        w(self.style.MIGRATE_HEADING("\nControl (retrain on CIC only — isolates the training pipeline)"))
        w(f"  CIC held-out : acc {ctrl_cic['accuracy']:.4f}  macroF1 {ctrl_cic['macro_f1']:.4f} "
          f"(baseline {base_cic['accuracy']:.4f}/{base_cic['macro_f1']:.4f})")

        # --- CANDIDATE B: CIC + real, UNWEIGHTED --------------------------
        Xb, yb, wb, man_b = rt.assemble_training(cic_X, cic_y, real.df, ftp_weight=1.0)
        cand_b = rt.train(Xb, yb, wb)
        b_cic = rt.evaluate(cand_b, cic_test_X, cic_test_names)
        logo_b = rt.leave_one_capture_out(cic_X, cic_y, _ftp_only(real), ftp_weight=1.0)
        w(self.style.MIGRATE_HEADING("\nCandidate B — CIC + real (unweighted)"))
        w(f"  CIC held-out : acc {b_cic['accuracy']:.4f}  macroF1 {b_cic['macro_f1']:.4f}")
        w(f"  LOGO real FTP recall (held-out): {logo_b['pooled'].get('FTP-BruteForce_recall')}")

        # --- CANDIDATE C: CIC + real, FTP UPWEIGHTED (saved artifact) -----
        Xcc, ycc, wcc, man_c = rt.assemble_training(cic_X, cic_y, real.df, ftp_weight=W)
        cand_c = rt.train(Xcc, ycc, wcc)
        c_cic = rt.evaluate(cand_c, cic_test_X, cic_test_names)
        logo_c = rt.leave_one_capture_out(cic_X, cic_y, real, ftp_weight=W)
        cand_real = _logo_pooled_metrics(logo_c)
        w(self.style.MIGRATE_HEADING(f"\nCandidate C — CIC + real (FTP upweighted x{W:g})"))
        w(f"  CIC held-out : acc {c_cic['accuracy']:.4f}  macroF1 {c_cic['macro_f1']:.4f}")
        w(f"  LOGO (held-out) real: acc {cand_real['accuracy']:.4f}  macroF1 {cand_real['macro_f1']:.4f}  "
          f"FTP recall {cand_real['ftp_recall']} Benign recall {cand_real['benign_recall']}")
        for f in logo_c["folds"]:
            w(f"    held-out {f['held_out_capture']:16} ({f['true_label']:14}) "
              f"n={f['n_flows']:2} recall={f['recall']:.3f}  preds={f['predictions']}")

        # --- save candidate C artifact ------------------------------------
        metadata = _metadata(W, man_c, base_cic, c_cic, cand_real, real)
        saved = rt.save_candidate(cand_c, metadata)
        w(self.style.SUCCESS(f"\n  candidate saved: {saved['model']}"))

        # --- SHAP comparison (baseline vs candidate C) --------------------
        shap_cmp = _shap_comparison(baseline, cand_c, cic_X)

        # --- baseline_vs_candidate table ----------------------------------
        table = _comparison_table(base_cic, c_cic, base_real, cand_real)
        w(self.style.MIGRATE_HEADING("\nBaseline vs Candidate C"))
        w(f"  {'Metric':26}{'Baseline':>12}{'Candidate':>12}{'Change':>10}")
        for r in table:
            w(f"  {r['metric']:26}{_fmt(r['baseline']):>12}{_fmt(r['candidate']):>12}{_fmt(r['change']):>10}")

        # --- write all evidence ------------------------------------------
        self._write_evidence(out, real, base_cic, base_real, ctrl_cic, b_cic, logo_b,
                             c_cic, logo_c, cand_real, table, man_b, man_c, shap_cmp, W, metadata)
        w(self.style.SUCCESS(f"\nEvidence written to {out}"))

        w(self.style.WARNING(
            "\nScientific note: only 5 real PCAPs (3 FTP, 2 Benign; 42 flows). "
            "Exploratory evidence only — not a statistically strong dataset. "
            "Real-PCAP candidate numbers are leave-one-capture-out (held-out); "
            "self-scoring the training flows would be contaminated and is not reported."))

    # -- evidence writer ---------------------------------------------------

    def _write_evidence(self, out, real, base_cic, base_real, ctrl_cic, b_cic, logo_b,
                        c_cic, logo_c, cand_real, table, man_b, man_c, shap_cmp, W, metadata):
        pd.DataFrame(table).to_csv(out / "baseline_vs_candidate.csv", index=False)

        pd.DataFrame([
            {"model": "baseline", **_flat(base_cic, "cic")},
            {"model": "candidate_C_cic+real_wx%g" % W, **_flat(c_cic, "cic")},
            {"model": "control_cic_only", **_flat(ctrl_cic, "cic")},
        ]).to_csv(out / "cic_metrics.csv", index=False)

        pd.DataFrame([
            {"model": "baseline (all 42 flows, self-scored)", **_flat(base_real, "real")},
            {"model": "candidate_C (LOGO held-out)", "real_accuracy": cand_real["accuracy"],
             "real_macro_f1": cand_real["macro_f1"], "real_ftp_recall": cand_real["ftp_recall"],
             "real_benign_recall": cand_real["benign_recall"], "real_n": cand_real["n"]},
            {"model": "candidate_B (LOGO held-out FTP, unweighted)",
             "real_ftp_recall": logo_b["pooled"].get("FTP-BruteForce_recall"),
             "real_n": logo_b["pooled"]["n"]},
        ]).to_csv(out / "real_pcap_metrics.csv", index=False)

        pd.DataFrame(base_cic["per_class"]).to_csv(out / "per_class_metrics.csv", index=False)
        _cm(base_real).to_csv(out / "confusion_matrix_baseline.csv")
        _cm_from_folds(logo_c).to_csv(out / "confusion_matrix_candidate.csv")

        # capture split manifest (which captures train/test in each fold)
        split_rows = []
        for f in logo_c["folds"]:
            split_rows.append({
                "fold_held_out": f["held_out_capture"], "test_label": f["true_label"],
                "test_flows": f["n_flows"],
                "train_captures": ";".join(f["train_manifest"]["included_captures"]),
                "train_cic_rows": f["train_manifest"]["cic_rows"],
                "train_real_rows": f["train_manifest"]["real_rows"],
            })
        pd.DataFrame(split_rows).to_csv(out / "capture_split_manifest.csv", index=False)

        pd.DataFrame([man_b, man_c]).to_csv(out / "training_manifest.csv", index=False)

        _feature_distribution(real).to_csv(out / "feature_distribution_comparison.csv", index=False)

        pd.DataFrame([
            {"model": "baseline_real", **base_real["prediction_distribution"]},
            {"model": "candidate_C_logo", **_logo_pred_dist(logo_c)},
        ]).to_csv(out / "prediction_distribution.csv", index=False)

        pd.DataFrame([
            {"model": "baseline_real", **base_real["confidence"]},
            {"model": "baseline_cic", **base_cic["confidence"]},
            {"model": "candidate_C_cic", **c_cic["confidence"]},
        ]).to_csv(out / "confidence_comparison.csv", index=False)

        if shap_cmp is not None:
            shap_cmp.to_csv(out / "shap_comparison.csv", index=False)

        pd.DataFrame([
            {"experiment": "realistic_pcap_retraining", "flows": len(real.df),
             "captures": len(real.captures), "invalid_flows": len(real.invalid),
             "baseline_real_ftp_recall": base_real["ftp_recall"],
             "candidate_real_ftp_recall": cand_real["ftp_recall"],
             "baseline_cic_acc": base_cic["accuracy"], "candidate_cic_acc": c_cic["accuracy"]},
        ]).to_csv(out / "experiment_summary.csv", index=False)

        (out / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2, default=str))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _ftp_only(real: rt.RealFlows) -> rt.RealFlows:
    return rt.RealFlows(df=real.df[real.df["Label"] == rt.FTP].reset_index(drop=True), invalid=[])


def _logo_pooled_metrics(logo) -> dict:
    truth, pred = [], []
    for f in logo["folds"]:
        truth += [f["true_label"]] * f["n_flows"]
    # rebuild pooled preds from fold prediction distributions
    for f in logo["folds"]:
        for cls, n in f["predictions"].items():
            pred += [cls] * n
    from sklearn.metrics import accuracy_score, f1_score, recall_score
    truth = np.array(truth); pred = np.array(pred)
    return {
        "n": int(len(truth)),
        "accuracy": float(accuracy_score(truth, pred)),
        "macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)),
        "ftp_recall": (float(recall_score(truth, pred, labels=[rt.FTP], average="micro", zero_division=0))
                       if rt.FTP in truth else None),
        "benign_recall": (float(recall_score(truth, pred, labels=[rt.BENIGN], average="micro", zero_division=0))
                          if rt.BENIGN in truth else None),
    }


def _logo_pred_dist(logo) -> dict:
    agg = {}
    for f in logo["folds"]:
        for cls, n in f["predictions"].items():
            agg[cls] = agg.get(cls, 0) + n
    return agg


def _comparison_table(base_cic, cand_cic, base_real, cand_real) -> list[dict]:
    def row(metric, b, c):
        change = (c - b) if (isinstance(b, (int, float)) and isinstance(c, (int, float))) else None
        return {"metric": metric, "baseline": b, "candidate": c, "change": change}
    return [
        row("CIC accuracy", base_cic["accuracy"], cand_cic["accuracy"]),
        row("CIC macro-F1", base_cic["macro_f1"], cand_cic["macro_f1"]),
        row("Real-PCAP accuracy", base_real["accuracy"], cand_real["accuracy"]),
        row("Real-PCAP macro-F1", base_real["macro_f1"], cand_real["macro_f1"]),
        row("FTP-BruteForce recall", base_real["ftp_recall"], cand_real["ftp_recall"]),
        row("Benign recall", base_real["benign_recall"], cand_real["benign_recall"]),
    ]


def _shap_comparison(baseline, candidate, cic_X):
    try:
        import shap
    except Exception:  # noqa: BLE001
        return None
    bg = cic_X.sample(min(300, len(cic_X)), random_state=rt.SEED)
    rows = []
    for name, model in (("baseline", baseline), ("candidate", candidate)):
        vals = np.abs(np.asarray(shap.TreeExplainer(model).shap_values(bg))).mean(axis=(0, 2))
        rows.append(pd.Series(vals, index=list(ml.FEATURES), name=name))
    df = pd.concat(rows, axis=1).reset_index().rename(columns={"index": "feature"})
    df["difference"] = df["candidate"] - df["baseline"]
    return df.reindex(df["baseline"].sort_values(ascending=False).index).reset_index(drop=True)


def _feature_distribution(real: rt.RealFlows) -> pd.DataFrame:
    feats = ["Fwd Seg Size Min", "Init Fwd Win Byts", "Dst Port", "Flow Duration",
             "Flow Pkts/s", "TotLen Fwd Pkts", "TotLen Bwd Pkts"]
    train = pd.read_parquet(ml.DATA_ROOT / "Processed_Data" / "balanced_train_selected.parquet")
    enc = rt._name_to_encoded()[rt.FTP]
    cic_ftp = train[train["Label"] == enc]
    real_ftp = real.df[real.df["Label"] == rt.FTP]
    rows = []
    for f in feats:
        rows.append({"feature": f,
                     "cic_ftp_median": float(cic_ftp[f].median()),
                     "real_ftp_median": float(pd.to_numeric(real_ftp[f], errors="coerce").median())})
    return pd.DataFrame(rows)


def _metadata(W, man_c, base_cic, c_cic, cand_real, real) -> dict:
    import sklearn
    return {
        "experiment": "realistic_pcap_retraining",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": rt.SEED,
        "model": "HistGradientBoostingClassifier",
        "hyperparameters": rt.hgb_params(),
        "features": list(ml.FEATURES),
        "n_features": len(ml.FEATURES),
        "scaler": None,
        "strategy": f"CIC + real flows; real FTP sample_weight = {W} (upweight to ~CIC FTP mass)",
        "training_manifest": man_c,
        "real_captures": real.captures,
        "real_flow_counts": real.df.groupby(["capture", "Label"]).size().reset_index(name="n").to_dict("records"),
        "split_strategy": "leave-one-capture-out (no PCAP in both train and test)",
        "baseline_cic": {"accuracy": base_cic["accuracy"], "macro_f1": base_cic["macro_f1"]},
        "candidate_cic": {"accuracy": c_cic["accuracy"], "macro_f1": c_cic["macro_f1"]},
        "candidate_real_logo": cand_real,
        "data_sources": {
            "cic_train": "webapp_data/Processed_Data/balanced_train_selected.parquet",
            "cic_test": "webapp_data/Processed_Data/test_selected.parquet",
            "real_pcaps": "sample_data/real_pcap/",
        },
        "versions": {"sklearn": sklearn.__version__},
        "limitation": "Only 5 real PCAPs (42 flows). Exploratory evidence only.",
    }


def _flat(m, prefix):
    return {f"{prefix}_accuracy": m["accuracy"], f"{prefix}_macro_f1": m["macro_f1"],
            f"{prefix}_macro_precision": m["macro_precision"], f"{prefix}_macro_recall": m["macro_recall"],
            f"{prefix}_weighted_f1": m["weighted_f1"], f"{prefix}_ftp_recall": m["ftp_recall"],
            f"{prefix}_benign_recall": m["benign_recall"]}


def _cm(metrics) -> pd.DataFrame:
    c = metrics["confusion"]
    return pd.DataFrame(c["matrix"], index=c["labels"], columns=c["labels"])


def _cm_from_folds(logo) -> pd.DataFrame:
    truth, pred = [], []
    for f in logo["folds"]:
        truth += [f["true_label"]] * f["n_flows"]
        for cls, n in f["predictions"].items():
            pred += [cls] * n
    from sklearn.metrics import confusion_matrix
    labels = sorted(set(truth) | set(pred))
    return pd.DataFrame(confusion_matrix(truth, pred, labels=labels), index=labels, columns=labels)


def _fmt(v):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:+.4f}" if abs(v) < 1 and v != int(v) else f"{v:.4f}"
    return str(v)
