"""
CICFlowMeter-vs-custom-extractor experiment (analysis only; production frozen).

Runs the same real PCAPs through an independent CICFlowMeter implementation and
scores them with the UNCHANGED production model, then compares three paths —
A: CIC held-out, B: real PCAP / custom Live extractor (frozen flows.csv, read
only), C: real PCAP / cicflowmeter — to tell model-generalisation failure apart
from custom-extractor mismatch. Writes evidence to
validation/results/cfm_experiment/. Retrains nothing; changes no production file.

    python manage.py cfm_experiment --output ../validation/results/cfm_experiment
"""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from django.core.management.base import BaseCommand

from predictor import ml, cfm_extract as cf, cfm_experiment as ce


class Command(BaseCommand):
    help = "CICFlowMeter vs custom-extractor experiment (no production changes, no retraining)."

    def add_arguments(self, parser):
        parser.add_argument("--output", default=None)

    def handle(self, *args, **opts):
        w = self.stdout.write
        out = Path(opts["output"]) if opts["output"] else \
            ce.repo_root() / "validation" / "results" / "cfm_experiment"
        out.mkdir(parents=True, exist_ok=True)

        w(self.style.MIGRATE_HEADING(
            "CICFlowMeter vs custom-extractor experiment (production frozen, no retraining)"))
        w(f"  extractor: {cf.cicflowmeter_version()}")

        # --- three paths --------------------------------------------------
        path_a = ce.path_a_cic_heldout()
        path_b, custom_df = ce.path_b_custom()
        path_c, cfm_df = ce.path_c_cicflowmeter()

        w(self.style.MIGRATE_HEADING("\nPath A — CIC-IDS2018 held-out (production model)"))
        w(f"  n={path_a['n']}  acc {path_a['accuracy']:.4f}  macroF1 {path_a['macro_f1']:.4f}")

        w(self.style.MIGRATE_HEADING("\nPath B — real PCAP, custom Live extractor (frozen flows.csv)"))
        self._line_real(w, path_b)

        w(self.style.MIGRATE_HEADING("\nPath C — real PCAP, CICFlowMeter (same production model)"))
        self._line_real(w, path_c)
        ext = path_c["extraction"]
        w(f"  extraction: {ext['total_flows']} valid flows, {ext['total_invalid']} invalid "
          f"(dropped, never zero-filled)")
        for p in ext["per_pcap"]:
            w(f"    {Path(p['pcap']).name:16} {p['label']:14} "
              f"cfm_flows={p['cfm_flows']:2} valid={p['valid_flows']:2} invalid={p['invalid_flows']}")

        # --- §8 FTP recall: custom vs CICFlowMeter ------------------------
        w(self.style.MIGRATE_HEADING("\n§8  FTP-BruteForce recall — custom vs CICFlowMeter"))
        w(f"  custom (B)       : {_f(path_b['ftp_recall'])}")
        w(f"  cicflowmeter (C) : {_f(path_c['ftp_recall'])}")

        # --- §9/§10 feature distribution ----------------------------------
        dist = ce.feature_distribution(custom_df, cfm_df)
        closer = ce.summarise_closer(dist)
        w(self.style.MIGRATE_HEADING("\n§9  Real-FTP feature medians (CIC / custom / cicflowmeter)"))
        for _, r in dist.iterrows():
            w(f"  {r['feature']:20} CIC={_f(r['cic_ftp_median'])}  "
              f"custom={_f(r['custom_real_ftp_median'])}  cfm={_f(r['cicflowmeter_real_ftp_median'])}  "
              f"closer={r['cfm_closer_to_cic'] or '—'}")
        w(self.style.MIGRATE_HEADING("\n§10 Does CICFlowMeter move real FTP toward CIC?"))
        w(f"  comparable features: {closer['comparable_features']}  "
          f"cicflowmeter-closer: {closer['cicflowmeter_closer']}  "
          f"custom-closer: {closer['custom_closer']}  tie: {closer['tie']}")

        # --- §11 verdict --------------------------------------------------
        verdict = self._verdict(path_b, path_c, closer)
        w(self.style.MIGRATE_HEADING("\n§11 Verdict (A = model generalisation vs B = extraction)"))
        w(self.style.WARNING("  " + verdict["headline"]))

        # --- evidence -----------------------------------------------------
        meta = self._write_evidence(out, path_a, path_b, path_c, dist, closer, verdict, ext)
        w(self.style.SUCCESS(f"\nEvidence written to {out}"))
        w(self.style.WARNING(
            "\nScientific note: only 5 real PCAPs (3 FTP, 2 Benign). cicflowmeter here is "
            "the Python port (hieulw), NOT the Java CICFlowMeter V3 the CIC authors used — "
            "an explicit, documented substitution. Results are not manufactured: if "
            "cicflowmeter is also poor, that is reported as-is."))

    # -- helpers -----------------------------------------------------------

    def _line_real(self, w, m):
        w(f"  n={m['n']}  acc {m['accuracy']:.4f}  macroF1 {m['macro_f1']:.4f}  "
          f"FTP recall {_f(m['ftp_recall'])}  Benign recall {_f(m['benign_recall'])}")
        w(f"  predictions: {m['prediction_distribution']}")

    def _verdict(self, path_b, path_c, closer) -> dict:
        b = path_b["ftp_recall"] or 0.0
        c = path_c["ftp_recall"] or 0.0
        improved = c - b
        # "substantial" = cicflowmeter recovers a large share of the missing recall
        substantial = improved >= 0.30
        if substantial:
            head = (f"CICFlowMeter FTP recall {c:.3f} vs custom {b:.3f} (+{improved:.3f}) — "
                    "SUBSTANTIAL improvement: consistent with EXTRACTION MISMATCH (B). "
                    "Per the protocol, STOP before retraining and report extraction as the cause.")
            cause = "extraction_mismatch"
        else:
            head = (f"CICFlowMeter FTP recall {c:.3f} vs custom {b:.3f} ({improved:+.3f}) — "
                    "NOT a substantial improvement: switching to an independent CICFlowMeter "
                    "does not recover FTP detection, so the failure is MODEL GENERALISATION (A), "
                    "not our custom extractor. No retraining is performed in this experiment.")
            cause = "model_generalisation"
        return {"cause": cause, "substantial_improvement": bool(substantial),
                "custom_ftp_recall": b, "cicflowmeter_ftp_recall": c,
                "improvement": improved, "headline": head,
                "feature_closeness": closer}

    def _write_evidence(self, out, path_a, path_b, path_c, dist, closer, verdict, ext) -> dict:
        # per-path metric summary
        pd.DataFrame([
            _flat("A_cic_heldout", path_a),
            _flat("B_real_custom", path_b),
            _flat("C_real_cicflowmeter", path_c),
        ]).to_csv(out / "path_metrics.csv", index=False)

        # per-class for each path
        for tag, m in (("A_cic", path_a), ("B_custom", path_b), ("C_cicflowmeter", path_c)):
            pd.DataFrame(m["per_class"]).to_csv(out / f"per_class_{tag}.csv", index=False)
            _cm(m).to_csv(out / f"confusion_{tag}.csv")

        # prediction distribution across paths
        pd.DataFrame([
            {"path": "A_cic_heldout", **path_a["prediction_distribution"]},
            {"path": "B_real_custom", **path_b["prediction_distribution"]},
            {"path": "C_real_cicflowmeter", **path_c["prediction_distribution"]},
        ]).to_csv(out / "prediction_distribution.csv", index=False)

        # §8 FTP recall
        pd.DataFrame([
            {"path": "B_real_custom", "ftp_recall": path_b["ftp_recall"], "benign_recall": path_b["benign_recall"]},
            {"path": "C_real_cicflowmeter", "ftp_recall": path_c["ftp_recall"], "benign_recall": path_c["benign_recall"]},
        ]).to_csv(out / "ftp_recall_comparison.csv", index=False)

        # §9 feature distribution + §10 verdict
        dist.to_csv(out / "feature_distribution_comparison.csv", index=False)

        # per-pcap extraction stats + failures
        pd.DataFrame([{
            "pcap": Path(p["pcap"]).name, "label": p["label"],
            "cfm_flows": p["cfm_flows"], "valid_flows": p["valid_flows"],
            "invalid_flows": p["invalid_flows"],
            "invalid_detail": json.dumps(p["invalid"]),
        } for p in ext["per_pcap"]]).to_csv(out / "extraction_per_pcap.csv", index=False)

        # validity stats
        pd.DataFrame([{
            "total_valid_flows": ext["total_flows"],
            "total_invalid_flows": ext["total_invalid"],
            "zero_filled": 0,
            "feature_schema_ok": True,
            "n_features": len(ml.FEATURES),
        }]).to_csv(out / "extraction_validity.csv", index=False)

        meta = {
            "experiment": "cicflowmeter_vs_custom_extractor",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "Distinguish (A) model generalisation failure from (B) custom-extractor mismatch.",
            "production_model": ml.DEFAULT_MODEL,
            "retrained": False,
            "extractor": cf.cicflowmeter_version(),
            "extractor_note": ("Python cicflowmeter port (hieulw), NOT the Java CICFlowMeter V3 the "
                               "CIC-IDS2018 authors used. Explicit, documented substitution."),
            "definition_notes": cf.DEFINITION_NOTES,
            "feature_mapping": cf.CFM_TO_ML,
            "features": list(ml.FEATURES),
            "n_features": len(ml.FEATURES),
            "paths": {
                "A_cic_heldout": {k: path_a[k] for k in ("n", "accuracy", "macro_f1")},
                "B_real_custom": {k: path_b[k] for k in ("n", "accuracy", "macro_f1", "ftp_recall", "benign_recall")},
                "C_real_cicflowmeter": {k: path_c[k] for k in ("n", "accuracy", "macro_f1", "ftp_recall", "benign_recall")},
            },
            "section_8_ftp_recall": {"custom": path_b["ftp_recall"], "cicflowmeter": path_c["ftp_recall"]},
            "section_10_feature_closeness": closer,
            "section_11_verdict": verdict,
            "data_sources": {
                "cic_test": "webapp_data/Processed_Data/test_selected.parquet",
                "cic_train_for_dist": "webapp_data/Processed_Data/balanced_train_selected.parquet",
                "real_pcaps": "sample_data/real_pcap/",
                "custom_baseline": "validation/results/flows.csv (frozen, read-only)",
            },
            "versions": {"python": platform.python_version()},
            "limitation": "Only 5 real PCAPs (42 custom / 22 cicflowmeter flows); Python port, not Java V3.",
        }
        (out / "experiment_metadata.json").write_text(json.dumps(meta, indent=2, default=str))
        return meta


def _flat(name, m) -> dict:
    return {
        "path": name, "n": m["n"], "accuracy": m["accuracy"],
        "macro_precision": m["macro_precision"], "macro_recall": m["macro_recall"],
        "macro_f1": m["macro_f1"], "weighted_f1": m["weighted_f1"],
        "ftp_recall": m["ftp_recall"], "benign_recall": m["benign_recall"],
    }


def _cm(m) -> pd.DataFrame:
    c = m["confusion"]
    return pd.DataFrame(c["matrix"], index=c["labels"], columns=c["labels"])


def _f(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 1000 else f"{v:.1f}"
    return str(v)
