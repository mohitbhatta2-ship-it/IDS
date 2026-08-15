"""
Run the real-PCAP validation workflow.

Score real captured traffic through the exact Live Capture flow engine + the
Dataset Testing model path, against known-from-capture ground truth, and report
honest generalisation metrics. Never retrains or forces predictions.

Examples
--------
    # one labelled capture
    python manage.py validate_pcaps --pcap validation/pcaps/ftp_bruteforce/s1.pcap \
        --label FTP-BruteForce

    # a whole tree (folder name = ground-truth class), writing CSV/JSON
    python manage.py validate_pcaps --input validation/pcaps --output validation/results \
        --compare-cic
"""

from __future__ import annotations

from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from predictor import ml, pcap_validation as pv


class Command(BaseCommand):
    help = "Validate the existing model against real labelled PCAPs (no retraining)."

    def add_arguments(self, parser):
        parser.add_argument("--pcap", help="a single .pcap/.pcapng file")
        parser.add_argument("--label", help="ground-truth class for --pcap (folder alias or exact class name)")
        parser.add_argument("--input", help="a directory of <class>/<*.pcap> captures")
        parser.add_argument("--output", help="directory to write CSV/JSON results into")
        parser.add_argument("--model", default=ml.DEFAULT_MODEL, help="model key (default: %(default)s)")
        parser.add_argument("--compare-cic", action="store_true",
                            help="also run the CIC held-out test set through Dataset Testing for comparison")

    def handle(self, *args, **opts):
        model_key = opts["model"]
        if model_key not in ml.MODEL_REGISTRY:
            raise CommandError(f"Unknown model {model_key!r}. Choose from: {', '.join(ml.MODEL_REGISTRY)}")

        if opts["pcap"]:
            if not opts["label"]:
                raise CommandError("--pcap requires --label (the capture's known class).")
            summary = self._single(opts["pcap"], opts["label"], model_key)
        elif opts["input"]:
            summary = pv.validate_directory(opts["input"], model_key)
        else:
            raise CommandError("Provide either --pcap (+ --label) or --input.")

        if opts["compare_cic"]:
            summary["cic_baseline"] = pv.cic_dataset_testing_metrics(model_key)

        if summary.get("combined") is not None:
            summary["distribution"] = pv.feature_distribution_comparison(summary["combined"])
            summary["descriptive"] = pv.descriptive_stats(summary["combined"])

        self._report(summary)

        if opts["output"]:
            written = pv.save_results(summary, opts["output"])
            self.stdout.write(self.style.SUCCESS("\nSaved:"))
            for name, path in written.items():
                self.stdout.write(f"  {name}: {path}")

    # -- helpers -----------------------------------------------------------

    def _single(self, pcap, label, model_key) -> dict:
        try:
            res = pv.validate_pcap(pcap, label, model_key)
        except pv.ValidationError as exc:
            raise CommandError(str(exc))
        import pandas as pd
        combined = res.frame.assign(_pcap=res.pcap) if res.frame is not None else None
        metrics = pv.compute_metrics(combined["Label"], combined["Predicted Class"]) if combined is not None and not combined.empty else None
        confusion = pv.confusion_table(combined["Label"], combined["Predicted Class"]) if combined is not None and not combined.empty else None
        return {"model_key": model_key, "per_pcap": [res], "flows": res.valid_flows,
                "metrics": metrics, "confusion": confusion,
                "invalid_flows": len(res.invalid_flows), "combined": combined}

    def _report(self, summary: dict):
        w = self.stdout.write
        w(self.style.MIGRATE_HEADING(f"\nReal-PCAP validation — model: {summary['model_key']}"))

        for r in summary.get("per_pcap", []):
            w(f"  {Path(r.pcap).name:32} label={r.label:22} "
              f"flows={r.valid_flows:5} correct={r.correct:5} invalid={len(r.invalid_flows)}")

        m = summary.get("metrics")
        if not m:
            w(self.style.WARNING("\nNo scored flows. Drop real captures into the pcap folders and re-run."))
            return

        w(f"\nTotal flows: {m['total_flows']}  correct: {m['correct']}  "
          f"incorrect: {m['incorrect']}  invalid/incomplete: {summary.get('invalid_flows', 0)}")
        w(f"Accuracy: {m['accuracy']:.4f}  Macro-F1: {m['macro_f1']:.4f}  Weighted-F1: {m['weighted_f1']:.4f}")

        w("\nClass | Ground Truth | Correct | Incorrect | Precision | Recall | F1")
        for row in m["per_class"]:
            w(f"  {row['class']:24} {row['ground_truth']:6} {row['correct']:8} {row['incorrect']:10} "
              f"{row['precision']:9.3f} {row['recall']:7.3f} {row['f1']:6.3f}")

        conf = summary.get("confusion")
        if conf:
            w("\nConfusion matrix (rows = truth, cols = predicted):")
            labels = conf["labels"]
            w("  " + " ".join(f"{l[:10]:>11}" for l in ["truth\\pred"] + labels))
            for lbl, rowvals in zip(labels, conf["matrix"]):
                w("  " + f"{lbl[:10]:>11} " + " ".join(f"{v:>11}" for v in rowvals))

        cic = summary.get("cic_baseline")
        if cic:
            w(self.style.MIGRATE_HEADING("\nCIC-IDS2018 (Dataset Testing) vs Real-PCAP"))
            w(f"  CIC held-out : accuracy {cic['accuracy']:.4f}  macro-F1 {cic['macro_f1']:.4f}  (n={cic['rows']})")
            w(f"  Real capture : accuracy {m['accuracy']:.4f}  macro-F1 {m['macro_f1']:.4f}  (n={m['total_flows']})")

        dist = summary.get("distribution")
        if dist is not None and not dist.empty:
            w(self.style.MIGRATE_HEADING("\nFeature medians: real capture vs CIC training"))
            w(f"  {'class':22}{'feature':20}{'real':>16}{'CIC':>16}")
            for _, d in dist.iterrows():
                w(f"  {d['class']:22}{d['feature']:20}{d['real_median']:16.3f}{d['cic_median']:16.3f}")

        desc = summary.get("descriptive")
        if desc is not None and not desc.empty:
            w(self.style.MIGRATE_HEADING(
                "\nDescriptive stats (real captures only; packet counts & Flow Bytes/s "
                "are NOT model features, so no CIC comparison)"))
            for _, d in desc.iterrows():
                w(f"  {d['class']:16} flows={int(d['flows']):3} "
                  f"FlowDur(us)~{d['median Flow Duration (us)']:.0f} "
                  f"FwdPkts~{d['median Total Fwd Packets']:.0f} BwdPkts~{d['median Total Bwd Packets']:.0f} "
                  f"Flow Pkts/s~{d['median Flow Pkts/s']:.1f} Flow Bytes/s~{d['median Flow Bytes/s (derived)']:.1f}")
