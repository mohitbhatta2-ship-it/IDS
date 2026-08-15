"""
Build the attack-family / severity dashboard from EXISTING validation results.

Read-only analysis: reuses the per-flow predictions + PCAP/dataset ground-truth
labels already written to validation/results/, rolls them up by attack family
and severity, and writes CSV/JSON + a self-contained HTML dashboard. Never
retrains, never calls the model, never regenerates labels or predictions.

    python manage.py attack_dashboard \
        --flows ../validation/results/flows.csv \
        --summary ../validation/results/summary.json \
        --output ../validation/results/attack_dashboard
"""

from __future__ import annotations

import json
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from predictor import attack_dashboard as ad


class Command(BaseCommand):
    help = "Family/severity dashboard from existing validation results (no model use)."

    def add_arguments(self, parser):
        parser.add_argument("--flows", default=None, help="per-flow results CSV (default: validation/results/flows.csv)")
        parser.add_argument("--summary", default=None, help="summary.json for the CIC baseline comparison")
        parser.add_argument("--output", default=None, help="output directory (default: validation/results/attack_dashboard)")

    def handle(self, *args, **opts):
        root = _repo_root()
        flows_path = Path(opts["flows"]) if opts["flows"] else root / "validation" / "results" / "flows.csv"
        summary_path = Path(opts["summary"]) if opts["summary"] else root / "validation" / "results" / "summary.json"
        out = Path(opts["output"]) if opts["output"] else root / "validation" / "results" / "attack_dashboard"

        if not flows_path.is_file():
            raise CommandError(
                f"No per-flow results at {flows_path}. Run `validate_pcaps ... --output "
                "../validation/results` first (this command does not regenerate them)."
            )

        flows = ad.load_flows(flows_path)

        cic_baseline = None
        if summary_path.is_file():
            try:
                cic_baseline = json.loads(summary_path.read_text()).get("cic_baseline")
            except Exception:  # noqa: BLE001
                cic_baseline = None

        dashboard = ad.build_dashboard(flows, cic_baseline=cic_baseline)
        written = ad.save_dashboard(dashboard, out)

        self._report(dashboard)

        # Supplementary: a full 15-class family/severity rollup from the EXISTING
        # cross-dataset per-class results (the real-PCAP set only spans 2 classes).
        cross = root / "validation" / "results" / "cross_dataset" / "aggregate_per_class.csv"
        if cross.is_file():
            import json as _json
            import pandas as pd
            roll = ad.build_from_per_class(pd.read_csv(cross), source="cross-dataset (all classes)")
            (out / "cross_dataset_family_rollup.csv").write_text(pd.DataFrame(roll["family"]).to_csv(index=False))
            (out / "cross_dataset_severity_rollup.csv").write_text(pd.DataFrame(roll["severity"]).to_csv(index=False))
            (out / "cross_dataset_rollup.json").write_text(_json.dumps(roll, indent=2, default=str))
            written["cross_dataset_rollup_json"] = str(out / "cross_dataset_rollup.json")
            self._report_rollup(roll)

        self.stdout.write(self.style.SUCCESS("\nSaved:"))
        for name, path in written.items():
            self.stdout.write(f"  {name}: {path}")

    def _report(self, d: dict):
        w = self.stdout.write
        o = d["overall"]
        w(self.style.MIGRATE_HEADING(f"\nAttack-family / severity dashboard — {d['source']}, {d['flows']} flows"))
        w(f"  accuracy {o['accuracy']:.4f}  macroP {o['macro_precision']:.4f}  "
          f"macroR {o['macro_recall']:.4f}  macroF1 {o['macro_f1']:.4f}  "
          f"correct {o['correct']}/{o['correct']+o['incorrect']}")

        if d.get("highlight_ftp_bruteforce"):
            f = d["highlight_ftp_bruteforce"]
            w(self.style.ERROR(
                f"  FTP-BruteForce [{f['family']} / severity {f['severity']}]: "
                f"support {f['support']}, correct {f['correct']}, recall {f['recall']:.3f}"))

        w(self.style.MIGRATE_HEADING("\nBy attack family (ground truth)"))
        w(f"  {'family':14}{'support':>8}{'correct':>9}{'incorrect':>11}{'class recall':>14}")
        for r in d["family"]["summary"]:
            w(f"  {r['group']:14}{r['support']:8}{r['correct_class']:9}{r['incorrect_class']:11}{r['class_recall']:14.3f}")

        w(self.style.MIGRATE_HEADING("\nBy severity (ground truth)"))
        w(f"  {'severity':14}{'support':>8}{'correct':>9}{'incorrect':>11}{'class recall':>14}")
        for r in d["severity"]["summary"]:
            w(f"  {r['group']:14}{r['support']:8}{r['correct_class']:9}{r['incorrect_class']:11}{r['class_recall']:14.3f}")

        if d.get("comparison"):
            c = d["comparison"]
            w(self.style.MIGRATE_HEADING("\nReal-PCAP vs CIC-IDS2018"))
            w(f"  CIC  : accuracy {c['cic']['accuracy']:.4f}  macroF1 {c['cic']['macro_f1']:.4f}  (n={c['cic']['rows']})")
            w(f"  Real : accuracy {c['real']['accuracy']:.4f}  macroF1 {c['real']['macro_f1']:.4f}  (n={c['real']['flows']})")

        if d.get("confidence"):
            cd = d["confidence"]
            w(self.style.MIGRATE_HEADING("\nConfidence distribution"))
            w(f"  mean {cd['overall_mean']:.3f}  median {cd['overall_median']:.3f}")
            w("  " + "  ".join(f"{k}:{v}" for k, v in cd["histogram"].items()))


    def _report_rollup(self, roll: dict):
        w = self.stdout.write
        w(self.style.MIGRATE_HEADING(
            f"\n{roll['source']} — {roll['classes']} classes, {roll['support']} flows, "
            f"{roll['correct']} correct"))
        w("  by family:")
        for r in roll["family"]:
            w(f"    {r['group']:12} support={r['support']:6} correct={r['correct_class']:6} "
              f"recall={r['class_recall']:.3f}")
        w("  by severity:")
        for r in roll["severity"]:
            w(f"    {r['group']:12} support={r['support']:6} correct={r['correct_class']:6} "
              f"recall={r['class_recall']:.3f}")


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "validation").is_dir() and (parent / "webapp_django").is_dir():
            return parent
    return here.parents[4]
