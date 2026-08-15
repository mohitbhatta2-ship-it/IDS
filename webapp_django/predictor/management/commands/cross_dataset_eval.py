"""
Cross-dataset evaluation of the existing model (read-only, no retraining).

Runs labelled datasets already in the repo through the SAME Dataset Testing path
(`ml.predict_batch`) and reports how the model generalises across data sources —
most importantly `sample_data/foreign_dataset_ids2017_style.csv` (CIC-IDS**2017**
naming, a genuine cross-dataset test), plus the per-attack labelled CSVs.

    python manage.py cross_dataset_eval --input ../sample_data --output ../validation/results/cross_dataset
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from django.core.management.base import BaseCommand, CommandError

from predictor import ml, pcap_validation as pv


class Command(BaseCommand):
    help = "Evaluate the existing model on other labelled datasets in the repo (no retraining)."

    def add_arguments(self, parser):
        parser.add_argument("--input", default=None,
                            help="a labelled CSV, or a directory searched for labelled CSVs "
                                 "(default: sample_data/)")
        parser.add_argument("--output", default=None, help="directory for CSV/JSON results")
        parser.add_argument("--model", default=ml.DEFAULT_MODEL)

    def handle(self, *args, **opts):
        if opts["model"] not in ml.MODEL_REGISTRY:
            raise CommandError(f"Unknown model {opts['model']!r}")

        target = Path(opts["input"]) if opts["input"] else _default_input()
        files = _discover(target)
        if not files:
            raise CommandError(f"No labelled CSVs found under {target}")

        w = self.stdout.write
        w(self.style.MIGRATE_HEADING(f"\nCross-dataset evaluation — model: {opts['model']}"))

        per_file = []
        frames = []
        for path in files:
            try:
                df = ml.read_upload(open(path, "rb"), path.name)
                result = ml.predict_batch(df, opts["model"])
            except (ml.BatchError, Exception) as exc:  # noqa: BLE001
                w(self.style.WARNING(f"  {path.name}: skipped ({exc})"))
                continue
            frame = result["frame"]
            ev = result["evaluation"]
            per_file.append({
                "dataset": str(path.relative_to(target)) if target in path.parents else path.name,
                "rows": result["rows"],
                "has_label": result["has_label_column"],
                "accuracy": (ev["accuracy"] if ev else None),
                "macro_f1": (ev["macro_f1"] if ev else None),
            })
            if "Label" in frame.columns and ev is not None:
                frames.append(frame[["Label", "Predicted Class"]].copy())
            acc = f"{ev['accuracy']:.4f}" if ev else "  (no usable Label)"
            mf1 = f"{ev['macro_f1']:.4f}" if ev else "   —"
            w(f"  {path.name:44} rows={result['rows']:7}  acc={acc}  macroF1={mf1}")

        summary = {"model_key": opts["model"], "per_file": per_file}

        if frames:
            combined = pd.concat(frames, ignore_index=True)
            metrics = pv.compute_metrics(combined["Label"], combined["Predicted Class"])
            confusion = pv.confusion_table(combined["Label"], combined["Predicted Class"])
            summary["aggregate_metrics"] = metrics
            summary["confusion"] = confusion

            w(self.style.MIGRATE_HEADING("\nAggregate over labelled datasets"))
            w(f"  flows={metrics['total_flows']}  correct={metrics['correct']}  "
              f"accuracy={metrics['accuracy']:.4f}  macroF1={metrics['macro_f1']:.4f}  "
              f"weightedF1={metrics['weighted_f1']:.4f}")
            w("\n  Class | Ground Truth | Correct | Incorrect | Precision | Recall | F1")
            for r in metrics["per_class"]:
                w(f"    {r['class']:24} {r['ground_truth']:6} {r['correct']:8} {r['incorrect']:10} "
                  f"{r['precision']:9.3f} {r['recall']:7.3f} {r['f1']:6.3f}")

        # Comparison line vs the CIC held-out baseline (same Dataset Testing path).
        cic = pv.cic_dataset_testing_metrics(opts["model"])
        if cic:
            summary["cic_baseline"] = cic
            w(self.style.MIGRATE_HEADING("\nCIC-IDS2018 held-out baseline (same path)"))
            w(f"  accuracy {cic['accuracy']:.4f}  macroF1 {cic['macro_f1']:.4f}  (n={cic['rows']})")

        if opts["output"]:
            out = Path(opts["output"])
            out.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(per_file).to_csv(out / "per_dataset.csv", index=False)
            if summary.get("aggregate_metrics"):
                pd.DataFrame(summary["aggregate_metrics"]["per_class"]).to_csv(
                    out / "aggregate_per_class.csv", index=False)
                conf = summary["confusion"]
                pd.DataFrame(conf["matrix"], index=conf["labels"], columns=conf["labels"]).to_csv(
                    out / "confusion_matrix.csv")
            (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
            w(self.style.SUCCESS(f"\nSaved -> {out}"))


def _default_input() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "sample_data"
        if cand.is_dir():
            return cand
    raise CommandError("Could not locate sample_data/.")


def _discover(target: Path) -> list[Path]:
    """Labelled CSVs only: skip *_no_labels* and anything under a no_labels dir."""
    if target.is_file():
        return [target]
    out = []
    for p in sorted(target.rglob("*.csv")):
        if "no_labels" in p.name or "no_labels" in p.parts:
            continue
        if p.name in ("MANIFEST.csv",):
            continue
        out.append(p)
    return out
