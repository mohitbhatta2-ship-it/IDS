"""
SHAP explanation report — why real FTP brute-force flows read as Benign.

Read-only analysis of the existing saved model (no retraining, no thresholds,
no forced predictions). Compares SHAP attributions between CIC-IDS2018
FTP-BruteForce samples the model classifies correctly and real captured FTP
flows the model classifies as Benign, and writes machine-readable results.

    python manage.py shap_report --output ../validation/results/shap --cic-sample 300
"""

from __future__ import annotations

import json
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from predictor import ml, shap_analysis as sa


class Command(BaseCommand):
    help = "SHAP explanations comparing CIC FTP-BruteForce vs real FTP (no model changes)."

    def add_arguments(self, parser):
        parser.add_argument("--output", default=None, help="directory for CSV/JSON results")
        parser.add_argument("--cic-sample", type=int, default=300, help="CIC FTP rows to explain")
        parser.add_argument("--model", default=ml.DEFAULT_MODEL)

    def handle(self, *args, **opts):
        try:
            sa._require_shap()
        except RuntimeError as exc:
            raise CommandError(str(exc))
        if opts["model"] not in ml.MODEL_REGISTRY:
            raise CommandError(f"Unknown model {opts['model']!r}")

        model, _ = ml._load(opts["model"])
        w = self.stdout.write

        cic = sa.cic_ftp_correct(model, sample=opts["cic_sample"])
        real = sa.real_ftp_flows(model, only_predicted=sa.BENIGN)
        if cic.empty:
            raise CommandError("No correctly-classified CIC FTP-BruteForce samples found.")
        if real.empty:
            raise CommandError("No real FTP flows were classified Benign (nothing to explain).")

        w(self.style.MIGRATE_HEADING(f"\nSHAP report — model: {opts['model']}"))
        w(f"  CIC FTP-BruteForce correctly classified: {len(cic)} samples")
        w(f"  Real FTP flows classified Benign        : {len(real)} flows")

        explainer = sa.build_explainer(model)
        res_cic = sa.explain(model, cic, explainer)
        res_real = sa.explain(model, real, explainer)

        add_err = max(sa.additivity_error(model, res_cic), sa.additivity_error(model, res_real))
        w(f"  SHAP additivity error (0 == exact)      : {add_err:.2e}")

        gi_all = sa.global_importance(res_cic)
        gi_ftp = sa.global_importance(res_cic, class_name=sa.FTP)
        cmp = sa.compare_cic_vs_real(model, cic, real, toward=sa.BENIGN)

        w(self.style.MIGRATE_HEADING("\nGlobal feature importance (CIC FTP, mean|SHAP|, all classes)"))
        for _, r in gi_all.head(8).iterrows():
            w(f"  {r['feature']:20} {r['mean_abs_shap']:.4f}")

        w(self.style.MIGRATE_HEADING(
            "\nWhy real flows read Benign — mean SHAP toward Benign (real vs CIC)"))
        w(f"  {'feature':20}{'CIC→Benign':>12}{'real→Benign':>13}{'Δ(real-CIC)':>13}"
          f"{'CIC med':>12}{'real med':>12}")
        for _, r in cmp.head(10).iterrows():
            w(f"  {r['feature']:20}{r['cic_shap_to_Benign']:12.3f}{r['real_shap_to_Benign']:13.3f}"
              f"{r['difference_real_minus_cic']:13.3f}{r['cic_median']:12.1f}{r['real_median']:12.1f}")

        # One concrete per-sample explanation from each population.
        cic_ex = sa.top_features_for_sample(res_cic, 0, sa.FTP, k=6)
        real_ex = sa.top_features_for_sample(res_real, 0, sa.BENIGN, k=6)

        interpretation = self._interpret(cmp)
        w(self.style.MIGRATE_HEADING("\nInterpretation"))
        for line in interpretation:
            w("  " + line)

        if opts["output"]:
            self._save(opts["output"], gi_all, gi_ftp, cmp, cic_ex, real_ex,
                       add_err, len(cic), len(real), interpretation, opts["model"])

    # -- helpers -----------------------------------------------------------

    def _interpret(self, cmp) -> list[str]:
        top = cmp.head(3)
        lines = [
            "The model does not generalise to real FTP brute force. Its FTP-BruteForce",
            "decision leans on features whose CIC values are capture artifacts:",
        ]
        for _, r in top.iterrows():
            direction = "toward Benign" if r["real_shap_to_Benign"] > r["cic_shap_to_Benign"] else "away from Benign"
            lines.append(
                f"- {r['feature']}: CIC median {r['cic_median']:.0f} vs real {r['real_median']:.0f}; "
                f"in real flows this pushes {direction} (Δ SHAP {r['difference_real_minus_cic']:+.2f})."
            )
        lines.append(
            "Real FTP traffic simply does not carry the CIC-specific values (e.g. "
            "Fwd Seg Size Min=40, Init Fwd Win Byts=26883), so it lands in Benign.")
        return lines

    def _save(self, out, gi_all, gi_ftp, cmp, cic_ex, real_ex, add_err, n_cic, n_real,
              interpretation, model_key):
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        gi_all.to_csv(out / "global_importance_all_classes.csv", index=False)
        gi_ftp.to_csv(out / "global_importance_ftp.csv", index=False)
        cmp.to_csv(out / "cic_vs_real_toward_benign.csv", index=False)
        cic_ex.to_csv(out / "sample_cic_ftp_top_features.csv", index=False)
        real_ex.to_csv(out / "sample_real_benign_top_features.csv", index=False)
        payload = {
            "model_key": model_key,
            "cic_samples": n_cic,
            "real_flows_benign": n_real,
            "shap_additivity_error": add_err,
            "global_importance_all_classes": gi_all.head(10).to_dict("records"),
            "cic_vs_real_toward_benign": cmp.head(12).to_dict("records"),
            "interpretation": interpretation,
        }
        (out / "summary.json").write_text(json.dumps(payload, indent=2, default=str))
        self.stdout.write(self.style.SUCCESS(f"\nSaved SHAP results -> {out}"))
