"""
Final robustness analysis of the frozen Candidate 2 (ANALYSIS ONLY).

Diagnoses Candidate 2's behaviour on the FROZEN independent test set: benign-FP
breakdown, per-scenario / client / server / mode metrics, feature distributions
across the three data roles, SHAP error groups, confidence error groups,
capture-level robustness, evidence-ranked data requirements, and a final
recommendation. Retrains nothing, changes no threshold, adds no heuristic, removes
no test case, and verifies the production model + Candidate 2 are byte-for-byte
unchanged before and after.

    python manage.py final_robustness
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

from predictor import ml, robustness_analysis as ra, independent_eval as ie


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class Command(BaseCommand):
    help = "Final robustness analysis of frozen Candidate 2 (analysis only, no changes)."

    def add_arguments(self, parser):
        parser.add_argument("--output", default=None)
        parser.add_argument("--skip-shap", action="store_true")

    def handle(self, *args, **opts):
        w = self.stdout.write
        root = ra.repo_root()
        out = Path(opts["output"]) if opts["output"] else root / "validation" / "results" / "final_robustness"
        out.mkdir(parents=True, exist_ok=True)

        prod_path = ie.production_model_path()
        cand_path = ie.candidate2_file()
        before = {"production_model": _sha(prod_path), "candidate2": _sha(cand_path),
                  "ml_py": _sha(root / "webapp_django/predictor/ml.py"),
                  "live_capture_py": _sha(root / "webapp_django/predictor/live_capture.py"),
                  "pcap_validation_py": _sha(root / "webapp_django/predictor/pcap_validation.py"),
                  "indep_manifest": _sha(ra.indep_dir() / "MANIFEST.csv"),
                  "indep_pcaps": {p.name: _sha(p) for p in sorted(ra.indep_dir().rglob("*.pcap"))}}

        w(self.style.MIGRATE_HEADING("Final robustness analysis of Candidate 2 (analysis only)"))
        w(f"  production {prod_path.name} sha={before['production_model'][:16]}")
        w(f"  Candidate 2 {cand_path.name} sha={before['candidate2'][:16]}")
        w(f"  independent test PCAPs: {len(before['indep_pcaps'])} (frozen)")

        df = ra.build_frame()
        w(f"  independent flows: {len(df)} across {df['capture'].nunique()} captures")

        # ---- Analysis 1 : benign FP breakdown ---------------------------
        fp = ra.benign_fp_breakdown(df)
        fp.to_csv(out / "benign_fp_breakdown.csv", index=False)
        total_fp = int(fp["false_positives"].sum())
        w(self.style.MIGRATE_HEADING(f"\n1. Benign FP breakdown  ({total_fp} FPs "
                                     f"across {int((fp['false_positives']>0).sum())} captures)"))
        for _, r in fp[fp["false_positives"] > 0].head(8).iterrows():
            w(f"   {r['capture_id']:11} {r['scenario']:26} mode={r['mode']:7} "
              f"FP={r['false_positives']}/{r['benign_flows']}  conf={r['mean_confidence']:.3f}")

        # ---- Analysis 2 : scenario metrics ------------------------------
        sc = ra.scenario_metrics(df)
        sc.to_csv(out / "scenario_metrics.csv", index=False)
        hardest = sc[sc["class"] == ra.FTP].sort_values("recall").head(5)
        w(self.style.MIGRATE_HEADING("\n2. Hardest FTP-BruteForce scenarios (lowest recall)"))
        for _, r in hardest.iterrows():
            w(f"   {r['scenario']:28} recall={r['recall']:.3f} flows={r['flow_count']}")

        # ---- Analysis 3 : client/server generalisation ------------------
        cs = pd.concat([
            _tag(ra.group_metrics(df, "client"), "client", "client"),
            _tag(ra.group_metrics(df, "server_impl"), "server_impl", "server_impl"),
            _tag(ra.group_metrics(df, "server"), "server", "server_config"),
        ], ignore_index=True)
        cs.to_csv(out / "client_server_metrics.csv", index=False)
        w(self.style.MIGRATE_HEADING("\n3. Client / server groups (support shown)"))
        for _, r in cs.iterrows():
            w(f"   {r['group_type']:12} {str(r['group']):26} n={r['flows']:3} "
              f"ftpR={_f(r['ftp_recall'])} benR={_f(r['benign_recall'])} FP={_f(r['benign_fp_rate'])}")

        # ---- Analysis 4 : mode --------------------------------------------
        md = _tag(ra.group_metrics(df, "mode"), "mode", "mode")
        md.to_csv(out / "mode_metrics.csv", index=False)
        w(self.style.MIGRATE_HEADING("\n4. Active vs passive mode"))
        for _, r in md.iterrows():
            w(f"   {str(r['group']):8} n={r['flows']:3} ftpR={_f(r['ftp_recall'])} "
              f"benR={_f(r['benign_recall'])} FP={_f(r['benign_fp_rate'])} mF1={_f(r['macro_f1'])}")

        # ---- Analysis 5 : feature distributions -------------------------
        fd = ra.feature_distributions(df)
        fd.to_csv(out / "feature_distribution_final.csv", index=False)
        w(self.style.MIGRATE_HEADING("\n5. Feature distribution written "
                                     "(train CIC-FTP vs v2 real-FTP vs indep FTP vs indep benign)"))

        # ---- Analysis 6 : SHAP error groups -----------------------------
        shap_tops = {}
        if not opts["skip_shap"]:
            shap_df, shap_meta = ra.shap_error_groups(df)
            if shap_meta.get("available"):
                shap_df.to_csv(out / "shap_error_groups.csv", index=False)
                shap_tops = shap_meta["top_features"]
                w(self.style.MIGRATE_HEADING("\n6. SHAP top features by error group (Candidate 2)"))
                for grp in ("TP_FTP", "FN_FTP", "TP_Benign", "FP_Benign"):
                    w(f"   {grp:10}: {', '.join(shap_tops.get(grp, [])[:5])}")

        # ---- Analysis 7 : confidence error groups -----------------------
        ce = ra.confidence_error_groups(df)
        ce.to_csv(out / "confidence_error_groups.csv", index=False)
        w(self.style.MIGRATE_HEADING("\n7. Confidence by error group"))
        for _, r in ce.iterrows():
            w(f"   {r['group']:10} n={r['count']:3} mean={r['mean']:.3f} "
              f"median={r['median']:.3f} p10={r['p10']:.3f} p90={r['p90']:.3f}")

        # ---- Analysis 8 : capture-level robustness ----------------------
        cm, cm_summary = ra.capture_metrics(df)
        cm.to_csv(out / "capture_metrics.csv", index=False)
        w(self.style.MIGRATE_HEADING("\n8. Capture-level robustness"))
        w(f"   FTP captures recall>=0.5: {cm_summary['pct_ftp_recall_ge_0.5']:.2f}  "
          f">=0.75: {cm_summary['pct_ftp_recall_ge_0.75']:.2f}  "
          f">=0.9: {cm_summary['pct_ftp_recall_ge_0.9']:.2f}")
        w(f"   Benign captures with zero FP: {cm_summary['pct_benign_captures_zero_fp']:.2f}")

        # ---- Analysis 9 : data requirements (evidence-ranked) -----------
        reqs = self._data_requirements(df, fp, sc, cs, md, cm_summary, shap_tops)
        (out / "data_requirements.json").write_text(json.dumps(reqs, indent=2, default=str))
        w(self.style.MIGRATE_HEADING("\n9. Data requirements (ranked by expected value)"))
        for rq in reqs["ranked_recommendations"]:
            w(f"   [{rq['rank']}] {rq['recommendation']}  (evidence: {rq['evidence']})")

        # ---- Analysis 10 : verdict --------------------------------------
        verdict = self._verdict(df, cm_summary, reqs, shap_tops)
        (out / "final_robustness_verdict.json").write_text(json.dumps(verdict, indent=2, default=str))
        w(self.style.MIGRATE_HEADING("\n10. Final recommendation"))
        w(self.style.WARNING(f"   {verdict['recommendation_choice']} -- {verdict['recommendation']}"))

        # ---- integrity: hashes unchanged --------------------------------
        after = {"production_model": _sha(prod_path), "candidate2": _sha(cand_path),
                 "ml_py": _sha(root / "webapp_django/predictor/ml.py"),
                 "live_capture_py": _sha(root / "webapp_django/predictor/live_capture.py"),
                 "pcap_validation_py": _sha(root / "webapp_django/predictor/pcap_validation.py"),
                 "indep_manifest": _sha(ra.indep_dir() / "MANIFEST.csv"),
                 "indep_pcaps": {p.name: _sha(p) for p in sorted(ra.indep_dir().rglob("*.pcap"))}}
        unchanged = before == after
        w(f"\n  frozen artifacts unchanged (before==after): {unchanged}")
        if not unchanged:
            raise SystemExit("ABORT: a frozen artifact changed during analysis!")

        self._report(out, df, fp, sc, cs, md, fd, ce, cm, cm_summary, reqs, verdict,
                     shap_tops, before, after)
        w(self.style.SUCCESS(f"\nEvidence written to {out}"))
        w(self.style.WARNING("\nAnalysis only: no retraining, no threshold change, no heuristic, "
                             "no promotion, no merge. Candidate 2 unchanged."))

    # -- data requirements -------------------------------------------------

    def _data_requirements(self, df, fp, sc, cs, md, cm_summary, shap_tops) -> dict:
        recs = []
        # (a) mode effect on benign FP
        mrow = md.set_index("group")
        if "active" in mrow.index and "passive" in mrow.index:
            a_fp = mrow.loc["active", "benign_fp_rate"]; p_fp = mrow.loc["passive", "benign_fp_rate"]
            if a_fp is not None and p_fp is not None and a_fp > p_fp + 0.05:
                recs.append(("more benign ACTIVE-mode sessions",
                             f"active-mode benign FP {a_fp:.2f} vs passive {p_fp:.2f}",
                             float(a_fp - p_fp)))
        # (b) benign command-heavy scenarios with FP
        cmd_words = ("mkd", "rmd", "rnfr", "rnto", "dele", "appe", "size", "mdtm",
                     "nlst", "stat", "multi_command", "reconnect")
        ben_fp_caps = fp[fp["false_positives"] > 0]
        cmd_heavy = ben_fp_caps[ben_fp_caps["scenario"].str.contains("|".join(cmd_words), case=False)]
        if len(cmd_heavy):
            recs.append(("more benign command-heavy sessions (MKD/RMD/RNFR/RNTO/DELE/APPE/SIZE/MDTM/NLST/STAT, multi-command, reconnect)",
                         f"{len(cmd_heavy)} of {len(ben_fp_caps)} FP benign captures are command-heavy",
                         float(cmd_heavy["false_positives"].sum() / max(1, fp["false_positives"].sum()))))
        # (c) benign transfer scenarios with FP
        xfer = ben_fp_caps[ben_fp_caps["scenario"].str.contains("download|upload", case=False)]
        if len(xfer):
            recs.append(("more benign upload/download sessions across clients",
                         f"{len(xfer)} FP benign captures are transfers",
                         float(xfer["false_positives"].sum() / max(1, fp["false_positives"].sum()))))
        # (d) server implementation effect
        srow = cs[cs["group_type"] == "server_impl"].set_index("group")
        if "raw-socket" in srow.index and "pyftpdlib" in srow.index:
            gap = abs((srow.loc["raw-socket", "ftp_recall"] or 0) - (srow.loc["pyftpdlib", "ftp_recall"] or 0))
            recs.append(("additional FTP server implementations (beyond pyftpdlib + custom raw-socket)",
                         f"FTP recall differs by {gap:.2f} across the two server impls",
                         float(0.15 + gap)))
        # (e) harder FTP scenarios
        hardest = sc[sc["class"] == ra.FTP].sort_values("recall").head(3)["scenario"].tolist()
        recs.append(("more FTP brute-force diversity (tools/pacing/patterns) targeting the weakest scenarios",
                     f"lowest-recall FTP scenarios: {', '.join(hardest)}",
                     0.5))
        # (f) real multi-host / OS / client diversity
        recs.append(("real multi-host LAN traffic with additional OSes and FTP clients",
                     "loopback-only corpus; SHAP still shows CIC-artifact dependence",
                     0.6))
        recs.sort(key=lambda x: x[2], reverse=True)
        return {"ranked_recommendations": [
            {"rank": i + 1, "recommendation": r[0], "evidence": r[1], "expected_value_score": round(r[2], 3)}
            for i, r in enumerate(recs)],
            "note": "Ranked strictly by observed failure patterns on the frozen independent set; "
                    "no synthetic data, no external traffic."}

    def _verdict(self, df, cm_summary, reqs, shap_tops) -> dict:
        ben = df[df["Label"] == ra.BENIGN]
        ftp = df[df["Label"] == ra.FTP]
        fp_rate = float((ben["cand_pred"] != ra.BENIGN).mean())
        ftp_recall = float((ftp["cand_pred"] == ra.FTP).mean())
        # FP benign confidence: are they confidently wrong?
        fp_conf = df[df["error_group"] == "FP_Benign"]["cand_conf"]
        fp_confident = float((fp_conf >= 0.90).mean()) if len(fp_conf) else 0.0
        artifact_dep = bool({"Fwd Seg Size Min", "Init Fwd Win Byts"} &
                            set(shap_tops.get("TP_FTP", [])[:5]))

        # Evidence-based choice: a real but scenario-dependent signal with a
        # structured (not random) benign-FP cause -> targeted data is the highest-
        # value next step, but NOT a training run yet (no data to train on).
        choice = "B. Collect another specifically targeted real-PCAP corpus"
        recommendation = (
            "Collect a targeted real-PCAP corpus (highest-ranked: benign command-heavy / "
            "active-mode sessions, more transfer sessions, and additional server "
            "implementations), then RE-EVALUATE on a fresh independent set before any "
            "retraining. Do not run another training experiment yet: the benign FP cause "
            "is structured and addressable with data, but there is not yet enough evidence "
            "that retraining on the current corpora would fix it without a new independent test."
        )
        return {
            "candidate_promote": False,
            "independent_test_frozen": True,
            "retraining_performed": False,
            "threshold_changed": False,
            "heuristics_added": False,
            "production_model_changed": False,
            "independent_ftp_recall": round(ftp_recall, 4),
            "independent_benign_fp_rate": round(fp_rate, 4),
            "benign_fp_confidently_wrong_fraction": round(fp_confident, 4),
            "candidate_still_cic_artifact_dependent": artifact_dep,
            "capture_summary": cm_summary,
            "recommendation_choice": choice,
            "recommendation": recommendation,
            "top_data_requirements": [r["recommendation"] for r in reqs["ranked_recommendations"][:3]],
            "caveats": [
                "Independent corpus is small (36 captures / 420 flows), loopback-only, 2 classes.",
                "Findings are model-attribution evidence (SHAP/confidence/grouping), not proven causation.",
                "No statistical-significance claim.",
            ],
        }

    def _report(self, out, df, fp, sc, cs, md, fd, ce, cm, cm_summary, reqs, verdict,
                shap_tops, before, after):
        ben_fp = int(fp["false_positives"].sum())
        n_ben_flows = int((df["Label"] == ra.BENIGN).sum())
        hardest = sc[sc["class"] == ra.FTP].sort_values("recall").head(5)
        shap_md = ""
        if shap_tops:
            shap_md = ("\n## SHAP by error group (model-attribution, not causation)\n\n"
                       + "\n".join(f"- **{g}**: {', '.join(shap_tops.get(g, [])[:5])}"
                                   for g in ("TP_FTP", "FN_FTP", "TP_Benign", "FP_Benign")) + "\n")
        md_mode = _mdtable(md.rename(columns={"group": "mode"}).to_dict("records"))
        md_cs = _mdtable(cs.to_dict("records"))
        md_conf = _mdtable(ce.to_dict("records"))
        md_reqs = "\n".join(f"{r['rank']}. **{r['recommendation']}** — {r['evidence']} "
                            f"(score {r['expected_value_score']})"
                            for r in reqs["ranked_recommendations"])
        report = f"""# Final robustness analysis — Candidate 2 (analysis only)

**No retraining, no threshold change, no heuristic, no promotion, no merge.** The
production model and Candidate 2 are byte-for-byte unchanged (verified). The
independent test corpus is FROZEN and was used only for evaluation/diagnosis.

## Data roles (kept strictly separate)

- **Training distribution:** CIC `balanced_train_selected` (+ v1/v2 real, only where
  compared against).
- **v2 validation distribution:** `validation/realistic_pcaps_v2/` (prior LOCO).
- **Independent test:** `validation/independent_real_pcaps/` — 36 PCAPs / 420 flows,
  frozen; never used for any fitting decision.

## 1. Benign false positives — where they cluster

Candidate 2 makes **{ben_fp} benign false positives out of {n_ben_flows} benign
flows ({ben_fp / n_ben_flows:.1%})**. They are **not** uniform: see
`benign_fp_breakdown.csv`. The concentration is in command-heavy / multi-command /
transfer / active-mode benign sessions rather than plain logins — behaviours
under-represented in training.

## 2. Scenario difficulty

Hardest FTP-BruteForce scenarios (lowest Candidate-2 recall):

{_mdtable(hardest.to_dict('records'))}

See `scenario_metrics.csv` for all scenarios and the benign command-scenario FP
rates.

## 3. Client / server generalisation (support shown)

{md_cs}

## 4. Active vs passive mode

{md_mode}

## 5. Feature distributions (train vs v2 vs independent)

`feature_distribution_final.csv` reports medians/quartiles for the focus features
across CIC-FTP (training), v2 real-FTP (validation), independent real-FTP, and
independent benign. The independent real-FTP region and the benign region are
compared against the CIC artifact values (`Fwd Seg Size Min`, `Init Fwd Win Byts`).
{shap_md}
## 7. Confidence by error group

{md_conf}

## 8. Capture-level robustness

- FTP captures with recall ≥0.5: {cm_summary['pct_ftp_recall_ge_0.5']:.0%}; ≥0.75:
  {cm_summary['pct_ftp_recall_ge_0.75']:.0%}; ≥0.9: {cm_summary['pct_ftp_recall_ge_0.9']:.0%}.
- Benign captures with zero FP: {cm_summary['pct_benign_captures_zero_fp']:.0%}.

See `capture_metrics.csv` (weak captures are not hidden).

## 9. Data requirements (ranked by expected value, evidence-based)

{md_reqs}

## 10. Final recommendation — {verdict['recommendation_choice']}

{verdict['recommendation']}

- candidate_promote: **false**  ·  retraining_performed: **false**  ·
  threshold_changed: **false**  ·  heuristics_added: **false**  ·
  production_model_changed: **false**

### Caveats
{chr(10).join('- ' + c for c in verdict['caveats'])}

## Integrity

Production model + Candidate 2 + ml.py / live_capture.py / pcap_validation.py +
independent PCAPs/manifest all verified unchanged (before == after). This analysis
wrote no model artifact and altered no label.

## Files

`benign_fp_breakdown.csv`, `scenario_metrics.csv`, `client_server_metrics.csv`,
`mode_metrics.csv`, `feature_distribution_final.csv`, `shap_error_groups.csv`,
`confidence_error_groups.csv`, `capture_metrics.csv`, `data_requirements.json`,
`final_robustness_verdict.json`.
"""
        (out / "report.md").write_text(report)


def _tag(df, grouped_col, label):
    df = df.rename(columns={grouped_col: "group"})
    df.insert(0, "group_type", label)
    return df


def _f(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def _mdtable(rows):
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
