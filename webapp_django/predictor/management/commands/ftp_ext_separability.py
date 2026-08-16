"""
Separability analysis of the EXTENDED behavioural features (approved corpora only).

Answers, using ONLY the approved training corpora (v1/v2/targeted/robust_train) and the
scenario METADATA (folder/manifest -- never a model prediction), which new features
distinguish the three groups:

  1. benign mistype / give-up  (Benign, with failed logins)
  2. genuine FTP brute force    (FTP-BruteForce, keeps failing)
  3. attacker eventually succeeds (FTP-BruteForce, gets a 230)

For every feature it reports per-group distribution stats and a rank-based
separability score (AUC of a single feature) for the two hardest contrasts:
benign-fail vs brute, and benign-mistype vs attacker-success. Scenario labels are used
only to GROUP rows for the analysis; no feature is constructed from them. The frozen
independent vsFTPD corpus is NOT touched here.

    python manage.py ftp_ext_separability
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from django.core.management.base import BaseCommand

from predictor import ml, ftp_behavioral as fb, ftp_behavioral_ext as fbx, \
    retraining_behavioral_ext as rbe, pcap_validation as pv, live_capture

FTP, BENIGN = "FTP-BruteForce", "Benign"
APPROVED = ("v1", "v2", "targeted", "robust_train")
# scenario families that map to the three groups (metadata only, for grouping)
BENIGN_FAIL_FAMS = {"mistype", "gave_up", "typo_user"}
BRUTE_FAMS = {"all_fail", "slow_brute", "fast_brute", "user_enum", "multi_conn"}
EVENTUAL_FAMS = {"eventual_success"}


def _auc(pos, neg):
    """Rank-based AUC of a single feature separating pos (higher) from neg. NaN-safe."""
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    pos = pos[~np.isnan(pos)]; neg = neg[~np.isnan(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return None
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv), float); ranks[order] = np.arange(1, len(allv) + 1)
    r_pos = ranks[:len(pos)].sum()
    auc = (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return float(max(auc, 1 - auc))       # direction-agnostic separability


class Command(BaseCommand):
    help = "Separability analysis of the extended behavioural features (approved corpora only)."

    def add_arguments(self, parser):
        parser.add_argument("--output", default=None)

    def handle(self, *args, **opts):
        w = self.stdout.write
        root = rbe.repo_root()
        out = Path(opts["output"]) if opts["output"] else root / "validation" / "results" / "ftp_ext_separability"
        out.mkdir(parents=True, exist_ok=True)
        live_capture._ensure_live_on_path()

        w(self.style.MIGRATE_HEADING("Extended-behavioural separability (approved corpora only)"))
        rows = self._collect(root)
        df = pd.DataFrame(rows)
        w(f"  captures analysed: {len(df)}  (group counts: {df['group'].value_counts().to_dict()})")

        feats = list(fb.BEHAV_FEATURES) + list(fbx.EXT_FEATURES)
        # per-group distribution stats
        stat_rows = []
        for f in feats:
            for g in ("benign_fail", "brute", "attacker_success", "benign_clean"):
                s = df[df["group"] == g][f].dropna()
                stat_rows.append({"feature": f, "group": g, "n": int(len(s)),
                                  "mean": float(s.mean()) if len(s) else None,
                                  "median": float(s.median()) if len(s) else None,
                                  "std": float(s.std()) if len(s) else None})
        pd.DataFrame(stat_rows).to_csv(out / "per_group_feature_stats.csv", index=False)

        # separability AUCs for the two hard contrasts
        bf = df[df["group"] == "benign_fail"]; br = df[df["group"] == "brute"]; ev = df[df["group"] == "attacker_success"]
        sep_rows = []
        for f in feats:
            sep_rows.append({
                "feature": f, "is_new_ext": f in set(fbx.EXT_FEATURES),
                "auc_benign_fail_vs_brute": _auc(bf[f], br[f]),
                "auc_benign_fail_vs_attacker_success": _auc(bf[f], ev[f]),
                "auc_brute_vs_attacker_success": _auc(br[f], ev[f]),
                "benign_fail_median": float(bf[f].median()) if bf[f].notna().any() else None,
                "brute_median": float(br[f].median()) if br[f].notna().any() else None,
                "attacker_success_median": float(ev[f].median()) if ev[f].notna().any() else None})
        sep = pd.DataFrame(sep_rows).sort_values("auc_benign_fail_vs_brute", ascending=False, na_position="last")
        sep.to_csv(out / "separability_auc.csv", index=False)

        w(self.style.MIGRATE_HEADING("\nTop features: benign-fail vs brute (AUC, higher=more separable)"))
        for _, r in sep.head(12).iterrows():
            tag = "NEW" if r["is_new_ext"] else "   "
            w(f"  [{tag}] {r['feature']:32} AUC {_f(r['auc_benign_fail_vs_brute'])}  "
              f"(benign-fail med {_f(r['benign_fail_median'])} vs brute med {_f(r['brute_median'])})")

        w(self.style.MIGRATE_HEADING("\nTop features: benign-mistype vs attacker-eventual-success"))
        sep2 = sep.sort_values("auc_benign_fail_vs_attacker_success", ascending=False, na_position="last")
        for _, r in sep2.head(10).iterrows():
            tag = "NEW" if r["is_new_ext"] else "   "
            w(f"  [{tag}] {r['feature']:32} AUC {_f(r['auc_benign_fail_vs_attacker_success'])}")

        summary = self._summary(df, sep)
        (out / "separability_summary.json").write_text(json.dumps(summary, indent=2, default=str))
        self._report(out, df, sep, summary)
        w(self.style.MIGRATE_HEADING("\nKey finding"))
        w("  " + summary["headline"])
        w(self.style.SUCCESS(f"\nEvidence written to {out}"))

    def _collect(self, root):
        man_cache, rows = {}, []
        base = root / "validation"
        dmap = {"v1": "realistic_pcaps", "v2": "realistic_pcaps_v2", "targeted": "targeted_benign_pcaps",
                "robust_train": "robust_train_pcaps"}
        for src in APPROVED:
            d = base / dmap[src]
            man = pd.read_csv(d / "MANIFEST.csv") if (d / "MANIFEST.csv").is_file() else None
            fam_by_id = {}
            if man is not None and "scenario_family" in man.columns:
                fam_by_id = dict(zip(man["capture_id"], man["scenario_family"]))
            for folder, label in (("benign", BENIGN), ("ftp_bruteforce", FTP)):
                fd = d / folder
                if not fd.is_dir():
                    continue
                for pcap in sorted(fd.glob("*.pcap")):
                    cap_id = "_".join(pcap.name.split("_")[:2])
                    fam = fam_by_id.get(cap_id, "unknown")
                    behav = fb.behavioural_features_for_pcap(pcap)
                    ext = fbx.ext_features_for_pcap(pcap)
                    group = self._group(label, fam, behav)
                    rows.append({"source": src, "capture": pcap.name, "label": label,
                                 "scenario_family": fam, "group": group, **behav, **ext})
        return rows

    def _group(self, label, fam, behav):
        # grouping is from scenario metadata + measured auth outcome (NOT a model prediction)
        if label == BENIGN:
            if fam in BENIGN_FAIL_FAMS or behav["ftp_failed_logins"] > 0:
                return "benign_fail"
            return "benign_clean"
        if fam in EVENTUAL_FAMS or behav["ftp_has_successful_auth"] > 0:
            return "attacker_success"
        return "brute"

    def _summary(self, df, sep):
        new = sep[sep["is_new_ext"]]
        best_bf = new.dropna(subset=["auc_benign_fail_vs_brute"]).sort_values("auc_benign_fail_vs_brute", ascending=False).head(5)
        best_ev = new.dropna(subset=["auc_benign_fail_vs_attacker_success"]).sort_values("auc_benign_fail_vs_attacker_success", ascending=False).head(5)
        # is the timing signal usable in the current corpora?
        inter = df["ftpx_interattempt_mean_s"].dropna()
        timing_usable = bool(inter.gt(0.05).mean() > 0.2) if len(inter) else False
        headline = (
            "New EXT features most separating benign-fail from brute: "
            + ", ".join(f"{r['feature']}(AUC {r['auc_benign_fail_vs_brute']:.2f})" for _, r in best_bf.iterrows())
            + ". Inter-attempt TIMING is NOT usable on the existing corpora (attempts were back-to-back), "
              "so a realistic-pacing benign failed-login training corpus is needed to activate it."
            if not timing_usable else
            "New EXT features separate the groups and timing is present in the corpora.")
        return {"headline": headline,
                "best_new_features_benign_fail_vs_brute": best_bf[["feature", "auc_benign_fail_vs_brute"]].to_dict("records"),
                "best_new_features_benign_fail_vs_attacker_success": best_ev[["feature", "auc_benign_fail_vs_attacker_success"]].to_dict("records"),
                "timing_usable_in_current_corpora": timing_usable,
                "recommend_collect_realistic_pacing": not timing_usable,
                "group_counts": df["group"].value_counts().to_dict()}

    def _report(self, out, df, sep, summary):
        top = sep.head(12)
        tbl = "\n".join(f"| {'NEW' if r['is_new_ext'] else ''} | `{r['feature']}` | {_f(r['auc_benign_fail_vs_brute'])} | "
                        f"{_f(r['auc_benign_fail_vs_attacker_success'])} | {_f(r['benign_fail_median'])} | {_f(r['brute_median'])} | "
                        f"{_f(r['attacker_success_median'])} |" for _, r in top.iterrows())
        (out / "report.md").write_text(f"""# Extended-behavioural separability analysis

**Approved training corpora only (v1/v2/targeted/robust_train). Scenario metadata is
used only to GROUP rows; no feature is built from a label or a prediction. The frozen
independent vsFTPD corpus is NOT used here.**

Groups (from scenario metadata + measured auth outcome): benign_fail
({summary['group_counts'].get('benign_fail', 0)}), brute
({summary['group_counts'].get('brute', 0)}), attacker_success
({summary['group_counts'].get('attacker_success', 0)}), benign_clean
({summary['group_counts'].get('benign_clean', 0)}).

## Single-feature separability (direction-agnostic AUC)

| new | feature | AUC benign-fail vs brute | AUC benign-fail vs attacker-success | benign-fail med | brute med | attacker-succ med |
|---|---|---|---|---|---|---|
{tbl}

## Key finding

{summary['headline']}

- Best NEW features (benign-fail vs brute): {', '.join(f"{d['feature']} ({d['auc_benign_fail_vs_brute']:.2f})" for d in summary['best_new_features_benign_fail_vs_brute'])}
- Timing usable in current corpora: **{summary['timing_usable_in_current_corpora']}** ->
  recommend collecting realistic-pacing benign failed-login data: **{summary['recommend_collect_realistic_pacing']}**

## Files
`separability_auc.csv`, `per_group_feature_stats.csv`, `separability_summary.json`.
""")


def _f(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v:.3f}" if isinstance(v, float) else str(v)
