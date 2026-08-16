"""
Collect the TARGETED benign real-PCAP corpus (experimental; models frozen).

Aims at the structured benign false positives found in final_robustness: active
mode, command-heavy sessions, transfers (upload/download/append/delete/mixed),
reconnect / multi-session, across a raw-socket FTP server + pyftpdlib (permissive
/ throttled). Fresh real captures only; loopback / host-local lab; labels from the
scenario folder. NEW addresses/ports keep it disjoint from v1/v2/independent.

    # from webapp_django/ (tcpdump needs root; controlled local lab only)
    python manage.py collect_targeted_benign

Does NOT retrain, and does not modify ml.py / live_capture.py / pcap_validation.py
/ production model / Candidate 2 / existing validation results.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from django.core.management.base import BaseCommand

from predictor import ml, targeted_capture as tc


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / ".git").is_dir():
            return p
    raise RuntimeError("repo root not found")


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _pcap_hashes(root: Path) -> set:
    return {_sha(p) for p in Path(root).rglob("*.pcap")}


class Command(BaseCommand):
    help = "Collect the targeted benign real-PCAP corpus (frozen models; no retraining)."

    def add_arguments(self, parser):
        parser.add_argument("--output", default=None)
        parser.add_argument("--limit", type=int, default=0)

    def handle(self, *args, **opts):
        w = self.stdout.write
        root = _repo_root()
        out = Path(opts["output"]) if opts["output"] else root / "validation" / "targeted_benign_pcaps"
        (out / "benign").mkdir(parents=True, exist_ok=True)
        (out / "metadata").mkdir(parents=True, exist_ok=True)

        targets = tc.build_targets(out / ".labhome")
        for t in targets.values():
            if not t.is_controlled_local():
                raise SystemExit(f"REFUSING: {t.addr} is not a controlled local address.")

        w(self.style.MIGRATE_HEADING("Targeted benign real-PCAP collection"))
        w(f"  environments: {tc.environments()}")
        w(f"  servers: custom(raw-socket), pyftpdlib(permissive/throttled)  ports={tc.SERVER_PORT}")
        w(f"  clients: python-ftplib, curl({shutil.which('curl')}), wget({shutil.which('wget')})")
        w(self.style.WARNING("  Aimed at the structured benign FPs from final_robustness. No retraining."))
        if os.geteuid() != 0:
            w(self.style.WARNING("  note: tcpdump usually needs root."))

        specs = tc.benign_specs()
        if opts["limit"]:
            specs = specs[:opts["limit"]]

        pool = tc.ServerPool(targets, out / ".labhome")
        metas, i = [], 0
        try:
            for spec in specs:
                i += 1
                metas.append(self._capture(w, spec, pool, out, i))
        finally:
            pool.stop_all()
            w("  all lab servers stopped.")

        manifest = pd.DataFrame([m.as_row() for m in metas], columns=tc.MANIFEST_COLUMNS)
        manifest.to_csv(out / "MANIFEST.csv", index=False)
        n_valid = sum(m.verification_status == "valid" for m in metas)
        w(self.style.SUCCESS(f"\n  manifest: {len(metas)} captures, {n_valid} verified valid"))

        extraction = self._extract(w, metas, out)
        leakage = self._leakage(w, root, out, metas)
        self._feature_distribution(root, out)
        self._report(out, metas, extraction, leakage)
        w(self.style.SUCCESS(f"\nReport: {out/'collection_report.md'}"))
        w(self.style.WARNING("\nSTOP: targeted benign corpus collected + validated. No retraining."))

    def _capture(self, w, spec, pool, out, idx):
        target = pool.get(spec.env, spec.server)
        cap_id = f"benign_{idx:02d}"
        pcap = out / "benign" / f"{cap_id}_{spec.server}_{spec.scenario}.pcap"
        cap = tc.Tcpdump(pcap_path=pcap, target=target)
        ts = datetime.now(timezone.utc).isoformat()
        cap.start()
        client_error = None
        try:
            stats = spec.fn(target) if spec.mode == "passive" else spec.fn(target, passive=False)
        except Exception as e:  # noqa: BLE001 - a client-side command error still leaves real captured traffic
            stats = {"attempts": 1, "client_error": str(e)}
            client_error = str(e)
        finally:
            cap.stop()
        chk = tc.verify_pcap(pcap, target)
        status = "valid" if chk.ok else "INVALID:" + ";".join(chk.errors)
        cmd = f"tcpdump -i {target.iface} -w <pcap> -U -n '{target.bpf()}'"
        meta = tc.CaptureMeta(
            capture_id=cap_id, scenario=spec.scenario, label="Benign", client=spec.client,
            server=f"{spec.server}" + ("(custom-raw-socket)" if spec.server == "custom" else "(pyftpdlib)"),
            environment=f"{spec.env} ({target.addr})", interface=target.iface, mode=spec.mode,
            connection_pattern=spec.pattern, attempts=int(stats.get("attempts", 0)),
            duration_s=round(chk.duration_s, 4), packet_count=chk.packets, capture_timestamp=ts,
            source=target.addr, destination=f"{target.addr}:{target.control_port}",
            capture_command=cmd, verification_status=status,
            detail={**stats, "wall_duration_s": round(cap.wall_duration, 3),
                    "check": {k: v for k, v in vars(chk).items() if k != "errors"},
                    "check_errors": chk.errors})
        (out / "metadata" / f"{cap_id}.json").write_text(
            json.dumps({**meta.as_row(), "detail": meta.detail}, indent=2, default=str))
        flag = self.style.SUCCESS("ok") if chk.ok else self.style.ERROR("FAIL")
        w(f"  [{flag}] {cap_id:10} {spec.server:11} {spec.mode:7} {spec.scenario:30} "
          f"pkts={chk.packets:4} dur={chk.duration_s:6.2f}")
        if not chk.ok:
            w(self.style.ERROR(f"        {chk.errors}"))
        return meta

    def _extract(self, w, metas, out):
        from predictor import pcap_validation as pv, live_capture
        live_capture._ensure_live_on_path()
        w(self.style.MIGRATE_HEADING("\nFeature extraction (existing pipeline, unchanged)"))
        rows, total, bad = [], 0, 0
        for m in metas:
            pcap = next((out / "benign").glob(f"{m.capture_id}_*.pcap"))
            n = ok = inc = 0; order_ok = True; probs = []
            try:
                flows = pv.replay_pcap(pcap)
            except Exception as e:  # noqa: BLE001
                rows.append({"capture_id": m.capture_id, "flows": 0, "valid_flows": 0,
                             "incomplete_flows": 0, "feature_order_ok": None, "extraction_error": str(e)})
                continue
            for fl in flows:
                n += 1
                problem = pv._feature_problem(fl["features"])
                if problem:
                    inc += 1; probs.append(problem); continue
                if [f for f in ml.FEATURES if f in fl["features"]] != list(ml.FEATURES):
                    order_ok = False
                ok += 1
            total += n; bad += inc
            rows.append({"capture_id": m.capture_id, "flows": n, "valid_flows": ok,
                         "incomplete_flows": inc, "feature_order_ok": order_ok,
                         "incomplete_detail": ";".join(sorted(set(probs))), "extraction_error": ""})
        pd.DataFrame(rows).to_csv(out / "feature_extraction_report.csv", index=False)
        w(f"  total flows: {total}, incomplete (reported, not zero-filled): {bad}")
        return {"per_capture": rows, "total_flows": int(total), "total_incomplete": int(bad),
                "n_features": len(ml.FEATURES)}

    def _leakage(self, w, root, out, metas):
        new = _pcap_hashes(out / "benign")
        v1 = _pcap_hashes(root / "validation/realistic_pcaps")
        v2 = _pcap_hashes(root / "validation/realistic_pcaps_v2")
        indep = _pcap_hashes(root / "validation/independent_real_pcaps")
        checks = {
            "disjoint_from_v1": new.isdisjoint(v1),
            "disjoint_from_v2": new.isdisjoint(v2),
            "disjoint_from_independent": new.isdisjoint(indep),
            "no_duplicate_within_corpus": len(new) == len(list((out / "benign").glob("*.pcap"))),
            "labels_only_from_folders": all(m.label == "Benign" for m in metas),
            "all_verified_valid": all(m.verification_status == "valid" for m in metas),
        }
        res = {"all_pass": all(checks.values()), "checks": checks,
               "n_new": len(new), "n_v1": len(v1), "n_v2": len(v2), "n_independent": len(indep)}
        (out / "leakage_validation.json").write_text(json.dumps(res, indent=2, default=str))
        w(self.style.MIGRATE_HEADING("\nLeakage / disjointness checks"))
        w(f"  all_pass={res['all_pass']}  " + " ".join(f"{k}={v}" for k, v in checks.items()))
        return res

    def _feature_distribution(self, root, out):
        """Diagnostic: does the new benign corpus populate the FP-prone benign region?
        Compares the new targeted benign against the independent benign and the
        independent FP-benign flows (read-only) on the FP-driving features."""
        from predictor import robustness_analysis as ra, pcap_validation as pv, live_capture
        live_capture._ensure_live_on_path()
        feats = ["Fwd Seg Size Min", "Init Fwd Win Byts", "Dst Port", "Flow Pkts/s",
                 "Flow Duration", "Fwd Pkts/s", "Pkt Len Mean"]
        # new targeted benign flows
        rows = []
        for pcap in sorted((out / "benign").glob("*.pcap")):
            for fl in pv.replay_pcap(pcap):
                if pv._feature_problem(fl["features"]) is None:
                    rows.append({f: float(fl["features"][f]) for f in ml.FEATURES})
        new_df = pd.DataFrame(rows)
        # independent benign + FP-benign region (diagnosis only, read-only)
        try:
            frame = ra.build_frame()
            ind_ben = frame[frame["Label"] == "Benign"]
            ind_fp = frame[frame["error_group"] == "FP_Benign"]
        except Exception:  # noqa: BLE001
            ind_ben = ind_fp = pd.DataFrame(columns=ml.FEATURES)

        def med(frame, f):
            if f not in frame.columns or len(frame) == 0:
                return None
            return float(pd.to_numeric(frame[f], errors="coerce").dropna().median())

        drows = []
        for f in feats:
            drows.append({"feature": f,
                          "targeted_benign_median": med(new_df, f),
                          "independent_benign_median": med(ind_ben, f),
                          "independent_FP_benign_median": med(ind_fp, f)})
        pd.DataFrame(drows).to_csv(out / "feature_distribution.csv", index=False)

    def _report(self, out, metas, extraction, leakage):
        def dist(key):
            return dict(Counter(getattr(m, key) for m in metas))
        summary = {
            "experiment": "targeted_benign_pcap_collection",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "Targeted benign corpus for the structured benign FPs in final_robustness.",
            "retrained": False, "labels_from": "scenario folder (never predictions)",
            "controlled_local_only": True,
            "pcap_count": len(metas), "benign_count": len(metas),
            "captures_passing_verification": sum(m.verification_status == "valid" for m in metas),
            "total_flows": extraction["total_flows"],
            "total_incomplete_flows": extraction["total_incomplete"],
            "n_model_features": extraction["n_features"],
            "diversity": {"by_scenario": len(set(m.scenario for m in metas)),
                          "by_client": dist("client"), "by_server": dist("server"),
                          "by_environment": dist("environment"), "by_mode": dist("mode"),
                          "by_connection_pattern": dist("connection_pattern")},
            "leakage": leakage,
            "targets_the_fp_categories": ["active-mode benign", "command-heavy sessions",
                                          "upload/download/append/delete/mixed transfers",
                                          "reconnect / multi-session"],
            "limitations": [
                "Loopback / host-local lab only (single host; same-host traffic stays on lo).",
                "Two server implementations (custom raw-socket + pyftpdlib configs); no third-party daemon; no FTPS/TLS.",
                "Benign-only corpus by design (targets the benign FP problem).",
                "For FUTURE evaluation/retraining decisions only after a fresh independent test; not used to tune anything here.",
            ],
            "versions": {"python": platform.python_version()},
        }
        (out / "collection_report.json").write_text(json.dumps(summary, indent=2, default=str))

        flowmap = {r["capture_id"]: r for r in extraction["per_capture"]}
        lines = []
        for m in metas:
            fr = flowmap.get(m.capture_id, {})
            lines.append(f"| {m.capture_id} | {m.scenario} | {m.client} | {m.server} | "
                         f"{m.environment} | {m.mode} | {m.connection_pattern} | {m.packet_count} | "
                         f"{m.duration_s:.2f} | {fr.get('flows','?')} | {fr.get('incomplete_flows','?')} | "
                         f"{m.verification_status} |")
        hdr = ("| id | scenario | client | server | env | mode | pattern | pkts | dur_s | flows | "
               "incomplete | verify |\n|" + "---|" * 12)
        md = f"""# Targeted benign real-PCAP corpus — collection report

**Collected to address the STRUCTURED benign false positives diagnosed in
`final_robustness` (real benign traffic entering the model's CIC-artifact-based FTP
region). No retraining, no model change, no threshold change.** Ground-truth labels
come only from the scenario folder. All traffic stayed inside the controlled local
lab; no external host was contacted. This corpus is content-hash **disjoint from
v1, v2, and the independent 36-PCAP test set**.

## Totals

| Metric | Value |
|---|---|
| Benign PCAPs | **{len(metas)}** |
| Verified valid | {summary['captures_passing_verification']} / {len(metas)} |
| Total flows (existing pipeline) | **{extraction['total_flows']}** |
| Incomplete flows (reported, not zero-filled) | {extraction['total_incomplete']} |
| Model features per flow | {extraction['n_features']} (exact order) |

## Diversity

- **Scenarios:** {summary['diversity']['by_scenario']} distinct
- **Clients:** {json.dumps(summary['diversity']['by_client'])}
- **Servers:** {json.dumps(summary['diversity']['by_server'])}
- **Environments:** {json.dumps(summary['diversity']['by_environment'])}
- **Mode:** {json.dumps(summary['diversity']['by_mode'])}
- **Connection patterns:** {json.dumps(summary['diversity']['by_connection_pattern'])}

Directly targets the FP categories: active-mode benign, command-heavy sessions
(MKD/RMD/RNFR/RNTO/DELE/APPE/SIZE/MDTM/NLST/STAT, multi-command, command-workout),
transfers (download/upload/append/delete/mixed), reconnect / multi-session.

## Leakage / disjointness — all pass: {leakage['all_pass']}

Content-hash disjoint from v1 ({leakage['n_v1']}), v2 ({leakage['n_v2']}), and the
independent set ({leakage['n_independent']}); no duplicates; labels from folders
only; all captures verified. See `leakage_validation.json`.

## Captures

{hdr}
{chr(10).join(lines)}

## Feature-distribution diagnostic

`feature_distribution.csv` compares the new targeted benign flows against the
independent benign and the independent FP-benign flows on the FP-driving features
(`Fwd Seg Size Min`, `Init Fwd Win Byts`, `Dst Port`, `Flow Pkts/s`, ...). This is
a read-only diagnostic to show the new corpus populates the FP-prone benign region;
it is NOT used to tune anything.

## Limitations

{chr(10).join('- ' + x for x in summary['limitations'])}

## Files

`MANIFEST.csv`, `benign/*.pcap`, `metadata/*.json`, `feature_extraction_report.csv`,
`feature_distribution.csv`, `leakage_validation.json`, `collection_report.json`,
`verify_pcaps.py`.
"""
        (out / "collection_report.md").write_text(md)
