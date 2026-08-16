"""
Collect REAL, controlled realistic PCAPs in the local loopback lab (experimental).

Runs a real pyftpdlib target on 127.0.0.1, captures with real tcpdump on the
loopback interface, drives real benign and brute-force FTP sessions with a real
ftplib client, verifies every resulting PCAP, records metadata + a manifest, and
runs the EXISTING validated feature-extraction pipeline (read-only) to confirm
each capture yields exactly the 30 finite model features. It does NOT retrain and
NEVER uses model predictions as labels — ground truth is the scenario folder.

    # from webapp_django/ (needs root for tcpdump; loopback only)
    python manage.py collect_realistic_pcaps

Output tree: validation/realistic_pcaps/{benign,ftp_bruteforce,metadata}/ +
MANIFEST.csv + collection_report.{md,json}.
"""

from __future__ import annotations

import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from django.core.management.base import BaseCommand

from predictor import ml, lab_capture as lab


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "sample_data" / "real_pcap").is_dir() or (parent / ".git").is_dir():
            return parent
    raise RuntimeError("repo root not found")


class Command(BaseCommand):
    help = "Collect real local-lab FTP PCAPs for the realistic-PCAP expansion (no retraining)."

    def add_arguments(self, parser):
        parser.add_argument("--output", default=None,
                            help="output root (default validation/realistic_pcaps)")
        parser.add_argument("--control-port", type=int, default=21)

    def handle(self, *args, **opts):
        w = self.stdout.write
        root = _repo_root()
        out = Path(opts["output"]) if opts["output"] else root / "validation" / "realistic_pcaps"
        (out / "benign").mkdir(parents=True, exist_ok=True)
        (out / "ftp_bruteforce").mkdir(parents=True, exist_ok=True)
        (out / "metadata").mkdir(parents=True, exist_ok=True)

        target = lab.LabTarget(control_port=opts["control_port"])

        # --- identify the target + confine to the lab (pre-flight) --------
        w(self.style.MIGRATE_HEADING("Realistic-PCAP data collection (local loopback lab)"))
        w(f"  target: {target.describe()}")
        w(f"  client tool: python-ftplib   capture: tcpdump on {target.iface}")
        if not target.is_loopback():
            raise SystemExit(f"REFUSING: target {target.host} is not loopback. Lab traffic only.")
        if os.geteuid() != 0:
            w(self.style.WARNING("  note: tcpdump usually needs root; if capture fails, run as root."))

        # --- start + verify the FTP lab target (task 1) -------------------
        server = lab.FtpLabServer(target=target, root=out / ".labhome").start()
        if not server.verify():
            server.stop()
            raise SystemExit("FTP lab target failed verification (control connection).")
        w(self.style.SUCCESS("  FTP lab target is up and accepting logins."))

        metas: list[lab.CaptureMeta] = []
        try:
            metas += self._collect(w, target, "Benign", "benign",
                                   lab.BENIGN_SCENARIOS, out / "benign", out)
            metas += self._collect(w, target, "FTP-BruteForce", "ftpbf",
                                   lab.BRUTEFORCE_SCENARIOS, out / "ftp_bruteforce", out)
        finally:
            server.stop()
            w("  FTP lab target stopped.")

        # --- manifest (task 13) -------------------------------------------
        manifest = pd.DataFrame([m.as_row() for m in metas], columns=lab.MANIFEST_COLUMNS)
        manifest.to_csv(out / "MANIFEST.csv", index=False)
        w(self.style.SUCCESS(f"\n  manifest: {out / 'MANIFEST.csv'} ({len(manifest)} captures)"))

        # --- feature extraction on the NEW pcaps (task 15-16, read-only) --
        extraction = self._extract_and_validate(w, metas, out)

        # --- report (task 19) ---------------------------------------------
        self._write_report(out, metas, extraction, target)
        w(self.style.SUCCESS(f"\nReport written to {out/'collection_report.md'}"))
        w(self.style.WARNING(
            "\nSTOP: data collection + validation only. No retraining performed; "
            "labels come from the capture scenario, never from model predictions."))

    # -- collect one label's scenarios ------------------------------------

    def _collect(self, w, target, label, prefix, scenarios, dest_dir, out) -> list:
        w(self.style.MIGRATE_HEADING(f"\nCapturing {label} scenarios"))
        metas = []
        for i, (scenario, fn) in enumerate(scenarios, 1):
            cap_id = f"{prefix}_{i:02d}"
            pcap = dest_dir / f"{cap_id}_{scenario}.pcap"
            ts = datetime.now(timezone.utc).isoformat()

            # each capture = a fresh tcpdump + fresh real FTP interaction (task 10)
            cap = lab.Tcpdump(pcap_path=pcap, target=target).start()
            try:
                stats = fn(target)
            finally:
                cap.stop()

            chk = lab.verify_pcap(pcap, target)
            status = "valid" if chk.ok else "INVALID:" + ";".join(chk.errors)
            meta = lab.CaptureMeta(
                capture_id=cap_id, label=label, scenario=scenario, timestamp=ts,
                source=target.host,
                destination=f"{target.host}:{target.control_port}",
                client_tool="python-ftplib",
                attempts=int(stats.get("attempts", 0)),
                capture_duration_s=round(chk.duration_s, 4),
                pcap_filename=str(pcap.relative_to(out)),
                validation_status=status,
                detail={**stats, "packets": chk.packets,
                        "wall_duration_s": round(cap.wall_duration, 3),
                        "check": {k: v for k, v in vars(chk).items() if k != "errors"},
                        "check_errors": chk.errors},
            )
            metas.append(meta)
            (out / "metadata" / f"{cap_id}.json").write_text(
                json.dumps({**meta.as_row(), "detail": meta.detail}, indent=2, default=str))
            flag = self.style.SUCCESS("ok") if chk.ok else self.style.ERROR("FAIL")
            w(f"  [{flag}] {cap_id:10} {scenario:32} pkts={chk.packets:4} "
              f"dur={chk.duration_s:6.2f}s attempts={meta.attempts:2}")
            if not chk.ok:
                w(self.style.ERROR(f"        {chk.errors}"))
        return metas

    # -- run the existing extraction pipeline on the new pcaps ------------

    def _extract_and_validate(self, w, metas, out) -> dict:
        from predictor import pcap_validation as pv
        w(self.style.MIGRATE_HEADING("\nFeature extraction on new PCAPs "
                                     "(existing pipeline, unchanged, read-only)"))
        rows, total_flows, total_bad = [], 0, 0
        for m in metas:
            pcap = out / m.pcap_filename
            n_flows = n_ok = n_bad = 0
            order_ok = True
            problems = []
            try:
                flows = pv.replay_pcap(pcap)
            except Exception as e:  # noqa: BLE001
                rows.append({"capture_id": m.capture_id, "label": m.label,
                             "flows": 0, "valid_flows": 0, "incomplete_flows": 0,
                             "feature_order_ok": None, "extraction_error": str(e)})
                continue
            for fl in flows:
                n_flows += 1
                problem = pv._feature_problem(fl["features"])
                if problem:
                    n_bad += 1
                    problems.append(problem)
                    continue
                # exactly-30, finite already guaranteed by _feature_problem;
                # verify we can materialise them in the exact ml.FEATURES order
                ordered = [float(fl["features"][f]) for f in ml.FEATURES]
                if len(ordered) != len(ml.FEATURES):
                    order_ok = False
                n_ok += 1
            total_flows += n_flows
            total_bad += n_bad
            rows.append({"capture_id": m.capture_id, "label": m.label,
                         "flows": n_flows, "valid_flows": n_ok,
                         "incomplete_flows": n_bad, "feature_order_ok": order_ok,
                         "incomplete_detail": ";".join(sorted(set(problems))) or "",
                         "extraction_error": ""})
            w(f"  {m.capture_id:10} {m.label:14} flows={n_flows:3} valid={n_ok:3} "
              f"incomplete={n_bad}")
        df = pd.DataFrame(rows)
        df.to_csv(out / "feature_extraction_report.csv", index=False)
        w(f"  total flows: {total_flows}, incomplete (reported, not zero-filled): {total_bad}")
        return {"per_capture": rows, "total_flows": int(total_flows),
                "total_incomplete": int(total_bad),
                "n_features": len(ml.FEATURES),
                "feature_order": list(ml.FEATURES)}

    # -- report ------------------------------------------------------------

    def _write_report(self, out, metas, extraction, target):
        benign = [m for m in metas if m.label == "Benign"]
        bf = [m for m in metas if m.label == "FTP-BruteForce"]
        valid = [m for m in metas if m.validation_status == "valid"]
        flow_by_cap = {r["capture_id"]: r for r in extraction["per_capture"]}

        summary = {
            "experiment": "realistic_pcap_data_expansion",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "target": target.describe(),
            "loopback_only": target.is_loopback(),
            "retrained": False,
            "labels_from": "capture scenario folder (never model predictions)",
            "pcap_count": len(metas),
            "benign_count": len(benign),
            "ftp_bruteforce_count": len(bf),
            "captures_passing_verification": len(valid),
            "total_flows_extracted": extraction["total_flows"],
            "total_incomplete_flows_reported": extraction["total_incomplete"],
            "n_model_features": extraction["n_features"],
            "client_tool": "python-ftplib",
            "capture_tool": f"tcpdump -i {target.iface}",
            "server": "pyftpdlib",
            "versions": {"python": platform.python_version()},
            "limitations": [
                "Loopback lab only (single host/OS/network); not diverse real-world traffic.",
                "One FTP server implementation (pyftpdlib) and one client (ftplib).",
                "Small set — an expansion of the 5-PCAP baseline, still exploratory.",
            ],
        }
        (out / "collection_report.json").write_text(json.dumps(summary, indent=2, default=str))

        def _rows(ms):
            lines = []
            for m in ms:
                fr = flow_by_cap.get(m.capture_id, {})
                lines.append(
                    f"| {m.capture_id} | {m.scenario} | {m.attempts} | "
                    f"{m.detail.get('packets','?')} | {m.capture_duration_s:.2f} | "
                    f"{fr.get('flows','?')} | {fr.get('incomplete_flows','?')} | "
                    f"{m.validation_status} |")
            return "\n".join(lines)

        md = f"""# Realistic-PCAP data expansion — collection report

**Experimental data collection only. No retraining. Ground-truth labels come from
the capture scenario, never from model predictions.**

## Target (identified before any traffic)

- {target.describe()}
- Loopback only: **{target.is_loopback()}** — all traffic stayed inside the local lab.
- Server: `pyftpdlib`  ·  client: `python-ftplib`  ·  capture: `tcpdump -i {target.iface}`

## Totals

| Metric | Value |
|---|---|
| PCAP count | **{len(metas)}** ({len(benign)} benign, {len(bf)} FTP-bruteforce) |
| Captures passing verification | {len(valid)} / {len(metas)} |
| Total flows extracted (existing pipeline) | **{extraction['total_flows']}** |
| Incomplete flows (reported, not zero-filled) | {extraction['total_incomplete']} |
| Model features per flow | {extraction['n_features']} (exact order preserved) |

## Benign captures

| capture_id | scenario | attempts | packets | duration_s | flows | incomplete | validation |
|---|---|---|---|---|---|---|---|
{_rows(benign)}

## FTP-BruteForce captures

| capture_id | scenario | attempts | packets | duration_s | flows | incomplete | validation |
|---|---|---|---|---|---|---|---|
{_rows(bf)}

## Capture diversity

- **Benign:** {", ".join(m.scenario for m in benign)}.
- **Brute force:** {", ".join(m.scenario for m in bf)} — varying speed, username
  and password sets, attempt counts, and connection/session patterns.

## Feature extraction & validity (existing pipeline, unchanged)

Every capture was replayed through `pcap_validation.replay_pcap` (the validated
Live-Capture flow engine) with no changes. Each flow is checked for **exactly the
{extraction['n_features']} `ml.FEATURES`, in order, all finite**; any flow missing
a feature is **reported as incomplete, never zero-filled**. Per-capture counts are
in `feature_extraction_report.csv`.

## Independence & integrity

Each capture is an independent new network interaction — a fresh `tcpdump` plus
fresh FTP connections — never a copy, replay, or relabel of an existing PCAP. No
packets were edited after capture.

## Limitations

- Loopback lab only: a single host / OS / FTP stack, not diverse real-world
  traffic. This expands the 5-PCAP baseline but remains exploratory.
- One server (`pyftpdlib`) and one client (`ftplib`).

## Files

`MANIFEST.csv`, `benign/*.pcap`, `ftp_bruteforce/*.pcap`, `metadata/*.json`,
`feature_extraction_report.csv`, `collection_report.json`, this report.
"""
        (out / "collection_report.md").write_text(md)
